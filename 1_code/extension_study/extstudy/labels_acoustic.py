"""Acoustic-only S1 and S2 labels for the control arm (Experiment 2; plan 'Next Study - Beat-Accurate Synthetic
ECG/EXP2_PREREG.md').

The same paired records, usable spans, and validity rules as labels.py, but no electrocardiogram enters any mark or
rule. Beat identity comes from the Springer segmenter (published parameters; the primary heart-rate setting of arm S,
40 to 220 beats per minute) run on the phonocardiogram alone, and S1 and S2 are the envelope maxima inside Springer's
S1 and S2 states. The cycle start (the Springer S1-state onset) takes the place of the R-peak as the beat reference:
lags, laws, and the consistency rule are measured from it. The R target is undefined (the control network has no R
head). Record selection and the usable 60 s spans of long records are those of the teacher-label pipeline, so both
arms train on the same audio; only the labels differ.

Per record: harmonised PCG (2 kHz) -> 4 kHz -> Springer states at 1 kHz in windows of at most 60 s (the usable spans
of a long record, else consecutive windows) -> complete S1/systole/S2/diastole cycles -> beat frame (r = S1-state
onset, r_next = the cycle's end; RR, local HR, regularity) -> envelope at 500 Hz (pcgecg.signals.pcg_envelope, as
labels.py) -> S1 window = the S1 state widened by WIDEN_MS on each side, S2 window = the S2 state widened likewise,
both in ms after r -> labels.find_events with those windows (peaks, z-floors, dominance, rival, lag consistency,
validity). Laws (per dataset, lag = intercept + slope x HR, from the cycle start) are fitted on the valid beats of
training subjects with labels.fit_laws so that the interval warping of training works unchanged; the windows do not
depend on the laws, so one pass labels every record.

Outputs under derived/<version>/: <dataset>/<record>.npz (labels.py layout, r_target_s = NaN), laws.json,
labels_records.csv, labels_summary.csv; and events_<pack>.npz with the events of every record of a pack in the
pack's record order (plus record_id), for embedding in the training notebook.
Usage:  python -m extstudy.labels_acoustic [--version labels_acoustic_v1] [--pack paired_v3] [--workers 12]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.signal as sg

from extstudy import labels as LB
from extstudy.config import LOCAL

FS_ENV = LB.FS                       # 500 Hz envelope, as labels.py
FS_SPR = 4000                        # the rate at which the Springer port runs in the study
WIDEN_MS = 30.0                      # search window = Springer state widened by this on each side
WINDOW_S = 60.0                      # Springer runs in windows of at most this length
SPRINGER_VARIANT = 'S_wpk'           # arm S's primary setting (40 to 220 bpm, highest local maximum, systole floor 0.10 s)
EVENT_KEYS = ('r_s', 'r_next_s', 's1_s', 's2_s', 'r_target_s', 's1_on_s', 's1_off_s', 's2_on_s', 's2_off_s', 'hr', 'rr_ms', 'valid')

_PARAMS = None


def springer_params():
    global _PARAMS
    if _PARAMS is None:
        from extstudy.springer import springer_py as sp
        _PARAMS = sp.load_sample2016_params(os.path.join(os.path.dirname(__file__), 'springer', 'params'))
    return _PARAMS


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return np.asarray(x, np.float64)
    fr = Fraction(int(fs_out), int(fs_in))
    return sg.resample_poly(np.asarray(x, np.float64), fr.numerator, fr.denominator)


def springer_cycles(x4, spans_s=None, fs=FS_SPR, window_s=WINDOW_S, variant=SPRINGER_VARIANT, states_fn=None):
    """Complete Springer cycles (4 kHz samples) over the usable spans, or over consecutive windows when no spans are
    given; cycles that cross a window edge are lost. `states_fn(x4w, fs)` -> states at 1 kHz (injectable for tests)."""
    from extstudy.springer import springer_arm as SA
    if states_fn is None:
        from extstudy.springer import springer_py as sp
        params = springer_params()
        states_fn = lambda xw, f: np.asarray(sp.run_springer(np.asarray(xw, float), f, params, return_info=False, **SA.VARIANTS[variant]))
    n = len(x4)
    if spans_s:
        wins = [(int(a * fs), min(n, int(b * fs))) for a, b in spans_s]
    else:
        step = int(window_s * fs)
        wins = [(a, min(n, a + step)) for a in range(0, n, step)]
    cyc = []
    for a, b in wins:
        if b - a < fs:                                        # under a second: nothing to segment
            continue
        states = states_fn(x4[a:b], fs)
        for c in SA.states_to_cycles(states, b - a):
            cyc.append({k: (c[k][0] + a, c[k][1] + a) for k in ('s1', 'systole', 's2', 'diastole')})
    cyc = sorted(cyc, key=lambda c: c['s1'][0])
    return [dict(c, cycle=i) for i, c in enumerate(cyc)]


def beat_frame_from_cycles(cyc, n_env, trainable, fs=FS_SPR):
    """labels.beat_frame over the cycle starts (plus the last cycle's end as the closing beat), with the Springer S1 and
    S2 state edges per beat in ms after the cycle start."""
    if len(cyc) < 2:
        return pd.DataFrame()
    k = FS_ENV / fs
    starts = [int(round(c['s1'][0] * k)) for c in cyc] + [int(round(cyc[-1]['diastole'][1] * k))]
    bf = LB.beat_frame(np.asarray(starts, int), trainable)
    if bf.empty:
        return bf
    ms = 1000.0 / fs
    rows = [cyc[i] for i in bf['i']]
    bf = bf.copy()
    bf['s1_end_ms'] = [(c['s1'][1] - c['s1'][0]) * ms for c in rows]
    bf['s2_lo_ms'] = [(c['s2'][0] - c['s1'][0]) * ms for c in rows]
    bf['s2_hi_ms'] = [(c['s2'][1] - c['s1'][0]) * ms for c in rows]
    return bf


def find_acoustic_events(env, bf):
    """labels.find_events with the Springer state windows; no rule reads an R-peak."""
    old = LB.LBL['record_s1_min_median_ms']
    LB.LBL['record_s1_min_median_ms'] = 0.0                  # S1 peaks follow the state onset closely; no record rule on that lag
    try:
        return LB.find_events(env, bf,
                              lambda row: (-WIDEN_MS, row.s1_end_ms + WIDEN_MS),
                              lambda row, s1_ms: (row.s2_lo_ms - WIDEN_MS, row.s2_hi_ms + WIDEN_MS))
    finally:
        LB.LBL['record_s1_min_median_ms'] = old


def _record_inputs(row):
    """Envelope (500 Hz), audio at 4 kHz, and the trainable mask (usable spans of a long record; else everything)."""
    if LOCAL['pcgecg_code'] not in sys.path:
        sys.path.insert(0, LOCAL['pcgecg_code'])
    from pcgecg.signals import pcg_envelope
    z = np.load(os.path.join(LOCAL['harmonised'], row['dataset'], row['record'] + '.npz'))
    pcg = np.asarray(z['pcg'], np.float64)
    fs = int(z['fs_pcg'])
    env = pcg_envelope(pcg, fs=fs).astype(np.float64)
    x4 = resample(pcg, fs, FS_SPR)
    n = len(env)
    mask = np.ones(n, bool)
    spans = row.get('spans')
    if spans is not None:
        mask[:] = False
        for s0, e0 in spans:
            mask[int(s0 * FS_ENV):int(e0 * FS_ENV)] = True
    return env, x4, mask, spans


def label_record(row, states_fn=None):
    """One record: (row, result frame or None)."""
    env, x4, mask, spans = _record_inputs(row)
    cyc = springer_cycles(x4, spans_s=spans, states_fn=states_fn)
    bf = beat_frame_from_cycles(cyc, len(env), mask)
    if bf.empty:
        return row, None
    return row, find_acoustic_events(env, bf)


def _run_one(row):
    try:
        return label_record(row)
    except Exception as e:                                     # a failing record is reported, not fatal
        return row, 'error: %s: %s' % (type(e).__name__, e)


def write_record(out, row, res, laws_sha):
    d = os.path.join(out, row['dataset'])
    os.makedirs(d, exist_ok=True)
    to_s = lambda c: (res[c].astype(float) / FS_ENV).values
    np.savez_compressed(os.path.join(d, row['record'] + '.npz'),
                        r_s=to_s('r'), r_next_s=to_s('r_next'), s1_s=to_s('s1'), s2_s=to_s('s2'),
                        s1_on_s=to_s('s1_on'), s1_off_s=to_s('s1_off'), s2_on_s=to_s('s2_on'), s2_off_s=to_s('s2_off'),
                        r_target_s=np.full(len(res), np.nan), hr=res['hr'].values, rr_ms=res['rr'].values,
                        z1=res['z1'].values, z2=res['z2'].values, q1=res['q1'].values, q2=res['q2'].values,
                        bg=res['bg'].values, dev1=res['dev1'].values, dev2=res['dev2'].values, regular=res['regular'].values,
                        in_mask=res['in_mask'].values, s1_ok=res['s1_ok'].values, s2_ok=res['s2_ok'].values,
                        valid=res['valid'].values, laws_sha256=np.array(laws_sha))


def summary_row(row, res):
    if res is None or isinstance(res, str):
        return dict(dataset=row['dataset'], record=row['record'], subject=row.get('subject'), group=row.get('group'), role=row['role'],
                    n_beats=0, n_valid=0, valid_s=0.0, error=res if isinstance(res, str) else '')
    v = res[res['valid']]
    return dict(dataset=row['dataset'], record=row['record'], subject=row['subject'], group=row['group'], role=row['role'],
                n_beats=len(res), n_in_mask=int(res['in_mask'].sum()), n_regular=int((res['in_mask'] & res['regular']).sum()),
                n_s1_ok=int((res['in_mask'] & res['regular'] & res['s1_ok']).sum()), n_valid=int(len(v)),
                valid_s=float(v['rr'].sum() / 1000.0), hr_median=float(v['hr'].median()) if len(v) else np.nan,
                s1_ms_median=float(((v['s1'] - v['r']) / FS_ENV * 1000).median()) if len(v) else np.nan,
                s2_ms_median=float(((v['s2'] - v['r']) / FS_ENV * 1000).median()) if len(v) else np.nan, error='')


def events_for_pack(version, pack):
    """events_<pack>.npz under derived/<version>/: the events of every record of the pack in the pack's record order,
    with record_id; a record without labels contributes no rows. Returns the manifest dict (written beside it)."""
    base = os.path.join(LOCAL['phd'], 'derived', version)
    R = pd.read_csv(os.path.join(LOCAL['phd'], 'derived', 'corpus', pack, 'records.csv'))
    cols = {k: [] for k in EVENT_KEYS + ('record_id',)}
    n_valid, missing = [], []
    for k, r in enumerate(R.itertuples()):
        f = os.path.join(base, r.dataset, r.record + '.npz')
        if not os.path.exists(f):
            missing.append(r.record)
            n_valid.append(0)
            continue
        z = np.load(f)
        for key in EVENT_KEYS:
            cols[key].append(np.asarray(z[key]))
        cols['record_id'].append(np.full(len(z['r_s']), k, np.int32))
        n_valid.append(int(np.asarray(z['valid']).sum()))
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out['valid'] = out['valid'].astype(bool)
    fp = os.path.join(base, 'events_%s.npz' % pack)
    np.savez_compressed(fp, **out)
    sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
    man = dict(version=version, pack=pack, file='events_%s.npz' % pack, sha256=sha, n_records=len(R), n_records_with_labels=len(R) - len(missing),
               n_beats=int(len(out['r_s'])), n_valid=int(out['valid'].sum()), records_without_labels=missing,
               n_valid_per_record=n_valid)
    json.dump(man, open(os.path.join(base, 'events_%s_manifest.json' % pack), 'w'), indent=1)
    return man


def main(version='labels_acoustic_v1', pack='paired_v3', workers=12):
    from multiprocessing import Pool
    from extstudy import splits
    out = os.path.join(LOCAL['phd'], 'derived', version)
    os.makedirs(out, exist_ok=True)
    T, rows = LB.record_rows()
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_run_one, rows, chunksize=1)
    print('labelled %d records in %.0f min' % (len(rows), (time.time() - t0) / 60))
    fit = []
    for r, df in res:
        if df is None or isinstance(df, str) or r['role'] != 'train':
            continue
        v = df[df['valid']]
        fit.append(pd.DataFrame(dict(dataset=r['dataset'], group=r['group'], hr=v['hr'].values,
                                     s1_ms=((v['s1'] - v['r']) / FS_ENV * 1000).values, s2_ms=((v['s2'] - v['r']) / FS_ENV * 1000).values)))
    fit = pd.concat(fit, ignore_index=True)
    laws = LB.fit_laws(fit)
    laws['split_table_sha256'] = splits.table_hash(T.drop(columns=[c for c in ('spans',) if c in T]))
    laws['params'] = dict(LB.LBL, record_s1_min_median_ms=0.0, widen_ms=WIDEN_MS, window_s=WINDOW_S, springer_variant=SPRINGER_VARIANT,
                          beat_reference='Springer S1-state onset')
    laws_sha = LB.laws_hash(laws)
    json.dump(dict(laws, sha256=laws_sha), open(os.path.join(out, 'laws.json'), 'w'), indent=1)
    print('laws:', json.dumps({k: laws[k] for k in ('s1', 's2', 'n_beats', 'n_subjects')}, indent=1))
    summ = []
    for r, df in res:
        if df is not None and not isinstance(df, str):
            write_record(out, r, df, laws_sha)
        summ.append(summary_row(r, df))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, 'labels_records.csv'), index=False)
    agg = S.groupby(['dataset', 'role']).agg(records=('record', 'size'), beats=('n_beats', 'sum'), valid=('n_valid', 'sum'),
                                              hours=('valid_s', lambda s: s.sum() / 3600))
    pd.set_option('display.width', 200)
    print(agg.round(2).to_string())
    agg.to_csv(os.path.join(out, 'labels_summary.csv'))
    errors = S[S['error'].astype(str).str.len() > 0]
    if len(errors):
        print('records with errors: %d, e.g.' % len(errors), errors[['dataset', 'record', 'error']].head(5).to_string())
    man = events_for_pack(version, pack)
    print('events for pack %s: %d beats, %d valid, %d records without labels (sha256 %s)' % (
        pack, man['n_beats'], man['n_valid'], len(man['records_without_labels']), man['sha256'][:16]))
    return laws, S, man


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='labels_acoustic_v1')
    ap.add_argument('--pack', default='paired_v3')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    main(a.version, a.pack, a.workers)
