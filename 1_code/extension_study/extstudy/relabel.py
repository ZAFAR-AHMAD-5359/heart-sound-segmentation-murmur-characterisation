"""Relabel every paired record with the teacher rule (qrs.py) and rebuild the usability flags from the new beats.

Stage 1 (per record): one ECG lead per dataset (LEAD_ORDER; the next lead only when the first fails the ECG check),
teacher beats, label mask, S1-lag gap of S-dominant against R-dominant beats, synchronisation statistics on the
labelled beats for the record and for every 60 s window, agreement with the first-audit consensus labels.
Writes derived/beats_v2/<dataset>/<record>.npz, beats_records.csv, beats_windows.csv.

Stage 2 (table only): S1-lag centres from development records, the calibrated synchronisation rule per record
(records under 120 s) or per 60 s window (longer records, 08 design), usable flags and labelled seconds, the
negative control (PCG of one subject against the beats of another), and a summary per dataset.
Writes derived/beats_v2/audit_v2.csv, negative_control_v2.csv, summary_v2.csv.

Usage:  python -m extstudy.relabel [--datasets a,b] [--limit N] [--workers W] [--stage 1|2|both]
"""
import argparse, json, os, sys, time, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd

from extstudy.config import BEATS, EXCLUDE_RECORDS, LEAD_ORDER, LOCAL
from extstudy import qrs

if LOCAL['pcgecg_code'] not in sys.path:
    sys.path.insert(0, LOCAL['pcgecg_code'])
from pcgecg import readers as R            # noqa: E402
from pcgecg import signals as S            # noqa: E402
from pcgecg.refresh_sync import SYNC_EFFECT, SYNC_LAG_HALFWIDTH, SYNC_IQR   # noqa: E402

OUT = LOCAL['beats']
FS = qrs.FS
LONG_S = 120.0                              # records at least this long get 60 s window flags
ECG_OK = dict(min_beats=5, min_coverage=0.50, min_confirmed=0.90)


def ecg_ok(rec):
    conf = 1.0 - rec['n_robust_rejected'] / max(1, rec['n_robust'])
    return bool(rec['n_beats'] >= ECG_OK['min_beats'] and rec['label_coverage'] >= ECG_OK['min_coverage']
                and conf >= ECG_OK['min_confirmed'])


def _leads(row, z):
    """Candidate leads at 500 Hz, in the fixed order for this dataset."""
    if row['dataset'] not in LEAD_ORDER:
        return [(str(z['ecg_lead']), np.asarray(z['ecg'], np.float64))]
    key = tuple(json.loads(row['key']))
    rec = [r for r in R.load(key) if r.record == row['record']][0]
    pool = {rec.ecg_lead: rec.ecg, **rec.ecg_alt}
    return [(name, S.resample(pool[name], rec.fs_ecg, FS)) for name in LEAD_ORDER[row['dataset']] if name in pool]


def _sync(env, beats):
    m, _ = S.sync_metrics(env, beats)
    return {k: m[k] for k in ('n_beats_used', 's1_lag_ms', 'lock_effect', 'perbeat_lag_iqr_ms', 'max_jump_ms', 'drift_ms')}


def process(row):
    t0 = time.time()
    ds, record = row['dataset'], row['record']
    try:
        z = np.load(os.path.join(LOCAL['harmonised'], ds, record + '.npz'), allow_pickle=False)
        env = S.pcg_envelope(z['pcg'])
        leads = _leads(row, z)
        chosen, tried = None, []
        for name, e in leads:
            t = qrs.teacher_beats(e, FS)
            ok = ecg_ok(t['rec'])
            tried.append('%s:%s' % (name, 'ok' if ok else 'fail'))
            if chosen is None:
                chosen = (name, t, ok)               # the primary lead is kept unless a later lead passes
            if ok:
                chosen = (name, t, ok)
                break
        lead, t, ok = chosen
        n = min(len(env), len(t['label_mask']))
        b, m = t['beats'], t['label_mask'][:n]
        b_in = b[b < n]
        lab = b_in[m[b_in]]
        gap, n_s = qrs.lag_gap_ms(env[:n], b_in, t['s_dominant'][:len(b_in)], keep=m[b_in])
        out = dict(dataset=ds, record=record, subject=row['subject'], lead=lead, lead_primary=leads[0][0],
                   lead_fallback=bool(lead != leads[0][0]), leads_tried=';'.join(tried), ecg_ok_v2=ok,
                   n_labelled_beats=int(len(lab)), lag_gap_ms=gap, n_s_dominant_labelled=n_s,
                   meta_split=row.get('meta_split', np.nan), duration_s=row['duration_s'])
        out.update({k: v for k, v in t['rec'].items()})
        out.update({'sync_' + k: v for k, v in _sync(env[:n], lab).items()})
        # agreement with the first-audit consensus labels (same lead only)
        v1 = np.asarray(z['rpeaks'], int)
        if lead == str(z['ecg_lead']) and len(v1) and len(b):
            ia, ib = qrs.match_pairs(v1, b, int(0.05 * FS))
            off = (v1[ia] - b[ib]) / FS * 1000
            out.update(v1_f1=qrs.f1_of(v1, b, int(0.05 * FS)), v1_minus_v2_med_ms=float(np.median(off)) if len(off) else np.nan,
                       v1_far15_share=float(np.mean(np.abs(off - np.median(off)) > 15)) if len(off) else np.nan)
        # 60 s windows: synchronisation on the labelled beats inside each window
        W = int(60 * FS)
        wins = []
        if n / FS >= LONG_S:
            for k, s0 in enumerate(range(0, n - W // 2, W)):
                e0 = min(n, s0 + W)
                bw = lab[(lab >= s0) & (lab < e0)]
                w = dict(dataset=ds, record=record, window=k, start_s=s0 / FS, end_s=e0 / FS,
                         label_share=float(m[s0:e0].mean()), n_labelled_beats=int(len(bw)))
                w.update(_sync(env[:n], bw))
                wins.append(w)
        d = os.path.join(OUT, ds); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, record + '.npz'), beats=b, source=t['source'], r_signed=t['r_signed'],
                            corr=t['corr'], s_dominant=t['s_dominant'], conf=t['conf'],
                            nolabel_intervals=qrs.mask_to_intervals(t['label_mask']), n_samples=np.int64(len(t['label_mask'])),
                            fs=np.int64(FS), lead=np.array(lead), detector_sha256=np.array(qrs.FROZEN_SHA256),
                            win_start=t['windows']['start'], win_end=t['windows']['end'],
                            win_f1_xg=t['windows']['f1_xg'], win_f1_rx=t['windows']['f1_rx'])
        out['seconds'] = round(time.time() - t0, 1)
        return out, wins
    except Exception as e:
        return dict(dataset=ds, record=record, error='%s: %s' % (type(e).__name__, e), trace=traceback.format_exc()[-800:]), []


# ------------------------------------------------------------------ stage 2
def sync_pass(lock_effect, lag, iqr, centre):
    return (lock_effect >= SYNC_EFFECT) & ((lag - centre).abs() <= SYNC_LAG_HALFWIDTH) & (iqr <= SYNC_IQR)


def stage2(seed=7, n_null=80):
    B = pd.read_csv(os.path.join(OUT, 'beats_records.csv'), low_memory=False)
    Wn = pd.read_csv(os.path.join(OUT, 'beats_windows.csv'), low_memory=False) if os.path.exists(os.path.join(OUT, 'beats_windows.csv')) else pd.DataFrame()
    A = pd.read_csv(LOCAL['audit_table'], low_memory=False)
    B = B.merge(A[['dataset', 'record', 'pcg_ok']], on=['dataset', 'record'], how='left')
    ok = B['error'].isna() if 'error' in B else pd.Series(True, index=B.index)
    excluded = B.apply(lambda r: (r['dataset'], r['record']) in EXCLUDE_RECORDS, axis=1)
    held_out = B['meta_split'].astype(str).eq('community_test')
    clean = ok & B['ecg_ok_v2'].astype(bool) & B['pcg_ok'].astype(bool) & (B['sync_lock_effect'] >= SYNC_EFFECT) & ~held_out & ~excluded
    centre = B[clean].groupby('dataset')['sync_s1_lag_ms'].median().to_dict()
    json.dump(centre, open(os.path.join(OUT, 'sync_lag_centres_v2.json'), 'w'), indent=1)
    B['s1_lag_centre_ms'] = B['dataset'].map(centre)
    B['long'] = B['duration_s'] >= LONG_S
    B['sync_ok_v2'] = sync_pass(B['sync_lock_effect'], B['sync_s1_lag_ms'], B['sync_perbeat_lag_iqr_ms'], B['s1_lag_centre_ms'])
    # long records: usable 60 s windows
    if len(Wn):
        Wn['centre'] = Wn['dataset'].map(centre)
        Wn['window_ok'] = (sync_pass(Wn['lock_effect'], Wn['s1_lag_ms'], Wn['perbeat_lag_iqr_ms'], Wn['centre'])
                           & (Wn['label_share'] >= 0.5) & (Wn['n_labelled_beats'] >= 10))
        Wn['usable_labelled_s'] = np.where(Wn['window_ok'], (Wn['end_s'] - Wn['start_s']) * Wn['label_share'], 0.0)
        Wn.to_csv(os.path.join(OUT, 'beats_windows.csv'), index=False)
        agg = Wn.groupby(['dataset', 'record']).agg(n_windows_ok=('window_ok', 'sum'), n_windows=('window_ok', 'size'),
                                                     usable_labelled_s=('usable_labelled_s', 'sum'),
                                                     usable_beats=('n_labelled_beats', lambda s: int(s[Wn.loc[s.index, 'window_ok']].sum())))
        B = B.merge(agg.reset_index(), on=['dataset', 'record'], how='left')
    short = ~B['long']
    base = ok & ~excluded & B['pcg_ok'].astype(bool) & B['ecg_ok_v2'].astype(bool)
    B['usable_v2'] = base & np.where(short, B['sync_ok_v2'], B.get('n_windows_ok', 0).fillna(0) > 0)
    B.loc[short, 'usable_labelled_s'] = np.where(B.loc[short, 'usable_v2'], B.loc[short, 'duration_s'] * B.loc[short, 'label_coverage'], 0.0)
    B.loc[short, 'usable_beats'] = np.where(B.loc[short, 'usable_v2'], B.loc[short, 'n_labelled_beats'], 0)
    B.loc[~B['usable_v2'], ['usable_labelled_s', 'usable_beats']] = 0
    B.to_csv(os.path.join(OUT, 'audit_v2.csv'), index=False)

    # negative control: PCG of one record against the labelled beats of another subject in the same dataset
    rng = np.random.default_rng(seed)
    G = B[base].reset_index(drop=True)
    rows = []
    for d, g in G.groupby('dataset'):
        g = g.reset_index(drop=True)
        for i in rng.choice(len(g), size=min(n_null, len(g)), replace=False):
            a = g.loc[i]
            others = g[g.subject.astype(str) != str(a.subject)]
            if not len(others):
                continue
            bb = others.iloc[int(rng.integers(len(others)))]
            env = S.pcg_envelope(np.load(os.path.join(LOCAL['harmonised'], d, a.record + '.npz'))['pcg'])
            zb = np.load(os.path.join(OUT, d, bb.record + '.npz'))
            mask = qrs.intervals_to_mask(zb['nolabel_intervals'], int(zb['n_samples']))
            beats = zb['beats'][(zb['beats'] < min(len(env), len(mask)))]
            beats = beats[mask[beats]]
            if a['long']:                                    # same unit as the rule: first 60 s window with beats
                beats = beats[beats < 60 * FS]
            m = _sync(env, beats)
            p = bool(np.isfinite(m['lock_effect']) and m['lock_effect'] >= SYNC_EFFECT and abs(m['s1_lag_ms'] - centre[d]) <= SYNC_LAG_HALFWIDTH
                     and m['perbeat_lag_iqr_ms'] <= SYNC_IQR)
            rows.append(dict(dataset=d, record=a.record, beats_from=bb.record, passes_rule=p, **m))
    N = pd.DataFrame(rows)
    N.to_csv(os.path.join(OUT, 'negative_control_v2.csv'), index=False)

    # summary per dataset
    S2 = []
    for d, g in B[ok].groupby('dataset'):
        u = g[g['usable_v2']]
        S2.append(dict(dataset=d, records=len(g), lead_fallback=int(g['lead_fallback'].sum()), ecg_ok=int(g['ecg_ok_v2'].sum()),
                       usable=int(len(u)), usable_hours=round(u['usable_labelled_s'].sum() / 3600, 2),
                       usable_beats=int(u['usable_beats'].sum()),
                       label_coverage_median=round(g['label_coverage'].median(), 3),
                       panel_rule_ok_share=round(g['rec_rule_ok'].mean(), 3),
                       robust_rejected_share=round(g['n_robust_rejected'].sum() / max(1, g['n_robust'].sum()), 4),
                       low_corr_share=round(g['n_low_corr'].sum() / max(1, g['n_beats'].sum()), 4),
                       lag_gap_median_ms=round(g['lag_gap_ms'].median(), 1), lag_gap_records=int(g['lag_gap_ms'].notna().sum()),
                       lag_gap_pass=bool(abs(g['lag_gap_ms'].median()) <= BEATS.lag_gap_max_ms) if g['lag_gap_ms'].notna().any() else None,
                       v1_f1_median=round(g['v1_f1'].median(), 4) if 'v1_f1' in g else np.nan,
                       v1_minus_v2_ms_median=round(g['v1_minus_v2_med_ms'].median(), 1) if 'v1_minus_v2_med_ms' in g else np.nan,
                       centre_ms=centre.get(d, np.nan),
                       null_pass_share=round(N[N.dataset == d]['passes_rule'].mean(), 3) if len(N) and (N.dataset == d).any() else np.nan))
    S2 = pd.DataFrame(S2)
    S2.to_csv(os.path.join(OUT, 'summary_v2.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(S2.T.to_string())
    return S2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(R.DATASETS))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--stage', default='both')
    a = ap.parse_args()
    if a.stage in ('1', 'both'):
        A = pd.read_csv(LOCAL['audit_table'], low_memory=False)
        A = A[A['error'].isna() & A['dataset'].isin(a.datasets.split(','))]
        if a.limit:
            A = A.groupby('dataset').head(a.limit)
        rows = A.to_dict('records')
        rows.sort(key=lambda r: -float(r['duration_s']))           # longest first
        os.makedirs(OUT, exist_ok=True)
        recs, wins, t0 = [], [], time.time()
        with Pool(a.workers) as pool:
            for i, (r, w) in enumerate(pool.imap_unordered(process, rows, chunksize=1)):
                recs.append(r); wins += w
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print('%d/%d records, %.0f s' % (i + 1, len(rows), time.time() - t0), flush=True)
        tag = '' if (not a.limit and set(a.datasets.split(',')) == set(R.DATASETS)) else '_partial'
        pd.DataFrame(recs).to_csv(os.path.join(OUT, 'beats_records%s.csv' % tag), index=False)
        pd.DataFrame(wins).to_csv(os.path.join(OUT, 'beats_windows%s.csv' % tag), index=False)
        print('errors:', sum('error' in r for r in recs))
    if a.stage in ('2', 'both'):
        stage2()


if __name__ == '__main__':
    main()
