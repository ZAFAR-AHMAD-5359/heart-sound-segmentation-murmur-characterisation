"""Acoustic S1 and S2 targets inside R-locked windows (08 design, section 2.6; Panel B's definitions).

The ECG gives beat identity (teacher beats, qrs.py); the heart sounds give position (PCG-envelope maxima).
Envelope: pcgecg.signals.pcg_envelope (25 to 400 Hz, Hilbert magnitude, 20 Hz low-pass, 500 Hz).

Pass 1, training subjects only (fits the window laws):
  S1 = envelope maximum in [-30 ms, min(0.45 RR, 200 ms)] after the beat; S2 = maximum in the QS2-like prior
  420 sqrt(RR) - 20 ms +/- 120 ms, at least 80 ms after S1 and before 0.85 RR. On valid beats (below), fit
  lag = intercept[dataset] + slope[dataset] x HR by median regression, separately for S1 and S2, at most 100 beats
  per subject. Each law labels only its own dataset's beats, inside that dataset's heart-rate range; a slope shared
  across datasets came out near 0 while S1 moves from 56 to 28 ms (EPHNOGRAM) and 90 to 44 ms (SensSmartTech) between
  rest and 150 bpm.
Pass 2, every record:
  S1 = maximum within law_S1(dataset, HR) +/- 40 ms (inside [-30 ms, 0.45 RR]); S2 = maximum within
  law_S2(dataset, HR) +/- 80 ms, at least 80 ms after S1 and before 0.85 RR.
Valid beat: its whole cycle inside the teacher label mask and inside a usable synchronisation span; RR before and
after within 20% of the 11-beat running median; S1 and S2 maxima off their window edges, at least 4 robust z-units
above the cycle's 10th percentile (the envelope is a median/MAD z-score of the whole record, as in Panel B), and at
least a quarter of the record's median height for that sound; and each mark within 40 ms of the centred running
median of the same sound's lag over the 11 neighbouring candidate beats (a record needs at least 5 candidates).
Labels v3 (18 Oct 2026, prereg/CHANGES_AFTER_FREEZE.md): S2 at least 6 z-units, at least 0.3 of the S1 height in the
same cycle, and no other local maximum in its search window more than 40 ms from the mark taller than 0.6 of it. The
blinded re-check of v2 failed (141 of 200 ok); faint or low S2 was the largest failure group and S2 height separated it
(AUC 0.88). v3 keeps fewer, surer S2 labels, as Springer et al. (2016) kept only periods where every reference detector
agreed; validated against the hand-corrected annotations of PhysioNet/CinC 2016 training-a (refcheck.py).
Labels v2 (17 Oct 2026, prereg/CHANGES_AFTER_FREEZE.md): the 200-beat check of labels v1 (floor 2 z-units, no lag
consistency) failed, 164 of 200 beats ok; 32 of the 36 failures were faint, flat or competing S2 marks at high heart
rate or in noisy records. On the checked beats, the floor of 4 and the 40 ms consistency rule (twice the check
tolerance; a mark on the wrong bump sits far from its neighbours' lag, while the lag of a sound changes little from
beat to beat) kept 150 beats, 4 of them doubtful, and removed 32 of the 36 failures and 18 of the 164 good beats.
Also evaluated on those beats and not adopted, as they added nothing to the two rules: a rival-peak limit (q1, q2,
the tallest other local maximum in the search window more than 40 ms from the mark, as a share of the mark's
height; still recorded per beat), a stricter background contrast, and an S1 lag floor of 10 to 20 ms (at 15 ms it
removed 1 doubtful beat and 10 good ones). A z-score against the cycle's own median and MAD was
tried first and rejected: at 160 bpm S1 and S2 fill over a third of the cycle, which inflated the MAD and failed 87%
of EPHNOGRAM S2 above 150 bpm. Peaks are refined by parabolic interpolation (labels and law fits are not tied to the
2 ms grid). Dominance: each labelled sound at least as tall as the largest envelope value in the rest of its cycle
(outside both search windows widened by 30 ms, and outside the last 30 ms before the next beat); in the 200-beat
check the doubtful marks were small bumps in cycles holding a larger event elsewhere (exercise noise, diastolic
events). Physiology: an S1 mark may not precede the QRS reference point, and a record whose valid S1 marks sit a
median of under 10 ms after it loses all labels (S1 peaks follow R by 24 to 66 ms in every dataset, Panel B; 69 HFrEF
and 7 training-a records had S1 marks at or before R, a QRS-coincident artefact or a mislabel). Cycles of invalid
beats are masked, never background.
HR is the local rate from the 11-beat running median RR.
R target (auxiliary head, beat identity only): the beat shifted by (mean S1 intercept - dataset S1 intercept), so that
every dataset shares one R-to-S1 relation.
"""
import json, hashlib, os
import numpy as np
import pandas as pd
import scipy.signal as sg

FS = 500
LBL = dict(s1_broad=(-30.0, 200.0), s1_rr_max=0.45, s2_prior_halfwidth=120.0, s2_min_after_s1=80.0, s2_rr_max=0.85,
           s1_halfwidth=40.0, s2_halfwidth=80.0, z_min=4.0, rel_height_min=0.25, dominance_min=1.0,
           bg_margin_ms=30.0, s1_min_lag_ms=0.0, record_s1_min_median_ms=10.0, rival_sep_ms=40.0,
           z_min_s2=6.0, s2_rel_s1_min=0.3, rival_max_s2=0.6,
           consist_beats=11, consist_min_beats=5, consist_max_ms=40.0,
           rr_dev_max=0.20, run_beats=11, max_beats_per_subject=100,
           onset_frac=0.5, onset_back_ms=100.0, offset_fwd_ms=120.0, rr_min_ms=250.0, rr_max_ms=2000.0)


def s2_prior_ms(rr_ms):
    return 420.0 * np.sqrt(np.asarray(rr_ms, float) / 1000.0) - 20.0


def running_median(x, k):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    k = min(k, len(x) if len(x) % 2 else len(x) - 1)
    return sg.medfilt(x, k) if k >= 3 else np.full(len(x), np.median(x))


def _peak_in(env, lo, hi):
    """Index of the envelope maximum in [lo, hi] (samples) and whether it sits on the window edge."""
    lo, hi = max(0, int(np.floor(lo))), min(len(env) - 1, int(np.ceil(hi)))
    if hi - lo < 3:
        return -1, True
    k = lo + int(np.argmax(env[lo:hi + 1]))
    return k, bool(k == lo or k == hi)


def _refine(env, k):
    """Sub-sample position of the maximum at k (parabolic interpolation)."""
    if 1 <= k < len(env) - 1:
        y0, y1, y2 = env[k - 1], env[k], env[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return k + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return float(k)


def _rival(env, lo, hi, k, base, sep):
    """Height of the tallest other local maximum in [lo, hi] (samples) lying more than sep samples from the mark at k,
    as a share of the mark's height (both above the cycle floor base); 0 when there is none."""
    lo, hi = max(0, int(np.floor(lo))), min(len(env) - 1, int(np.ceil(hi)))
    h = env[k] - base
    if hi - lo < 3 or not h > 0:
        return np.nan
    pk = lo + sg.find_peaks(env[lo:hi + 1])[0]
    pk = pk[np.abs(pk - k) > sep]
    return float(max(0.0, (env[pk].max() - base) / h)) if len(pk) else 0.0


def lag_deviation(res, cand):
    """Per sound, each candidate mark's lag minus the centred running median of that lag over the neighbouring
    candidate beats (consist_beats of them, at least consist_min_beats in the record), in ms; NaN elsewhere."""
    idx = np.flatnonzero(np.asarray(cand, bool))
    out = {}
    for k in ('1', '2'):
        dev = np.full(len(res), np.nan)
        if len(idx) >= LBL['consist_min_beats']:
            lag = ((res['s' + k] - res['r']) / FS * 1000.0).values[idx]
            ref = pd.Series(lag).rolling(LBL['consist_beats'], center=True, min_periods=max(1, LBL['consist_min_beats'])).median().values
            dev[idx] = lag - ref
        out[k] = dev
    return out


def _crossing(env, pk, thr, back, fwd):
    """Fractional sample of the last crossing of thr before pk (onset) and the first after pk (offset)."""
    a = max(0, pk - back); seg = env[a:pk + 1]; below = np.flatnonzero(seg < thr)
    on = np.nan
    if len(below):
        i = a + below[-1]
        on = i + (thr - env[i]) / (env[i + 1] - env[i] + 1e-12)
    b = min(len(env) - 1, pk + fwd); seg = env[pk:b + 1]; below = np.flatnonzero(seg < thr)
    off = np.nan
    if len(below):
        i = pk + below[0]
        off = (i - 1) + (env[i - 1] - thr) / (env[i - 1] - env[i] + 1e-12)
    return on, off


def beat_frame(beats, trainable):
    """Per beat with a following beat: RR, local HR, regularity, and whether its cycle is trainable."""
    b = np.asarray(beats, int)
    if len(b) < 3:
        return pd.DataFrame()
    rr = np.diff(b) / FS * 1000.0
    run = running_median(rr, LBL['run_beats'])
    dev = np.abs(rr - run) > LBL['rr_dev_max'] * run
    i = np.arange(len(rr))
    regular = ~dev & ~np.r_[True, dev[:-1]]                     # RR after and RR before both regular
    cs = np.r_[0, np.cumsum(~trainable)]
    inside = np.array([b[k + 1] <= len(trainable) and cs[b[k + 1]] - cs[b[k]] == 0 for k in i])
    ok_rr = (rr >= LBL['rr_min_ms']) & (rr <= LBL['rr_max_ms'])
    return pd.DataFrame(dict(i=i, r=b[:-1], r_next=b[1:], rr=rr, hr=60000.0 / run, regular=regular & ok_rr,
                             in_mask=inside))


def find_events(env, bf, s1_win, s2_win):
    """s1_win(row) -> (lo_ms, hi_ms) after r; s2_win(row, s1_ms) -> (lo_ms, hi_ms). Adds peaks, z, edges, onsets."""
    out = []
    ms = FS / 1000.0
    for row in bf.itertuples():
        r, rn = row.r, row.r_next
        lo, hi = s1_win(row)
        lo, hi = max(lo, LBL['s1_broad'][0]), min(hi, LBL['s1_rr_max'] * row.rr)
        k1, e1 = _peak_in(env, r + lo * ms, r + hi * ms)
        lo2, hi2 = lo, hi
        d = dict(s1=np.nan, s2=np.nan, s1_edge=True, s2_edge=True, z1=np.nan, z2=np.nan, h1=np.nan, h2=np.nan,
                 q1=np.nan, q2=np.nan, s1_on=np.nan, s1_off=np.nan, s2_on=np.nan, s2_off=np.nan)
        sep = LBL['rival_sep_ms'] * ms
        if k1 >= 0:
            s1_ms = (k1 - r) / ms
            base = float(np.percentile(env[r:rn], 10))
            on, off = _crossing(env, k1, base + LBL['onset_frac'] * (env[k1] - base), int(LBL['onset_back_ms'] * ms), int(LBL['offset_fwd_ms'] * ms))
            d.update(s1=_refine(env, k1), s1_edge=e1, z1=env[k1] - base, h1=env[k1] - base, s1_on=on, s1_off=off,
                     q1=_rival(env, r + lo * ms, r + hi * ms, k1, base, sep))
            lo2, hi2 = s2_win(row, s1_ms)
            lo2, hi2 = max(lo2, s1_ms + LBL['s2_min_after_s1']), min(hi2, LBL['s2_rr_max'] * row.rr)
            k2, e2 = _peak_in(env, r + lo2 * ms, r + hi2 * ms)
            if k2 >= 0:
                on, off = _crossing(env, k2, base + LBL['onset_frac'] * (env[k2] - base), int(LBL['onset_back_ms'] * ms), int(LBL['offset_fwd_ms'] * ms))
                d.update(s2=_refine(env, k2), s2_edge=e2, z2=env[k2] - base, h2=env[k2] - base, s2_on=on, s2_off=off,
                         q2=_rival(env, r + lo2 * ms, r + hi2 * ms, k2, base, sep))
        # background: the cycle outside both windows (widened) and outside the end of the cycle
        bg = np.ones(rn - r, bool)
        mg = LBL['bg_margin_ms'] * ms
        for (wl, wh) in ((lo, hi), (lo2, hi2) if k1 >= 0 else (lo, hi)):
            a0, b0 = int(max(0, wl * ms - mg)), int(min(rn - r, wh * ms + mg))
            bg[a0:b0] = False
        bg[max(0, rn - r - int(mg)):] = False
        if k1 >= 0 and bg.any():
            d['bg'] = float(env[r:rn][bg].max() - base)
        else:
            d['bg'] = np.nan
        out.append(d)
    ev = pd.DataFrame(out, index=bf.index)
    res = pd.concat([bf, ev], axis=1)
    for k in ('1', '2'):
        z_floor = max(LBL['z_min'], LBL['z_min_s2']) if k == '2' else LBL['z_min']
        base_ok = res['s' + k].notna() & ~res['s' + k + '_edge'].astype(bool) & (res['z' + k] >= z_floor)
        ref = res.loc[base_ok & res['regular'] & res['in_mask'], 'h' + k].median()
        if not np.isfinite(ref):
            ref = res.loc[base_ok, 'h' + k].median()
        dom = res['bg'].isna() | (res['h' + k] >= LBL['dominance_min'] * res['bg'])
        res['s%s_ok' % k] = (base_ok & (res['h' + k] >= LBL['rel_height_min'] * ref) & dom) if np.isfinite(ref) else False
    # labels v3: S2 also at least s2_rel_s1_min of the cycle's S1 height, and no rival peak in its window taller than
    # rival_max_s2 of it (more than rival_sep_ms away); a NaN ratio (no S1 peak) fails
    res['s2_ok'] = res['s2_ok'] & (res['h2'] >= LBL['s2_rel_s1_min'] * res['h1']) & (res['q2'] <= LBL['rival_max_s2'])
    s1_lag = (res['s1'] - res['r']) / ms
    res['s1_ok'] = res['s1_ok'] & (s1_lag >= LBL['s1_min_lag_ms'])
    cand = res['regular'] & res['in_mask'] & res['s1_ok'] & res['s2_ok']
    dev = lag_deviation(res, cand)
    res['dev1'], res['dev2'] = dev['1'], dev['2']
    res['valid'] = cand & (res['dev1'].abs() <= LBL['consist_max_ms']) & (res['dev2'].abs() <= LBL['consist_max_ms'])
    med = s1_lag[res['valid']].median() if res['valid'].any() else np.nan
    res['early_s1_record'] = bool(np.isfinite(med) and med < LBL['record_s1_min_median_ms'])
    if res['early_s1_record'].iloc[0] if len(res) else False:
        res['valid'] = False
    return res


def pass1(env, bf):
    return find_events(env, bf, lambda row: LBL['s1_broad'],
                       lambda row, s1: (s2_prior_ms(row.rr) - LBL['s2_prior_halfwidth'], s2_prior_ms(row.rr) + LBL['s2_prior_halfwidth']))


def law_ms(law, dataset, hr):
    slope = law['slope'][dataset] if isinstance(law['slope'], dict) else law['slope']
    return law['intercept'][dataset] + slope * np.asarray(hr, float)


def pass2(env, bf, laws, dataset):
    s1, s2 = laws['s1'], laws['s2']
    return find_events(env, bf,
                       lambda row: (law_ms(s1, dataset, row.hr) - LBL['s1_halfwidth'], law_ms(s1, dataset, row.hr) + LBL['s1_halfwidth']),
                       lambda row, _: (law_ms(s2, dataset, row.hr) - LBL['s2_halfwidth'], law_ms(s2, dataset, row.hr) + LBL['s2_halfwidth']))


def fit_laws(beats_df, seed=42):
    """beats_df: pass-1 valid beats from training subjects with columns dataset, group, hr, s1_ms, s2_ms.
    Median regression lag = intercept[dataset] + slope x HR, at most max_beats_per_subject per subject."""
    from sklearn.linear_model import QuantileRegressor
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in beats_df.groupby('group'):
        parts.append(g.iloc[rng.permutation(len(g))[:LBL['max_beats_per_subject']]])
    d = pd.concat(parts)
    ds = sorted(d['dataset'].unique())
    laws = {ev: dict(slope={}, intercept={}) for ev in ('s1', 's2')}
    for k in ds:
        g = d[d['dataset'] == k]
        hr0 = 80.0                                          # centre HR so the intercept is the lag at 80 bpm
        X = np.column_stack([np.ones(len(g)), g['hr'].values - hr0])
        for ev in ('s1', 's2'):
            q = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False, solver='highs').fit(X, g[ev + '_ms'].values)
            laws[ev]['slope'][k] = float(q.coef_[1])
            laws[ev]['intercept'][k] = float(q.coef_[0] - q.coef_[1] * hr0)
        laws.setdefault('hr_range', {})[k] = [float(np.percentile(g['hr'], 5)), float(np.percentile(g['hr'], 95))]
    laws['n_beats'] = int(len(d)); laws['n_subjects'] = int(d['group'].nunique())
    laws['per_dataset_beats'] = d.groupby('dataset').size().astype(int).to_dict()
    return laws


def r_target_shift_ms(laws, dataset, hr0=80.0):
    """Shift of the R target so every dataset shares one R-to-S1 lag at 80 bpm (mean over datasets)."""
    a = {k: law_ms(laws['s1'], k, hr0) for k in laws['s1']['intercept']}
    return float(np.mean(list(a.values())) - a[dataset])


def laws_hash(laws):
    return hashlib.sha256(json.dumps(laws, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ run over the paired corpus
def _record_inputs(row):
    """Envelope (500 Hz), teacher beats, and the trainable mask (teacher labels inside usable sync spans)."""
    import sys
    from extstudy import qrs
    from extstudy.config import LOCAL
    if LOCAL['pcgecg_code'] not in sys.path:
        sys.path.insert(0, LOCAL['pcgecg_code'])
    from pcgecg.signals import pcg_envelope
    z = np.load(os.path.join(LOCAL['harmonised'], row['dataset'], row['record'] + '.npz'))
    env = pcg_envelope(z['pcg']).astype(np.float64)
    b = np.load(os.path.join(LOCAL['beats'], row['dataset'], row['record'] + '.npz'))
    n = min(len(env), int(b['n_samples']))
    mask = qrs.intervals_to_mask(b['nolabel_intervals'], int(b['n_samples']))[:n]
    spans = row.get('spans')
    if spans is not None:                                       # long records: usable 60 s windows only
        keep = np.zeros(n, bool)
        for s0, e0 in spans:
            keep[int(s0 * FS):int(e0 * FS)] = True
        mask &= keep
    beats = np.asarray(b['beats'], int)
    return env[:n], beats[beats < n], mask


def _run_one(args):
    row, laws = args
    env, beats, mask = _record_inputs(row)
    bf = beat_frame(beats, mask)
    if bf.empty:
        return row, None
    res = pass1(env, bf) if laws is None else pass2(env, bf, laws, row['dataset'])
    return row, res


def record_rows():
    """The split table as a list of records (longest first), each with its usable sync spans if it is a long record."""
    from extstudy import splits
    from extstudy.config import LOCAL
    T = splits.paired_splits()
    W = pd.read_csv(os.path.join(LOCAL['beats'], 'beats_windows.csv'), low_memory=False)
    W = W[W['window_ok'].astype(bool)]
    spans = {(d, r): list(zip(g['start_s'], g['end_s'])) for (d, r), g in W.groupby(['dataset', 'record'])}
    B = pd.read_csv(os.path.join(LOCAL['beats'], 'audit_v2.csv'), low_memory=False)
    long_ = set(map(tuple, B.loc[B['long'].astype(bool), ['dataset', 'record']].values))
    rows = []
    for r in T.to_dict('records'):
        k = (r['dataset'], r['record'])
        r['spans'] = spans.get(k, []) if k in long_ else None
        rows.append(r)
    rows.sort(key=lambda r: -float(r['duration_s']))
    return T, rows


def main(workers=14):
    from multiprocessing import Pool
    from extstudy import splits
    from extstudy.config import LOCAL, LABELS
    out = os.path.join(LOCAL['phd'], 'derived', LABELS)
    os.makedirs(out, exist_ok=True)
    T, rows = record_rows()
    # pass 1 on training records -> laws
    train = [r for r in rows if r['role'] == 'train']
    with Pool(workers) as pool:
        res1 = pool.map(_run_one, [(r, None) for r in train], chunksize=1)
    fit = []
    for r, res in res1:
        if res is None:
            continue
        v = res[res['valid']]
        fit.append(pd.DataFrame(dict(dataset=r['dataset'], group=r['group'], hr=v['hr'].values,
                                     s1_ms=((v['s1'] - v['r']) / FS * 1000).values, s2_ms=((v['s2'] - v['r']) / FS * 1000).values)))
    fit = pd.concat(fit, ignore_index=True)
    laws = fit_laws(fit)
    laws['split_table_sha256'] = splits.table_hash(T.drop(columns=[c for c in ('spans',) if c in T]))
    laws['params'] = LBL
    json.dump(dict(laws, sha256=laws_hash(laws)), open(os.path.join(out, 'laws.json'), 'w'), indent=1)
    print('laws:', json.dumps({k: laws[k] for k in ('s1', 's2', 'n_beats', 'n_subjects')}, indent=1))
    # pass 2 on every record
    with Pool(workers) as pool:
        res2 = pool.map(_run_one, [(r, laws) for r in rows], chunksize=1)
    summ = []
    for r, res in res2:
        d = os.path.join(out, r['dataset']); os.makedirs(d, exist_ok=True)
        if res is None:
            summ.append(dict(dataset=r['dataset'], record=r['record'], role=r['role'], n_beats=0, n_valid=0)); continue
        sh = r_target_shift_ms(laws, r['dataset'])
        to_s = lambda c: (res[c].astype(float) / FS).values
        np.savez_compressed(os.path.join(d, r['record'] + '.npz'),
                            r_s=to_s('r'), r_next_s=to_s('r_next'), s1_s=to_s('s1'), s2_s=to_s('s2'),
                            s1_on_s=to_s('s1_on'), s1_off_s=to_s('s1_off'), s2_on_s=to_s('s2_on'), s2_off_s=to_s('s2_off'),
                            r_target_s=to_s('r') + sh / 1000.0, hr=res['hr'].values, rr_ms=res['rr'].values,
                            z1=res['z1'].values, z2=res['z2'].values, q1=res['q1'].values, q2=res['q2'].values,
                            bg=res['bg'].values, dev1=res['dev1'].values, dev2=res['dev2'].values, regular=res['regular'].values,
                            in_mask=res['in_mask'].values, s1_ok=res['s1_ok'].values, s2_ok=res['s2_ok'].values,
                            valid=res['valid'].values, laws_sha256=np.array(laws_hash(laws)))
        v = res[res['valid']]
        summ.append(dict(dataset=r['dataset'], record=r['record'], subject=r['subject'], group=r['group'], role=r['role'],
                         early_s1_record=bool(res['early_s1_record'].iloc[0]), n_beats=len(res), n_in_mask=int(res['in_mask'].sum()), n_regular=int((res['in_mask'] & res['regular']).sum()),
                         n_s1_ok=int((res['in_mask'] & res['regular'] & res['s1_ok']).sum()), n_valid=int(len(v)),
                         valid_s=float(v['rr'].sum() / 1000.0), hr_median=float(v['hr'].median()) if len(v) else np.nan,
                         s1_ms_median=float(((v['s1'] - v['r']) / FS * 1000).median()) if len(v) else np.nan,
                         s2_ms_median=float(((v['s2'] - v['r']) / FS * 1000).median()) if len(v) else np.nan))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, 'labels_records.csv'), index=False)
    agg = S.groupby(['dataset', 'role']).agg(records=('record', 'size'), beats=('n_beats', 'sum'), in_mask=('n_in_mask', 'sum'),
                                              regular=('n_regular', 'sum'), valid=('n_valid', 'sum'), hours=('valid_s', lambda s: s.sum() / 3600))
    agg['valid_share_of_masked_regular'] = (agg['valid'] / agg['regular'].clip(lower=1)).round(3)
    pd.set_option('display.width', 200)
    print(agg.round(2).to_string())
    agg.to_csv(os.path.join(out, 'labels_summary.csv'))
    return laws, S


if __name__ == '__main__':
    main()
