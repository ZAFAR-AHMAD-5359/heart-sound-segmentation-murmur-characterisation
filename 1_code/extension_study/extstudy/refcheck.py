"""Label validation against a reference standard: the hand-corrected heart-sound state annotations of PhysioNet/CinC
2016 training-a (Liu et al. 2016, Physiol Meas 37:2181), the MGH database of Springer et al. (2016, IEEE TBME 63(4)),
recorded with a synchronous ECG. Rules fixed in prereg/CHANGES_AFTER_FREEZE.md (18 Oct 2026) before any annotation
was opened.

Correct mark (definition 1): the mark lies on the annotated complex of the same sound in the same cycle, from its
onset minus TOL_S to its offset plus TOL_S (a split S1 or S2 is one complex). Secondary: within 100 ms of the complex
centre (the tolerance of Springer et al.); signed offset mark - centre. Units: valid beats of a label version in
training-a records, outside annotated noise and inside the annotated span. Share correct per sound with a
record-clustered bootstrap 95% CI (N_BOOT draws, seed 42).
Calibration: Claude's panel verdicts from the visual checks on training-a against the reference (both marks correct).
Usage:  python -m extstudy.refcheck
"""
import glob, json, os
import numpy as np
import pandas as pd

from extstudy.config import LOCAL

FS_ANN = 2000
TOL_S = 0.020
WIDE_S = 0.100
N_BOOT, SEED = 2000, 42
ANN_DIR = os.path.join(LOCAL['phd'], 'Datasets', 'Dataset-4', 'Dataset-Cinc2016', 'annotations_physionet')


def parse_states(rows):
    """rows: sequence of (start_sample, label) in file order. Returns (segments, noise, span): segments is a DataFrame
    of state, start_s, end_s for S1/systole/S2/diastole (end = next boundary); noise is a list of (start_s, end_s)
    between '(N' and 'N)' markers; span = (first boundary, last boundary) in seconds."""
    seg, noise, open_n, bounds = [], [], None, []
    states = []
    for start, lab in rows:
        lab = str(lab).strip()
        t = float(start) / FS_ANN
        if lab.startswith('(N'):
            open_n = t
            continue
        if lab.endswith('N)'):
            if open_n is not None:
                noise.append((open_n, t))
            open_n = None
            continue
        states.append((t, lab))
        bounds.append(t)
    for (t0, lab), (t1, _) in zip(states[:-1], states[1:]):
        seg.append(dict(state=lab, start_s=t0, end_s=t1))
    if open_n is not None and bounds:
        noise.append((open_n, max(bounds)))
    span = (min(bounds), max(bounds)) if bounds else (np.nan, np.nan)
    return pd.DataFrame(seg, columns=['state', 'start_s', 'end_s']), noise, span


def mark_scores(t, segs, sound):
    """Correct under definition 1, within WIDE_S of the nearest complex centre, and the signed offset to that centre."""
    g = segs[segs.state == sound]
    if not np.isfinite(t) or g.empty:
        return False, False, np.nan
    inside = bool(((g.start_s - TOL_S <= t) & (t <= g.end_s + TOL_S)).any())
    c = (g.start_s.values + g.end_s.values) / 2
    j = int(np.argmin(np.abs(c - t)))
    off = float(t - c[j])
    return inside, abs(off) <= WIDE_S, off


def score_record(ev, segs, noise, span, valid_only=True):
    """ev: label arrays of one record (r_s, r_next_s, s1_s, s2_s, valid). Rows for the beats that can be judged."""
    out = []
    for i in range(len(ev['r_s'])):
        if valid_only and not bool(ev['valid'][i]):
            continue
        r, rn = float(ev['r_s'][i]), float(ev['r_next_s'][i])
        if not (span[0] <= r and rn <= span[1]):
            continue
        if any(a < rn and r < b for a, b in noise):
            continue
        row = dict(beat=i, hr=float(ev['hr'][i]))
        for k, sound in (('1', 'S1'), ('2', 'S2')):
            ok, wide, off = mark_scores(float(ev['s%s_s' % k][i]), segs, sound)
            row.update({'s%s_ok' % k: ok, 's%s_wide' % k: wide, 's%s_off_ms' % k: off * 1000})
        out.append(row)
    return out


def _unwrap(x):
    """MATLAB cell contents arrive as nested object arrays (labels as 'S1' or as [['S1']]); returns the scalar."""
    while isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else ''
    return x


def load_annotations():
    """{record: (segments, noise, span)} from the hand-corrected training-a annotation files."""
    import scipy.io as sio
    files = sorted(glob.glob(os.path.join(ANN_DIR, '**', 'hand_corrected', '**', 'a*_StateAns.mat'), recursive=True))
    ann = {}
    for f in files:
        rec = os.path.basename(f).split('_')[0]
        m = sio.loadmat(f)
        key = [k for k in m if not k.startswith('__')][0]
        cells = m[key]
        rows = [(int(_unwrap(cells[j, 0])), str(_unwrap(cells[j, 1]))) for j in range(cells.shape[0])]
        ann[rec] = parse_states(rows)
    return ann


def bootstrap_share(df, col, n=N_BOOT, seed=SEED):
    recs = df['record'].unique()
    by = {r: g[col].values for r, g in df.groupby('record')}
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(n):
        pick = rng.choice(recs, len(recs), replace=True)
        v = np.concatenate([by[r] for r in pick])
        est.append(v.mean())
    return float(df[col].mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def evaluate(version, ann):
    base = os.path.join(LOCAL['phd'], 'derived', version)
    rows = []
    for rec, (segs, noise, span) in ann.items():
        f = os.path.join(base, 'cinc2016a', rec + '.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for r in score_record(z, segs, noise, span):
            rows.append(dict(record=rec, **r))
    D = pd.DataFrame(rows)
    res = dict(version=version, beats=int(len(D)), records=int(D.record.nunique()) if len(D) else 0)
    for col in ('s1_ok', 's2_ok', 's1_wide', 's2_wide'):
        res[col] = bootstrap_share(D, col) if len(D) else None
    for k in ('1', '2'):
        o = D['s%s_off_ms' % k].dropna()
        res['s%s_offset_ms' % k] = dict(median=float(o.median()), p25=float(o.quantile(.25)), p75=float(o.quantile(.75)),
                                         abs_p95=float(o.abs().quantile(.95))) if len(o) else None
    return res, D


def calibrate(ann):
    """Claude's verdicts on training-a panels of every visual check against the reference (both marks correct)."""
    rows = []
    for version in ('labels_v1', 'labels_v2', 'labels_v3'):
        sheet = os.path.join(LOCAL['phd'], 'derived', version, 'check', 'check_beats.csv')
        if not os.path.exists(sheet):
            continue
        C = pd.read_csv(sheet, dtype={'verdict': str})
        key = os.path.join(os.path.dirname(sheet), 'check_key.csv')
        C = C.merge(pd.read_csv(key), on='panel', how='left') if os.path.exists(key) else C.assign(set='sample')
        C = C[(C.dataset == 'cinc2016a') & C.verdict.fillna('').str.strip().isin(['ok', 'wrong'])]
        for _, c in C.iterrows():
            if c.record not in ann:
                continue
            segs, noise, span = ann[c.record]
            z = np.load(os.path.join(LOCAL['phd'], 'derived', version, 'cinc2016a', c.record + '.npz'))
            sc = score_record(z, segs, noise, span, valid_only=False)
            hit = [s for s in sc if s['beat'] == int(c.beat)]
            if not hit:
                rows.append(dict(check=version, panel=c.panel, set=c.get('set', 'sample'), claude=c.verdict, judged=False))
                continue
            h = hit[0]
            rows.append(dict(check=version, panel=c.panel, set=c.get('set', 'sample'), claude=c.verdict.strip(), judged=True,
                             ref_ok=bool(h['s1_ok'] and h['s2_ok']), s1_ok=h['s1_ok'], s2_ok=h['s2_ok'],
                             s1_off_ms=h['s1_off_ms'], s2_off_ms=h['s2_off_ms'], notes=c.get('notes', '')))
    K = pd.DataFrame(rows)
    J = K[K.judged == True] if len(K) else K
    out = dict(panels=int(len(K)), judged=int(len(J)))
    if len(J):
        cw, rw = J.claude == 'wrong', ~J.ref_ok.astype(bool)
        tab = pd.crosstab(J.claude, J.ref_ok.map({True: 'ref_ok', False: 'ref_wrong'}))
        po = float((cw == rw).mean())
        pe = float(cw.mean() * rw.mean() + (1 - cw.mean()) * (1 - rw.mean()))
        out.update(table=tab.to_dict(), agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan,
                   wrong_confirmed=int((cw & rw).sum()), wrong_total=int(cw.sum()),
                   wrong_confirmed_share=float((cw & rw).sum() / cw.sum()) if cw.sum() else np.nan,
                   ref_wrong_total=int(rw.sum()))
    return out, K


def main():
    ann = load_annotations()
    out_dir = os.path.join(LOCAL['phd'], 'derived', 'refcheck')
    os.makedirs(out_dir, exist_ok=True)
    report = dict(annotation_files=len(ann), tol_ms=TOL_S * 1000, wide_ms=WIDE_S * 1000, versions={})
    for v in ('labels_v1', 'labels_v2', 'labels_v3'):
        if os.path.exists(os.path.join(LOCAL['phd'], 'derived', v, 'laws.json')):
            res, D = evaluate(v, ann)
            report['versions'][v] = res
            D.to_csv(os.path.join(out_dir, 'beats_%s.csv' % v), index=False)
    cal, K = calibrate(ann)
    report['calibration'] = cal
    K.to_csv(os.path.join(out_dir, 'calibration_panels.csv'), index=False)
    json.dump(report, open(os.path.join(out_dir, 'refcheck_report.json'), 'w'), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    return report


if __name__ == '__main__':
    main()
