"""Pre-CirCor check of the Run 2 cycle export on paired test records (descriptive; constants already fixed in
extstudy/cycles.py; nothing is tuned here). Each record is processed as CirCor audio will be: PCG resampled to
4 kHz, first 60 s, max-abs normalised; frozen T and W (seed averages) -> beats -> cycles. Truth: labels v3 S1 and S2
50% onsets of valid beats. A ceiling: arm B's construction from the true ECG R-peaks (teacher beats).
Usage:  python notebooks/check_export_paired.py <checkpoint folder> [max records per dataset]
"""
import json, os, sys
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import cycles as C, infer as I
from extstudy.config import LOCAL, LABELS, PACK

SECONDS = 60.0


def match(a, b, tol):
    """Hungarian one-to-one matching within tol (seconds) -> (n matched, signed a - b of the matches)."""
    if len(a) == 0 or len(b) == 0:
        return 0, np.array([])
    D = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e9))
    ok = D[ri, ci] <= tol
    return int(ok.sum()), np.asarray(a)[ri[ok]] - np.asarray(b)[ci[ok]]


def main(folder, cap=40):
    T, _ = I.load_models(folder, 'T')
    W, _ = I.load_models(folder, 'W')
    R = pd.read_csv(os.path.join(LOCAL['phd'], 'derived', 'corpus', PACK, 'records.csv'))
    R = R[R.role == 'test']
    rows = []
    for d, g in R.groupby('dataset'):
        for _, r in g.sort_values('record').head(cap).iterrows():
            h = np.load(os.path.join(LOCAL['harmonised'], d, r.record + '.npz'))
            x = sg.resample_poly(np.asarray(h['pcg'], np.float64), 4000, int(round(float(h['fs_pcg']))))
            x4 = x[:int(SECONDS * 4000)]
            x4 = (x4 / (np.abs(x4).max() + 1e-10)).astype(np.float32)
            L = np.load(os.path.join(LOCAL['phd'], 'derived', LABELS, d, r.record + '.npz'))
            v = L['valid'].astype(bool) & (L['r_next_s'] < len(x4) / 4000)
            if v.sum() < 3:
                continue
            spec = I.spectrogram(x4, 4000)
            tt, tw = I.forward(T, spec, 'T'), I.forward(W, spec, 'W')
            bt, _ = I.beats(tt, 'T')
            bw, _ = I.beats(tw, 'W')
            arms = {'T': C.t_cycles(x4, bt, tt[1]), 'W': C.b_cycles(x4, (bw * 4000).astype(int)),
                    'ECG ceiling': C.b_cycles(x4, (L['r_s'][L['r_s'] < len(x4) / 4000] * 4000).astype(int))}
            t1, t2 = L['s1_on_s'][v], L['s2_on_s'][v]
            for arm, (cyc, dg) in arms.items():
                st = np.array([c['s1'][0] for c in cyc]) / 4000
                s2 = np.array([c['s2'][0] for c in cyc]) / 4000
                m150, o1 = match(st, t1, 0.150)
                m50, _ = match(st, t1, 0.050)
                _, o2 = match(s2, t2, 0.150)
                rows.append(dict(dataset=d, record=r.record, arm=arm, n_truth=int(v.sum()), n_cycles=len(cyc),
                                 found150=m150 / v.sum(), found50=m50 / v.sum(), s1_off_med=np.median(o1) * 1000 if len(o1) else np.nan,
                                 s1_off_sd=np.std(o1) * 1000 if len(o1) > 1 else np.nan,
                                 s2_off_med=np.median(o2) * 1000 if len(o2) else np.nan, s2_off_sd=np.std(o2) * 1000 if len(o2) > 1 else np.nan))
        print(d, 'done')
    D = pd.DataFrame(rows)
    out = os.path.join(folder, 'export_check_paired.csv')
    D.to_csv(out, index=False)
    pd.set_option('display.width', 220)
    agg = D.groupby(['dataset', 'arm']).agg(records=('record', 'size'), found150=('found150', 'median'), found50=('found50', 'median'),
                                            s1_off_med=('s1_off_med', 'median'), s1_off_sd=('s1_off_sd', 'median'),
                                            s2_off_med=('s2_off_med', 'median'), s2_off_sd=('s2_off_sd', 'median')).round(3)
    print(agg.to_string())
    return D


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
