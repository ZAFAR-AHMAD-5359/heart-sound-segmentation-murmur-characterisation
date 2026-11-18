"""Second exploratory step (19 Oct 2026, after s2_phase_analysis.py showed that median phase agreement does not
explain the arms' gain): is it the cycle-to-cycle SPREAD of an arm's boundary errors within a recording, rather
than their size, that costs gain? (Experiment 0's jitter is pure spread; a constant offset shifts every cycle alike.)
Measures, fixed before computing: per recording and arm, over cycles paired with the annotation (as before), the SD
and the IQR of the S2 onset error and of the start error, and the share of paired cycles whose S2 error lies more
than 40 ms from the recording's own median S2 error (scatter around its own offset); per arm the median over
recordings, set against the composite retention; per recording (S, T, W) the loss rate by tertile of S2-error SD.
Usage:  python notebooks/s2_variability_analysis.py <run1 cycles dir> <exp0 audio cache dir> <output dir>
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_phase_analysis as SP


def main(run1_cycles_dir, audio_dir, out_dir):
    keep = np.load(os.path.join(SP.FOLDS, 'detection.npz'))['keep_names'].astype(str).tolist()
    cyc = SP.load_cycles(run1_cycles_dir, audio_dir, keep)
    o1 = np.load(os.path.join(SP.RUN1, 'oof_predictions.npz'), allow_pickle=False)
    o2 = np.load(os.path.join(SP.RUN2, 'oof_predictions.npz'), allow_pickle=False)
    names3 = sorted(set().union(*[set(o1['%s__pcg_seg__names' % t].astype(str)) for t in SP.TASKS]))
    rows = []
    for k in SP.ARMS:
        for n in names3:
            P = pd.DataFrame(SP.pair_metrics(cyc[k][n], cyc['C'][n]))
            if len(P) < 3:
                continue
            e2, e1 = P.s2_err_ms.values, P.start_err_ms.values
            rows.append(dict(arm=k, recording=n, n_pairs=len(P), s2_sd=float(np.std(e2)), s2_iqr=float(np.subtract(*np.percentile(e2, [75, 25]))),
                             start_sd=float(np.std(e1)), start_iqr=float(np.subtract(*np.percentile(e1, [75, 25]))),
                             s2_scatter40=float(np.mean(np.abs(e2 - np.median(e2)) > 40)), s2_bias=float(np.median(e2)), start_bias=float(np.median(e1))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_dir, 's2_variability_per_recording.csv'), index=False)
    A = pd.read_csv(os.path.join(out_dir, 's2_phase_per_arm.csv'))[['arm', 'retention', 'timing_retention', 's2_abs_err_ms', 'iou_mean']]
    agg = R.groupby('arm').agg(s2_sd=('s2_sd', 'median'), s2_iqr=('s2_iqr', 'median'), start_sd=('start_sd', 'median'), start_iqr=('start_iqr', 'median'),
                               s2_scatter40=('s2_scatter40', 'median'), s2_bias=('s2_bias', 'median'), start_bias=('start_bias', 'median'),
                               s2_sd_p75=('s2_sd', lambda v: float(np.percentile(v, 75)))).reset_index()
    A = A.merge(agg, on='arm').sort_values('retention')
    rank = {c: float(spearmanr(-A[c], A.retention).correlation) for c in ('s2_sd', 's2_iqr', 'start_sd', 'start_iqr', 's2_scatter40', 's2_sd_p75', 's2_abs_err_ms')}
    loss_rows = []
    for k, src in (('S', o1), ('T', o2), ('W', o2)):
        key = SP.ARMS[k]
        g = R[R.arm == k].set_index('recording')
        for t in SP.TASKS:
            names = o1['%s__pcg_seg__names' % t].astype(str)
            y = o1['%s__pcg_seg__y' % t]
            loss = ((o1['%s__pcg_seg__mlp__pred' % t] == y) & (src['%s__%s__mlp__pred' % (t, key)] != y)).astype(float)
            sd = g.reindex(names)['s2_sd'].values
            ok = np.isfinite(sd)
            q = np.percentile(sd[ok], [100 / 3, 200 / 3])
            terc = np.digitize(sd, q)
            r = spearmanr(sd[ok], loss[ok])
            loss_rows.append(dict(arm=k, task=t, n=int(ok.sum()), loss_low=float(loss[ok & (terc == 0)].mean()), loss_mid=float(loss[ok & (terc == 1)].mean()),
                                  loss_high=float(loss[ok & (terc == 2)].mean()), sd_low=float(np.median(sd[ok & (terc == 0)])), sd_high=float(np.median(sd[ok & (terc == 2)])),
                                  spearman=float(r.correlation), p=float(r.pvalue)))
    L = pd.DataFrame(loss_rows)
    L.to_csv(os.path.join(out_dir, 's2_variability_loss.csv'), index=False)
    json.dump(dict(rank_correlations_with_retention=rank), open(os.path.join(out_dir, 's2_variability_summary.json'), 'w'), indent=1)
    pd.set_option('display.width', 230)
    print(A.round(3).to_string(index=False))
    print('rank correlation of LOW spread with retention (8 arms):', {k: round(v, 2) for k, v in rank.items()})
    print(L.round(3).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
