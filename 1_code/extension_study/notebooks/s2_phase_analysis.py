"""Exploratory follow-up to Experiment 0 (19 Oct 2026; not pre-registered, prompted by its result): do the automatic
arms lose the gain of accurate gating through where they place S2 and the phase windows, rather than through beat
detection?

Measures, fixed before computing:
- Pairing: inside each recording, cycles of arm X paired one to one with the annotation's (arm C) cycles by start
  (Hungarian matching within 150 ms).
- Per pair: S2 onset error (X minus C, ms); for each phase (S1, systole, S2, diastole) the overlap of X's window with
  C's (intersection over union); their mean.
- Per arm (610 recordings labelled for grading, timing or shape): medians over recordings of the per-recording medians;
  share of paired cycles with |S2 error| <= 40 ms; set against the composite gain recovered (Run 1 and Run 2 scores).
- Per recording (arms S, T, W; tasks grading, timing, shape): tertiles of the median |S2 error|; in each tertile the
  share of recordings the annotation classifies correctly and arm X wrongly (a loss), the reverse (a win), and the net
  loss; a rank correlation of |S2 error| with the loss indicator.
- S2 error by murmur timing class (holosystolic, early, mid).
Inputs are verified first: every arm's cycles reproduce the cycle hashes its run recorded (T and W are rebuilt from
Run 2's traces and the analysed audio with extstudy/cycles.py, exactly as Run 2 built them).
Usage:  python notebooks/s2_phase_analysis.py <run1 cycles dir> <exp0 audio cache dir> <output dir>
"""
import hashlib, json, os, pickle, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import cycles as TWC, infer as TWI

PHD = 'E:/NUST MS Thesis/PhD Level Research'
RUN1 = PHD + '/Results/A100 Results/extracted/runs/run1/20261016_1516'
RUN2 = PHD + '/extension_study/runs/run2/20261018_1358'
FOLDS = PHD + '/extension_study/runs/folds_child'
PHASES = ('s1', 'systole', 's2', 'diastole')
SR = 4000
MATCH_MS = 150
ARMS = {'A': 'syn_fixed', 'B': 'syn_fusion', 'S': 'springer_wpk', 'S_pub': 'springer_pub', 'T': 'timing_net', 'W': 'waveform_net',
        'D': 'pcg_seg_fixed', 'E': 'equal_time'}
TASKS = ('grading', 'timing', 'shape')
TIMING_NAMES = {0: 'holosystolic', 1: 'early', 2: 'mid'}


def cycles_hash(cyc_of, names):
    """The run notebooks' cycles_hash (boundaries as plain ints, '|' between recordings, first 16 hex of SHA-1)."""
    h = hashlib.sha1()
    for n in names:
        h.update(json.dumps([[int(x) for k in PHASES for x in c[k]] for c in cyc_of[n]]).encode())
        h.update(b'|')
    return h.hexdigest()[:16]


def equal_time_cycles(n_samples, sr=4000, bpm=75):
    """R1 notebook, verbatim."""
    cycle_len = int(sr * 60 / bpm); n_cycles = max(1, n_samples // cycle_len); out = []
    for i in range(n_cycles):
        s = i*cycle_len; e = min(n_samples, (i+1)*cycle_len); rr = e - s
        if rr <= 0: continue
        out.append({'cycle': i, 's1': (s, s+int(0.25*rr)), 'systole': (s+int(0.25*rr), s+int(0.50*rr)),
                    's2': (s+int(0.50*rr), s+int(0.75*rr)), 'diastole': (s+int(0.75*rr), e)})
    return out


def load_cycles(run1_cycles_dir, audio_dir, keep):
    cyc = {k: {} for k in ['C'] + list(ARMS)}
    o = pickle.load(open(os.path.join(run1_cycles_dir, 'cycles_arms_cdee2.pkl'), 'rb'))['data']
    for n in keep:
        c, _, d, _, e, _, _, _ = o[n]
        cyc['C'][n], cyc['D'][n], cyc['E'][n] = c, d, e
    o = pickle.load(open(os.path.join(run1_cycles_dir, 'cycles_arms_springer.pkl'), 'rb'))['data']
    for n in keep:
        cyc['S'][n], cyc['S_pub'][n] = o[n]['springer_wpk'][0], o[n]['springer_pub'][0]
    o = pickle.load(open(os.path.join(run1_cycles_dir, 'cycles_arms_ab.pkl'), 'rb'))['data']
    for n in keep:
        cyc['A'][n], cyc['B'][n] = o[n][0], o[n][3]
    tw = pickle.load(open(os.path.join(RUN2, 'tw_traces.pkl'), 'rb'))['data']
    need = set(keep)
    for f in sorted(os.listdir(audio_dir)):
        if not (f.startswith('audio_') and f.endswith('.npz')):
            continue
        z = np.load(os.path.join(audio_dir, f), allow_pickle=False)
        meta = json.loads(str(z['meta']))
        for n in [x for x in z['names'].astype(str) if x in need]:
            L4 = meta[n]['L4']
            x4 = np.asarray(z[n + '__4k'][:L4], np.float64)
            raw = tw[n]
            ct, _ = TWC.t_cycles(x4, raw['bt'], raw['p_s2'], TWI.FRAME_T0, TWI.FRAME_S)
            pk4 = np.round(raw['bw'] * SR).astype(int)
            cw, _ = TWC.b_cycles(x4, pk4[(pk4 >= 0) & (pk4 < L4)])
            cyc['T'][n] = ct if len(ct) >= 2 else equal_time_cycles(L4, SR)
            cyc['W'][n] = cw if len(cw) >= 2 else equal_time_cycles(L4, SR)
    assert all(n in cyc['T'] for n in keep), 'audio missing for some recordings'
    return cyc


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else np.nan


def pair_metrics(cx, cc):
    """Paired cycles of X and C (by start, Hungarian within MATCH_MS): per pair S2 error (ms) and phase IoUs."""
    if not cx or not cc:
        return []
    a = np.array([c['s1'][0] for c in cx], float)
    b = np.array([c['s1'][0] for c in cc], float)
    D = np.abs(a[:, None] - b[None, :])
    tol = MATCH_MS / 1000 * SR
    ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e9))
    out = []
    for i, j in zip(ri, ci):
        if D[i, j] > tol:
            continue
        x, c = cx[i], cc[j]
        r = dict(s2_err_ms=(x['s2'][0] - c['s2'][0]) / SR * 1000, start_err_ms=(x['s1'][0] - c['s1'][0]) / SR * 1000)
        for k in PHASES:
            r['iou_' + k] = iou(x[k], c[k])
        r['iou_mean'] = np.nanmean([r['iou_' + k] for k in PHASES])
        out.append(r)
    return out


def main(run1_cycles_dir, audio_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keep = np.load(os.path.join(FOLDS, 'detection.npz'))['keep_names'].astype(str).tolist()
    cyc = load_cycles(run1_cycles_dir, audio_dir, keep)
    rec1 = json.load(open(os.path.join(RUN1, 'run_manifest.json')))['hashes']['feature_cycles_hashes']
    rec2 = json.load(open(os.path.join(RUN2, 'run_manifest.json')))['hashes']['feature_cycles_hashes']
    verify = {}
    for k in ['C'] + list(ARMS):
        key = 'pcg_seg' if k == 'C' else ARMS[k]
        ref = rec2.get(key) if k in ('T', 'W') else rec1.get(key)
        verify[k] = dict(recomputed=cycles_hash(cyc[k], keep), recorded=ref)
    print('cycle hashes:', {k: v['recomputed'] == v['recorded'] for k, v in verify.items()})
    assert all(v['recomputed'] == v['recorded'] for v in verify.values()), 'an arm does not reproduce its run'

    o1 = np.load(os.path.join(RUN1, 'oof_predictions.npz'), allow_pickle=False)
    o2 = np.load(os.path.join(RUN2, 'oof_predictions.npz'), allow_pickle=False)
    names3 = sorted(set().union(*[set(o1['%s__pcg_seg__names' % t].astype(str)) for t in TASKS]))
    rows = []
    for k in ARMS:
        for n in names3:
            pm = pair_metrics(cyc[k][n], cyc['C'][n])
            if not pm:
                rows.append(dict(arm=k, recording=n, n_pairs=0))
                continue
            P = pd.DataFrame(pm)
            rows.append(dict(arm=k, recording=n, n_pairs=len(P), n_c=len(cyc['C'][n]),
                             s2_abs_err_ms=float(P.s2_err_ms.abs().median()), s2_err_ms=float(P.s2_err_ms.median()),
                             s2_within40=float((P.s2_err_ms.abs() <= 40).mean()), start_abs_err_ms=float(P.start_err_ms.abs().median()),
                             **{c: float(P[c].median()) for c in P.columns if c.startswith('iou_')}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_dir, 's2_phase_per_recording.csv'), index=False)

    st = json.load(open(PHD + '/extension_study/runs/run2_scoring/stats_results.json'))['mlp']
    comp, pt = st['composite'], st['per_task']
    Cc, Ec = comp['pcg_seg']['point'], comp['equal_time']['point']
    arm_rows = []
    for k in ARMS:
        g = R[R.arm == k]
        key = ARMS[k]
        arm_rows.append(dict(arm=k, recordings_paired=int((g.n_pairs > 0).sum()), pairs_per_annotated_cycle=float(g.n_pairs.sum() / g.n_c.sum()),
                             s2_abs_err_ms=float(g.s2_abs_err_ms.median()), s2_err_ms=float(g.s2_err_ms.median()), s2_within40=float(g.s2_within40.median()),
                             start_abs_err_ms=float(g.start_abs_err_ms.median()), iou_s1=float(g.iou_s1.median()), iou_systole=float(g.iou_systole.median()),
                             iou_s2=float(g.iou_s2.median()), iou_diastole=float(g.iou_diastole.median()), iou_mean=float(g.iou_mean.median()),
                             composite=comp[key]['point'], retention=(comp[key]['point'] - Ec) / (Cc - Ec),
                             timing_retention=(pt['timing/' + key]['point'] - pt['timing/equal_time']['point']) / (pt['timing/pcg_seg']['point'] - pt['timing/equal_time']['point'])))
    A = pd.DataFrame(arm_rows)
    A.to_csv(os.path.join(out_dir, 's2_phase_per_arm.csv'), index=False)
    rank = dict(iou_mean=spearmanr(A.iou_mean, A.retention).correlation, iou_systole=spearmanr(A.iou_systole, A.retention).correlation,
                s2_within40=spearmanr(A.s2_within40, A.retention).correlation, start_abs=spearmanr(-A.start_abs_err_ms, A.retention).correlation)

    loss_rows = []
    for k, src in (('S', o1), ('T', o2), ('W', o2)):
        key = ARMS[k]
        g = R[R.arm == k].set_index('recording')
        for t in TASKS:
            names = o1['%s__pcg_seg__names' % t].astype(str)
            y = o1['%s__pcg_seg__y' % t]
            c_ok = o1['%s__pcg_seg__mlp__pred' % t] == y
            x_ok = src['%s__%s__mlp__pred' % (t, key)] == y
            assert list(src['%s__%s__names' % (t, key)].astype(str)) == list(names)
            err = g.loc[names, 's2_abs_err_ms'].values
            ok = np.isfinite(err)
            q = np.nanpercentile(err[ok], [100 / 3, 200 / 3])
            terc = np.digitize(err, q)
            loss = (c_ok & ~x_ok).astype(float)
            win = (~c_ok & x_ok).astype(float)
            for b in range(3):
                m = ok & (terc == b)
                loss_rows.append(dict(arm=k, task=t, tertile=b + 1, n=int(m.sum()), s2_abs_err_median_ms=float(np.median(err[m])),
                                      loss_rate=float(loss[m].mean()), win_rate=float(win[m].mean()), net_loss=float(loss[m].mean() - win[m].mean())))
            rho = spearmanr(err[ok], loss[ok])
            loss_rows.append(dict(arm=k, task=t, tertile='all', n=int(ok.sum()), s2_abs_err_median_ms=float(np.median(err[ok])),
                                  loss_rate=float(loss[ok].mean()), win_rate=float(win[ok].mean()), net_loss=float(loss[ok].mean() - win[ok].mean()),
                                  spearman_err_loss=float(rho.correlation), spearman_p=float(rho.pvalue)))
    L = pd.DataFrame(loss_rows)
    L.to_csv(os.path.join(out_dir, 's2_loss_by_tertile.csv'), index=False)

    tim_names = o1['timing__pcg_seg__names'].astype(str)
    tim_y = o1['timing__pcg_seg__y']
    cls_rows = []
    for k in ('S', 'T', 'W', 'B'):
        g = R[R.arm == k].set_index('recording')
        for c, nm in TIMING_NAMES.items():
            sel = tim_names[tim_y == c]
            v = g.loc[sel, 's2_abs_err_ms'].dropna()
            cls_rows.append(dict(arm=k, timing_class=nm, n=len(v), s2_abs_err_median_ms=float(v.median()),
                                 s2_err_median_ms=float(g.loc[sel, 's2_err_ms'].median()), iou_systole=float(g.loc[sel, 'iou_systole'].median())))
    K = pd.DataFrame(cls_rows)
    K.to_csv(os.path.join(out_dir, 's2_by_timing_class.csv'), index=False)
    summary = dict(verified_cycle_hashes=verify, arm_rank_correlations_with_retention=rank, n_recordings=len(names3))
    json.dump(summary, open(os.path.join(out_dir, 's2_phase_summary.json'), 'w'), indent=1)
    pd.set_option('display.width', 230)
    print(A.round(3).to_string(index=False))
    print('rank correlations with composite retention (8 arms):', {k: round(v, 2) for k, v in rank.items()})
    print(L.round(3).to_string(index=False))
    print(K.round(1).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
