"""Revision checks defined in prereg/CHANGES_AFTER_FREEZE.md (23 Oct 2026) before computing, after an independent
pre-submission review. All descriptive, with the shared participant bootstrap of stats.py (2,000 draws, seed 42):
(1) T - S_pub, W - S_pub, S_pub - S;
(2) contrasts with the reference at matched coverage only: C - T_annot, C - W_annot, C - S_annot;
(3) sensitivity without the pregnant participants: composite per arm, T - E, W - E, T - S, W - S, recovery fractions;
(4) end-to-end characterisation (Reviewer 3, R3-M3): a murmur-present recording that the same arm's detection model
    calls absent counts as a characterisation failure (a false negative for its true class, a false positive for no
    class); macro-F1 over the true classes and the composite, for E, S, S_pub, T, W, D, C, and the CNN.
Usage:  python notebooks/revision_checks.py <merged oof> <cnn oof> <circor training_data.csv> <output folder>
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extstudy import stats
import run2_estimation as RE

K = stats.ARM_KEYS
TASKS3 = stats.TASKS3
iv = stats.interval


def diff(res, a, b):
    d = stats.difference(res, a, b)
    return dict(point=d['point'], ci95=d['ci95'])


def macro_f1_end_to_end(y, pred, missed, ncls, W):
    """W: (B, n) weights. A missed recording adds a false negative for its true class only."""
    out = np.zeros(len(W))
    f1s = []
    for c in range(ncls):
        tp = W @ ((y == c) & (pred == c) & ~missed).astype(float)
        fp = W @ ((y != c) & (pred == c) & ~missed).astype(float)
        fn = W @ ((y == c) & ((pred != c) | missed)).astype(float)
        den = 2 * tp + fp + fn
        f1s.append(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0))
    return 100 * np.mean(f1s, 0)


def end_to_end(data, children):
    D = stats.child_draws(children)
    det = {tag: dict(zip(data[('detection', tag)]['names'], data[('detection', tag)]['mlp'])) for tag in
           [K[a] for a in ('E', 'S', 'S_pub', 'T', 'W', 'D', 'C')] + ['cnn']}
    out = {}
    for tag in det:
        comp_p, comp_d, per = [], [], {}
        for t in TASKS3:
            d = data[(t, tag)]
            W = D[:, np.searchsorted(children, d['groups'])].astype(float)
            ncls = int(d['y'].max() + 1)
            missed = np.array([det[tag][n] == 0 for n in d['names']])
            pt = float(macro_f1_end_to_end(d['y'], d['mlp'], missed, ncls, np.ones((1, len(d['y']))))[0])
            dr = macro_f1_end_to_end(d['y'], d['mlp'], missed, ncls, W)
            per[t] = dict(point=pt, ci95=iv(dr), share_missed=float(missed.mean()))
            comp_p.append(pt)
            comp_d.append(dr)
        out[tag] = dict(composite=dict(point=float(np.mean(comp_p)), ci95=iv(np.mean(comp_d, 0)), draws=np.mean(comp_d, 0)), per_task=per)
    contrasts = {}
    for a, b in (('timing_net', 'equal_time'), ('waveform_net', 'equal_time'), ('timing_net', 'springer_wpk'), ('waveform_net', 'springer_wpk'),
                 ('timing_net', 'cnn'), ('waveform_net', 'cnn')):
        dd = out[a]['composite']['draws'] - out[b]['composite']['draws']
        contrasts['%s-%s' % (a, b)] = dict(point=out[a]['composite']['point'] - out[b]['composite']['point'], ci95=iv(dd))
    for v in out.values():
        v['composite'].pop('draws')
    return out, contrasts


def main(merged, cnn_path, meta_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data, tasks, tags = stats.load_oof(merged)
    res = RE.masked_analyse(data, tasks, tags)
    out = dict(spub=dict(T_minus_Spub=diff(res, K['T'], K['S_pub']), W_minus_Spub=diff(res, K['W'], K['S_pub']), Spub_minus_S=diff(res, K['S_pub'], K['S'])),
               matched_coverage_vs_C={x: diff(res, K['C'], K[x]) for x in ('T_annot', 'W_annot', 'S_annot')})
    meta = pd.read_csv(meta_csv)
    preg = set(meta.loc[meta['Pregnancy status'] == True, 'Patient ID'].astype(str))
    names = data[('detection', K['C'])]['names']
    keep = set(n for n in names if n.split('_')[0] not in preg)
    r = RE.masked_analyse(data, tasks, tags, keep)
    out['without_pregnant'] = dict(n_removed_detection=int(len(names) - len(keep)), n=r['n'],
                                   composite={a: dict(point=r['comp_point'][K[a]], ci95=iv(r['comp_draws'][K[a]])) for a in ('E', 'S', 'S_pub', 'T', 'W', 'D', 'C')},
                                   contrasts={'%s-%s' % (a, b): diff(r, K[a], K[b]) for a, b in (('T', 'E'), ('W', 'E'), ('T', 'S'), ('W', 'S'))},
                                   recovery_fraction={x: stats.recovery_fraction(r, K[x]) for x in ('T_annot', 'W_annot', 'S_annot')})
    RE.cnn_detection(data, cnn_path)                       # adds the CNN predictions to `data` for every task (checked against the ledger)
    children = np.unique(np.concatenate([data[(t, K['C'])]['groups'] for t in tasks]))
    out['end_to_end'], out['end_to_end_contrasts'] = end_to_end(data, children)
    json.dump(out, open(os.path.join(out_dir, 'revision_checks.json'), 'w'), indent=1, default=float)
    f = lambda d: '%.2f [%.2f, %.2f]' % (d['point'], *d['ci95'])
    print('S_pub:', {k: f(v) for k, v in out['spub'].items()})
    print('matched coverage C - X:', {k: f(v) for k, v in out['matched_coverage_vs_C'].items()})
    w = out['without_pregnant']
    print('without pregnant (removed %d):' % w['n_removed_detection'], w['n'], {k: f(v) for k, v in w['composite'].items()})
    print('   contrasts', {k: f(v) for k, v in w['contrasts'].items()}, {k: '%.2f' % v['point'] for k, v in w['recovery_fraction'].items()})
    print('end-to-end composite:', {k: f(v['composite']) for k, v in out['end_to_end'].items()})
    print('   share missed (grading):', {k: round(v['per_task']['grading']['share_missed'], 3) for k, v in out['end_to_end'].items()})
    print('   contrasts:', {k: f(v) for k, v in out['end_to_end_contrasts'].items()})
    return out


if __name__ == '__main__':
    main(*sys.argv[1:5])
