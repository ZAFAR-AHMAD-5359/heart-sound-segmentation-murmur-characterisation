"""The remaining pre-registered Run 2 estimation analyses (prereg section 3), with the definitions logged in
prereg/CHANGES_AFTER_FREEZE.md on 22 Oct 2026 before any of them was computed: heart-rate bands and infants, the
sensitivity analysis without fallback recordings, and T against the CNN. No p-values. The frozen stats.py supplies the
scoring, the shared child draws (2,000, seed 42), the differences and the recovery fraction; a subset keeps the draws
of the full set, each of its recordings carrying its child's multiplicity.
Usage:  python notebooks/run2_estimation.py <merged oof> <run1 diagnostics.csv> <run2 diagnostics.csv> <cnn oof> <output folder>
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

CNN_SHA = 'd921b7c99e6587bfdb0c88c802a8f7e40d0ae96d88ee8eb0363b7652ddf38689'
K = stats.ARM_KEYS
ARMS = ('E', 'A', 'B', 'S', 'S_pub', 'T', 'W', 'D', 'C')
ANNOT = ('S_annot', 'E_annot', 'T_annot', 'W_annot')
CONTRASTS = (('T', 'E'), ('W', 'E'), ('T', 'S'), ('W', 'S'))
BANDS = (('<100', 0.0, 100.0), ('100-125', 100.0, 125.0), ('125-150', 125.0, 150.0), ('>=150', 150.0, np.inf))
INFANT_AGES = ('Neonate', 'Infant')
CNN_TASK = {'detection': 'murmur', 'grading': 'grading', 'timing': 'timing', 'shape': 'shape'}


def masked_analyse(data, tasks, tags, keep=None, clf='mlp'):
    """stats.analyse restricted to the recordings in `keep` (None: all), with the draws of the full set."""
    stats.check_alignment(data, tasks, tags)
    children = np.unique(np.concatenate([data[(t, tags[0])]['groups'] for t in tasks]))
    D = stats.child_draws(children)
    point, draws, n = {}, {}, {}
    for t in tasks:
        d0 = data[(t, tags[0])]
        m = np.ones(len(d0['y']), bool) if keep is None else np.isin(d0['names'], sorted(keep))
        n[t] = int(m.sum())
        W = D[:, np.searchsorted(children, d0['groups'][m])].astype(float)
        ncls = int(max(data[(t, tag)]['y'].max() for tag in tags) + 1)          # all classes of the task, as stats.analyse
        for tag in tags:
            d = data[(t, tag)]
            if clf not in d:
                continue
            y, p = d['y'][m], d[clf][m]
            point[(t, tag)] = float(stats.macro_f1_from_confusion(stats.weighted_confusions(y, p, ncls, np.ones((1, len(y)))))[0])
            draws[(t, tag)] = stats.macro_f1_from_confusion(stats.weighted_confusions(y, p, ncls, W))
    comp_point, comp_draws = {}, {}
    if all(t in tasks for t in stats.TASKS3):
        for tag in tags:
            if all((t, tag) in point for t in stats.TASKS3):
                comp_point[tag] = float(np.mean([point[(t, tag)] for t in stats.TASKS3]))
                comp_draws[tag] = np.mean([draws[(t, tag)] for t in stats.TASKS3], 0)
    return dict(children=children, point=point, draws=draws, comp_point=comp_point, comp_draws=comp_draws, n=n)


def summarise(res, with_detection=True):
    out = dict(n=res['n'],
               composite={a: dict(point=res['comp_point'][K[a]], ci95=stats.interval(res['comp_draws'][K[a]])) for a in ARMS},
               contrasts={'%s-%s' % (a, b): {k: v for k, v in stats.difference(res, K[a], K[b]).items() if k in ('point', 'ci95')}
                          for a, b in CONTRASTS},
               recovery_fraction={a: stats.recovery_fraction(res, K[a]) for a in ('T_annot', 'W_annot', 'S_annot')})
    if with_detection:
        out['detection'] = {a: dict(point=res['point'][('detection', K[a])], ci95=stats.interval(res['draws'][('detection', K[a])])) for a in ARMS}
    return out


def subsets(diag1):
    c = diag1[diag1.arm == 'pcg_seg'].set_index('recording')
    hr = pd.to_numeric(c['ref_bpm'], errors='coerce')
    out = {name: set(hr.index[(hr >= lo) & (hr < hi)]) for name, lo, hi in BANDS}
    out['infants'] = set(c.index[c['age_band'].isin(INFANT_AGES)])
    return out, int(hr.isna().sum())


def fallback_recordings(diag1, diag2):
    rows = []
    for d, arms in ((diag1, ('springer_wpk_annot', 'equal_time_annot')), (diag2, ('timing_net_annot', 'waveform_net_annot'))):
        s = d[d.arm.isin(arms)]
        fb = pd.to_numeric(s['arm_fallback'].replace({'True': 1, 'False': 0, True: 1, False: 0}), errors='coerce').fillna(0) > 0
        rows.append(s.loc[fb, ['recording', 'arm']])
    r = pd.concat(rows)
    return set(r.recording), r.groupby('arm').size().to_dict()


def cnn_detection(data, cnn_path):
    assert hashlib.sha256(open(cnn_path, 'rb').read()).hexdigest() == CNN_SHA, 'CNN predictions differ from the ledger'
    z = np.load(cnn_path, allow_pickle=True)                      # our own CNN output; object arrays hold the names
    for t, tc in CNN_TASK.items():
        ref = data[(t, K['C'])]
        o = {n: i for i, n in enumerate(z['%s__rec_names' % tc].astype(str))}
        ix = np.array([o[n] for n in ref['names']])
        assert np.array_equal(z['%s__y_true' % tc][ix].astype(int), ref['y']), 'CNN labels differ on %s' % t
        data[(t, 'cnn')] = dict(y=ref['y'], groups=ref['groups'], names=ref['names'], mlp=z['%s__pred' % tc][ix].astype(int))
    tags = [K[a] for a in ('C', 'T', 'W', 'S', 'E')] + ['cnn']
    res = masked_analyse(data, list(CNN_TASK), tags)
    det = {a: dict(point=res['point'][('detection', K[a] if a != 'CNN' else 'cnn')],
                   ci95=stats.interval(res['draws'][('detection', K[a] if a != 'CNN' else 'cnn')])) for a in ('C', 'T', 'W', 'S', 'E', 'CNN')}
    vs = {}
    for a in ('T', 'W', 'S', 'C'):
        d = res['draws'][('detection', K[a])] - res['draws'][('detection', 'cnn')]
        vs['%s-CNN' % a] = dict(point=res['point'][('detection', K[a])] - res['point'][('detection', 'cnn')], ci95=stats.interval(d))
    comp = dict(point=res['comp_point']['cnn'], ci95=stats.interval(res['comp_draws']['cnn']))
    tc = res['comp_draws'][K['T']] - res['comp_draws']['cnn']
    return dict(detection=det, detection_vs_cnn=vs, cnn_composite=comp,
                composite_T_minus_CNN=dict(point=res['comp_point'][K['T']] - res['comp_point']['cnn'], ci95=stats.interval(tc)))


def main(merged, diag1_path, diag2_path, cnn_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data, tasks, tags = stats.load_oof(merged)
    diag1 = pd.read_csv(diag1_path, low_memory=False)
    diag2 = pd.read_csv(diag2_path, low_memory=False)
    out = dict(merged_oof_sha256=hashlib.sha256(open(merged, 'rb').read()).hexdigest(),
               stats_py_sha256=hashlib.sha256(open(stats.__file__, 'rb').read()).hexdigest())
    full = masked_analyse(data, tasks, tags)
    ref = stats.analyse(data, tasks, tags)
    assert all(abs(full['comp_point'][k] - ref['comp_point'][k]) < 1e-12 and np.array_equal(full['comp_draws'][k], ref['comp_draws'][k])
               for k in ref['comp_point']), 'the masked analysis does not reproduce stats.analyse'
    out['full'] = summarise(full)
    sets, n_no_hr = subsets(diag1)
    out['subsets'] = {name: summarise(masked_analyse(data, tasks, tags, keep)) for name, keep in sets.items()}
    out['subsets_note'] = dict(recordings_without_heart_rate=n_no_hr, infant_ages=list(INFANT_AGES))
    fb, by_arm = fallback_recordings(diag1, diag2)
    nofb = set(data[('detection', K['C'])]['names']) - fb
    r = masked_analyse(data, tasks, tags, nofb)
    out['without_fallback'] = dict(n_recordings_removed=len(fb), fallbacks_by_arm=by_arm, n=r['n'],
                                   recovery_fraction={a: stats.recovery_fraction(r, K[a]) for a in ('T_annot', 'W_annot', 'S_annot')})
    out['cnn'] = cnn_detection(data, cnn_path)
    rec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'run1_stats', 'cnn_stats.json')
    if os.path.exists(rec):                                     # the CNN as scored on 17 Oct: same point and interval
        r1 = json.load(open(rec))['per_task']['detection']
        d = out['cnn']['detection']['CNN']
        assert abs(d['point'] - r1['point']) < 1e-9 and np.allclose(d['ci95'], r1['ci95'], atol=1e-9), 'CNN detection does not reproduce cnn_stats.json'
        out['cnn']['reproduces_cnn_stats'] = True
    json.dump(out, open(os.path.join(out_dir, 'run2_estimation.json'), 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    o = main(*sys.argv[1:6])
    f = lambda d: '%.1f [%.1f, %.1f]' % (d['point'], *d['ci95'])
    for name in ['full'] + list(o['subsets']):
        s = o['full'] if name == 'full' else o['subsets'][name]
        print('%-8s n %s | C %s | T %s | W %s | S %s | E %s' % (name, s['n'], f(s['composite']['C']), f(s['composite']['T']), f(s['composite']['W']),
                                                             f(s['composite']['S']), f(s['composite']['E'])))
        print('         ' + ' | '.join('%s %s' % (k, f(v)) for k, v in s['contrasts'].items()))
        print('         recovery: ' + ' | '.join('%s %.2f %s' % (k, v['point'], None if v['ci95'] is None else '[%.2f, %.2f]' % tuple(v['ci95']))
                                                 for k, v in s['recovery_fraction'].items()))
    w = o['without_fallback']
    print('without fallback (%d removed, %s): ' % (w['n_recordings_removed'], w['fallbacks_by_arm']) +
          ' | '.join('%s %.2f [%.2f, %.2f]' % (k, v['point'], *v['ci95']) for k, v in w['recovery_fraction'].items()))
    c = o['cnn']
    print('detection: ' + ' | '.join('%s %s' % (k, f(v)) for k, v in c['detection'].items()))
    print('vs CNN: ' + ' | '.join('%s %s' % (k, f(v)) for k, v in c['detection_vs_cnn'].items()) + ' | composite T-CNN ' + f(c['composite_T_minus_CNN']))
