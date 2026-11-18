"""Experiment 2 analysis, as fixed in 'Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md': the acoustic-label
control arm T_ac (Run 3) against T (Run 2) on the shared child bootstrap over the predictions of Runs 1, 2 and 3; arm E
of Run 3 must reproduce Run 1. H1 (T minus T_ac, two-sided 95%), the estimation contrasts (T_ac against E, S and S_pub;
recovery at matched coverage; per task; random forest; detection), and the descriptive label-gate and training-beat
figures when their files are present.
Usage:  python notebooks/exp2_analysis.py <run1 oof_predictions.npz> <run2 oof_predictions.npz> <run3 run folder> <output folder>
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

PLAN_SHA = 'f0b9fe2b9ddc76a2ffaa0c170cf13d0392e5e68f7a65a4dcdfbf190e5a31a184'
TASKS3 = ('grading', 'timing', 'shape')
SUFFIXES = ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba', 'rf__pred', 'rf__proba')
FROM_RUN1 = ('pcg_seg', 'equal_time', 'equal_time_annot', 'springer_wpk', 'springer_pub', 'springer_wpk_annot')
FROM_RUN2 = ('timing_net', 'timing_net_annot', 'waveform_net')
FROM_RUN3 = ('timing_net_ac', 'timing_net_ac_annot')
AC_LABELS = 'E:/NUST MS Thesis/PhD Level Research/derived/labels_acoustic_v1'


def _same(za, ka, zb, kb, t, label):
    pa, pb = '%s__%s__' % (t, ka), '%s__%s__' % (t, kb)
    for k in ('names', 'y', 'groups'):
        assert np.array_equal(za[pa + k], zb[pb + k]), '%s: rows differ on %s (%s)' % (label, t, k)
    same = bool(np.array_equal(za[pa + 'mlp__pred'], zb[pb + 'mlp__pred']))
    diff = float(np.max(np.abs(za[pa + 'mlp__proba'].astype(float) - zb[pb + 'mlp__proba'].astype(float))))
    rf = bool(np.array_equal(za[pa + 'rf__pred'], zb[pb + 'rf__pred']))
    assert same and rf and diff < 1e-5, '%s does not reproduce its run on %s' % (label, t)
    return dict(predictions_identical=same, rf_identical=rf, mlp_proba_max_abs_diff=diff)


def merge(oof1, oof2, oof3):
    z1, z2, z3 = (np.load(p, allow_pickle=False) for p in (oof1, oof2, oof3))
    tasks = [t for t in z3['tasks'].astype(str).tolist()]
    check = {t: _same(z3, 'equal_time', z1, 'equal_time', t, 'Run 3 arm E') for t in tasks}
    out = {}
    for t in tasks:
        for k in ('y', 'names', 'patient_ids', 'fold_id'):
            out['%s__%s' % (t, k)] = z1['%s__%s' % (t, k)]
        for z, tags in ((z1, FROM_RUN1), (z2, FROM_RUN2), (z3, FROM_RUN3)):
            for tag in tags:
                for s in SUFFIXES:
                    k = '%s__%s__%s' % (t, tag, s)
                    if k in z.files:
                        out[k] = z[k]
    out['arms'] = np.array(list(FROM_RUN1 + FROM_RUN2 + FROM_RUN3))
    out['tasks'] = np.array(tasks)
    return out, tasks, check


def _contrast(res, a, b, two_sided=True):
    d = res['comp_draws'][a] - res['comp_draws'][b]
    out = dict(point=float(res['comp_point'][a] - res['comp_point'][b]), ci95=stats.interval(d),
               per_task={t: dict(point=float(res['point'][(t, a)] - res['point'][(t, b)]), ci95=stats.interval(res['draws'][(t, a)] - res['draws'][(t, b)]))
                         for t in TASKS3 if (t, a) in res['point']})
    return out


def analyse(merged, tasks):
    tmp = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(tmp, 'oof_predictions.npz'), **merged)
    data, tsk, tags = stats.load_oof(os.path.join(tmp, 'oof_predictions.npz'))
    t3 = [t for t in tsk if t in TASKS3]
    res = stats.analyse(data, t3, sorted(tags))
    rf = stats.analyse(data, t3, sorted(tags), clf='rf')
    T, A, E, S, SP = 'timing_net', 'timing_net_ac', 'equal_time', 'springer_wpk', 'springer_pub'
    h1 = _contrast(res, T, A)
    lo, hi = h1['ci95']
    reading = ('T above T_ac: the ECG teacher contributes to the gain' if lo > 0 else
               'T_ac above T: the acoustic teacher does at least as well' if hi < 0 else 'no difference shown at this precision')
    out = dict(plan_sha256=PLAN_SHA,
               composite={a: dict(point=float(res['comp_point'][a]), ci95=stats.interval(res['comp_draws'][a])) for a in sorted(tags)},
               H1=dict(contrast='timing_net - timing_net_ac', point=h1['point'], ci95=h1['ci95'], reading=reading, per_task=h1['per_task'],
                       rule='two-sided 95% participant-bootstrap interval; no equivalence margin'),
               estimation=dict(T_ac_minus_E=_contrast(res, A, E), T_ac_minus_S=_contrast(res, A, S), T_ac_minus_S_pub=_contrast(res, A, SP),
                               T_minus_E=_contrast(res, T, E), T_minus_S=_contrast(res, T, S),
                               recovery_T_ac=stats.recovery_fraction(res, 'timing_net_ac_annot'),
                               recovery_T=stats.recovery_fraction(res, 'timing_net_annot'),
                               rf_composite={a: float(rf['comp_point'][a]) for a in (E, S, SP, T, A)}),
               per_task={a: {t: float(res['point'][(t, a)]) for t in t3} for a in (E, S, T, A, 'pcg_seg')})
    if 'detection' in tsk:
        det = stats.analyse(data, ['detection'], sorted(tags))
        out['detection'] = {a: dict(point=float(det['point'][('detection', a)]), ci95=stats.interval(det['draws'][('detection', a)]))
                            for a in (E, S, T, A)}
    return out


def descriptive():
    d = {}
    for f, key in (('refcheck_acoustic.json', 'label_gate'), ('events_paired_v3_manifest.json', 'events'), ('laws.json', 'laws')):
        p = os.path.join(AC_LABELS, f)
        if os.path.exists(p):
            v = json.load(open(p))
            d[key] = v if key != 'laws' else {k: v[k] for k in ('s1', 's2', 'n_beats', 'n_subjects') if k in v}
    return d


def main(oof1, oof2, run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    r3 = json.load(open(os.path.join(run_dir, 'results.json')))
    flags = r3['run'].get('run_flags', {})
    assert flags.get('exp2_plan_sha256') == PLAN_SHA, 'Run 3 ran under a different plan'
    merged, tasks, check = merge(oof1, oof2, os.path.join(run_dir, 'oof_predictions.npz'))
    out = analyse(merged, tasks)
    out['reproduction_E'] = check
    out['run3_diagnostics'] = dict(fallback_share=r3['run'].get('fallback_share'), diagnostics_summary=r3.get('diagnostics_summary', {}),
                                   validation_gate={k: v for k, v in (r3.get('validation_gate') or {}).items() if k not in ('folds', 'run1_e_predictions')})
    out['descriptive'] = descriptive()
    json.dump(out, open(os.path.join(out_dir, 'exp2_results.json'), 'w'), indent=1, default=float)
    c = out['composite']
    print('composite: ' + ', '.join('%s %.1f' % (k, c[k]['point']) for k in ('equal_time', 'springer_wpk', 'springer_pub', 'timing_net', 'timing_net_ac', 'pcg_seg')))
    print('H1 T - T_ac: %+.2f [%+.2f, %+.2f] -> %s' % (out['H1']['point'], *out['H1']['ci95'], out['H1']['reading']))
    e = out['estimation']
    for k in ('T_ac_minus_E', 'T_ac_minus_S', 'T_ac_minus_S_pub'):
        print('%-16s %+.2f [%+.2f, %+.2f]' % (k, e[k]['point'], *e[k]['ci95']))
    print('recovery at matched coverage: T_ac %.2f %s | T %.2f %s' % (e['recovery_T_ac']['point'], e['recovery_T_ac']['ci95'],
                                                                     e['recovery_T']['point'], e['recovery_T']['ci95']))
    if 'detection' in out:
        print('detection macro-F1: ' + ', '.join('%s %.1f' % (k, v['point']) for k, v in out['detection'].items()))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:5])
