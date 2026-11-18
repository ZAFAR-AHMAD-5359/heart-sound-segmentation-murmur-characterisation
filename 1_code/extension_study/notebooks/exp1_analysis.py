"""Experiment 1 analysis, as fixed in 'Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md': composite macro-F1
(MLP; grading, timing, shape) with the shared child bootstrap over every arm of the run; the five clean controls must
reproduce Runs 1 and 2; H1 (T_seg_gauss0 minus E above 0, one-sided 0.025) and the estimation contrasts: change from
clean, retention of the clean gain under corrupted segmentation, gain over E per condition, T against Springer under the
same interference, the both-stage contrasts, and the extractor's own sensitivity.
Usage:  python notebooks/exp1_analysis.py <run1 oof_predictions.npz> <run2 oof_predictions.npz> <exp1 run folder> <output folder>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

PLAN_SHA = 'c0f0269d27bfed486240ff3a1ce1ca53c05f57531f6c5e05625d875bc1f3df9b'
TASKS3 = ('grading', 'timing', 'shape')
KINDS = ('gauss', 'burst', 'esc50')
LEVELS = (10, 0)
CONDS = ['%s%d' % (k, l) for k in KINDS for l in LEVELS]
CONTROLS = {'pcg_seg': ('run1', 'pcg_seg'), 'equal_time': ('run1', 'equal_time'), 'springer_wpk': ('run1', 'springer_wpk'),
            'timing_net': ('run2', 'timing_net'), 'waveform_net': ('run2', 'waveform_net')}
BASE = {'S': 'springer_wpk', 'T': 'timing_net', 'W': 'waveform_net', 'E': 'equal_time', 'C': 'pcg_seg'}
SEG_ARMS = ('S', 'T', 'W')
BOTH_ARMS = ('E', 'C', 'S', 'T', 'W')
H1_TAG = 'T_seg_gauss0'


def tags_for(conds):
    out = []
    for c in conds:
        out += ['%s_seg_%s' % (a, c) for a in SEG_ARMS] + ['%s_both_%s' % (a, c) for a in BOTH_ARMS]
    return out


def _same(za, ka, zb, kb, t, label):
    pa, pb = '%s__%s__' % (t, ka), '%s__%s__' % (t, kb)
    for k in ('names', 'y', 'groups'):
        assert np.array_equal(za[pa + k], zb[pb + k]), '%s: rows differ on %s (%s)' % (label, t, k)
    same = bool(np.array_equal(za[pa + 'mlp__pred'], zb[pb + 'mlp__pred']))
    diff = float(np.max(np.abs(za[pa + 'mlp__proba'].astype(float) - zb[pb + 'mlp__proba'].astype(float))))
    rf = bool(np.array_equal(za[pa + 'rf__pred'], zb[pb + 'rf__pred']))
    assert same and rf and diff < 1e-5, '%s does not reproduce its run on %s' % (label, t)
    return dict(predictions_identical=same, rf_identical=rf, mlp_proba_max_abs_diff=diff)


def check_controls(oof1, oof2, oofx):
    z = dict(run1=np.load(oof1, allow_pickle=False), run2=np.load(oof2, allow_pickle=False))
    zx = np.load(oofx, allow_pickle=False)
    return {tag: {t: _same(zx, tag, z[run], src, t, 'Experiment 1 control %s' % tag) for t in TASKS3} for tag, (run, src) in CONTROLS.items()}


def _ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def _contrast(res, a, b):
    d = res['comp_draws'][a] - res['comp_draws'][b]
    return dict(point=float(res['comp_point'][a] - res['comp_point'][b]), ci95=_ci(d), p2_5=float(np.percentile(d, 2.5)),
                per_task={t: dict(point=float(res['point'][(t, a)] - res['point'][(t, b)]), ci95=_ci(res['draws'][(t, a)] - res['draws'][(t, b)]))
                          for t in TASKS3})


def _retention(res, x_c, x, e='equal_time'):
    num = res['comp_draws'][x_c] - res['comp_draws'][e]
    den = res['comp_draws'][x] - res['comp_draws'][e]
    ok = den > 0
    pt = (res['comp_point'][x_c] - res['comp_point'][e]) / (res['comp_point'][x] - res['comp_point'][e])
    return dict(point=float(pt), ci95=_ci(num[ok] / den[ok]) if ok.sum() > 10 else None,
                n_draws_denominator_not_positive=int((~ok).sum()))


def analyse(oof_path, conds):
    data, tasks, tags = stats.load_oof(oof_path)
    assert set(TASKS3) <= set(tasks), 'the three characterisation tasks are needed'
    want = set(tags_for(conds)) | set(CONTROLS)
    assert want <= set(tags), 'arms missing from the predictions: %s' % sorted(want - set(tags))
    res = stats.analyse(data, [t for t in tasks if t in TASKS3], sorted(want))
    E = 'equal_time'
    out = dict(composite={a: dict(point=float(res['comp_point'][a]), ci95=stats.interval(res['comp_draws'][a])) for a in sorted(want)},
               clean_gain={a: _contrast(res, BASE[a], E) for a in SEG_ARMS}, conditions={})
    for c in conds:
        oc = dict(seg={}, both={})
        for a in SEG_ARMS:
            xc, x = '%s_seg_%s' % (a, c), BASE[a]
            oc['seg'][a] = dict(change_from_clean=_contrast(res, xc, x), gain_over_E=_contrast(res, xc, E),
                                retention=_retention(res, xc, x))
        oc['seg']['T_minus_S'] = _contrast(res, 'T_seg_%s' % c, 'S_seg_%s' % c)
        eb, cb = 'E_both_%s' % c, 'C_both_%s' % c
        for a in ('S', 'T', 'W', 'C'):
            xb = '%s_both_%s' % (a, c)
            g = _contrast(res, xb, eb)
            clean = res['comp_draws'][BASE[a]] - res['comp_draws'][E]
            both = res['comp_draws'][xb] - res['comp_draws'][eb]
            oc['both'][a] = dict(gain_over_E_both=g,
                                 change_of_gain=dict(point=float(g['point'] - (res['comp_point'][BASE[a]] - res['comp_point'][E])), ci95=_ci(both - clean)))
        oc['both']['T_minus_S'] = _contrast(res, 'T_both_%s' % c, 'S_both_%s' % c)
        oc['extractor'] = dict(E_both_minus_E=_contrast(res, eb, E), C_both_minus_C=_contrast(res, cb, 'pcg_seg'))
        out['conditions'][c] = oc
    if 'gauss0' in conds:
        h = _contrast(res, H1_TAG, E)
        out['H1'] = dict(contrast='%s - equal_time' % H1_TAG, point=h['point'], ci95=h['ci95'], p2_5=h['p2_5'],
                         rejected=bool(h['p2_5'] > 0), rule='one-sided 0.025: 2.5th percentile of the composite difference above 0')
    else:
        out['H1'] = dict(rejected=None, note='gauss0 not run')
    return out


def main(oof1, oof2, run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    b = json.load(open(os.path.join(run_dir, 'exp1_boundaries.json')))
    assert b['plan_sha256'] == PLAN_SHA, 'Experiment 1 ran under a different plan'
    conds = [c for c in CONDS if c in b['conditions_run']]
    not_run = [c for c in CONDS if c not in conds]
    oofx = os.path.join(run_dir, 'oof_predictions.npz')
    check = check_controls(oof1, oof2, oofx)
    out = analyse(oofx, conds)
    out['reproduction'] = check
    out['plan_sha256'] = b['plan_sha256']
    out['conditions_run'], out['conditions_not_run'] = conds, not_run
    out['diagnostics'] = b.get('diagnostics', {})
    json.dump(out, open(os.path.join(out_dir, 'exp1_results.json'), 'w'), indent=1, default=float)
    print('clean composites: ' + ', '.join('%s %.1f' % (k, out['composite'][BASE[k]]['point']) for k in ('E', 'S', 'T', 'W', 'C')))
    if out['H1'].get('rejected') is not None:
        print('H1 %s: %+.2f [%+.2f, %+.2f], 2.5th pct %+.2f -> %s' % (out['H1']['contrast'], out['H1']['point'], *out['H1']['ci95'],
                                                                         out['H1']['p2_5'], 'REJECTED (gain holds)' if out['H1']['rejected'] else 'not rejected'))
    for c in conds:
        oc = out['conditions'][c]
        print('%-8s seg: ' % c + ' | '.join('%s %+.1f over E [%+.1f, %+.1f], R %.2f' % (
            a, oc['seg'][a]['gain_over_E']['point'], *oc['seg'][a]['gain_over_E']['ci95'], oc['seg'][a]['retention']['point']) for a in SEG_ARMS)
              + ' | T-S %+.1f' % oc['seg']['T_minus_S']['point'])
        print('%-8s both:' % '' + ' | '.join('%s %+.1f over E_both [%+.1f, %+.1f]' % (
            a, oc['both'][a]['gain_over_E_both']['point'], *oc['both'][a]['gain_over_E_both']['ci95']) for a in ('S', 'T', 'W', 'C'))
              + ' | E_both-E %+.1f' % oc['extractor']['E_both_minus_E']['point'])
    if not_run:
        print('conditions not run:', not_run)
    return out


if __name__ == '__main__':
    main(*sys.argv[1:5])
