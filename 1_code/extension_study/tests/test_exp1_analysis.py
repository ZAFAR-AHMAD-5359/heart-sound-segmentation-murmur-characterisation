"""Experiment 1 analysis on stand-in predictions built from Run 1's and Run 2's own files (no Experiment 1 result is
involved): condition arms are copies of the clean controls, so the change from clean is 0, retention is 1 (copy of the
arm) or 0 (copy of E), and H1 reproduces the T minus E verdict of Run 2.
Run:  python -m pytest tests/test_exp1_analysis.py -q   (needs the Run 1 and Run 2 results folders)"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'notebooks'))
import exp1_analysis as EX

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
OOF2 = os.path.join(HERE, 'runs', 'run2', '20261018_1358', 'oof_predictions.npz')
pytestmark = pytest.mark.skipif(not (os.path.exists(OOF1) and os.path.exists(OOF2)), reason='Run 1 or Run 2 results not present')
SUF = ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba', 'rf__pred', 'rf__proba', 'mlp__proba_seeds')


def _copy(z, out, tag, src):
    for t in EX.TASKS3:
        for s in SUF:
            k = '%s__%s__%s' % (t, src, s)
            if k in z.files:
                out['%s__%s__%s' % (t, tag, s)] = z[k]


def _standin(tmp_path, seg_src, conds=('gauss10', 'gauss0')):
    """seg_src: 'self' (copies of each arm) or 'equal_time' (every corrupted arm a copy of E)."""
    z1, z2 = np.load(OOF1, allow_pickle=False), np.load(OOF2, allow_pickle=False)
    out = {}
    for tag, (run, src) in EX.CONTROLS.items():
        _copy(z1 if run == 'run1' else z2, out, tag, src)
    for t in EX.TASKS3:
        for k in ('y', 'names', 'patient_ids', 'fold_id'):
            out['%s__%s' % (t, k)] = z1['%s__%s' % (t, k)]
    for c in conds:
        for a in EX.SEG_ARMS:
            base = EX.BASE[a]
            src = base if seg_src == 'self' else 'equal_time'
            _copy(z2 if base in ('timing_net', 'waveform_net') and src == base else z1, out, '%s_seg_%s' % (a, c), src)
        for a in EX.BOTH_ARMS:
            base = EX.BASE[a]
            _copy(z2 if base in ('timing_net', 'waveform_net') else z1, out, '%s_both_%s' % (a, c), base)
    tags = EX.tags_for(list(conds)) + list(EX.CONTROLS)
    d = tmp_path / 'exp1'
    d.mkdir()
    np.savez_compressed(d / 'oof_predictions.npz', arms=np.array(tags), tasks=np.array(list(EX.TASKS3)), **out)
    json.dump(dict(plan_sha256=EX.PLAN_SHA, conditions_run=list(conds), diagnostics={}), open(d / 'exp1_boundaries.json', 'w'))
    return str(d)


def test_copies_of_the_arms_give_zero_change_and_unit_retention(tmp_path):
    out = EX.main(OOF1, OOF2, _standin(tmp_path, 'self'), str(tmp_path / 'out'))
    for c in ('gauss10', 'gauss0'):
        for a in EX.SEG_ARMS:
            s = out['conditions'][c]['seg'][a]
            assert abs(s['change_from_clean']['point']) < 1e-9 and abs(s['retention']['point'] - 1.0) < 1e-9
            assert abs(s['gain_over_E']['point'] - out['clean_gain'][a]['point']) < 1e-9
        for a in ('S', 'T', 'W', 'C'):
            assert abs(out['conditions'][c]['both'][a]['change_of_gain']['point']) < 1e-9
        assert abs(out['conditions'][c]['extractor']['E_both_minus_E']['point']) < 1e-9
    assert out['H1']['rejected'] is True and abs(out['H1']['point'] - 6.25) < 0.05       # Run 2's T minus E
    assert all(v['predictions_identical'] for r in out['reproduction'].values() for v in r.values())
    assert out['conditions_not_run'] == ['burst10', 'burst0', 'esc5010', 'esc500']


def test_copies_of_e_give_zero_retention_and_no_rejection(tmp_path):
    out = EX.main(OOF1, OOF2, _standin(tmp_path, 'equal_time'), str(tmp_path / 'out'))
    for a in EX.SEG_ARMS:
        s = out['conditions']['gauss0']['seg'][a]
        assert abs(s['retention']['point']) < 1e-9 and abs(s['gain_over_E']['point']) < 1e-9
    assert out['H1']['rejected'] is False and abs(out['H1']['p2_5']) < 1e-9


def test_a_control_that_does_not_reproduce_stops_the_analysis(tmp_path):
    d = _standin(tmp_path, 'self')
    z = dict(np.load(os.path.join(d, 'oof_predictions.npz')))
    z['timing__timing_net__mlp__pred'] = z['timing__timing_net__mlp__pred'].copy()
    z['timing__timing_net__mlp__pred'][0] = (z['timing__timing_net__mlp__pred'][0] + 1) % 3
    np.savez_compressed(os.path.join(d, 'oof_predictions.npz'), **z)
    with pytest.raises(AssertionError, match='timing_net does not reproduce'):
        EX.main(OOF1, OOF2, d, str(tmp_path / 'out'))
