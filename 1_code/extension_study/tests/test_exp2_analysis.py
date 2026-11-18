"""Experiment 2 analysis on stand-in predictions built from Run 1's and Run 2's own files (no Experiment 2 result is
involved): T_ac is a copy of T, so H1 is exactly 0 with an interval covering 0; T_ac's recovery equals T's; arm E of the
stand-in Run 3 reproduces Run 1. A perturbed E stops the analysis.
Run:  python -m pytest tests/test_exp2_analysis.py -q   (needs the Run 1 and Run 2 results folders)"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'notebooks'))
import exp2_analysis as EX

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
OOF2 = os.path.join(HERE, 'runs', 'run2', '20261018_1358', 'oof_predictions.npz')
pytestmark = pytest.mark.skipif(not (os.path.exists(OOF1) and os.path.exists(OOF2)), reason='Run 1 or Run 2 results not present')
TASKS = ('detection', 'grading', 'timing', 'shape')


def _copy(z, out, tag, src):
    for t in TASKS:
        for s in EX.SUFFIXES + ('mlp__proba_seeds',):
            k = '%s__%s__' % (t, src) + s
            if k in z.files:
                out['%s__%s__' % (t, tag) + s] = z[k]


def _standin(tmp_path, ac_src='timing_net', perturb_e=False):
    z1, z2 = np.load(OOF1, allow_pickle=False), np.load(OOF2, allow_pickle=False)
    out = {}
    _copy(z1, out, 'equal_time', 'equal_time')
    _copy(z2 if ac_src.startswith('timing_net') else z1, out, 'timing_net_ac', ac_src)
    _copy(z2 if ac_src.startswith('timing_net') else z1, out, 'timing_net_ac_annot', ac_src + '_annot' if ac_src != 'equal_time' else 'equal_time_annot')
    for t in TASKS:
        for k in ('y', 'names', 'patient_ids', 'fold_id'):
            out['%s__%s' % (t, k)] = z1['%s__%s' % (t, k)]
    if perturb_e:
        p = out['grading__equal_time__mlp__pred'].copy()
        p[0] = (p[0] + 1) % 3
        out['grading__equal_time__mlp__pred'] = p
    d = tmp_path / 'run3'
    d.mkdir()
    np.savez_compressed(d / 'oof_predictions.npz', arms=np.array(['equal_time', 'timing_net_ac', 'timing_net_ac_annot']), tasks=np.array(list(TASKS)), **out)
    json.dump(dict(run=dict(run_flags=dict(exp2_plan_sha256=EX.PLAN_SHA), fallback_share={'timing_net_ac': 0.0}), diagnostics_summary={},
                   validation_gate=dict(ok=True, folds={}, run1_e_predictions={})), open(d / 'results.json', 'w'))
    return str(d)


def test_a_copy_of_t_gives_h1_zero_and_the_same_recovery(tmp_path):
    out = EX.main(OOF1, OOF2, _standin(tmp_path), str(tmp_path / 'out'))
    assert abs(out['H1']['point']) < 1e-9 and out['H1']['reading'].startswith('no difference')
    assert out['H1']['ci95'][0] <= 0 <= out['H1']['ci95'][1]
    e = out['estimation']
    assert abs(e['T_ac_minus_E']['point'] - e['T_minus_E']['point']) < 1e-9 and abs(e['T_minus_E']['point'] - 6.25) < 0.05
    assert abs(e['recovery_T_ac']['point'] - e['recovery_T']['point']) < 1e-9 and abs(e['recovery_T']['point'] - 0.63) < 0.01
    assert all(v['predictions_identical'] and v['rf_identical'] for v in out['reproduction_E'].values())
    assert abs(out['composite']['timing_net_ac']['point'] - 49.9) < 0.1 and abs(out['composite']['pcg_seg']['point'] - 53.5) < 0.1
    assert 'detection' in out and abs(out['detection']['timing_net_ac']['point'] - out['detection']['timing_net']['point']) < 1e-9


def test_a_copy_of_e_reads_t_above_t_ac(tmp_path):
    out = EX.main(OOF1, OOF2, _standin(tmp_path, ac_src='equal_time'), str(tmp_path / 'out'))
    assert out['H1']['reading'].startswith('T above T_ac') and out['H1']['ci95'][0] > 0
    assert abs(out['estimation']['T_ac_minus_E']['point']) < 1e-9 and abs(out['estimation']['recovery_T_ac']['point']) < 1e-9


def test_an_e_that_does_not_reproduce_run_1_stops_the_analysis(tmp_path):
    with pytest.raises(AssertionError, match='Run 3 arm E does not reproduce'):
        EX.main(OOF1, OOF2, _standin(tmp_path, perturb_e=True), str(tmp_path / 'out'))


def test_a_different_plan_hash_stops_the_analysis(tmp_path):
    d = _standin(tmp_path)
    r = json.load(open(os.path.join(d, 'results.json')))
    r['run']['run_flags']['exp2_plan_sha256'] = 'x'
    json.dump(r, open(os.path.join(d, 'results.json'), 'w'))
    with pytest.raises(AssertionError, match='different plan'):
        EX.main(OOF1, OOF2, d, str(tmp_path / 'out'))
