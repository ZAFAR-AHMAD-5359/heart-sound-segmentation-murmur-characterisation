"""Merge of Run 2 into Run 1 and the frozen stats with t_frozen=True, on stand-in predictions built from Run 1's own
file (T and W copied from S and E, so the hypothesis chain is exercised; no Run 2 result is involved).
Run:  python -m pytest tests/test_merge_run2.py -q   (needs the Run 1 results folder)"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
import merge_run2 as MR

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
pytestmark = pytest.mark.skipif(not os.path.exists(OOF1), reason='Run 1 results not present')


def _standin(tmp_path, perturb_e=False):
    z = np.load(OOF1, allow_pickle=False)
    out = {}
    for t in MR.TASKS:
        for suffix in ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba', 'mlp__proba_seeds', 'rf__pred', 'rf__proba'):
            out['%s__equal_time__%s' % (t, suffix)] = z['%s__equal_time__%s' % (t, suffix)]
            out['%s__timing_net__%s' % (t, suffix)] = z['%s__springer_wpk__%s' % (t, suffix)]
            out['%s__waveform_net__%s' % (t, suffix)] = z['%s__equal_time__%s' % (t, suffix)]
            out['%s__timing_net_annot__%s' % (t, suffix)] = z['%s__springer_wpk_annot__%s' % (t, suffix)]
            out['%s__waveform_net_annot__%s' % (t, suffix)] = z['%s__equal_time_annot__%s' % (t, suffix)]
    if perturb_e:
        p = out['detection__equal_time__mlp__pred'].copy(); p[0] = 1 - p[0]
        out['detection__equal_time__mlp__pred'] = p
    out['arms'] = np.array(['equal_time', 'timing_net', 'waveform_net', 'timing_net_annot', 'waveform_net_annot'])
    path = tmp_path / 'run2_oof.npz'
    np.savez_compressed(path, **out)
    return str(path)


def test_merge_and_hypotheses(tmp_path):
    res, rep = MR.main(OOF1, _standin(tmp_path), str(tmp_path / 'out'))
    h = res['mlp']['hypotheses']
    assert set(h) >= {'H1', 'H2'} and all(v['predictions_identical'] for v in rep['e_reproduced'].values())
    # T stand-in = S, so T - E equals S - E and both get the same verdict
    assert abs(h['H2']['point'] - h['H1']['point']) < 1e-9 and h['H2']['rejected'] == h['H1']['rejected']
    assert 'timing_net_annot' in res['mlp']['recovery_fraction']


def test_merge_refuses_when_e_does_not_reproduce(tmp_path):
    with pytest.raises(AssertionError):
        MR.main(OOF1, _standin(tmp_path, perturb_e=True), str(tmp_path / 'out'))
