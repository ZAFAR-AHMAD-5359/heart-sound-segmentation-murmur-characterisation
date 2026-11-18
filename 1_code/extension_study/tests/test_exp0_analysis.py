"""Experiment 0 analysis on stand-in predictions built from Run 1's own file (no Experiment 0 result is involved):
conditions are copies of arms C, D, E2 and E, so retention must come out as 1 for C copies and 0 for E copies.
Run:  python -m pytest tests/test_exp0_analysis.py -q   (needs the Run 1 results folder)"""
import json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
import exp0_analysis as EA

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
pytestmark = pytest.mark.skipif(not os.path.exists(OOF1), reason='Run 1 results not present')
SRC = {'del': 'pcg_seg', 'ins': 'equal_time', 'jit': 'pcg_seg_fixed'}


def _standin(tmp_path):
    z = np.load(OOF1, allow_pickle=False)
    out, b = {}, {}
    for typ, src in SRC.items():
        for li, lv in enumerate((0.1, 0.2) if typ != 'jit' else (5, 10)):
            for s in (1, 2):
                tag = '%s%03d_s%d' % (typ, int(lv * 100) if typ != 'jit' else lv, s)
                b[tag] = dict(type=typ, level=lv, seed=s, f1_median=1.0 - 0.1 * (li + 1), offset_sd_ms_median=float(lv) if typ == 'jit' else 0.0,
                              fallback_frac=0.0)
                for t in EA.TASKS3:
                    for sfx in EA.SUFFIXES:
                        out['%s__%s__%s' % (t, tag, sfx)] = z['%s__%s__%s' % (t, src, sfx)]
    for t in EA.TASKS3:
        for sfx in EA.SUFFIXES:
            out['%s__pcg_seg__%s' % (t, sfx)] = z['%s__pcg_seg__%s' % (t, sfx)]
    run = tmp_path / 'exp0run'
    run.mkdir()
    np.savez_compressed(run / 'oof_predictions.npz', **out)
    json.dump(dict(plan_sha256='x', conditions=b), open(run / 'exp0_boundaries.json', 'w'))
    return str(run)


def test_retention_and_read_off(tmp_path):
    out = EA.main(OOF1, _standin(tmp_path), str(tmp_path / 'out'))
    assert abs(out['levels']['del_0.1']['retention'] - 1.0) < 1e-9       # copies of C
    assert abs(out['levels']['ins_0.1']['retention'] - 0.0) < 1e-9       # copies of E
    ro = out['read_off']
    assert ro['threshold_missed'] == 0.2 and ro['threshold_spurious'] is None and ro['gate1_boundary_f1_target'] is None
    assert all(v['predictions_identical'] for v in out['arm_c_reproduction'].values())
