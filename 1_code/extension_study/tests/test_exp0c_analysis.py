"""Experiment 0c analysis on stand-in predictions built from Run 1's and Run 2's own files (no Experiment 0c result is
involved): sets are copies of arms C, E, Springer and W, so retention comes out as 1, 0 or the arm's own, Delta as +1 or
-1, and the decompositions close exactly.
Run:  python -m pytest tests/test_exp0c_analysis.py -q   (needs the Run 1 and Run 2 results folders)"""
import json, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'notebooks'))
import exp0c_analysis as EC

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
OOF2 = os.path.join(HERE, 'runs', 'run2', '20261018_1358', 'oof_predictions.npz')
pytestmark = pytest.mark.skipif(not (os.path.exists(OOF1) and os.path.exists(OOF2)), reason='Run 1 or Run 2 results not present')
SUF = EC.SUFFIXES + ('rf__pred',)


def _copy(z, out, tag, src):
    for t in EC.TASKS3:
        for s in SUF:
            out['%s__%s__%s' % (t, tag, s)] = z['%s__%s__%s' % (t, src, s)]


def _standin(tmp_path, random_src, fs_src):
    z1, z2 = np.load(OOF1, allow_pickle=False), np.load(OOF2, allow_pickle=False)
    names = sorted(set().union(*[set(z1['%s__pcg_seg__names' % t].astype(str)) for t in EC.TASKS3]))
    f_s, f_w = names[:163], names[163:232]
    out, b, off = {}, {}, {}
    _copy(z1, out, 'pcg_seg', 'pcg_seg'); _copy(z1, out, 's_all', 'springer_wpk'); _copy(z2, out, 'w_all', 'waveform_net')
    spec = [('s2fs_s%d' % s, z1, fs_src, f_s) for s in range(1, 6)] + [('s2rn_s%02d' % s, z1, random_src, names[-163:]) for s in range(1, 11)]
    spec += [('etfs', z1, fs_src, f_s)] + [('etrn_s%02d' % s, z1, random_src, names[-163:]) for s in range(1, 11)]
    spec += [('s_inside', z1, 'springer_wpk', f_s), ('s_outside', z1, 'pcg_seg', f_s), ('w_inside', z2, 'waveform_net', f_w), ('w_outside', z1, 'pcg_seg', f_w)]
    for tag, z, src, sel in spec + [('pcg_seg', None, None, []), ('s_all', None, None, []), ('w_all', None, None, [])]:
        if z is not None:
            _copy(z, out, tag, src)
        b[tag] = dict(s2_off_gt60_share=0.0, fallback_frac=0.0)
        off[tag] = dict(selected=[n in set(sel) for n in names])
    d = tmp_path / 'exp0c'
    d.mkdir()
    np.savez_compressed(d / 'oof_predictions.npz', **out)
    json.dump(dict(plan_sha256=EC.PLAN_SHA, f_s=f_s, f_w=f_w, conditions=b), open(d / 'exp0c_boundaries.json', 'w'))
    json.dump(dict(plan_sha256=EC.PLAN_SHA, names=names, conditions=off), open(d / 'exp0c_s2_offsets.json', 'w'))
    return str(d)


@pytest.mark.parametrize('random_src,fs_src,decision', [('pcg_seg', 'equal_time', "placement on Springer's failures costs more"),
                                                        ('equal_time', 'pcg_seg', "placement on Springer's failures costs less")])
def test_primary_secondary_descriptive(tmp_path, random_src, fs_src, decision):
    out = EC.main(OOF1, OOF2, _standin(tmp_path, random_src, fs_src), str(tmp_path / 'out'))
    p = out['primary']
    sign = 1.0 if decision.endswith('more') else -1.0
    assert p['decision'] == decision and abs(p['pooled']['delta'] - sign) < 1e-9
    assert all(abs(r['delta'] - sign) < 1e-9 for r in p['per_form'].values())
    for k in ('springer_decomposition', 'w_decomposition'):
        d = out['secondary'][k]
        assert abs(d['loss_outside']) < 1e-9 and abs(d['whole_minus_sum']) < 1e-9 and abs(d['loss_inside'] - d['loss_whole_arm']) < 1e-9
    assert abs(out['groups']['S']['R'] - 0.3344) < 1e-3 and abs(out['groups']['W']['R'] - 0.7021) < 1e-3
    assert all(v['predictions_identical'] for r in out['reproduction'].values() for v in r.values())
    nl, gh = out['descriptive']['net_loss'], out['descriptive']['gain_held']['F_S']
    for t in EC.TASKS3:
        if fs_src == 'equal_time':                        # copies of E lose exactly C's net gain over E, inside and outside F_S
            assert nl['s2fs'][t]['selected'] == pytest.approx(gh[t]['inside'], abs=1e-12)
            assert nl['s2fs'][t]['rest'] == pytest.approx(gh[t]['outside'], abs=1e-12)
        else:                                             # copies of C lose nothing
            assert nl['s2fs'][t]['selected'] == 0.0 and nl['s2fs'][t]['rest'] == 0.0
        assert gh[t]['n_inside'] > 0 and gh[t]['n_outside'] > 0


def test_a_control_that_does_not_reproduce_stops_the_analysis(tmp_path):
    d = _standin(tmp_path, 'pcg_seg', 'equal_time')
    z = dict(np.load(os.path.join(d, 'oof_predictions.npz')))
    z['timing__s_all__mlp__pred'] = z['timing__s_all__mlp__pred'].copy()
    z['timing__s_all__mlp__pred'][0] = (z['timing__s_all__mlp__pred'][0] + 1) % 3
    np.savez_compressed(os.path.join(d, 'oof_predictions.npz'), **z)
    with pytest.raises(AssertionError, match='s_all does not reproduce'):
        EC.main(OOF1, OOF2, d, str(tmp_path / 'out'))
