"""Experiment 0b analysis on stand-in predictions built from Run 1's own file (no Experiment 0 or 0b result is
involved): conditions are copies of arms C and E, so retention must come out as 1 or 0 and the pooled Delta as +1 or -1,
with the decision that follows.
Run:  python -m pytest tests/test_exp0b_analysis.py -q   (needs the Run 1 results folder)"""
import json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
import exp0b_analysis as EB

OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
pytestmark = pytest.mark.skipif(not os.path.exists(OOF1), reason='Run 1 results not present')
LEVELS = {'s2c': EB.S2_LEVELS, 's2s': EB.S2_LEVELS, 'delc': EB.DELC_LEVELS, 'fail': EB.FAIL_LEVELS}
NAMES = ['r%02d' % i for i in range(10)]


def _copy(z, out, tag, src):
    for t in EB.TASKS3:
        for sfx in EB.SUFFIXES:
            out['%s__%s__%s' % (t, tag, sfx)] = z['%s__%s__%s' % (t, src, sfx)]


def _standin(tmp_path, src):
    z = np.load(OOF1, allow_pickle=False)
    o0, b0 = {}, {}
    _copy(z, o0, 'pcg_seg', 'pcg_seg')
    for lv in EB.DEL_LEVELS:
        for s in range(1, 6):
            tag = 'del%03d_s%d' % (round(lv * 100), s)
            _copy(z, o0, tag, src['del'])
            b0[tag] = dict(type='del', level=lv, seed=s, f1_median=0.9, fallback_frac=0.0)
    _copy(z, o0, 'del050_s1', 'equal_time')                       # an Experiment 0 condition that is not reused
    b0['del050_s1'] = dict(type='del', level=0.5, seed=1, f1_median=0.7, fallback_frac=0.0)
    ob, bb, off = {}, {}, {}
    _copy(z, ob, 'pcg_seg', 'pcg_seg')
    for typ, lvls in LEVELS.items():
        for li, lv in enumerate(lvls):
            for s in range(1, 6):
                tag = '%s%03d_s%d' % (typ, round(lv * 100), s)
                _copy(z, ob, tag, src[typ])
                k = int(round(lv * len(NAMES))) if typ != 's2s' else 0
                sel = [i < k for i in range(len(NAMES))]
                cyc = [20] * len(NAMES)
                disp = ([20 * x for x in sel] if typ == 's2c' else [int(round(lv * 20))] * len(NAMES) if typ == 's2s' else [0] * len(NAMES))
                offs = [100.0 if (x and typ in ('s2c', 'fail')) else 0.0 for x in sel]
                bb[tag] = dict(type=typ, level=lv, seed=10000 + 100 * li + s, n_selected=sum(sel), cycles=sum(cyc), cycles_displaced=sum(disp),
                               share_cycles_displaced=sum(disp) / sum(cyc), s2_off_gt60_share=float(np.mean(np.abs(offs) > 60)),
                               f1_median=1.0, fallback_frac=0.0)
                off[tag] = dict(offset_ms=offs, pairs=[20] * len(NAMES), selected=sel, sign=[1] * len(NAMES), cycles=cyc, displaced=disp)
    d0, db = tmp_path / 'exp0', tmp_path / 'exp0b'
    d0.mkdir(); db.mkdir()
    np.savez_compressed(d0 / 'oof_predictions.npz', **o0)
    np.savez_compressed(db / 'oof_predictions.npz', **ob)
    json.dump(dict(plan_sha256='exp0', conditions=b0), open(d0 / 'exp0_boundaries.json', 'w'))
    json.dump(dict(plan_sha256=EB.PLAN_SHA, conditions=bb), open(db / 'exp0b_boundaries.json', 'w'))
    json.dump(dict(plan_sha256=EB.PLAN_SHA, names=NAMES, conditions=off), open(db / 'exp0b_s2_offsets.json', 'w'))
    return str(d0), str(db)


@pytest.mark.parametrize('conc,spread,decision', [('pcg_seg', 'equal_time', 'concentration costs less'),
                                                  ('equal_time', 'pcg_seg', 'concentration costs more')])
def test_delta_and_decision(tmp_path, conc, spread, decision):
    src = dict(s2c=conc, s2s=spread, delc='pcg_seg', fail='pcg_seg_fixed', **{'del': 'equal_time'})
    d0, db = _standin(tmp_path, src)
    out = EB.main(OOF1, d0, db, str(tmp_path / 'out'))
    p = out['primary_s2']
    sign = 1.0 if decision.endswith('more') else -1.0
    assert abs(p['pooled']['delta'] - sign) < 1e-9 and p['decision'] == decision
    assert all(abs(r['delta'] - sign) < 1e-9 for r in p['per_level'])
    assert abs(out['secondary_missed']['pooled']['delta'] + 1.0) < 1e-9                 # spread (E copies) minus concentrated (C copies)
    assert 'del_0.5' not in out['levels'] and len(out['levels']) == 15
    assert all(v['predictions_identical'] for r in out['arm_c_reproduction'].values() for v in r.values())
    assert out['offsets_check']['recomputed'] and all(abs(r['concentrated'] - r['spread']) < 1e-9 for r in out['displacement_check_per_recording'])
    assert any(r['source'] == 'Run 1 and Run 2 arms' for r in out['transfer']) == os.path.exists(EB.ARMS_CSV)


def test_plan_hash_is_enforced(tmp_path):
    src = dict(s2c='pcg_seg', s2s='equal_time', delc='pcg_seg', fail='pcg_seg', **{'del': 'equal_time'})
    d0, db = _standin(tmp_path, src)
    b = json.load(open(os.path.join(db, 'exp0b_boundaries.json')))
    b['plan_sha256'] = 'other'
    json.dump(b, open(os.path.join(db, 'exp0b_boundaries.json'), 'w'))
    with pytest.raises(AssertionError, match='different plan'):
        EB.main(OOF1, d0, db, str(tmp_path / 'out'))
