"""Experiment 0c placement functions. Run:  python -m pytest tests/test_placement.py -q"""
import json, os, pickle, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'notebooks'))
from extstudy import corrupt as K, placement as P
from test_corrupt import annotated, eqt

SETS = os.path.join(HERE, 'runs', 'exp0c_prep', 'exp0c_sets.json')
RUN1_CYC = ('C:/Users/ahmad/AppData/Local/Temp/claude/e--NUST-MS-Thesis-Final-Draft/15d5e8b9-9b00-4710-9b69-d0614bc76cda/'
            'scratchpad/run1_cycles/runs/run1/20261016_1516/cache/cycles_arms_cdee2.pkl')


def test_select_count():
    names = ['r%03d' % i for i in range(610)]
    a = P.select_count(names, 163, 90001)
    assert len(a) == 163 and a == P.select_count(list(reversed(names)), 163, 90001) and a != P.select_count(names, 163, 90002)


def test_unpack_round_trip():
    import exp0c_prep as EP
    cyc = {'a': annotated(n=5), 'b': annotated(n=3, rr=2800)}
    b64, _ = EP.pack(cyc, ['a', 'b'])
    assert P.unpack(b64) == cyc


def test_failure_set_rule():
    ref = annotated(n=20)
    shifted, _ = K.displace_s2(ref, 1.0, K.rng_for(1, 'x'), 1, 100)
    small, _ = K.displace_s2(ref, 1.0, K.rng_for(1, 'x'), 1, 50)
    far = [dict(c, s1=(c['s1'][0] + 2000, c['s1'][1] + 2000)) for c in ref]          # no start within 150 ms
    cx = {'exact': ref, 'shift100': shifted, 'shift50': small, 'nopairs': far}
    cc = {k: ref for k in cx}
    assert P.failure_set(cx, cc, sorted(cx)) == ['nopairs', 'shift100']


def test_conditions():
    ref, oth = annotated(n=10), annotated(n=8, rr=3600)
    L = 10 ** 6
    s, _ = P.exp0c_cycles(ref, 'r1', 's2', 80101, {'r1'}, None, eqt, L)
    assert s == K.exp0b_cycles(ref, 'r1', 's2c', 1.0, 80101, {'r1'}, eqt, L)[0] and s != ref
    assert P.exp0c_cycles(ref, 'r2', 's2', 80101, {'r1'}, None, eqt, L)[0] == ref
    assert P.exp0c_cycles(ref, 'r1', 'fail', 0, {'r1'}, None, eqt, L)[0] == eqt(L)
    assert P.exp0c_cycles(ref, 'r1', 'inside', 0, {'r1'}, oth, eqt, L)[0] == oth
    assert P.exp0c_cycles(ref, 'r2', 'inside', 0, {'r1'}, oth, eqt, L)[0] == ref
    assert P.exp0c_cycles(ref, 'r1', 'outside', 0, {'r1'}, oth, eqt, L)[0] == ref
    assert P.exp0c_cycles(ref, 'r2', 'outside', 0, {'r1'}, oth, eqt, L)[0] == oth
    assert P.exp0c_cycles(ref, 'r1', 'all', 0, set(), oth, eqt, L)[0] == oth
    with pytest.raises(ValueError):
        P.exp0c_cycles(ref, 'r1', 'nope', 0, set(), oth, eqt, L)


@pytest.mark.skipif(not (os.path.exists(SETS) and os.path.exists(RUN1_CYC)), reason='Experiment 0c sets or Run 1 cycles not present')
def test_embedded_cycles_and_sets_recompute():
    import s2_phase_analysis as SP
    d = json.load(open(SETS))
    names = d['names']
    o = pickle.load(open(RUN1_CYC, 'rb'))['data']
    cc = {n: o[n][0] for n in names}
    assert SP.cycles_hash(cc, names) == P.cycles_hash(cc, names) == d['cycles_hash_610_C']
    for k in ('S', 'W'):
        e = d['embedded'][k]
        cyc = P.unpack(e['b64'])
        assert sorted(cyc) == names and P.cycles_hash(cyc, names) == e['cycles_hash_610']
        assert P.failure_set(cyc, cc, names) == d['sets'][k]['recordings']
    assert len(d['sets']['S']['recordings']) == 163 and len(d['sets']['W']['recordings']) == 69
