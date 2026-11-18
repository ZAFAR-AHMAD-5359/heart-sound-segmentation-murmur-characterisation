"""extstudy.labels_acoustic on a synthetic phonocardiogram with known S1 and S2 bursts and injected Springer states:
the marks land on the bursts, regular beats are valid, lags are measured from the cycle start, the R target is
undefined, and the pack events file follows the pack's record order. Run:  python -m pytest tests/test_labels_acoustic.py -q"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from extstudy import labels_acoustic as LA
from extstudy import labels as LB

FS = 4000
PCGECG = 'E:/NUST MS Thesis/PhD Level Research/code'
pytestmark = pytest.mark.skipif(not os.path.isdir(PCGECG), reason='pcgecg code not present')


def _synthetic(n_beats=40, rr_s=0.8, s1_lag=0.04, sys_s=0.30, seed=0):
    """4 kHz signal: S1 burst 40 ms after each cycle start, S2 burst at start + sys_s; state track at 1 kHz."""
    rng = np.random.default_rng(seed)
    n = int((n_beats + 1) * rr_s * FS)
    t = np.arange(n) / FS
    x = 0.002 * rng.standard_normal(n)
    starts = np.arange(n_beats) * rr_s + 0.3
    states = np.zeros(int(n / 4), np.int8)                      # 1 kHz
    for s in starts:
        for c, w in ((s + s1_lag, 0.06), (s + sys_s, 0.05)):
            m = (t >= c - w / 2) & (t < c + w / 2)
            x[m] += np.sin(2 * np.pi * 70 * t[m]) * np.hanning(m.sum())
        a = int(s * 1000)
        states[a:a + 90] = 1                                    # S1 state 90 ms
        states[a + 90:int((s + sys_s) * 1000) - 40] = 2         # systole
        states[int((s + sys_s) * 1000) - 40:int((s + sys_s) * 1000) + 50] = 3   # S2 state 90 ms
        states[int((s + sys_s) * 1000) + 50:int((s + rr_s) * 1000)] = 4        # diastole
    return x, states, starts


def test_marks_land_on_the_bursts_and_lags_are_from_the_cycle_start():
    x4, states, starts = _synthetic()
    if PCGECG not in sys.path:
        sys.path.insert(0, PCGECG)
    from pcgecg.signals import pcg_envelope
    from scipy.signal import resample_poly
    x2 = resample_poly(x4, 1, 2)
    env = pcg_envelope(x2, fs=2000).astype(np.float64)
    cyc = LA.springer_cycles(x4, states_fn=lambda xw, f: states[:int(len(xw) / f * 1000)])
    assert len(cyc) >= 35
    bf = LA.beat_frame_from_cycles(cyc, len(env), np.ones(len(env), bool))
    res = LA.find_acoustic_events(env, bf)
    v = res[res['valid']]
    assert len(v) >= 25
    s1_lag = ((v['s1'] - v['r']) / LB.FS).values
    s2_lag = ((v['s2'] - v['r']) / LB.FS).values
    assert np.all(np.abs(s1_lag - 0.04) < 0.015) and np.all(np.abs(s2_lag - 0.30) < 0.015)
    assert np.all((v['hr'] > 70) & (v['hr'] < 80))


def test_events_for_pack_follows_record_order(tmp_path, monkeypatch):
    phd = tmp_path / 'phd'
    (phd / 'derived' / 'corpus' / 'packX').mkdir(parents=True)
    (phd / 'derived' / 'labX' / 'dsA').mkdir(parents=True)
    pd.DataFrame(dict(record_id=[0, 1, 2], dataset=['dsA', 'dsA', 'dsA'], record=['r0', 'r1', 'r2'])).to_csv(
        phd / 'derived' / 'corpus' / 'packX' / 'records.csv', index=False)
    for rec, n in (('r0', 3), ('r2', 2)):
        np.savez(phd / 'derived' / 'labX' / 'dsA' / (rec + '.npz'),
                 **{k: np.arange(n, dtype=float) for k in LA.EVENT_KEYS if k != 'valid'}, valid=np.array([True] * n))
    monkeypatch.setitem(LA.LOCAL, 'phd', str(phd))
    man = LA.events_for_pack('labX', 'packX')
    z = np.load(phd / 'derived' / 'labX' / 'events_packX.npz')
    assert z['record_id'].tolist() == [0, 0, 0, 2, 2] and man['records_without_labels'] == ['r1']
    assert man['n_valid'] == 5 and np.all(np.isnan(z['r_target_s']) | np.isfinite(z['r_target_s']))
    assert json.load(open(phd / 'derived' / 'labX' / 'events_packX_manifest.json'))['sha256'] == man['sha256']
