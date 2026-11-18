"""Shared beat decoder (Panel H, decision 2). Run:  python -m pytest tests/test_decoder.py -q"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import decoder as D, robust_beats as RB, sim_ecg as SIM

HRS = [40, 60, 80, 100, 120, 150, 180, 200, 220]


def _prf(truth_s, beats_s, hr):
    """Precision, recall, F1 at the pre-registered tolerance min(50 ms, 10% RR)."""
    tol = min(0.050, 0.10 * 60.0 / hr)
    tp, fp, fn, _ = RB.match(np.round(np.asarray(truth_s) * 1000).astype(int), np.round(np.asarray(beats_s) * 1000).astype(int),
                             int(round(tol * 1000)))
    return tp / max(1, tp + fp), tp / max(1, tp + fn), RB.f1(tp, fp, fn)


def _f1(truth_s, beats_s, hr):
    return _prf(truth_s, beats_s, hr)[2]


@pytest.fixture(scope='module')
def suite():
    rows = []
    for i, hr in enumerate(HRS):
        for j in range(4):
            prem = 0.05 if j == 3 else 0.0
            x, truth, shown = SIM.simulate_events(hr, 500 + 10 * i + j, premature=prem, return_shown=True)
            o = D.decode_beats(x, 250, kind='event')
            p, r, f = _prf(truth[shown], o['beats_s'], hr)          # beats that leave no trace cannot be decoded
            rows.append(dict(hr=hr, rep=j, f1=f, precision=p, cov=o['rec']['coverage']))
    return rows


def test_event_suite(suite):
    """Against beats that carry a bump: median F1 >= 0.97, 10th percentile >= 0.90, median precision >= 0.98."""
    f = np.array([r['f1'] for r in suite])
    assert np.median([r['precision'] for r in suite]) >= 0.98
    by_hr = {hr: round(float(np.mean([r['f1'] for r in suite if r['hr'] == hr])), 3) for hr in HRS}
    print('event suite F1 by rate:', by_hr, '| median %.3f, 10th percentile %.3f' % (np.median(f), np.percentile(f, 10)))
    assert np.median(f) >= 0.97 and np.percentile(f, 10) >= 0.90


def test_s2_leak_removed_by_s2_head():
    """Beat head leaking S2 at 0.6 to 0.9 with 15% of beats missing: decoding the raw head can follow the S2 chain;
    decoding beat minus S2 head must keep the beat sequence."""
    for hr in (60, 90, 150):
        x, truth, s2, shown = SIM.simulate_events(hr, 77 + hr, s2_amp=(0.6, 0.9), p_miss=0.15, spur_per_min=0,
                                                  return_s2_head=True, return_shown=True)
        raw = _f1(truth[shown], D.decode_beats(x, 250, kind='event')['beats_s'], hr)
        con = _f1(truth[shown], D.decode_beats(D.beat_trace_from_heads(x, s2), 250, kind='event')['beats_s'], hr)
        print('S2 leak at %d bpm: raw head F1 %.3f, beat minus S2 head F1 %.3f' % (hr, raw, con))
        assert con >= 0.95, (hr, con)


def test_missing_beats_are_masked_not_invented():
    x, truth = SIM.simulate_events(80, 9, p_miss=0.15, spur_per_min=0)
    o = D.decode_beats(x, 250, kind='event')
    b = o['beats_s']
    tp, fp, fn, _ = RB.match(np.round(truth * 1000).astype(int), np.round(b * 1000).astype(int), 50)
    assert fp <= 2, fp
    assert o['rec']['coverage'] < 0.95


@pytest.mark.parametrize('cond', ['clean', 'tallT', 'deepS_inv', 'premature', 'missing'])
def test_ecg_kind_on_simulated_ecg(cond):
    """W's read-out on an ECG-like trace over 40 to 200 bpm: mean precision >= 0.99 everywhere; mean F1 >= 0.97,
    except 0.94 with premature ventricular beats (amplitude alone misses part of the wide beats; the gap is masked)."""
    pr = []
    for i, hr in enumerate(HRS[:-1]):
        x, truth, _, _ = SIM.simulate(hr, cond, 900 + 7 * i)
        o = D.decode_beats(x, SIM.FS, kind='ecg')
        pr.append(_prf(truth / SIM.FS, o['beats_s'], hr))
    pr = np.array(pr)
    print(cond, 'precision', np.round(pr[:, 0], 3), 'F1', np.round(pr[:, 2], 3))
    assert pr[:, 0].mean() >= 0.99 and pr[:, 2].mean() >= (0.94 if cond == 'premature' else 0.97), cond


def test_flat_and_empty_traces():
    o = D.decode_beats(np.zeros(250 * 20), 250, kind='event')
    assert o['rec']['n_beats'] == 0 and not o['mask500'].any()
