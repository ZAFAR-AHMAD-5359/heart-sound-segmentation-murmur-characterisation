"""Acoustic S1/S2 labels on synthetic envelopes with known positions. Run:  python -m pytest tests/test_labels.py -q"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import labels as L

FS = L.FS


def synth(rr_s=0.8, n_beats=40, s1_ms=50.0, s2_ms=300.0, seed=0, jitter_rr=0.0, drop_s2=(), irregular_at=None):
    rng = np.random.default_rng(seed)
    rr = np.full(n_beats, rr_s) * (1 + jitter_rr * rng.standard_normal(n_beats))
    if irregular_at is not None:
        rr[irregular_at] *= 0.6
    beats = np.round(np.cumsum(np.r_[0.5, rr]) * FS).astype(int)
    n = beats[-1] + FS
    t = np.arange(n)
    import scipy.signal as sg
    w = sg.sosfiltfilt(sg.butter(4, 20, fs=FS, output='sos'), rng.standard_normal(n))
    env = 0.3 * np.abs(w) / np.std(w)                 # smooth like the 20 Hz low-passed envelope
    for k, b in enumerate(beats):
        env += 6.0 * np.exp(-0.5 * ((t - b - s1_ms / 1000 * FS) / (0.015 * FS)) ** 2)
        if k not in drop_s2:
            env += 4.0 * np.exp(-0.5 * ((t - b - s2_ms / 1000 * FS) / (0.012 * FS)) ** 2)
    return env, beats


def test_pass1_recovers_positions():
    env, beats = synth()
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    v = res[res.valid]
    assert len(v) >= len(res) - 2
    e1, e2 = np.abs((v.s1 - v.r) / FS * 1000 - 50), np.abs((v.s2 - v.r) / FS * 1000 - 300)
    assert np.median(e1) <= 1 and np.median(e2) <= 1 and e1.max() <= 3 and e2.max() <= 3     # noise moves single peaks


def test_irregular_and_masked_beats_are_invalid():
    env, beats = synth(irregular_at=20)
    mask = np.ones(len(env), bool); mask[beats[5]:beats[6] + 10] = False
    res = L.pass1(env, L.beat_frame(beats, mask))
    assert not res.loc[res.i == 20, 'regular'].iloc[0] and not res.loc[res.i == 21, 'regular'].iloc[0]
    assert not res.loc[res.i == 5, 'in_mask'].iloc[0] and not res.loc[res.i == 5, 'valid'].iloc[0]


def test_missing_s2_is_invalid():
    env, beats = synth(drop_s2=(10,))
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert not res.loc[res.i == 10, 's2_ok'].iloc[0]


def test_pass2_window_follows_law_and_edge_rule():
    env, beats = synth(s1_ms=60.0)
    bf = L.beat_frame(beats, np.ones(len(env), bool))
    good = dict(s1=dict(slope=0.0, intercept={'x': 60.0}), s2=dict(slope=0.0, intercept={'x': 300.0}))
    res = L.pass2(env, bf, good, 'x')
    assert res.valid.mean() > 0.9
    off = dict(s1=dict(slope=0.0, intercept={'x': 150.0}), s2=dict(slope=0.0, intercept={'x': 300.0}))
    res = L.pass2(env, bf, off, 'x')                  # S1 outside 150 +/- 40 ms: maximum on the window edge
    assert res.s1_ok.mean() < 0.1


def test_fit_laws_recovers_parameters():
    rng = np.random.default_rng(3)
    rows = []
    for d, a1, b1, a2 in (('a', 60.0, -0.25, 470.0), ('b', 85.0, -0.60, 460.0)):
        for s in range(30):
            hr = rng.uniform(55, 160, 200)
            rows.append(pd.DataFrame(dict(dataset=d, group='%s:%d' % (d, s), hr=hr,
                                          s1_ms=a1 + b1 * hr + 5 * rng.standard_normal(200),
                                          s2_ms=a2 - 1.9 * hr + 8 * rng.standard_normal(200))))
    laws = L.fit_laws(pd.concat(rows))
    assert abs(laws['s1']['slope']['a'] + 0.25) < 0.05 and abs(laws['s1']['slope']['b'] + 0.60) < 0.05
    assert abs(laws['s2']['slope']['a'] + 1.9) < 0.08
    assert abs(L.law_ms(laws['s1'], 'b', 100) - (85 - 60)) < 2 and abs(L.law_ms(laws['s2'], 'a', 100) - (470 - 190)) < 3
    assert laws['n_beats'] == 60 * 100                 # at most 100 beats per subject
    # at 80 bpm: a = 40, b = 37; mean 38.5 -> shift for a is -1.5 ms
    assert abs(L.r_target_shift_ms(laws, 'a') + 1.5) < 2


def test_cycle_with_larger_event_elsewhere_is_invalid():
    """A diastolic event taller than S2 (beat 12): its S2 cannot be verified, so the beat is invalid; others stay."""
    env, beats = synth()
    t = np.arange(len(env))
    b = beats[12]
    env = env + 7.0 * np.exp(-0.5 * ((t - b - 0.60 * FS) / (0.012 * FS)) ** 2)
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert not res.loc[res.i == 12, 'valid'].iloc[0]
    assert res.loc[res.i != 12, 'valid'].mean() > 0.9


def test_record_with_s1_at_the_qrs_loses_labels():
    """S1 bumps 2 ms after the beat in every cycle: physiologically implausible, so the record carries no labels."""
    env, beats = synth(s1_ms=2.0)
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert res['early_s1_record'].all() and not res['valid'].any()
