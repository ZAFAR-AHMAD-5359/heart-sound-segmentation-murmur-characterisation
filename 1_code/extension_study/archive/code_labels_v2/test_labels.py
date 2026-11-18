"""Acoustic S1/S2 labels on synthetic envelopes with known positions. Run:  python -m pytest tests/test_labels.py -q"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import labels as L

FS = L.FS


def synth(rr_s=0.8, n_beats=40, s1_ms=50.0, s2_ms=300.0, seed=0, jitter_rr=0.0, drop_s2=(), irregular_at=None,
          a1=8.0, a2=6.0, s2_amp_at=None, s2_shift_at=None):
    """Envelope with Gaussian S1/S2 bumps on smooth noise (amplitudes well above the height floor of 4 units);
    s2_amp_at / s2_shift_at: {beat: amplitude} / {beat: extra lag in ms} for single beats."""
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
        env += a1 * np.exp(-0.5 * ((t - b - s1_ms / 1000 * FS) / (0.015 * FS)) ** 2)
        if k not in drop_s2:
            a, lag = (s2_amp_at or {}).get(k, a2), s2_ms + (s2_shift_at or {}).get(k, 0.0)
            env += a * np.exp(-0.5 * ((t - b - lag / 1000 * FS) / (0.012 * FS)) ** 2)
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


def test_faint_s2_below_height_floor_is_invalid():
    """Labels v2: an S2 bump 3 units tall (floor 4) cannot be confirmed, so its beat is invalid; neighbours stay."""
    env, beats = synth(s2_amp_at={15: 3.0})
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert not res.loc[res.i == 15, 's2_ok'].iloc[0] and not res.loc[res.i == 15, 'valid'].iloc[0]
    assert res.loc[res.i.isin([14, 16]), 'valid'].all()


def test_mark_far_from_its_neighbours_lag_is_invalid():
    """Labels v2: S2 80 ms later than in the surrounding beats (a mark on another bump) fails the 40 ms consistency
    rule; the deviation is measured against the neighbours' running median, so the other beats keep their labels."""
    env, beats = synth(s2_shift_at={20: 80.0})
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    b = res[res.i == 20].iloc[0]
    assert abs(b['dev2'] - 80.0) < 3 and not b['valid'] and b['s2_ok']
    assert res.loc[(res.i != 20) & res.regular, 'valid'].mean() > 0.95
    assert res.loc[res.valid, 'dev2'].abs().max() < 5


def test_record_needs_five_candidate_beats():
    """Labels v2: with fewer than 5 candidate beats the lag consistency cannot be judged and the record carries no
    labels; 5 candidates are enough (the first beat is never regular, so n beats give n - 1 candidates)."""
    env, beats = synth(n_beats=5)
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert (res.regular & res.s1_ok & res.s2_ok).sum() == 4 and not res.valid.any()
    env, beats = synth(n_beats=6)
    res = L.pass1(env, L.beat_frame(beats, np.ones(len(env), bool)))
    assert res.valid.sum() == 5


def test_lag_deviation_is_relative_to_the_running_median():
    n = 30
    r = np.arange(n) * 400.0
    s1 = r + 25.0; s1[12] = r[12] + 45.0                  # 50 ms lag, one beat at 90 ms (samples at 500 Hz)
    s2 = r + 150.0
    res = pd.DataFrame(dict(r=r, s1=s1, s2=s2))
    cand = np.ones(n, bool); cand[3] = False
    dev = L.lag_deviation(res, cand)
    assert abs(dev['1'][12] - 40.0) < 1e-9 and np.nanmax(np.abs(np.delete(dev['1'], 12))) < 1e-9
    assert np.isnan(dev['1'][3]) and np.isnan(dev['2'][3]) and np.nanmax(np.abs(dev['2'])) < 1e-9
