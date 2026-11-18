"""Interval warping and augmentation. Run:  python -m pytest tests/test_augment.py -q"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import augment as A

LAWS = dict(s1=dict(intercept={'x': 64.0}, slope={'x': -0.3}), s2=dict(intercept={'x': 470.0}, slope={'x': -1.8}))
FS = A.FRAME_S


def _beats(hr=75.0, n=12, t_first=0.6):
    rr = 60.0 / hr
    s1 = t_first + rr * np.arange(n + 1)
    s2 = s1[:-1] + A.systole_ms(LAWS, 'x', hr) / 1000.0
    return s1, s2


def _spec(n_frames, s1, s2, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 0.1, (24, n_frames)).astype(np.float32)
    for t, v in [(x, 10.0) for x in s1] + [(x, 6.0) for x in s2]:
        L[:, int(round(t / FS))] += v
    return L


@pytest.mark.parametrize('hr_t', [70.0, 110.0, 150.0, 180.0])
def test_kept_windows_copied_exactly_and_labels_follow(hr_t):
    s1, s2 = _beats(75.0)
    L = _spec(4000, s1, s2)
    rng = np.random.default_rng(1)
    src, dst, hr = A.warp_map(s1, s2, np.full(len(s2), 75.0), hr_t, LAWS, 'x', rng)
    assert np.all(np.diff(dst) > 0)
    out, valid = A.warp_frames(L, 0.0, src, dst, 4000, 0.0)
    k = int(round(A.KEEP_S / FS))
    for t in list(s1[:-1]) + list(s2):
        js = int(round(t / FS)); jt = int(round(A.map_times(t, src, dst) / FS))
        assert np.array_equal(out[:, jt - k:jt + k + 1], L[:, js - k:js + k + 1])   # the heart sound, untouched
    for t, v in zip(s1[:-1], s1[1:]):                                              # labels sit on the moved sounds
        a, b = int(round(A.map_times(t, src, dst) / FS)), int(round(A.map_times(v, src, dst) / FS))
        assert a + int(np.argmax(out[0, a - 3:a + 4])) - 3 == a
    rr_out = np.diff(A.map_times(s1, src, dst))
    assert abs(np.median(60.0 / rr_out) - hr) < 0.08 * hr                      # target rate within the 3% jitter


def test_systole_follows_the_paired_law():
    s1, s2 = _beats(75.0)
    rng = np.random.default_rng(2)
    src, dst, hr = A.warp_map(s1, s2, np.full(len(s2), 75.0), 120.0, LAWS, 'x', rng, jitter_cv=0.0)
    sys_out = A.map_times(s2, src, dst) - A.map_times(s1[:-1], src, dst)
    expect = (s2 - s1[:-1]) * A.systole_ms(LAWS, 'x', hr) / A.systole_ms(LAWS, 'x', 75.0)
    assert np.max(np.abs(sys_out - expect)) <= FS / 2 + 1e-9                   # within half a frame (gap rounding)


def test_infeasible_target_rate_is_lowered():
    s1 = 0.6 + 0.9 * np.arange(9); s2 = s1[:-1] + 0.42                         # slow source with a long systole
    rng = np.random.default_rng(3)
    src, dst, hr = A.warp_map(s1, s2, np.full(8, 67.0), 180.0, LAWS, 'x', rng, jitter_cv=0.0)
    assert hr < 180.0
    s1o, s2o = A.map_times(s1, src, dst), A.map_times(s2, src, dst)
    assert np.all(s2o - s1o[:-1] - 2 * A.KEEP_S >= A.MIN_SYS_GAP_S - FS)
    assert np.all(s1o[1:] - s2o - 2 * A.KEEP_S >= A.MIN_DIA_GAP_S - FS)


def test_outside_run_is_zero_and_flagged():
    s1, s2 = _beats(75.0, n=4, t_first=2.0)
    L = _spec(3000, s1, s2)
    src, dst, hr = A.warp_map(s1, s2, np.full(4, 75.0), 140.0, LAWS, 'x', np.random.default_rng(4))
    out, valid = A.warp_frames(L, 0.0, src, dst, 3000, 0.0)
    assert not valid[: int((dst[0] - FS) / FS)].any() and np.all(out[:, ~valid] == 0)


def test_waveform_and_spectrogram_augmentations():
    rng = np.random.default_rng(5)
    fs = 1000; x = np.sin(2 * np.pi * 50 * np.arange(10 * fs) / fs)
    y, snr = A.add_noise(x, rng)
    meas = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert abs(meas - snr) < 0.5
    s1 = np.array([1.0, 2.0, 3.0]); s2 = s1 + 0.3
    z, info = A.murmur_noise(x, s1, s2, fs, rng)
    d = np.abs(z - x) > 0
    inside = np.zeros(len(x), bool)
    for a, b in zip(s1, s2):
        inside[int((a + 0.03) * fs):int((b - 0.02) * fs)] = True
    assert d.any() and not (d & ~inside).any()                                 # only between S1 and S2
    L = np.zeros((24, 10), np.float32)
    T, a = A.device_tilt(L, rng)
    assert abs(T[0, 0] + a) < 1e-6 and abs(T[-1, 0] - a) < 1e-6
    M, (s, w) = A.freq_mask(np.ones((24, 10), np.float32), rng)
    assert np.all(M[s:s + w] == 0) and M.sum() == (24 - w) * 10
