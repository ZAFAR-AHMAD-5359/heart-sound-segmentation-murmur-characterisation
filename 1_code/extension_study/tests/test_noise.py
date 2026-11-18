"""extstudy.noise: SNR reached within 0.1 dB, determinism per recording name, burst structure, ESC-50 bank loading from
synthetic wav files, and the unit-peak convention. Run:  python -m pytest tests/test_noise.py -q"""
import csv
import os
import sys

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import noise as NZ

SR = 4000


def _heart_like(n, rng):
    """A crude cycle-locked signal: tone bursts at 80 bpm plus faint wideband content, unit peak."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in np.arange(0, n / SR, 0.75):
        m = (t >= k) & (t < k + 0.08)
        x[m] += np.sin(2 * np.pi * 60 * t[m]) * np.hanning(m.sum())
    x += 0.01 * rng.standard_normal(n)
    return x / np.max(np.abs(x))


@pytest.mark.parametrize('kind', ['gauss', 'burst'])
@pytest.mark.parametrize('snr', [10.0, 0.0])
def test_snr_is_reached_and_peak_is_one(kind, snr):
    x = _heart_like(SR * 20, np.random.default_rng(0))
    y, info = NZ.mix(x, kind, snr, NZ.rng_for(1, 'rec_a'), SR)
    assert y.dtype == np.float32 and abs(float(np.max(np.abs(y))) - 1.0) < 1e-6
    assert abs(info['achieved_snr_db'] - snr) < 0.1
    assert info['kind'] == kind and info['peak_rescale'] > 0


def test_deterministic_per_name_and_independent_of_order():
    x = _heart_like(SR * 12, np.random.default_rng(1))
    a1, _ = NZ.mix(x, 'gauss', 0.0, NZ.rng_for(7, 'rec_x'), SR)
    a2, _ = NZ.mix(x, 'gauss', 0.0, NZ.rng_for(7, 'rec_x'), SR)
    b, _ = NZ.mix(x, 'gauss', 0.0, NZ.rng_for(7, 'rec_y'), SR)
    c, _ = NZ.mix(x, 'gauss', 0.0, NZ.rng_for(8, 'rec_x'), SR)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b) and not np.array_equal(a1, c)


def test_noise_is_in_band():
    rng = NZ.rng_for(3, 'r')
    g = NZ.gaussian_noise(SR * 30, SR, rng)
    f = np.fft.rfftfreq(len(g), 1 / SR)
    P = np.abs(np.fft.rfft(g)) ** 2
    inb = P[(f >= 20) & (f <= 500)].sum()
    out_hi = P[f > 800].sum()
    assert inb > 50 * out_hi


def test_bursts_cover_the_expected_time_and_do_not_overlap():
    n = SR * 30
    w, k, secs = NZ.burst_window(n, SR, NZ.rng_for(5, 'b'))
    assert k == 6 and 0.2 * 6 - 0.01 <= secs <= 0.5 * 6 + 0.01
    on = np.flatnonzero(np.diff((w > 0).astype(int)) == 1)
    assert len(on) == k                                   # six separate bursts
    nz, mask, k2, secs2 = NZ.burst_noise(n, SR, NZ.rng_for(5, 'b'))
    assert k2 == k and np.all(nz[~mask] == 0)


def _fake_esc50(root, sr_raw=44100):
    os.makedirs(os.path.join(root, 'meta'))
    os.makedirs(os.path.join(root, 'audio'))
    rng = np.random.default_rng(0)
    rows = []
    for i, cat in enumerate(['crying_baby', 'breathing', 'coughing', 'dog', 'crying_baby']):
        fn = '1-%d-A-%d.wav' % (i, i)
        a = (0.3 * rng.standard_normal(sr_raw * 5) * 32767).astype(np.int16)
        wavfile.write(os.path.join(root, 'audio', fn), sr_raw, a)
        rows.append(dict(filename=fn, fold=1, target=i, category=cat, esc10='False', src_file=str(i), take='A'))
    with open(os.path.join(root, 'meta', 'esc50.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_esc50_bank_and_mix(tmp_path):
    _fake_esc50(str(tmp_path))
    bank = NZ.load_esc50_bank(str(tmp_path), SR)
    assert len(bank['clips']) == 4 and all(c[1] in NZ.ESC50_CLASSES for c in bank['files'])
    assert all(len(c) == SR * 5 for c in bank['clips']) and len(bank['bank_sha256']) == 64
    x = _heart_like(SR * 23, np.random.default_rng(2))
    y, info = NZ.mix(x, 'esc50', 10.0, NZ.rng_for(9, 'rec_e'), SR, bank=bank)
    assert len(y) == len(x) and abs(info['achieved_snr_db'] - 10.0) < 0.1 and len(info['clips']) >= 5


def test_audio_hash_depends_on_content_and_order():
    a = {'x': np.zeros(10, np.float32), 'y': np.ones(10, np.float32)}
    assert NZ.audio_hash(a, ['x', 'y']) != NZ.audio_hash(a, ['y', 'x'])
    b = dict(a, y=np.full(10, 2.0, np.float32))
    assert NZ.audio_hash(a, ['x', 'y']) != NZ.audio_hash(b, ['x', 'y'])
