"""Front end: crop equals whole, gain invariance, torch equals numpy. Run:  python -m pytest tests/test_frontend.py -q"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import frontend as F


@pytest.fixture(scope='module')
def pcg():
    rng = np.random.default_rng(0)
    n = 20 * F.FS
    t = np.arange(n) / F.FS
    x = 0.05 * rng.standard_normal(n)
    for b in np.arange(0.3, 20, 0.8):                        # S1 and S2 bursts
        for c, f, a in ((b + 0.05, 60, 1.0), (b + 0.33, 90, 0.7)):
            x += a * np.exp(-0.5 * ((t - c) / 0.015) ** 2) * np.sin(2 * np.pi * f * t)
    return F.bandpass(x)


def test_shape_and_bins():
    freqs = np.fft.rfftfreq(F.NFFT, 1 / F.FS)[F.BINS]
    assert len(freqs) == 24 and abs(freqs[0] - 31.25) < 1e-9 and abs(freqs[-1] - 390.625) < 1e-9


def test_crop_equals_whole(pcg):
    med = F.bin_medians(pcg)
    whole = F.logspec(pcg, med)
    start = 4 * 1234                                         # on the 4-sample grid
    crop = F.logspec(pcg[start:start + 12 * F.FS], med)
    j0 = start // F.HOP
    assert np.abs(crop - whole[:, j0:j0 + crop.shape[1]]).max() < 1e-4
    assert np.allclose(F.frame_times_s(12 * F.FS, start)[:5], F.frame_times_s(len(pcg))[j0:j0 + 5])


def test_gain_invariance(pcg):
    a, b = F.logspec(pcg), F.logspec(7.3 * pcg)
    assert np.abs(a - b).max() < 1e-3


def test_torch_matches_numpy(pcg):
    torch = pytest.importorskip('torch')
    med = F.bin_medians(pcg)
    x = torch.tensor(pcg[: 12 * F.FS][None, :], dtype=torch.float64)
    t = F.logspec_torch(x, torch.tensor(med[None, :], dtype=torch.float64)).numpy()[0]
    assert np.abs(t - F.logspec(pcg[: 12 * F.FS], med)).max() < 1e-4


def test_bandpass_stable():
    import scipy.signal as sg
    assert np.abs(sg.sos2zpk(F.bandpass_sos())[1]).max() < 0.9999
