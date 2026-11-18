"""Model shape and size; loss behaviour. Run:  python -m pytest tests/test_model_losses.py -q"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import model as M, losses as L


def test_model_sizes_and_shapes():
    t, w = M.UNet1D('T'), M.UNet1D('W')
    nt = M.n_params(t)
    print('parameters: T %d, W %d' % (nt, M.n_params(w)))
    assert 0.8e6 <= nt <= 1.0e6
    for n in (2993, 3000, 1001):                         # any frame count, including odd lengths
        x = torch.randn(2, 24, n)
        assert t(x).shape == (2, 3, n) and w(x).shape == (2, 1, n)


def test_deterministic_forward():
    torch.manual_seed(0); a = M.UNet1D('T').eval()
    torch.manual_seed(0); b = M.UNet1D('T').eval()
    x = torch.randn(1, 24, 500)
    assert torch.equal(a(x), b(x))


def test_gaussian_targets_peak_and_shape():
    y, peak = L.gaussian_targets([0.200, 0.6012], 300, 0.016, 0.012)
    k = int(round((0.200 - 0.016) * 250))
    assert peak[k] and y[k] == 1.0 and peak.sum() == 2
    assert abs(y[k + 3] - np.exp(-0.5 * (0.012 / 0.012) ** 2)) < 1e-6          # 3 frames = 12 ms = one sigma
    y2, p2 = L.gaussian_targets([5.0], 300, 0.016, 0.012)                       # event outside the crop
    assert not p2.any() and y2.max() == 0


def test_focal_loss_masks_and_prefers_correct_peaks():
    n = 200
    y, peak = L.gaussian_targets([0.4], n, 0.016, 0.012)
    tgt, pk = torch.tensor(y)[None], torch.tensor(peak)[None]
    mask = torch.ones(1, n, dtype=torch.bool)
    good = torch.full((1, n), -6.0); good[0, int(peak.argmax())] = 6.0
    bad = torch.full((1, n), -6.0); bad[0, int(peak.argmax()) + 40] = 6.0
    assert L.focal_event_loss(good, tgt, pk, mask) < L.focal_event_loss(bad, tgt, pk, mask)
    m2 = mask.clone(); m2[0, int(peak.argmax()) + 30:] = False                   # masked frames contribute zero
    junk = good.clone(); junk[0, int(peak.argmax()) + 40:] = 6.0
    assert torch.isclose(L.focal_event_loss(junk, tgt, pk, m2), L.focal_event_loss(good, tgt, pk, m2))


def test_w_loss_masked():
    ecg = torch.randn(2, 100); pred = ecg.clone(); mask = torch.ones(2, 100, dtype=torch.bool)
    assert L.w_loss(pred, ecg, mask)[0].item() == 0.0
    pred[:, 50:] += 5; mask[:, 50:] = False
    assert L.w_loss(pred, ecg, mask)[0].item() < 1e-6
