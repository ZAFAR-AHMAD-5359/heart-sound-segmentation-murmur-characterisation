"""Losses for T and W (08 design, section 2.7; prereg section 7).

T: penalty-reduced focal loss (CenterNet: alpha 2, beta 4) on Gaussian event targets at the frame rate (sigma 12 ms
for S1 and S2, 8 ms for R), head weights 1, 1, 0.5, normalised by the number of events, masked frames contributing
zero (invalid cycles are masked, never background).
W: L1 on the ECG and on its first difference, masked the same way.
Targets are built from event times in seconds; frame j is centred at frame_t0 + j / 250 s.
"""
import numpy as np
import torch

FRAME_HZ = 250.0
SIGMA_S = {'s1': 0.012, 's2': 0.012, 'r': 0.008}
HEAD_WEIGHT = {'s1': 1.0, 's2': 1.0, 'r': 0.5}
ALPHA, BETA = 2.0, 4.0


def gaussian_targets(event_s, n_frames, frame_t0, sigma_s):
    """(n_frames,) heatmap: max over events of exp(-(t - e)^2 / 2 sigma^2), with exactly 1 at each event's nearest frame
    (the CenterNet peak), and a boolean array marking those peak frames."""
    t = frame_t0 + np.arange(n_frames) / FRAME_HZ
    y = np.zeros(n_frames, np.float32)
    peak = np.zeros(n_frames, bool)
    for e in np.asarray(event_s, float):
        if not np.isfinite(e) or e < t[0] - 5 * sigma_s or e > t[-1] + 5 * sigma_s:
            continue
        lo, hi = np.searchsorted(t, e - 5 * sigma_s), np.searchsorted(t, e + 5 * sigma_s)
        y[lo:hi] = np.maximum(y[lo:hi], np.exp(-0.5 * ((t[lo:hi] - e) / sigma_s) ** 2))
        k = int(np.clip(np.round((e - frame_t0) * FRAME_HZ), 0, n_frames - 1))
        if abs(t[k] - e) <= 0.5 / FRAME_HZ + 1e-9:
            y[k], peak[k] = 1.0, True
    return y, peak


def focal_event_loss(logits, target, peak, mask):
    """logits, target (float), peak (bool), mask (bool): (B, n). Penalty-reduced focal loss summed over unmasked frames,
    divided by the number of unmasked peaks (at least 1)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    m = mask.float()
    pos = -((1 - p) ** ALPHA) * torch.log(p) * peak.float()
    neg = -((1 - target) ** BETA) * (p ** ALPHA) * torch.log(1 - p) * (1 - peak.float())
    n_pos = (peak.float() * m).sum().clamp(min=1.0)
    return ((pos + neg) * m).sum() / n_pos


def t_loss(logits, targets, peaks, mask, heads=('s1', 's2', 'r')):
    """logits (B, len(heads), n) in head order; targets/peaks dicts of (B, n) tensors; mask (B, n). The acoustic-label
    control arm passes heads ('s1', 's2')."""
    total = 0.0
    parts = {}
    for i, h in enumerate(heads):
        parts[h] = focal_event_loss(logits[:, i], targets[h], peaks[h], mask)
        total = total + HEAD_WEIGHT[h] * parts[h]
    return total, parts


def w_loss(pred, ecg, mask):
    """pred, ecg, mask: (B, n). Masked L1 on the signal plus masked L1 on the first difference, each a mean over unmasked
    frames."""
    m = mask.float()
    l1 = (torch.abs(pred - ecg) * m).sum() / m.sum().clamp(min=1.0)
    md = m[:, 1:] * m[:, :-1]
    l1d = (torch.abs(torch.diff(pred, dim=1) - torch.diff(ecg, dim=1)) * md).sum() / md.sum().clamp(min=1.0)
    return l1 + l1d, dict(l1=l1, l1_diff=l1d)
