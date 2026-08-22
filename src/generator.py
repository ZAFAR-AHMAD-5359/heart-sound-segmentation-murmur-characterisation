"""
Synthetic-ECG reconstruction from the phonocardiogram.

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

The generator is a conditional adversarial network trained on EPHNOGRAM, where the
ECG and the phonocardiogram were recorded simultaneously. At inference it maps an
8 s phonocardiogram window, sampled at 8 kHz, to a synthetic ECG of the same length.
The architecture below matches the released checkpoint exactly; training code and
reconstruction metrics are reported in the companion reconstruction paper.

Recordings longer than 8 s are processed in consecutive 8 s windows and the
reconstructions are concatenated, so the per-beat phase reference covers the whole
recording. A trailing window shorter than 2 s is dropped rather than zero-padded,
because the generator was trained on full windows and a mostly-padded input
produces an unreliable trace.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sig

GAN_SR = 8000                 # the generator operates at 8 kHz
WINDOW_S = 8                  # seconds per generator window
WINDOW = GAN_SR * WINDOW_S    # 64000 samples
MIN_TAIL_S = 2                # drop a trailing window shorter than this


# ---------------------------------------------------------------------------
# architecture: exact copy of the training definition for the released weights
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, T)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(2, 3).reshape(B, C, T)
        return self.proj(out) + x


class MultiScaleDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c1 = channels // 3 + (channels % 3 > 0)
        c2 = channels // 3 + (channels % 3 > 1)
        c3 = channels // 3
        self.conv_d1 = nn.Conv1d(channels, c1, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(channels, c2, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(channels, c3, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        concat = torch.cat([self.conv_d1(x), self.conv_d2(x), self.conv_d4(x)], dim=1)
        return self.fusion(concat) + x


class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels, use_attention=False, use_dilated=True):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.use_dilated = use_dilated
        if use_dilated:
            self.dilated_block = MultiScaleDilatedBlock(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadSelfAttention(channels, num_heads=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use_dilated:
            out = self.dilated_block(out)
        out = self.bn2(self.conv2(out))
        if self.use_attention:
            out = self.attention(out)
        return self.relu(out + residual)


class ImprovedGeneratorNoTanh(nn.Module):
    """U-Net encoder-decoder with self-attention and multi-scale dilated blocks.

    The output layer is BatchNorm1d(1) rather than a saturating nonlinearity, so
    R-peak amplitudes are not compressed toward a fixed ceiling.
    """

    def __init__(self, signal_length=WINDOW, base_channels=64, num_scenarios=3):
        super().__init__()
        self.scenario_embed = nn.Linear(num_scenarios, base_channels)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 8), nn.ReLU(inplace=True))
        self.enc5 = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 8), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(
            EnhancedResidualBlock(base_channels * 8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(base_channels * 8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(base_channels * 8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(base_channels * 8, use_attention=True, use_dilated=True))
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 8), nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(
            nn.Conv1d(base_channels * 16, base_channels * 8, kernel_size=1),
            nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 4, kernel_size=1),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=1),
            nn.ConvTranspose1d(base_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1))

    def forward(self, x, scenario_onehot):
        scenario_emb = self.scenario_embed(scenario_onehot).unsqueeze(2)
        e1 = self.enc1(x) + scenario_emb
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = torch.cat([self.dec5(b), e4], dim=1)
        d4 = torch.cat([self.dec4(d5), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        return self.dec1(d2)


# ---------------------------------------------------------------------------
# loading and inference
# ---------------------------------------------------------------------------

def load_generator(weights_path, device='cpu'):
    """Load the released checkpoint into the architecture above.

    `strict=True` throughout: a checkpoint that does not match this architecture
    raises instead of silently loading a partial model, because a partially loaded
    generator produces a plausible-looking but meaningless trace.
    """
    gen = ImprovedGeneratorNoTanh()
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=True)
    except Exception:
        # older checkpoints may carry non-tensor objects (e.g. optimiser state)
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for name in ('generator_state_dict', 'state_dict'):
            if name in ckpt:
                ckpt = ckpt[name]
                break
    gen.load_state_dict(ckpt, strict=True)
    return gen.to(device).eval()


def _z_score(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)


def analysed_portion(pcg_8k):
    """The portion of an 8 kHz phonocardiogram the pipeline analyses: whole 8 s
    windows plus one trailing window of at least MIN_TAIL_S seconds. A shorter
    trailing remainder is dropped. A recording under MIN_TAIL_S is zero-padded
    up to one full window.

    Both arms of the study call this, so they analyse identical audio and any
    between-arm difference is attributable to the boundaries alone.
    """
    pcg_8k = pcg_8k.astype(np.float32)
    n_full = len(pcg_8k) // WINDOW
    tail = len(pcg_8k) - n_full * WINDOW
    n_windows = n_full + (1 if tail >= MIN_TAIL_S * GAN_SR else 0)
    if n_windows == 0:                      # recording shorter than MIN_TAIL_S
        return np.pad(pcg_8k, (0, WINDOW - len(pcg_8k)))
    n = min(n_windows * WINDOW, len(pcg_8k))
    return pcg_8k[:n]


def reconstruct_ecg(generator, pcg, sr, device='cpu', scenario='rest'):
    """Map a phonocardiogram of any length to a synthetic ECG at 8 kHz.

    Returns (synthetic_ecg, pcg_8k): both at 8 kHz and of equal length, covering
    the analysed portion of the recording (see analysed_portion). The final
    partial window is zero-padded for inference and trimmed back afterwards.

    Each window is z-scored independently before inference, matching training.
    The scenario one-hot conditions the generator on activity state; auscultation
    recordings are resting, so 'rest' is the default and the sensible choice.
    """
    if sr != GAN_SR:
        pcg = sig.resample(pcg.astype(np.float32), int(len(pcg) * GAN_SR / sr))
    pcg = analysed_portion(pcg)
    n_windows = (len(pcg) + WINDOW - 1) // WINDOW

    scenario_map = {'rest': [1, 0, 0], 'exercise': [0, 1, 0], 'recovery': [0, 0, 1]}
    onehot = torch.tensor([scenario_map[scenario]], dtype=torch.float32, device=device)

    pieces = []
    with torch.no_grad():
        for w in range(n_windows):
            chunk = pcg[w * WINDOW:(w + 1) * WINDOW]
            pad = WINDOW - len(chunk)
            if pad:
                chunk = np.pad(chunk, (0, pad))
            x = torch.tensor(_z_score(chunk), dtype=torch.float32,
                             device=device).view(1, 1, -1)
            y = generator(x, onehot).squeeze().cpu().numpy()
            pieces.append(y[:WINDOW - pad] if pad else y)

    ecg = np.concatenate(pieces)
    return ecg, pcg[:len(ecg)]
