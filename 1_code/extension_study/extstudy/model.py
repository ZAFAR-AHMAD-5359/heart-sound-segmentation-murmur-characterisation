"""1-D U-Net on the log-power spectrogram (08 design, section 2.7; prereg section 7).

Input (B, 24, frames) at 250 Hz. Encoder 64 -> 96 -> 128 channels (two stride-2 steps, 3-tap convolutions),
eight dilated residual blocks at 128 channels (a dilated 3-tap convolution and a 1x1 convolution; dilations 1, 2, 4, 8,
1, 2, 4, 8; receptive field about 1 s), decoder back to 64 channels with skip connections, and 1x1 heads at the input
frame rate. T: three event logits (S1, S2, R). W: one ECG channel. GroupNorm (independent of batch
composition), 872,387 parameters with the T heads. Any frame count works: the input is padded to a multiple of 4
by reflection and the output cropped back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = {'T': ('s1', 's2', 'r'), 'W': ('ecg',), 'A': ('s1', 's2')}       # A: the acoustic-label control arm (Experiment 2), no R head


def _norm(c):
    return nn.GroupNorm(8, c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), _norm(cout), nn.GELU(),
                                 nn.Conv1d(cout, cout, k, padding=k // 2), _norm(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation), _norm(c), nn.GELU(), nn.Dropout(dropout),
                                 nn.Conv1d(c, c, 1), _norm(c))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class UNet1D(nn.Module):
    def __init__(self, kind='T', n_in=24, widths=(64, 96, 128), dilations=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        assert kind in HEADS
        self.kind = kind
        w0, w1, w2 = widths
        self.enc0 = ConvBlock(n_in, w0)
        self.down1 = nn.Conv1d(w0, w1, 4, stride=2, padding=1)
        self.enc1 = ConvBlock(w1, w1)
        self.down2 = nn.Conv1d(w1, w2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(*[ResBlock(w2, d) for d in dilations])
        self.up2 = nn.ConvTranspose1d(w2, w1, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(2 * w1, w1)
        self.up1 = nn.ConvTranspose1d(w1, w0, 4, stride=2, padding=1)
        self.dec0 = ConvBlock(2 * w0, w0)
        self.head = nn.Conv1d(w0, len(HEADS[kind]), 1)
        if kind != 'W':                                   # event heads start near a low prior (focal-loss practice)
            nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        n = x.shape[-1]
        pad = (-n) % 4
        if pad:
            x = F.pad(x, (0, pad), mode='reflect')
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        m = self.mid(self.down2(e1))
        d1 = self.dec1(torch.cat([self.up2(m), e1], 1))
        d0 = self.dec0(torch.cat([self.up1(d1), e0], 1))
        return self.head(d0)[..., :n]


def n_params(model):
    return sum(p.numel() for p in model.parameters())
