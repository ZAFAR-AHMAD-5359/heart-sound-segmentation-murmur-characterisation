"""Model input: log-power spectrogram of the PCG (08 design, section 2.7).

PCG at 1 kHz, band-pass 25 to 400 Hz (second-order sections, pole assertion) applied once to the whole recording;
frames of 32 samples (32 ms Hann) every 4 samples (250 Hz), zero-padded to 64 points, bins 2 to 25 (31.25 to 390.6 Hz,
24 bins); log power; each bin's median over the whole recording subtracted. Frame j covers samples [4j, 4j + 32)
and is centred at (4j + 16) / 1000 s. Frames use only samples inside their window, so a crop that starts on the 4-sample
grid yields exactly the frames of the whole recording. The numpy version builds the corpus; the torch version runs
on Colab, and a test holds them equal.
"""
import numpy as np
import scipy.signal as sg

FS = 1000
WIN, HOP, NFFT = 32, 4, 64
BINS = slice(2, 26)                        # 31.25 to 390.6 Hz
BAND = (25.0, 400.0)
EPS = 1e-10


def bandpass_sos(fs=FS):
    sos = sg.butter(4, BAND, btype='bandpass', fs=fs, output='sos')
    assert np.abs(sg.sos2zpk(sos)[1]).max() < 0.9999, 'unstable band-pass'
    return sos


def bandpass(pcg, fs=FS):
    return np.ascontiguousarray(sg.sosfiltfilt(bandpass_sos(fs), np.asarray(pcg, np.float64)))


def to_1k(pcg, fs_in):
    from math import gcd
    fs_in = int(round(fs_in))
    if fs_in == FS:
        return np.asarray(pcg, np.float64)
    g = gcd(fs_in, FS)
    return sg.resample_poly(np.asarray(pcg, np.float64), FS // g, fs_in // g)


def n_frames(n_samples):
    return max(0, (n_samples - WIN) // HOP + 1)


def frame_times_s(n_samples, start_sample=0):
    return (start_sample + np.arange(n_frames(n_samples)) * HOP + WIN / 2) / FS


def logpower(x):
    """Band-passed PCG (1 kHz) -> (24, frames) log power, before median subtraction."""
    x = np.asarray(x, np.float64)
    nf = n_frames(len(x))
    if nf == 0:
        return np.zeros((BINS.stop - BINS.start, 0))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    fr = x[idx] * np.hanning(WIN + 2)[1:-1][None, :]
    spec = np.abs(np.fft.rfft(fr, n=NFFT, axis=1)) ** 2
    return np.log(spec[:, BINS] + EPS).T


def bin_medians(x_bp):
    return np.median(logpower(x_bp), axis=1)


def logspec(x_bp, medians=None):
    """Median-normalised spectrogram; medians from the whole recording (pass them for crops)."""
    L = logpower(x_bp)
    m = np.median(L, axis=1) if medians is None else np.asarray(medians)
    return (L - m[:, None]).astype(np.float32)


def logspec_torch(x_bp, medians):
    """Torch twin of logspec for batches: x_bp (B, n) float32, medians (B, 24). Returns (B, 24, frames)."""
    import torch
    win = torch.hann_window(WIN + 2, periodic=False, dtype=x_bp.dtype, device=x_bp.device)[1:-1]
    fr = x_bp.unfold(-1, WIN, HOP) * win                      # (B, frames, 32)
    spec = torch.fft.rfft(fr, n=NFFT, dim=-1).abs() ** 2
    L = torch.log(spec[..., BINS] + EPS).transpose(-1, -2)    # (B, 24, frames)
    return L - medians[..., None]
