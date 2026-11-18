"""In-band interference added to the analysed 4 kHz audio at test time (Experiment 1; plan in
'Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md').

Three kinds of interference, all confined to the murmur band BAND (20 to 500 Hz) and none locked to the cardiac cycle:
  gauss : stationary Gaussian noise band-passed to BAND;
  burst : short band-limited bursts (handling or friction-like): 0.2 to 0.5 s long with 20 ms raised-cosine edges, two
          per 10 s of audio on average, at uniformly random non-overlapping onsets, silence elsewhere;
  esc50 : recorded environmental sounds from the ESC-50 corpus (classes crying_baby, breathing, coughing), resampled
          to the target rate, band-passed to BAND, and concatenated with 50 ms crossfades to cover the recording.
SNR is defined on the in-band root-mean-square (RMS) over the analysed portion: the clean in-band RMS over the noise's
in-band RMS; for bursts the noise RMS is taken inside the bursts only. After mixing, the signal is re-normalised to
unit peak, as the loader normalises every recording. Draws are seeded per recording (condition seed, CRC-32 of the
recording name), so they do not depend on processing order.
"""
import csv
import hashlib
import os
import zlib
from fractions import Fraction

import numpy as np
from scipy import signal as sig
from scipy.io import wavfile

BAND = (20.0, 500.0)
KINDS = ('gauss', 'burst', 'esc50')
ESC50_CLASSES = ('crying_baby', 'breathing', 'coughing')
BURST_DUR_S = (0.2, 0.5)
BURST_RATE_PER_10S = 2.0
BURST_EDGE_S = 0.02
CROSSFADE_S = 0.05


def rng_for(seed, name):
    return np.random.default_rng([int(seed), zlib.crc32(str(name).encode())])


def band_sos(sr, band=BAND):
    return sig.butter(4, [band[0], band[1]], btype='bandpass', fs=sr, output='sos')


def inband(x, sr, band=BAND):
    x = np.asarray(x, np.float64)
    if len(x) < 30:                                    # too short to filter: keep as is
        return x.copy()
    return sig.sosfiltfilt(band_sos(sr, band), x)


def inband_rms(x, sr, band=BAND, mask=None):
    f = inband(x, sr, band)
    if mask is not None:
        f = f[np.asarray(mask, bool)]
    return float(np.sqrt(np.mean(f ** 2))) if len(f) else 0.0


def gaussian_noise(n, sr, rng, band=BAND):
    """Stationary band-limited Gaussian noise of length n (unit variance before the band-pass)."""
    return inband(rng.standard_normal(int(n)), sr, band)


def burst_window(n, sr, rng, dur_s=BURST_DUR_S, rate_per_10s=BURST_RATE_PER_10S, edge_s=BURST_EDGE_S):
    """A 0/1 window with raised-cosine edges marking the bursts; (window, number of bursts placed, burst seconds)."""
    n = int(n)
    k = max(1, int(round(rate_per_10s * n / sr / 10.0)))
    w = np.zeros(n)
    placed = []
    edge = max(1, int(round(edge_s * sr)))
    for _ in range(k):
        d = min(n, max(2 * edge + 1, int(round(rng.uniform(*dur_s) * sr))))
        for _try in range(100):
            s = int(rng.integers(0, max(1, n - d + 1)))
            if all(s + d <= a or s >= b for a, b in placed):
                break
        else:
            continue
        placed.append((s, s + d))
        seg = np.ones(d)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        seg[:edge] = ramp
        seg[-edge:] = ramp[::-1]
        w[s:s + d] = seg
    return w, len(placed), float(sum(b - a for a, b in placed) / sr)


def burst_noise(n, sr, rng, band=BAND):
    """(noise, burst mask, number of bursts, burst seconds): band-limited Gaussian noise gated by the burst window."""
    g = gaussian_noise(n, sr, rng, band)
    w, k, secs = burst_window(n, sr, rng)
    return g * w, w > 0, k, secs


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def resample(x, sr_raw, sr):
    if sr_raw == sr:
        return np.asarray(x, np.float64)
    fr = Fraction(int(sr), int(sr_raw))
    return sig.resample_poly(np.asarray(x, np.float64), fr.numerator, fr.denominator)


def load_esc50_bank(root, sr, classes=ESC50_CLASSES, band=BAND):
    """Clips of the named ESC-50 classes from a corpus folder (meta/esc50.csv and audio/*.wav), sorted by file name,
    resampled to sr and band-passed. Returns dict(files=[(filename, category, sha256)], clips=[array], bank_sha256)."""
    meta = os.path.join(root, 'meta', 'esc50.csv')
    rows = []
    with open(meta, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['category'] in classes:
                rows.append((r['filename'], r['category']))
    rows.sort()
    assert rows, 'no ESC-50 clips of %s under %s' % (classes, root)
    files, clips, h = [], [], hashlib.sha256()
    for fn, cat in rows:
        p = os.path.join(root, 'audio', fn)
        sr_raw, a = wavfile.read(p)
        a = np.asarray(a, np.float64)
        if a.ndim > 1:
            a = a[:, 0]
        if np.issubdtype(np.asarray(wavfile.read(p)[1]).dtype, np.integer):
            a = a / 32768.0
        x = inband(resample(a, sr_raw, sr), sr, band)
        x = x / (np.max(np.abs(x)) + 1e-12)
        sha = file_sha256(p)
        files.append((fn, cat, sha))
        clips.append(x.astype(np.float64))
        h.update(sha.encode())
    return dict(files=files, clips=clips, bank_sha256=h.hexdigest())


def esc50_noise(n, sr, rng, bank, crossfade_s=CROSSFADE_S):
    """Random clips of the bank concatenated with linear crossfades until n samples are covered; (noise, clip indices)."""
    n = int(n)
    cf = max(1, int(round(crossfade_s * sr)))
    out = np.zeros(0)
    used = []
    while len(out) < n:
        i = int(rng.integers(0, len(bank['clips'])))
        used.append(i)
        c = bank['clips'][i]
        if len(out) == 0:
            out = c.copy()
            continue
        k = min(cf, len(out), len(c))
        ramp = np.linspace(0, 1, k)
        joined = np.concatenate([out[:-k], out[-k:] * (1 - ramp) + c[:k] * ramp, c[k:]])
        out = joined
    return out[:n], used


def mix(x4, kind, snr_db, rng, sr, bank=None, band=BAND):
    """Add interference of `kind` at `snr_db` (in-band) to the analysed audio x4 and re-normalise to unit peak.
    Returns (float32 signal, info dict)."""
    x = np.asarray(x4, np.float64)
    n = len(x)
    rc = inband_rms(x, sr, band)
    mask = None
    extra = {}
    if kind == 'gauss':
        nz = gaussian_noise(n, sr, rng, band)
    elif kind == 'burst':
        nz, mask, k, secs = burst_noise(n, sr, rng, band)
        extra = dict(n_bursts=int(k), burst_seconds=float(secs))
    elif kind == 'esc50':
        assert bank is not None, 'esc50 needs a clip bank'
        nz, used = esc50_noise(n, sr, rng, bank)
        extra = dict(clips=[int(i) for i in used])
    else:
        raise ValueError('unknown interference kind %r' % kind)
    rn = inband_rms(nz, sr, band, mask=mask)
    target = rc * 10.0 ** (-float(snr_db) / 20.0)
    scale = target / rn if rn > 0 else 0.0
    nz = nz * scale
    y = x + nz
    peak = float(np.max(np.abs(y)) + 1e-10)
    y = (y / peak).astype(np.float32)
    rn2 = inband_rms(nz, sr, band, mask=mask)
    info = dict(kind=kind, snr_db=float(snr_db), clean_inband_rms=rc, noise_inband_rms=rn2,
                achieved_snr_db=float(20 * np.log10(rc / rn2)) if rn2 > 0 and rc > 0 else None, peak_rescale=peak, **extra)
    return y, info


def audio_hash(arrays, names):
    """Content hash of a set of float32 signals in a fixed order (for the run manifest)."""
    h = hashlib.sha256()
    for n in names:
        h.update(np.ascontiguousarray(np.asarray(arrays[n], np.float32)).tobytes())
        h.update(b'|')
    return h.hexdigest()
