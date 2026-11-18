"""Training augmentation for T and W (08 design, section 2.7; prereg section 7). Paired data only.

Interval warping (on spectrogram frames): a run of consecutive valid beats is re-timed to a target heart rate in 70 to
180 bpm. Each heart sound keeps a window of plus or minus 50 ms around its label unchanged; only the systolic gap
(S1 window end to S2 window start) and the diastolic gap (S2 window end to the next S1 window start) are resampled.
Target RR = 60 / HR with a per-beat jitter (CV 3%); target S1-to-S2 interval = the beat's own interval x
f(HR_target) / f(HR_source), with f = law_S2 - law_S1 of its dataset (the paired law, <config.LABELS>/laws.json). Every gap
change is rounded to whole frames, so the kept windows move by whole frames and are copied exactly. Labels, cycle
bounds and the loss mask move with the same piecewise-linear time map. A beat whose target gaps would fall under
10 ms (systole) or 20 ms (diastole) lowers the target rate for the whole run to the fastest feasible one.

Waveform augmentation (before the spectrogram): additive noise at 0 to 20 dB SNR; murmur-like band noise between S1
and S2 (plateau, crescendo, decrescendo, or diamond envelope). Spectrogram augmentation: device-response tilt across
the 24 bins; frequency masking.
"""
import numpy as np

FRAME_S = 0.004
KEEP_S = 0.050
MIN_SYS_GAP_S, MIN_DIA_GAP_S = 0.010, 0.020
HR_RANGE = (70.0, 180.0)


def law_ms(law, dataset, hr):
    slope = law['slope'][dataset] if isinstance(law['slope'], dict) else law['slope']
    return law['intercept'][dataset] + slope * hr


def systole_ms(laws, dataset, hr):
    """Paired law of the S1-to-S2 interval (envelope peaks) for a dataset at a heart rate."""
    return law_ms(laws['s2'], dataset, hr) - law_ms(laws['s1'], dataset, hr)


def _knots(s1, s2, rr_t, sys_t):
    """Source and target knot times for a run of beats. s1, s2: (k,) source label times; the run ends at the S1 window
    of beat k (s1 has k + 1 entries). rr_t, sys_t: (k,) target RR and S1-to-S2 intervals. Gap changes are rounded to
    whole frames."""
    src, dst = [], []
    t = s1[0] - KEEP_S                                    # the first window starts where it starts in the source
    for i in range(len(s2)):
        a1, b1 = s1[i] - KEEP_S, s1[i] + KEEP_S
        a2, b2 = s2[i] - KEEP_S, s2[i] + KEEP_S
        a_next = s1[i + 1] - KEEP_S
        gs_src, gd_src = a2 - b1, a_next - b2
        gs_t = sys_t[i] - 2 * KEEP_S
        gd_t = rr_t[i] - sys_t[i] - 2 * KEEP_S
        gs_t = gs_src + np.round((gs_t - gs_src) / FRAME_S) * FRAME_S
        gd_t = gd_src + np.round((gd_t - gd_src) / FRAME_S) * FRAME_S
        src += [a1, b1, a2, b2]
        dst += [t, t + 2 * KEEP_S, t + 2 * KEEP_S + gs_t, t + 4 * KEEP_S + gs_t]
        t = t + 4 * KEEP_S + gs_t + gd_t
    src.append(s1[-1] - KEEP_S); dst.append(t)
    src.append(s1[-1] + KEEP_S); dst.append(t + 2 * KEEP_S)
    return np.array(src), np.array(dst)


def feasible_hr(s1, s2, hr_target, sys_src, f_src, laws, dataset):
    """Fastest rate <= hr_target at which every beat keeps its minimum gaps."""
    for hr in np.arange(hr_target, HR_RANGE[0] - 1e-9, -2.0):
        rr = 60.0 / hr
        sys_t = sys_src * systole_ms(laws, dataset, hr) / f_src
        if np.all(sys_t - 2 * KEEP_S >= MIN_SYS_GAP_S) and np.all(rr - sys_t - 2 * KEEP_S >= MIN_DIA_GAP_S):
            return float(hr)
    return float(HR_RANGE[0])


def warp_map(s1, s2, hr_src, hr_target, laws, dataset, rng, jitter_cv=0.03):
    """Piecewise-linear time map for a run of k beats: returns (src_knots, dst_knots, hr_used).
    s1 has k + 1 entries (the run's last S1 closes the last cycle), s2 has k entries; hr_src: (k,) local source rates."""
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    sys_src = s2 - s1[:-1]
    f_src = systole_ms(laws, dataset, np.asarray(hr_src, float)) / 1000.0
    hr = feasible_hr(s1, s2, hr_target, sys_src, f_src, laws, dataset)
    rr_t = 60.0 / hr * (1.0 + jitter_cv * rng.standard_normal(len(s2)))
    sys_t = sys_src * (systole_ms(laws, dataset, hr) / 1000.0) / f_src
    sys_t = np.minimum(sys_t, rr_t - 2 * KEEP_S - MIN_DIA_GAP_S)
    sys_t = np.maximum(sys_t, 2 * KEEP_S + MIN_SYS_GAP_S)
    src, dst = _knots(s1, s2, rr_t, sys_t)
    return src, dst, hr


def map_times(x, src, dst):
    """Source times -> target times; linear between knots, slope 1 outside the run."""
    x = np.asarray(x, float)
    y = np.interp(x, src, dst)
    y = np.where(x < src[0], dst[0] - (src[0] - x), y)
    return np.where(x > src[-1], dst[-1] + (x - src[-1]), y)


def warp_frames(L, t0, src, dst, n_out, out_t0):
    """Resample spectrogram frames L (bins, n) with frame j at t0 + j * FRAME_S onto n_out target frames starting at
    out_t0, through the inverse of the time map (linear interpolation between source frames; whole-frame shifts inside
    the kept windows copy frames exactly). Target frames outside the run are zero (the median level) and reported."""
    tt = out_t0 + np.arange(n_out) * FRAME_S
    inside = (tt >= dst[0]) & (tt <= dst[-1])
    ts = np.interp(tt, dst, src)                          # inverse map (dst is increasing)
    pos = (ts - t0) / FRAME_S
    pos_r = np.round(pos)
    pos = np.where(np.abs(pos - pos_r) < 1e-6, pos_r, pos)   # whole frames stay whole (no interpolation drift)
    j0 = np.clip(np.floor(pos).astype(int), 0, L.shape[1] - 1)
    j1 = np.clip(j0 + 1, 0, L.shape[1] - 1)
    w = np.clip(pos - np.floor(pos), 0.0, 1.0)
    out = L[:, j0] * (1 - w) + L[:, j1] * w
    valid = inside & (pos >= 0) & (pos <= L.shape[1] - 1)
    out[:, ~valid] = 0.0
    return out.astype(np.float32), valid


# ------------------------------------------------------------------ waveform and spectrogram augmentation
def add_noise(x, rng, snr_db_range=(0.0, 20.0)):
    snr = rng.uniform(*snr_db_range)
    p = np.mean(np.asarray(x, np.float64) ** 2) + 1e-12
    return x + rng.standard_normal(len(x)) * np.sqrt(p / 10 ** (snr / 10)), float(snr)


def murmur_noise(x, s1, s2, fs, rng, rel_db_range=(-20.0, 0.0)):
    """Band-limited noise between S1 and S2 of each beat (envelope shape drawn per call), scaled to the S1 window RMS."""
    import scipy.signal as sg
    lo = rng.uniform(60, 200); hi = min(lo + rng.uniform(100, 250), 420)
    sos = sg.butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
    shape = rng.choice(['plateau', 'crescendo', 'decrescendo', 'diamond'])
    rel = 10 ** (rng.uniform(*rel_db_range) / 20)
    y = np.array(x, np.float64, copy=True)
    for a, b in zip(s1, s2):
        i0, i1 = int((a + 0.03) * fs), int((b - 0.02) * fs)
        if i1 - i0 < int(0.03 * fs) or i0 < 0 or i1 > len(y):
            continue
        n = i1 - i0
        u = np.linspace(0, 1, n)
        env = {'plateau': np.ones(n), 'crescendo': u, 'decrescendo': 1 - u, 'diamond': 1 - np.abs(2 * u - 1)}[shape]
        noise = sg.sosfilt(sos, rng.standard_normal(n + 200))[200:]
        w0, w1 = max(0, int((a - 0.05) * fs)), min(len(y), int((a + 0.05) * fs))
        ref = np.sqrt(np.mean(y[w0:w1] ** 2)) + 1e-12
        y[i0:i1] += noise / (np.std(noise) + 1e-12) * ref * rel * env
    return y, dict(band=(float(lo), float(hi)), shape=str(shape), rel=float(rel))


def device_tilt(L, rng, max_tilt=1.0):
    """Linear tilt across the bins in log-power units (plus or minus max_tilt at the band edges)."""
    a = rng.uniform(-max_tilt, max_tilt)
    return (L + a * np.linspace(-1, 1, L.shape[0])[:, None]).astype(np.float32), float(a)


def freq_mask(L, rng, max_width=3):
    w = int(rng.integers(1, max_width + 1)); s = int(rng.integers(0, L.shape[0] - w + 1))
    L = L.copy(); L[s:s + w] = 0.0
    return L, (s, w)
