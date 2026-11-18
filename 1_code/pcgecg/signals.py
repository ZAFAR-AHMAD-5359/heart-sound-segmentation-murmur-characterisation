"""Signal utilities: resampling, consensus R-peak detection, PCG envelope, and synchronisation checks."""
from math import gcd
import numpy as np
import scipy.signal as sg

FS_ECG, FS_PCG, FS_ENV = 500, 2000, 500
DETECTORS = ('neurokit', 'pantompkins1985', 'hamilton2002', 'elgendi2010')


def resample(x, fs_in, fs_out):
    fs_in, fs_out = int(round(fs_in)), int(round(fs_out))
    if fs_in == fs_out:
        return np.asarray(x, np.float32)
    g = gcd(fs_in, fs_out)
    return sg.resample_poly(np.asarray(x, np.float64), fs_out // g, fs_in // g).astype(np.float32)


def sos_band(x, lo, hi, fs, order=4):
    sos = sg.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    assert np.all(np.abs(sg.sos2zpk(sos)[1]) < 1), 'unstable filter'   # lesson from the June detector bug
    return sg.sosfiltfilt(sos, x)


# ------------------------------------------------------------------ ECG
def match_f1(a, b, tol):
    """One-to-one greedy matching of two sorted event lists within tol samples -> F1."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    used = np.zeros(len(b), bool); hits = 0
    j0 = 0
    for x in a:
        j = np.searchsorted(b, x - tol, side='left')
        while j < len(b) and b[j] <= x + tol:
            if not used[j]:
                used[j] = True; hits += 1; break
            j += 1
    return 2 * hits / (len(a) + len(b))


def refine(peaks, xc, fs, win_ms=60):
    """Snap each detection to the local maximum of the polarity-corrected cleaned ECG, so detectors that
    mark different fiducial points (onset, slope, peak) are compared on the same R maximum."""
    w = int(win_ms / 1000 * fs); out = []
    for p in peaks:
        lo, hi = max(0, p - w), min(len(xc), p + w + 1)
        if hi > lo:
            out.append(lo + int(np.argmax(np.abs(xc[lo:hi]))))
    return np.unique(np.asarray(out, int))


def consensus_rpeaks(ecg, fs=FS_ECG, tol_ms=50, min_votes=3):
    """Four detectors, polarity-corrected; keep beats found by >= min_votes detectors.
    Returns (rpeaks, info) with rpeaks at fs, refined to the local maximum within +/-40 ms."""
    import neurokit2 as nk
    x = np.asarray(ecg, np.float64)
    x = np.nan_to_num(x - np.nanmedian(x))
    try:
        x, inverted = nk.ecg_invert(x, sampling_rate=fs)
    except Exception:
        inverted = False
    xc = nk.ecg_clean(x, sampling_rate=fs, method='neurokit')
    tol = int(tol_ms / 1000 * fs)
    dets = {}
    for m in DETECTORS:
        try:
            _, info = nk.ecg_peaks(xc, sampling_rate=fs, method=m, correct_artifacts=False)
            dets[m] = refine(np.asarray(info['ECG_R_Peaks'], int), xc, fs)
        except Exception:
            dets[m] = np.array([], int)
    allp = sorted((p, m) for m, ps in dets.items() for p in ps)
    groups, cur = [], []
    for p, m in allp:
        if cur and p - cur[0][0] > tol:
            groups.append(cur); cur = []
        cur.append((p, m))
    if cur:
        groups.append(cur)
    w = int(0.040 * fs); peaks, votes = [], []
    for g in groups:
        v = len({m for _, m in g})
        if v >= min_votes:
            c = int(np.median([p for p, _ in g]))
            lo, hi = max(0, c - w), min(len(xc), c + w + 1)
            peaks.append(lo + int(np.argmax(np.abs(xc[lo:hi])))); votes.append(v)
    peaks = np.unique(np.asarray(peaks, int))
    names = list(dets)
    pair = [match_f1(dets[a], dets[b], tol) for i, a in enumerate(names) for b in names[i + 1:]]
    dvc = [match_f1(dets[m], peaks, tol) for m in names]
    info = dict(inverted=bool(inverted), n_by_detector={m: int(len(p)) for m, p in dets.items()},
                det_vs_cons_f1_median=float(np.median(dvc)) if dvc else 0.0,
                det_vs_cons_f1_min=float(np.min(dvc)) if dvc else 0.0,
                pairwise_f1_mean=float(np.mean(pair)) if pair else 0.0,
                pairwise_f1_min=float(np.min(pair)) if pair else 0.0,
                frac_unanimous=float(np.mean(np.asarray(votes) == len(DETECTORS))) if votes else 0.0)
    return peaks, info, xc.astype(np.float32)


def rr_stats(rpeaks, fs=FS_ECG):
    """Global and local RR statistics; the local outlier share compares each RR with an 11-beat running
    median, so slow heart-rate trends (exercise, recovery) do not count as errors."""
    if len(rpeaks) < 3:
        return dict(hr_median=np.nan, hr_p05=np.nan, hr_p95=np.nan, rr_cv=np.nan, frac_rr_outlier=np.nan,
                    frac_rr_outlier_local=np.nan)
    rr = np.diff(rpeaks) / fs
    med = np.median(rr)
    k = min(11, len(rr) if len(rr) % 2 else len(rr) - 1)
    run = sg.medfilt(rr, k) if k >= 3 else np.full_like(rr, med)
    return dict(hr_median=float(60 / med), hr_p05=float(60 / np.percentile(rr, 95)), hr_p95=float(60 / np.percentile(rr, 5)),
                rr_cv=float(np.std(rr) / np.mean(rr)), frac_rr_outlier=float(np.mean(np.abs(rr - med) > 0.2 * med)),
                frac_rr_outlier_local=float(np.mean(np.abs(rr - run) > 0.2 * run)))


# ------------------------------------------------------------------ PCG
def pcg_envelope(pcg2k, fs=FS_PCG, fs_out=FS_ENV):
    """25-400 Hz band, Hilbert magnitude, 20 Hz low-pass, decimated to fs_out, robust z-score."""
    y = sos_band(np.asarray(pcg2k, np.float64), 25, 400, fs)
    env = np.abs(sg.hilbert(y))
    sos = sg.butter(4, 20, btype='lowpass', fs=fs, output='sos')
    env = sg.sosfiltfilt(sos, env)[::fs // fs_out]
    med = np.median(env); mad = np.median(np.abs(env - med)) + 1e-12
    return ((env - med) / (1.4826 * mad)).astype(np.float32)


def pcg_quality(pcg, fs):
    x = np.asarray(pcg, np.float64)
    a = np.max(np.abs(x)) + 1e-12
    sd = np.std(x) + 1e-12
    return dict(pcg_rms=float(np.sqrt(np.mean(x ** 2))), pcg_clip_frac=float(np.mean(np.abs(x) >= 0.999 * a)),
                pcg_flat_frac=float(np.mean(np.abs(np.diff(x)) < 1e-4 * sd)))


# ------------------------------------------------------------------ synchronisation
PRE_MS, POST_MS = 200, 700
S1_WIN_MS = (-50, 250)


def triggered(env, rpeaks, fs=FS_ENV):
    pre, post = int(PRE_MS / 1000 * fs), int(POST_MS / 1000 * fs)
    seg = [env[r - pre:r + post] for r in rpeaks if r - pre >= 0 and r + post <= len(env)]
    return np.array(seg, np.float32) if seg else np.zeros((0, pre + post), np.float32)


def sync_metrics(env, rpeaks, fs=FS_ENV, rng=None):
    """R-triggered average of the PCG envelope.
    s1_lag_ms: lag of the averaged envelope maximum in [-50, 250] ms after R.
    lock_t: paired t-statistic across beats of (envelope at that lag minus the beat's own median); near 0 when the
            PCG is unrelated to the ECG beats, large when S1 locks to every R-peak.
    perbeat_lag_iqr_ms: spread of per-beat S1 lags; drift_ms: range of the S1 lag over 60 s windows."""
    pre = int(PRE_MS / 1000 * fs)
    lag_ms = (np.arange(int((PRE_MS + POST_MS) / 1000 * fs)) - pre) / fs * 1000
    win = (lag_ms >= S1_WIN_MS[0]) & (lag_ms <= S1_WIN_MS[1])
    seg = triggered(env, rpeaks, fs)
    out = dict(n_beats_used=int(len(seg)))
    if len(seg) < 5:
        return dict(out, s1_lag_ms=np.nan, lock_contrast=np.nan, lock_t=np.nan, lock_effect=np.nan, perbeat_lag_iqr_ms=np.nan,
                    drift_ms=np.nan, max_jump_ms=np.nan, r_hr_lag=np.nan, lag_slope_ms_per_bpm=np.nan,
                    hr_range_windows=np.nan, n_windows_60s=0), np.full(len(lag_ms), np.nan, np.float32)
    avg = seg.mean(0)
    k = np.flatnonzero(win)[np.argmax(avg[win])]
    s1 = float(lag_ms[k])
    d = seg[:, k] - np.median(seg, axis=1)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
    eff = float(d.mean() / (d.std(ddof=1) + 1e-12))          # length-free effect size of the S1 lock
    c = float((avg[win].max() - np.median(avg)) / (avg.std() + 1e-9))
    per = lag_ms[win][np.argmax(seg[:, win], axis=1)]
    iqr = float(np.subtract(*np.percentile(per, [75, 25])))
    lags, hrs = [], []
    step = 60 * fs
    for w0 in range(0, len(env), step):
        rr = rpeaks[(rpeaks >= w0) & (rpeaks < w0 + step)]
        sw = triggered(env, rr, fs)
        if len(sw) >= 10:
            a = sw.mean(0); lags.append(lag_ms[win][np.argmax(a[win])]); hrs.append(60 * fs / np.median(np.diff(rr)))
    lags, hrs = np.array(lags), np.array(hrs)
    drift = float(lags.max() - lags.min()) if len(lags) >= 2 else np.nan
    jump = float(np.max(np.abs(np.diff(lags)))) if len(lags) >= 2 else np.nan
    if len(lags) >= 4 and np.std(lags) > 0 and np.std(hrs) > 0:
        r_hl = float(np.corrcoef(hrs, lags)[0, 1]); slope = float(np.polyfit(hrs, lags, 1)[0])
    else:
        r_hl, slope = np.nan, np.nan
    out.update(s1_lag_ms=s1, lock_contrast=c, lock_t=t, lock_effect=eff, perbeat_lag_iqr_ms=iqr, drift_ms=drift, max_jump_ms=jump,
               r_hr_lag=r_hl, lag_slope_ms_per_bpm=slope, hr_range_windows=float(np.ptp(hrs)) if len(hrs) >= 2 else np.nan,
               n_windows_60s=len(lags))
    return out, avg.astype(np.float32)
