"""
springer_py.py -- line-by-line Python port of Springer's LR-HSMM heart sound segmenter.

Source ported: PhysioNet/CinC 2016 sample entry (sample2016.zip, GPL-3), which is the
davidspringer/Springer-Segmentation-Code MATLAB code (commit 853cb53, 2016) with
include_wavelet_feature = true and use_mex = true (C Viterbi, sum-of-log emissions).
Licence of this port: GPL-3.0 (derivative of GPL-3 code).

Fidelity notes (what is identical, what is not):
  * filtfilt padding set to MATLAB's 3*(nfilt-1) odd reflection.
  * MATLAB resample(x,p,q) == scipy.signal.resample_poly(x,p,q) (Kaiser beta=5, N=10,
    same delay removal) up to floating point.
  * std uses N-1 (MATLAB default).
  * spectrogram at F = 1..500 Hz evaluated by direct DFT with hamming(25), noverlap 13.
    Absolute PSD scale differs from MATLAB but the feature is z-scored, so irrelevant.
  * wavedec 'rbio3.9' level 3, mode 'sym' == pywt mode 'symmetric'.
  * Viterbi follows viterbi_Springer.c (sum of logs, start clamp [0,T-2], window
    start..end inclusive, i.e. the original's d+1-sample emission window) in float64
    (the C code uses float32 for some accumulators; ties may break differently).
  * Backtracking: identical, including the 1000-segment cap (exposed as max_segments).
Only deliberate, optional deviation: heart-rate search range and systolic search floor
(hr_min, hr_max, sys_min_s). Defaults reproduce getHeartRateSchmidt.m (30-120 bpm, 0.2 s).
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
from scipy import signal
from math import gcd

try:
    import pywt
except ImportError:  # wavelet feature optional
    pywt = None

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

AUDIO_FS = 1000
FEATURES_FS = 50
REALMIN = np.finfo(np.float64).tiny     # MATLAB realmin
FLT_MIN = np.float64(np.finfo(np.float32).tiny)


# ----------------------------------------------------------------------------- helpers
def matlab_round(x):
    """MATLAB round: half away from zero."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64) if x.ndim else int(np.sign(x) * np.floor(abs(x) + 0.5))


def mresample(x, p, q):
    """MATLAB resample(x, p, q)."""
    p, q = int(p), int(q)
    g = gcd(p, q)
    p, q = p // g, q // g
    if p == q:
        return np.asarray(x, float).copy()
    return signal.resample_poly(np.asarray(x, float), p, q, window=("kaiser", 5.0))


def mfiltfilt(b, a, x):
    nfilt = max(len(a), len(b))
    return signal.filtfilt(b, a, x, padtype="odd", padlen=3 * (nfilt - 1))


def butter_lp(x, order, cutoff, fs):
    b, a = signal.butter(order, 2 * cutoff / fs, "low")
    return mfiltfilt(b, a, x)


def butter_hp(x, order, cutoff, fs):
    b, a = signal.butter(order, 2 * cutoff / fs, "high")
    return mfiltfilt(b, a, x)


def normalise_signal(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def schmidt_spike_removal(original, fs):
    original = np.asarray(original, float)
    ws = int(matlab_round(fs / 2))
    trailing = len(original) % ws
    n_use = len(original) - trailing
    if n_use < ws:
        return original.copy()
    # MATLAB reshape is column-major: frames[:, k] = original[k*ws:(k+1)*ws]
    frames = original[:n_use].reshape(-1, ws).T.copy()
    maas = np.max(np.abs(frames), axis=0)
    guard = 0
    while np.any(maas > np.median(maas) * 3) and guard < 10000:
        guard += 1
        w = int(np.argmax(maas))
        col = frames[:, w]
        spike_pos = int(np.argmax(np.abs(col)))           # 0-based
        zc = np.concatenate([np.abs(np.diff(np.sign(col))) > 1, [False]])
        idx = np.flatnonzero(zc[: spike_pos + 1])
        spike_start = idx[-1] if idx.size else 0          # max([1 find(...,'last')]) in 1-based
        zc2 = zc.copy()
        zc2[: spike_pos + 1] = False
        idx2 = np.flatnonzero(zc2)
        spike_end = min(idx2[0], ws - 1) if idx2.size else ws - 1
        frames[spike_start: spike_end + 1, w] = 0.0001
        maas = np.max(np.abs(frames), axis=0)
    despiked = frames.T.reshape(-1)
    return np.concatenate([despiked, original[len(despiked):]])


def homomorphic_envelope(x, fs, lpf=8):
    b, a = signal.butter(1, 2 * lpf / fs, "low")
    env = np.exp(mfiltfilt(b, a, np.log(np.abs(signal.hilbert(x)))))
    env[0] = env[1]
    return env


def hilbert_envelope(x):
    return np.abs(signal.hilbert(x))


def psd_feature(x, fs, flo=40, fhi=60):
    """get_PSD_feature_Springer_HMM: spectrogram(x, fs/40, round(fs/80), 1:round(fs/2), fs)."""
    nwin = int(fs / 40)
    nover = int(matlab_round(fs / 80))
    hop = nwin - nover
    nseg = (len(x) - nover) // hop
    win = signal.windows.hamming(nwin, sym=True)
    F = np.arange(1, int(matlab_round(fs / 2)) + 1)
    lo = int(np.argmin(np.abs(F - flo)))
    hi = int(np.argmin(np.abs(F - fhi)))
    f_sel = F[lo: hi + 1]
    n = np.arange(nwin)
    E = np.exp(-2j * np.pi * np.outer(f_sel, n) / fs)          # nf x nwin
    idx = np.arange(nseg)[:, None] * hop + n[None, :]
    segs = x[idx] * win[None, :]                                # nseg x nwin
    P = np.abs(segs @ E.T) ** 2                                 # nseg x nf (unscaled PSD)
    return P.mean(axis=1)


def get_dwt_detail(x, level, name):
    """getDWT.m, detail part: cD(k,:) for k=1..level, nearest-upsampled and centre-kept."""
    coeffs = pywt.wavedec(x, name, mode="symmetric", level=level)
    # pywt order: [cA_n, cD_n, ..., cD_1]
    L = len(x)
    out = np.zeros((level, L))
    for k in range(1, level + 1):
        d = coeffs[-k]
        d = np.repeat(d, 2 ** k)
        sx = len(d)
        dd = (sx - L) / 2
        first = int(np.floor(dd))
        last = sx - int(np.ceil(dd))
        out[k - 1] = d[first:last]
    out[np.abs(out) < np.sqrt(np.finfo(float).eps)] = 0
    return out


# ----------------------------------------------------------------------------- features
def get_springer_pcg_features(audio, fs=AUDIO_FS, include_wavelet=True, features_fs=FEATURES_FS):
    audio = butter_lp(audio, 2, 400, fs)
    audio = butter_hp(audio, 2, 25, fs)
    audio = schmidt_spike_removal(audio, fs)
    homo = homomorphic_envelope(audio, fs)
    d_homo = normalise_signal(mresample(homo, features_fs, fs))
    hil = hilbert_envelope(audio)
    d_hil = normalise_signal(mresample(hil, features_fs, fs))
    psd = psd_feature(audio, fs, 40, 60)
    psd = normalise_signal(mresample(psd, len(d_homo), len(psd)))
    feats = [d_homo, d_hil, psd]
    if include_wavelet:
        a = audio
        if len(a) < fs * 1.025:
            a = np.concatenate([a, np.zeros(int(matlab_round(0.025 * fs)))])
        cD = get_dwt_detail(a, 3, "rbio3.9")
        wf = np.abs(cD[2])[: len(homo)]
        feats.append(normalise_signal(mresample(wf, features_fs, fs)))
    n = min(len(f) for f in feats)
    return np.column_stack([f[:n] for f in feats]), features_fs


# ----------------------------------------------------------------------------- heart rate
def get_heart_rate_schmidt(audio, fs=AUDIO_FS, hr_min=30.0, hr_max=120.0, sys_min_s=0.2, method="argmax",
                           n_candidates=3):
    """getHeartRateSchmidt.m. Defaults = original (lags 0.5-2.0 s, systole floor 0.2 s, argmax).
    method="argmax": original rule (global max in the lag window; can sit on the window edge).
    method="peak":   highest *local maximum* in the lag window (deviation; avoids edge picks).
    method="multi":  returns up to n_candidates local maxima, best first (for likelihood selection).
    """
    a = butter_lp(audio, 2, 400, fs)
    a = butter_hp(a, 2, 25, fs)
    a = schmidt_spike_removal(a, fs)
    env = homomorphic_envelope(a, fs)
    y = env - env.mean()
    n = len(y)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    Y = np.fft.rfft(y, nfft)
    c = np.fft.irfft(Y * np.conj(Y), nfft)[:n]
    c = c / c[0]
    sa = c[1:]                      # sa[k-1] == MATLAB signal_autocorrelation(k) == lag k
    min_index = int(matlab_round(60.0 / hr_max * fs))
    max_index = min(int(matlab_round(60.0 / hr_min * fs)), len(sa))
    seg = sa[min_index - 1: max_index]
    if method == "argmax":
        lags = [int(np.argmax(seg)) + 1 + min_index - 1]
    else:
        pk, prop = signal.find_peaks(seg)
        if pk.size == 0:
            lags = [int(np.argmax(seg)) + 1 + min_index - 1]
        else:
            order = np.argsort(seg[pk])[::-1]
            lags = [int(pk[o]) + 1 + min_index - 1 for o in order[: (n_candidates if method == "multi" else 1)]]
    out = []
    for true_index in lags:
        heart_rate = 60.0 / (true_index / fs)
        max_sys = int(matlab_round(((60.0 / heart_rate) * fs) / 2))
        min_sys = int(matlab_round(sys_min_s * fs))
        if max_sys <= min_sys:          # only reachable with widened ranges
            max_sys = min_sys + 1
        win = sa[min_sys - 1: max_sys]
        if method == "argmax":
            pos = int(np.argmax(win)) + 1
        else:
            pk2, _ = signal.find_peaks(win)
            pos = (int(pk2[np.argmax(win[pk2])]) + 1) if pk2.size else (int(np.argmax(win)) + 1)
        out.append((heart_rate, (min_sys + pos) / fs))
    if method == "multi":
        return out
    return out[0]


# ----------------------------------------------------------------------------- durations
def get_duration_distributions(heartrate, systolic_time, ffs=FEATURES_FS):
    mean_S1 = matlab_round(0.122 * ffs); std_S1 = matlab_round(0.022 * ffs)
    mean_S2 = matlab_round(0.094 * ffs); std_S2 = matlab_round(0.022 * ffs)
    mean_sys = matlab_round(systolic_time * ffs) - mean_S1
    std_sys = (25 / 1000) * ffs
    mean_dia = ((60 / heartrate) - systolic_time - 0.094) * ffs
    std_dia = 0.07 * mean_dia + (6 / 1000) * ffs
    dist = [(mean_S1, std_S1 ** 2), (mean_sys, std_sys ** 2), (mean_S2, std_S2 ** 2), (mean_dia, std_dia ** 2)]
    min_sys = mean_sys - 3 * (std_sys + std_S1); max_sys = mean_sys + 3 * (std_sys + std_S1)
    min_dia = mean_dia - 3 * std_dia; max_dia = mean_dia + 3 * std_dia
    min_S1 = max(mean_S1 - 3 * std_S1, ffs / 50); min_S2 = max(mean_S2 - 3 * std_S2, ffs / 50)
    max_S1 = mean_S1 + 3 * std_S1; max_S2 = mean_S2 + 3 * std_S2
    bounds = [(min_S1, max_S1), (min_sys, max_sys), (min_S2, max_S2), (min_dia, max_dia)]
    return dist, bounds


# ----------------------------------------------------------------------------- viterbi
@njit(cache=True)
def _viterbi_forward(logobs, log_dur, log_pi, max_D):
    T, N = logobs.shape
    L = T + max_D - 1
    delta = np.full((L, N), -np.inf)
    psi = np.zeros((L, N), np.int64)          # stores 1-based preceding state
    psi_d = np.zeros((L, N), np.int64)
    cs = np.zeros((T + 1, N))
    for t in range(T):
        for j in range(N):
            cs[t + 1, j] = cs[t, j] + logobs[t, j]
    for j in range(N):
        delta[0, j] = log_pi[j] + logobs[0, j]
        psi[0, j] = -1
    for t in range(1, L):
        for j in range(N):
            i_prev = (j - 1) % N               # a_matrix: only (j-1) -> j allowed
            for d in range(1, max_D + 1):
                start = t - d
                if start < 0:
                    start = 0
                if start > T - 2:
                    start = T - 2
                end_t = t
                if end_t > T - 1:
                    end_t = T - 1
                max_delta = delta[start, i_prev]
                if max_delta == -np.inf:
                    continue
                em = cs[end_t + 1, j] - cs[start, j]
                if em == 0.0:
                    em = 1.1754943508222875e-38
                dt = max_delta + em + log_dur[j, d - 1]
                if dt > delta[t, j]:
                    delta[t, j] = dt
                    psi[t, j] = i_prev + 1
                    psi_d[t, j] = d
    return delta, psi, psi_d


def viterbi_decode(features, pi_vector, B, total_obs, heartrate, systolic_time, ffs=FEATURES_FS,
                   max_segments=1000, return_score=False):
    T = features.shape[0]
    N = 4
    max_D = int(matlab_round((60 / heartrate) * ffs))
    # observation probabilities
    from scipy.stats import multivariate_normal
    po = multivariate_normal(mean=total_obs[0], cov=total_obs[1], allow_singular=True).pdf(features)
    obs = np.zeros((T, N))
    for n in range(N):
        b = np.asarray(B[n], float).ravel()
        eta = b[0] + features @ b[1:]
        pihat2 = 1.0 / (1.0 + np.exp(eta))       # mnrval category 2 (= "in state")
        obs[:, n] = pihat2 * po / pi_vector[n]
    # duration probabilities
    dist, bounds = get_duration_distributions(heartrate, systolic_time, ffs)
    ncol = max(3 * ffs, max_D)
    dur = np.zeros((N, ncol))
    d = np.arange(1, max_D + 1, dtype=float)
    for j in range(N):
        mu, var = dist[j]
        p = np.exp(-0.5 * (d - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        lo, hi = bounds[j]
        p[(d < lo) | (d > hi)] = REALMIN
        dur[j, :max_D] = p
    dsum = dur.sum(axis=1)
    with np.errstate(divide="ignore"):
        log_dur = np.log(dur[:, :max(max_D, 1)] / dsum[:, None])
    obs_c = obs.copy()
    obs_c[obs_c == 0] = FLT_MIN
    logobs = np.log(obs_c)
    log_pi = np.log(np.asarray(pi_vector, float))
    delta, psi, psi_d = _viterbi_forward(logobs, log_dur, log_pi, max_D)
    # backtracking, MATLAB 1-based bookkeeping
    L = delta.shape[0]
    qt = np.zeros(L, np.int64)
    temp = delta[T:, :]
    flat = temp.T.ravel()                       # column-major like temp_delta(:)
    idx = int(np.argmax(flat))
    row = idx % temp.shape[0]
    pos = row + 1 + T                           # 1-based row in delta
    state = int(np.argmax(delta[pos - 1, :])) + 1
    offset = pos
    preceding = psi[offset - 1, state - 1]
    onset = offset - psi_d[offset - 1, state - 1] + 1
    qt[max(onset, 1) - 1: offset] = state
    state = preceding
    count = 0
    while onset > 2:
        offset = onset - 1
        preceding = psi[offset - 1, state - 1]
        onset = offset - psi_d[offset - 1, state - 1] + 1
        if onset < 2:
            onset = 1
        qt[onset - 1: offset] = state
        state = preceding
        count += 1
        if count > max_segments:
            break
    if return_score:
        return qt[:T], float(flat[idx])
    return qt[:T]


def expand_qt(qt, old_fs, new_fs, new_length):
    qt = np.asarray(qt).ravel()
    out = np.zeros(new_length, np.int64)
    changes = list(np.flatnonzero(np.diff(qt)) + 1) + [len(qt)]   # MATLAB 1-based end indices
    start = 0
    for end in changes:
        mid = int(matlab_round((end - start) / 2)) + start
        val = qt[mid - 1]
        es = int(matlab_round(start / old_fs * new_fs)) + 1
        ee = int(matlab_round(end / old_fs * new_fs))
        ee = min(ee, new_length)
        out[es - 1: ee] = val
        start = end
    return out


# ----------------------------------------------------------------------------- model I/O
def load_sample2016_params(folder):
    B = sio.loadmat(f"{folder}/Springer_B_matrix.mat", squeeze_me=True)["Springer_B_matrix"]
    pi = sio.loadmat(f"{folder}/Springer_pi_vector.mat", squeeze_me=True)["Springer_pi_vector"]
    tod = sio.loadmat(f"{folder}/Springer_total_obs_distribution.mat", squeeze_me=True)["Springer_total_obs_distribution"]
    B = [np.asarray(b, float) for b in B]
    pi = np.asarray([float(p) for p in pi])
    tod = (np.asarray(tod[0], float), np.asarray(tod[1], float))
    return B, pi, tod


def run_springer(audio, fs_in, params, hr_min=30.0, hr_max=120.0, sys_min_s=0.2, include_wavelet=True,
                 max_segments=1000, return_info=False, hr_method="argmax", n_candidates=3, hr_fixed=None):
    """challenge.m pipeline: resample to 1 kHz, features, HR, Viterbi, expand to 1 kHz.
    hr_method="multi": decode each candidate heart rate and keep the highest Viterbi score (deviation)."""
    B, pi, tod = params
    x = mresample(np.asarray(audio, float), AUDIO_FS, fs_in)
    feats, ffs = get_springer_pcg_features(x, AUDIO_FS, include_wavelet)
    if hr_fixed is not None:            # diagnostic only: externally supplied (heart rate, systolic time)
        hr, st = hr_fixed
        qt = viterbi_decode(feats, pi, B, tod, hr, st, ffs, max_segments)
    elif hr_method == "multi":
        best = None
        for hr_c, st_c in get_heart_rate_schmidt(x, AUDIO_FS, hr_min, hr_max, sys_min_s, "multi", n_candidates):
            q_c, sc = viterbi_decode(feats, pi, B, tod, hr_c, st_c, ffs, max_segments, return_score=True)
            sc_norm = sc / len(q_c)
            if best is None or sc_norm > best[0]:
                best = (sc_norm, q_c, hr_c, st_c)
        _, qt, hr, st = best
    else:
        hr, st = get_heart_rate_schmidt(x, AUDIO_FS, hr_min, hr_max, sys_min_s, hr_method)
        qt = viterbi_decode(feats, pi, B, tod, hr, st, ffs, max_segments)
    states = expand_qt(qt, ffs, AUDIO_FS, len(x))
    if return_info:
        return states, dict(hr=hr, systolic_time=st, n_feat=len(qt))
    return states


# ----------------------------------------------------------------------------- training
def label_pcg_states(envelope, s1_pos, s2_pos, fs=FEATURES_FS):
    """labelPCGStates.m (positions 1-based sample indices at fs, as in MATLAB)."""
    n = len(envelope)
    states = np.zeros(n, np.int64)
    mean_S1 = 0.122 * fs
    mean_S2 = 0.092 * fs
    std_S2 = 0.022 * fs
    s1_pos = np.asarray(s1_pos, float); s2_pos = np.asarray(s2_pos, float)
    for p in s1_pos:
        ub = int(matlab_round(min(n, p + mean_S1)))
        lo = int(max(1, p))
        states[lo - 1: min(ub, n)] = 1
    for p in s2_pos:
        lb = int(max(p - np.floor(mean_S2 + std_S2), 1))
        ub = int(min(n, np.ceil(p + np.floor(mean_S2 + std_S2))))
        win = envelope[lb - 1: ub] * (states[lb - 1: ub] != 1)
        s2i = min(n, lb + int(np.argmax(win)))
        ub2 = int(min(n, np.ceil(s2i + mean_S2 / 2)))
        states[int(max(np.ceil(s2i - mean_S2 / 2), 1)) - 1: ub2] = 3
        diffs = s1_pos - p
        diffs[diffs < 0] = np.inf
        if np.all(np.isinf(diffs)):
            end_pos = n
        else:
            end_pos = int(s1_pos[int(np.argmin(diffs))]) - 1
        states[int(np.ceil(s2i + mean_S2 / 2)) - 1: end_pos] = 4
    nz = np.flatnonzero(states != 0)
    if nz.size:
        first = nz[0]   # 0-based index of first definite == MATLAB first_location (1-based) - 1 + ... see below
        # MATLAB: first_location_of_definite_state = find(states~=0,1)-1 (1-based count of leading zeros)
        f = first
        if f > 1:
            if states[f] == 1:
                states[:f] = 4
            if states[f] == 3:
                states[:f] = 2
        last = nz[-1]
        if last + 1 > 1:
            if states[last] == 1:
                states[last:] = 2
            if states[last] == 3:
                states[last:] = 4
    states[states == 0] = 2
    return states


def train_band_pi(state_obs, rng=None):
    """trainBandPiMatricesSpringer.m with sklearn-free IRLS logistic regression (== mnrfit)."""
    rng = np.random.default_rng(0) if rng is None else rng
    pools = [np.vstack([so[s] for so in state_obs if len(so[s])]) for s in range(4)]
    total = np.vstack(pools)
    tod = (total.mean(axis=0), np.cov(total, rowvar=False))
    B = []
    for s in range(4):
        per_other = len(pools[s]) // 3
        per_other = min(per_other, min(len(pools[o]) for o in range(4) if o != s))
        pos = pools[s][rng.choice(len(pools[s]), per_other * 3, replace=False)]
        neg = np.vstack([pools[o][rng.choice(len(pools[o]), per_other, replace=False)] for o in range(4) if o != s])
        X = np.vstack([pos, neg])
        # mnrfit: log(P(cat1)/P(cat2)) = X b ; cat2 = state (label 2), cat1 = others
        y_cat1 = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])
        B.append(_irls_logit(X, y_cat1))
    return B, np.array([0.25] * 4), tod


def _irls_logit(X, y, iters=100, tol=1e-10):
    Xd = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        eta = Xd @ b
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-12, None)
        z = eta + (y - p) / W
        b_new = np.linalg.solve(Xd.T @ (Xd * W[:, None]), Xd.T @ (W * z))
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    return b
