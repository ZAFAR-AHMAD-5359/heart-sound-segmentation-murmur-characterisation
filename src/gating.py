"""
Cardiac-phase gating: R-peak detection on the synthetic ECG and cycle boundaries.

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

This module is the study's independent variable. Two boundary sources exist:

  synthetic_cycles(...)   R-peaks detected on the reconstructed ECG divide each
                          cycle into S1, systole, S2 and diastole
  equal_time_cycles(...)  equal-duration windows that ignore the cardiac cycle,
                          the ablation control

Everything downstream (features, classifier, evaluation) is identical across the
two, so the paired difference isolates what cycle alignment contributes.

The R-peak detector is deliberately modular. It is a bandpass-plus-envelope
detector in the Pan-Tompkins style with physiological post-filters, and the phase
boundaries consume nothing but an array of peak indices. A stronger detector, or
a segmentation model fused with acoustic S1/S2 evidence, can be swapped in behind
the same interface, and better peak timing should carry through to better
phase alignment.
"""
import numpy as np
from scipy import signal as sig

# Phase boundaries as fractions of the RR interval, from R-peak to R-peak:
# S1 0.00-0.15, systole 0.15-0.50, S2 0.50-0.60, diastole 0.60-1.00.
PHASE_SPLITS = (0.15, 0.50, 0.60)


def ensure_positive_polarity(ecg):
    """Flip the trace when the dominant deflection is negative, so the R-peak is
    always the maximum. The generator carries no polarity constraint."""
    if np.abs(ecg.min()) > np.abs(ecg.max()):
        return -ecg
    return ecg


def _filter_peaks(peaks, ecg):
    """Physiological post-filters on candidate peaks.

    1. Amplitude: drop candidates below 30% of the median peak amplitude.
    2. RR regularity: when two candidates sit closer than 60% of the median RR
       interval, keep the larger one; a double detection of a single beat would
       otherwise split one cycle into two half-cycles.
    """
    if len(peaks) < 3:
        return peaks
    amps = np.abs(ecg[peaks])
    peaks = peaks[amps >= 0.30 * np.median(amps)]
    if len(peaks) < 3:
        return peaks
    median_rr = np.median(np.diff(peaks))
    if median_rr <= 0:
        return peaks
    valid = np.ones(len(peaks), dtype=bool)
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] < 0.60 * median_rr:
            if np.abs(ecg[peaks[i]]) < np.abs(ecg[peaks[i - 1]]):
                valid[i] = False
            else:
                valid[i - 1] = False
    return peaks[valid]


def _align_to_maxima(peaks, ecg, fs, window_ms=40):
    """Move each envelope peak to the local maximum of the raw trace. The
    smoothed-envelope peak sits 10 to 30 ms off the R-peak apex."""
    half = int(window_ms / 1000 * fs)
    out = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(ecg), p + half + 1)
        out.append(lo + int(np.argmax(ecg[lo:hi])))
    return np.array(out, dtype=int)


def detect_r_peaks(ecg, fs, min_distance_ms=350):
    """R-peaks on a synthetic ECG trace.

    1. Bandpass 5 to 30 Hz (second-order sections, which stay numerically stable
       at this ratio of band to sampling rate), square, smooth over 80 ms.
    2. Peak-pick at descending prominence thresholds until at least three
       candidates survive.
    3. Realign each candidate to the raw-trace maximum, then apply the
       physiological post-filters.

    Falls back to envelope peak-picking, and finally to evenly spaced beats at
    80 bpm, so downstream stages always receive a usable peak train. The
    fallbacks exist for pathological traces; the caller can compare the returned
    peak count against the recording duration to judge detection quality.
    """
    min_dist = int(min_distance_ms / 1000 * fs)

    sos = sig.butter(4, [5, 30], btype='bandpass', fs=fs, output='sos')
    filtered = sig.sosfiltfilt(sos, ecg)
    smoothed = np.convolve(filtered ** 2,
                           np.ones(int(0.08 * fs)) / int(0.08 * fs), mode='same')
    if np.all(np.isfinite(smoothed)):
        for prom in (0.4, 0.25, 0.15, 0.08):
            peaks, _ = sig.find_peaks(smoothed, distance=min_dist,
                                      prominence=np.std(smoothed) * prom)
            if len(peaks) >= 3:
                peaks = _filter_peaks(_align_to_maxima(peaks, ecg, fs), ecg)
                if len(peaks) >= 3:
                    return peaks

    # fallback: envelope of the raw trace
    win = int(0.05 * fs)
    env = np.convolve(np.abs(ecg), np.ones(win) / win, mode='same')
    peaks, _ = sig.find_peaks(env, distance=min_dist,
                              height=np.percentile(env, 75))
    if len(peaks) >= 3:
        peaks = _filter_peaks(_align_to_maxima(peaks, ecg, fs), ecg)
        if len(peaks) >= 3:
            return peaks

    # last resort: evenly spaced beats at 80 bpm
    n_beats = max(2, int(len(ecg) / fs * 80 / 60))
    return np.linspace(0, len(ecg) - 1, n_beats + 1, dtype=int)[:-1]


def synthetic_cycles(r_peaks, out_sr, ecg_sr):
    """Cardiac-phase boundaries from consecutive R-peaks.

    Peaks are detected at ecg_sr and the boundaries are returned as sample
    indices at out_sr, the rate the feature extractor runs at. Each RR interval
    is divided by the fixed fractions in PHASE_SPLITS; without P and T wave
    morphology to anchor finer landmarks, fixed fractions of the RR interval are
    the standard first-order model of phase durations.
    """
    scale = out_sr / ecg_sr
    a, b, c = PHASE_SPLITS
    out = []
    for i in range(len(r_peaks) - 1):
        r0, rr = r_peaks[i] * scale, (r_peaks[i + 1] - r_peaks[i]) * scale
        s1_end = int(r0 + a * rr)
        sys_end = int(r0 + b * rr)
        s2_end = int(r0 + c * rr)
        out.append({'cycle': i,
                    's1': (int(r0), s1_end),
                    'systole': (s1_end, sys_end),
                    's2': (sys_end, s2_end),
                    'diastole': (s2_end, int(r0 + rr))})
    return out


def equal_time_cycles(n_samples, sr, bpm=75):
    """The ablation control: equal-length pseudo-cycles at a fixed 75 bpm, each
    split into four equal quarters. Same feature extractor, same classifier, no
    cardiac information in the boundaries."""
    cycle_len = int(sr * 60 / bpm)
    out = []
    for i in range(max(1, n_samples // cycle_len)):
        s = i * cycle_len
        e = min(n_samples, (i + 1) * cycle_len)
        rr = e - s
        if rr <= 0:
            continue
        out.append({'cycle': i,
                    's1': (s, s + int(0.25 * rr)),
                    'systole': (s + int(0.25 * rr), s + int(0.50 * rr)),
                    's2': (s + int(0.50 * rr), s + int(0.75 * rr)),
                    'diastole': (s + int(0.75 * rr), e)})
    return out
