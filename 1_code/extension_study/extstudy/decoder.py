"""Beat decoder shared by arms T and W (Panel H, decision 2), so that W against T compares traces, not decoders.

Input: an event trace (T: the R or S1 head output) or a synthetic ECG (W: derivative energy of its 5 to 25 Hz band).
Steps, all at 500 Hz:
  1. detection energy (ECG) or the trace itself (events); 60 ms smoothing for candidates;
  2. expected RR from autocorrelation (frozen detector's rr_prior: 30 to 242 bpm search, highest local maximum,
     half-lag switch at 0.6; 20 s windows accepted within 0.55 to 1.8 x the global value);
  3. candidates at least 120 ms apart with prominence at least 2% of the local beat level;
  4. emission from amplitude alone (relative to the local beat level);
  5. two-pass Viterbi over log-RR transitions (sinus SD 0.25, then 0.15 around a 9-beat running median; premature
     0.10 over 0.4 to 1.0 RR; long 0.05 over 1.0 to 1.8 RR; missed beat 0.01 at 2 and 3 RR), refractory
     max(200 ms, 0.35 x expected RR), restart penalty -12 (the frozen detector's values);
  6. beat position: energy centroid (ECG) or the trace maximum with parabolic refinement (events);
  7. no labels across gaps longer than 1.8 x the expected RR.
Nothing here reads a dataset; the trace comes from the caller.
"""
import numpy as np
import scipy.signal as sg

from extstudy import robust_beats as RB

FS = RB.FS
# emission 4 log2(amplitude / local beat level) + 2: zero at 0.71 of the local beat level (as in the frozen detector,
# 2 log2 + 1), but a good beat scores +2, not +1. With +1, dropping a stretch of beats through one restart (-12) beat
# keeping it whenever a third of its beats were missing (each missed beat costs about -3.4): 11 true beats lost in
# a simulated record with a 3 s pause
PARAMS = dict(cand_min_ms=120, cand_prom=0.02, ref_win_s=5.0, w_amp=4.0, amp_clip=(-4.0, 1.0), e_bias=2.0,
              sigma1=RB.P['sigma1'], max_gap1=RB.P['max_gap1'], sigma2=RB.P['sigma'], max_gap2=RB.P['max_gap_ratio'],
              rr_rel_min=RB.P['rr_rel_min'], gap_flag=1.8, cent_ms=60)


def beat_trace_from_heads(p_beat, p_s2):
    """Arm T's input to the decoder: the beat head (R or S1) minus the S2 head, floored at 0. An S2 sequence is as
    regular as the beat sequence; with S1 missing in a stretch, amplitude and rhythm alone preferred the S2 chain
    (simulated traces), so S2 evidence enters through the trace and the decoder stays the same for T and W."""
    return np.clip(np.asarray(p_beat, np.float64) - np.asarray(p_s2, np.float64), 0.0, None)


def ecg_energy(ecg500):
    """Squared 5-point derivative of the 5 to 25 Hz band (Pan-Tompkins form), and the band itself."""
    q = RB._filt_segments(np.asarray(ecg500, np.float64), RB._sos('qrs'), [0, len(ecg500)])
    q = q / (np.median(np.abs(q)) + 1e-12)
    d = np.zeros_like(q)
    d[2:-2] = (-q[4:] - 2 * q[3:-1] + 2 * q[1:-3] + q[:-4]) / 8.0
    return d ** 2, q


def _local_level(pk, h, rre, W):
    ref = np.empty(len(pk))
    for i, p in enumerate(pk):
        a, b = np.searchsorted(pk, p - W), np.searchsorted(pk, p + W)
        k = max(2, int(round(2 * PARAMS['ref_win_s'] / max(rre[min(p, len(rre) - 1)], 0.2))))
        ref[i] = np.median(np.sort(h[a:b])[::-1][:k])
    return ref


def _refine_peak(y, p):
    if 1 <= p < len(y) - 1:
        y0, y1, y2 = y[p - 1], y[p], y[p + 1]
        den = y0 - 2 * y1 + y2
        return p + (np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) if den != 0 else 0.0)
    return float(p)


def decode_beats(trace, fs, kind='event'):
    """trace: event probability (kind='event') or synthetic ECG (kind='ecg') at fs.
    Returns dict(beats_s, beats500, mask500 (True = labelled), rre_s, rec)."""
    x = RB.resample_to(np.nan_to_num(np.asarray(trace, np.float64)), fs, FS)
    n = len(x)
    if kind == 'ecg':
        e, q = ecg_energy(x)
        eraw = q ** 2
    elif kind == 'event':
        e, eraw = np.maximum(x, 0.0), None
    else:
        raise ValueError(kind)
    e60 = RB._movavg(e, 0.060 * FS) if kind == 'ecg' else RB._movavg(e, 0.020 * FS)
    rre, acf_strength = RB.rr_prior(e60 if kind == 'ecg' else RB._movavg(e, 0.060 * FS))
    pk, pr = sg.find_peaks(e60, distance=int(PARAMS['cand_min_ms'] / 1000 * FS), prominence=0)
    empty = dict(beats_s=np.array([]), beats500=np.array([], int), mask500=np.zeros(n, bool), rre_s=rre,
                 rec=dict(n_beats=0, acf_strength=acf_strength, coverage=0.0, hr_median=np.nan))
    if len(pk) < 3:
        return empty
    h = e60[pk]
    ref = _local_level(pk, h, rre, int(PARAMS['ref_win_s'] * FS))
    keep = pr['prominences'] >= PARAMS['cand_prom'] * ref
    pk, h, ref = pk[keep], h[keep], ref[keep]
    if len(pk) < 3:
        return empty
    em = PARAMS['w_amp'] * np.clip(np.log2(h / (ref + 1e-12) + 1e-12), *PARAMS['amp_clip']) + PARAMS['e_bias']
    if kind == 'ecg':
        half = int(PARAMS['cent_ms'] / 1000 * FS)
        pos = np.array([RB._centroid(eraw, p, half, 2) for p in pk])
    else:
        pos = np.array([_refine_peak(e60, p) for p in pk])
    tc = np.round(pos).astype(int)
    order = np.argsort(tc)
    tc, em_o = tc[order], em[order]
    path = RB._viterbi(tc, em_o, rre[np.clip(tc, 0, n - 1)], FS, [], n, PARAMS['sigma1'], PARAMS['max_gap1'])
    beats = tc[path]
    if len(beats) >= 5:
        rr = np.diff(beats) / FS
        okrr = rr < 1.7 * np.median(rr)
        if okrr.sum() >= 3:
            mid = 0.5 * (beats[1:] + beats[:-1])
            k = min(9, int(okrr.sum()) | 1)
            rr_run = sg.medfilt(rr[okrr], k) if okrr.sum() >= k else np.full(okrr.sum(), np.median(rr[okrr]))
            rre = np.interp(np.arange(n), mid[okrr], rr_run)
            path = RB._viterbi(tc, em_o, rre[np.clip(tc, 0, n - 1)], FS, [], n, PARAMS['sigma2'], PARAMS['max_gap2'],
                               PARAMS['rr_rel_min'])
    sel = order[path]
    if kind == 'ecg':
        b = np.array([RB._centroid(eraw, pk[i], int(min(PARAMS['cent_ms'] / 1000, 0.15 * rre[min(pk[i], n - 1)]) * FS), 2)
                      for i in sel])
    else:
        b = pos[sel]
    b500 = np.round(b).astype(int)
    mask = np.ones(n, bool)
    rre_b = rre[np.clip(b500, 0, n - 1)] * FS
    for a, c, r in zip(b500[:-1], b500[1:], rre_b[:-1]):
        if c - a > PARAMS['gap_flag'] * r:
            mask[a:c] = False
    if len(b500):
        if b500[0] > PARAMS['gap_flag'] * rre_b[0]:
            mask[:b500[0]] = False
        if n - b500[-1] > PARAMS['gap_flag'] * rre_b[-1]:
            mask[b500[-1]:] = False
    rec = dict(n_beats=int(len(b500)), acf_strength=float(acf_strength), coverage=float(mask.mean()),
               hr_median=float(60 * FS / np.median(np.diff(b500))) if len(b500) > 2 else np.nan,
               n_candidates=int(len(pk)))
    return dict(beats_s=b / FS, beats500=b500, mask500=mask, rre_s=rre, rec=rec)
