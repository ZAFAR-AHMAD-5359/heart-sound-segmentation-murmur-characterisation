"""Teacher R-peaks on real ECG.

Detector: Panel H's robust_beats.py, copied unchanged (its SHA-256 is checked by the tests). SOS filters with a pole
assertion; candidates on QRS slope energy; a heart-rate prior from autocorrelation; an emission score that penalises
narrow spikes, T waves taller than the QRS, and wide deflections; a two-pass Viterbi decoder with physiological RR
transitions (premature, long, and missed-beat components). The beat position is the polarity-free centroid of the
5 to 25 Hz energy, never the largest absolute deflection (which lands on the S trough in part of the beats).

Two-of-three rule: a robust beat is kept when XQRS or GQRS also finds it within 50 ms, after removing each wfdb
detector's own record offset (GQRS marks the QRS onset, 20 to 50 ms before the centroid, with a 2 ms spread inside
a record). Beats found only by XQRS and GQRS are not added (both accept narrow spikes together), but the cycle that
holds one carries no labels (bigeminy: the decoder keeps a regular normal-to-normal chain and every cycle hides a
premature beat). The same holds for a rejected candidate with at least half the local beat level's QRS energy that
one wfdb detector confirms (wide premature beats that XQRS misses).

Good beat: confirmed and template correlation >= 0.8 (0.5 to 40 Hz, +/- 100 ms, from the robust detector). In 20 s of
electrode noise the rhythm model places beats and XQRS and GQRS fire 2 to 3 times per second, so chance confirmation
is common; the correlation of such beats has median 0.5. No labels: the two cycles next to a beat that is not good;
neighbourhoods where under 80% of the robust beats within +/- 5 s are good; RR intervals above 1.5 x the local median (missed or dropped beats), and NaN
stretches. Agreement per 60 s window is reported, not used as a mask: XQRS and GQRS accept narrow spikes together
and GQRS loses beats above about 180 bpm, so reference-against-reference checks masked exactly the windows where the
robust detector was right (stress tests, 15 Oct 2026).
"""
import numpy as np

from extstudy import robust_beats as RB
from extstudy.config import BEATS

FS = RB.FS
FROZEN_SHA256 = '09bb9d7f22105eefd31cd7ae883da95af98e7d05dd8e126c8e20799035d4c077'


# ------------------------------------------------------------------ independent detectors (wfdb)
def mv_scale(x, fs=FS):
    """QRS amplitude near 1 mV: median over 2 s blocks of the block maximum |x - median|. A 99.5th-percentile scale
    let one artefact shrink the QRS below XQRS's learning threshold (0 to 1 beats on 6 of 60 paired records)."""
    x = np.asarray(x, np.float64)
    x = x - np.median(x)
    B = int(2 * fs)
    nb = len(x) // B
    a = np.median(np.abs(x[:nb * B]).reshape(nb, -1).max(1)) if nb >= 1 else np.max(np.abs(x))
    return x / (a + 1e-12)


def _wfdb_input(x, fs=FS):
    """Scaled ECG clipped at +/- wfdb_clip QRS amplitudes. One unclipped artefact raised XQRS's adaptive threshold
    so far that it missed the next 347 of 375 beats in a 300 s test (F1 0.12; clipped 0.997; no change on any stress
    condition)."""
    return np.clip(mv_scale(x, fs), -BEATS.wfdb_clip, BEATS.wfdb_clip)


def xqrs(x, fs=FS):
    import logging
    import wfdb.processing as wp
    logging.disable(logging.CRITICAL)
    try:
        q = wp.XQRS(sig=_wfdb_input(x, fs), fs=fs)
        q.detect(verbose=False)
        return np.sort(np.asarray(q.qrs_inds, int))
    except Exception:
        return np.array([], int)


def gqrs(x, fs=FS):
    import wfdb.processing as wp
    try:
        return np.sort(np.asarray(wp.gqrs_detect(sig=_wfdb_input(x, fs), fs=fs), int))
    except Exception:
        return np.array([], int)


# ------------------------------------------------------------------ matching
def match_pairs(a, b, tol):
    """One-to-one matching of sorted event lists within tol samples, greedy by distance.
    Returns (ia, ib): indices of matched events in a and in b."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.array([], int), np.array([], int)
    cand = []
    for i, x in enumerate(a):
        lo, hi = np.searchsorted(b, x - tol), np.searchsorted(b, x + tol, side='right')
        cand += [(abs(int(b[j]) - int(x)), i, j) for j in range(lo, hi)]
    cand.sort()
    ua, ub = set(), set()
    for _, i, j in cand:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j)
    ia = np.array(sorted(ua), int)
    ib = np.array(sorted(ub), int)
    return ia, ib


def f1_of(a, b, tol):
    ia, _ = match_pairs(a, b, tol)
    return RB.f1(len(ia), len(b) - len(ia), len(a) - len(ia))


def nearest_dist(a, b):
    """Distance from each event of a to the nearest event of b (inf when b is empty)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    d = np.abs(b[j] - a)
    if len(b) > 1:
        d = np.minimum(d, np.abs(b[j - 1] - a))
    return d.astype(float)


def record_offset(ref, det, fs=FS):
    """Median signed distance (det - nearest ref) over pairs within offset_search_ms; 0 with fewer than 5 pairs."""
    ref, det = np.asarray(ref), np.asarray(det)
    if len(ref) < 2 or len(det) < 2:
        return 0
    j = np.clip(np.searchsorted(ref, det), 1, len(ref) - 1)
    near = np.where(np.abs(ref[j] - det) < np.abs(ref[j - 1] - det), ref[j], ref[j - 1])
    off = det - near
    off = off[np.abs(off) <= BEATS.offset_search_ms / 1000 * fs]
    return int(np.round(np.median(off))) if len(off) >= 5 else 0


# ------------------------------------------------------------------ teacher rule
def _fill_nan(x):
    x = np.asarray(x, np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad]) if (~bad).any() else 0.0
    return x, bad


def _local_rr(beats, n):
    """Expected RR (samples) at every sample: 9-beat running median of the RR series, held at the ends."""
    if len(beats) < 3:
        return np.full(n, 0.8 * FS)
    rr = np.diff(beats).astype(float)
    k = min(9, len(rr))
    k = max(1, k - 1 if k % 2 == 0 else k)
    pad = k // 2
    rp = np.r_[np.full(pad, rr[0]), rr, np.full(pad, rr[-1])]
    med = np.array([np.median(rp[i:i + k]) for i in range(len(rr))])
    mid = 0.5 * (beats[1:] + beats[:-1])
    return np.interp(np.arange(n), mid, med)


def _windows(n, fs=FS):
    W = int(BEATS.window_s * fs)
    starts = list(range(0, n, W))
    if len(starts) > 1 and n - starts[-1] < W // 2:
        starts.pop()
    ends = starts[1:] + [n]
    return np.array(starts, int), np.array(ends, int)


def _counts_by_window(a, b, tol, starts):
    """Per-window TP/FP/FN of b against a (a = reference), attributing each event to the window it falls in."""
    ia, ib = match_pairs(a, b, tol)
    nw = len(starts)
    wa = np.clip(np.searchsorted(starts, a, side='right') - 1, 0, nw - 1)
    wb = np.clip(np.searchsorted(starts, b, side='right') - 1, 0, nw - 1)
    tp = np.bincount(wa[ia], minlength=nw)
    fn = np.bincount(wa, minlength=nw) - tp
    fp = np.bincount(wb, minlength=nw) - np.bincount(wb[ib], minlength=nw)
    return tp, fp, fn


def teacher_beats(ecg, fs=FS):
    """Teacher R-peaks for one ECG lead. Returns a dict:
    beats (500 Hz samples, energy centroid), source (bit 0 robust, 1 XQRS, 2 GQRS), r_signed (record-polarity
    extremum within 40 ms), corr (template correlation), s_dominant (S deflection larger than R), conf (robust beat
    confidence),
    label_mask (bool per 500 Hz sample), windows (start, end, f1_xg, f1_rx, ok: reporting only), rec (record metrics)."""
    x, bad = _fill_nan(ecg)
    o = RB.detect(x, fs, return_debug=True)
    x5 = RB.resample_to(x, fs, FS)
    n = len(x5)
    nan5 = RB.resample_to(bad.astype(float), fs, FS) > 0.5 if bad.any() else np.zeros(n, bool)
    dbg = o.get('dbg')
    r = np.sort(np.asarray(o['beats'], int))
    xq, gq = xqrs(x5), gqrs(x5)
    tol = int(round(BEATS.tol_ms / 1000 * FS))
    ox, og = record_offset(r, xq), record_offset(r, gq)
    xqc, gqc = xq - ox, gq - og

    # votes on the robust beats
    vx = nearest_dist(r, xqc) <= tol
    vg = nearest_dist(r, gqc) <= tol
    keep_r = vx | vg
    src_r = 1 + 2 * vx.astype(int) + 4 * vg.astype(int)
    corr_r = (np.asarray(dbg['corr'], float)[dbg['sel']][np.argsort(o['beats'])] if (dbg is not None and len(r))
              else np.ones(len(r)))
    good_r = keep_r & (corr_r >= BEATS.corr_min)

    # beats XQRS and GQRS agree on but the robust detector rejected: never added (both wfdb detectors accept narrow
    # spikes together: 7 false beats per minute at 40 bpm when such beats were added back), but the cycle that
    # holds one carries no labels. In bigeminy the decoder keeps the regular normal-to-normal chain, every RR looks
    # regular, and the gap mask stays silent while each cycle hides a premature beat (30 usable HFrEF records,
    # first relabelling, 15 Oct 2026)
    ix, ig = match_pairs(xqc, gqc, tol)
    xg = np.round(0.5 * (xqc[ix] + gqc[ig])).astype(int) if len(ix) else np.array([], int)
    xg_only = xg[nearest_dist(xg, r) > tol] if len(xg) else xg
    n_xg_only = int(len(xg_only))
    # rejected candidates that carry QRS energy (at least half the local beat level) and that one wfdb detector also
    # finds: wide premature beats, which XQRS's matched filter misses (simulated bigeminy: XQRS 0%, GQRS 100%).
    # Narrow spikes are removed before candidates are formed, so they do not reach this rule
    extra = np.array([], int)
    if dbg is not None and len(dbg['pk']):
        cen = np.round(np.asarray(dbg['cen'])).astype(int)
        rej = np.setdiff1d(np.arange(len(cen)), np.asarray(dbg['sel'], int))
        rej = rej[np.asarray(dbg['amp'])[rej] >= BEATS.extra_amp_min]
        c = cen[rej]
        if len(c):
            far = nearest_dist(c, r) > int(0.120 * FS)
            ref1 = (nearest_dist(c, xqc) <= tol) | (nearest_dist(c, gqc) <= tol)
            extra = c[far & ref1]
    n_extra_candidates = int(len(extra))
    beats, source, corr_b = r[keep_r], src_r[keep_r], corr_r[keep_r]
    conf_all = np.asarray(o['conf'], float)[np.argsort(o['beats'])][keep_r] if len(r) else np.array([])

    # fiducials for every final beat: record polarity from the robust templates
    pol = float(o['rec'].get('polarity', 1.0)) or 1.0
    if dbg is not None:
        bb = dbg['bb']
    else:
        bb = RB._filt_segments(x5, RB._sos('bb'), [0, n])
    w40 = int(0.040 * FS)
    r_signed, s_dom = np.zeros(len(beats), int), np.zeros(len(beats), bool)
    for k, c in enumerate(beats):
        a, b = max(0, c - w40), min(n, c + w40 + 1)
        seg = pol * bb[a:b]
        r_signed[k] = a + int(np.argmax(seg))
        s_dom[k] = bool(np.max(-seg) > np.max(seg))

    # masks: gaps, NaN, unconfirmed neighbourhoods
    gap = nan5.copy()
    rre = _local_rr(beats, n)
    for a, b in zip(beats[:-1], beats[1:]):
        if b - a > BEATS.gap_ratio * rre[a]:
            gap[a:b] = True
    if len(beats):
        if beats[0] > BEATS.gap_ratio * rre[beats[0]]:
            gap[:beats[0]] = True
        if n - beats[-1] > BEATS.gap_ratio * rre[beats[-1]]:
            gap[beats[-1]:] = True
    else:
        gap[:] = True
    kept = beats
    for p in np.r_[xg_only, extra]:                     # the cycle of kept beats that holds the extra beat
        j = np.searchsorted(kept, p)
        a = kept[j - 1] if j > 0 else 0
        b = kept[j] if j < len(kept) else n
        gap[a:b] = True
    n_unconfirmed_zones = 0
    if len(r):
        # kept beats that fail the correlation check: their two cycles carry no labels
        for i in np.flatnonzero(keep_r & ~good_r):
            a = r[i - 1] if i > 0 else 0
            b = r[i + 1] if i + 1 < len(r) else n
            gap[a:b] = True
        cs = np.r_[0, np.cumsum(good_r)]
        L = int(BEATS.local_s * FS)
        lo, hi = np.searchsorted(r, r - L), np.searchsorted(r, r + L, side='right')
        share = (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)
        for i in np.flatnonzero(share < BEATS.local_confirm_min):
            a = r[i - 1] if i > 0 else 0
            b = r[i + 1] if i + 1 < len(r) else n
            gap[a:b] = True
            n_unconfirmed_zones += 1
    label_mask = ~gap
    starts, ends = _windows(n)
    tp, fp, fn = _counts_by_window(xqc, gqc, tol, starts)
    f1_xg_w = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    tp, fp, fn = _counts_by_window(xqc, r, tol, starts)
    f1_rx_w = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    win_ok = (f1_xg_w >= BEATS.xg_f1_min) & (f1_rx_w >= BEATS.rx_f1_min)

    f1_xg, f1_rx, f1_rg = f1_of(xqc, gqc, tol), f1_of(xqc, r, tol), f1_of(gqc, r, tol)
    rec = dict(
        n_beats=int(len(beats)), n_robust=int(len(r)), n_xqrs=int(len(xq)), n_gqrs=int(len(gq)),
        n_robust_rejected=int((~keep_r).sum()), n_low_corr=int((keep_r & ~good_r).sum()), n_xg_only=n_xg_only, n_extra_candidates=n_extra_candidates,
        offset_xqrs_ms=ox / FS * 1000, offset_gqrs_ms=og / FS * 1000,
        f1_xg=f1_xg, f1_rx=f1_rx, f1_rg=f1_rg,
        rec_rule_ok=bool(f1_xg >= BEATS.xg_f1_min and f1_rx >= BEATS.rx_f1_min and len(beats) >= 5),
        n_windows=int(len(starts)), share_windows_agree=float(win_ok.mean()) if len(win_ok) else 0.0,
        n_unconfirmed_beats_flagged=n_unconfirmed_zones,
        label_coverage=float(label_mask.mean()), frac_s_dominant=float(s_dom.mean()) if len(s_dom) else np.nan,
        hr_median=float(60 * FS / np.median(np.diff(beats))) if len(beats) > 2 else np.nan,
        polarity=pol, robust_abstain=bool(o['rec']['abstain']), robust_coverage=float(o['rec']['coverage']),
        robust_median_corr=float(o['rec']['median_corr']), n_despiked=int(o['rec'].get('n_despiked', 0)),
        duration_s=n / FS)
    return dict(beats=beats.astype(np.int32), source=source.astype(np.uint8), r_signed=r_signed.astype(np.int32),
                corr=corr_b.astype(np.float32),
                s_dominant=s_dom, conf=conf_all.astype(np.float32), label_mask=label_mask,
                windows=dict(start=starts, end=ends, f1_xg=f1_xg_w, f1_rx=f1_rx_w, ok=win_ok), rec=rec, fs=FS)


def mask_to_intervals(mask):
    """Boolean mask -> (k, 2) array of [start, end) runs where the mask is False (no labels)."""
    m = np.r_[False, ~np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)].astype(np.int64)


def intervals_to_mask(iv, n):
    mask = np.ones(n, bool)
    for a, b in np.asarray(iv).reshape(-1, 2):
        mask[a:b] = False
    return mask


# ------------------------------------------------------------------ S1-lag check of the fiducial
def s1_lags_ms(env500, beats, fs=FS):
    """Per beat: delay (ms) of the PCG-envelope maximum in [-50, +250] ms after the beat."""
    out = np.full(len(beats), np.nan)
    a0, b0 = int(0.05 * fs), int(0.25 * fs)
    for i, p in enumerate(beats):
        a, b = p - a0, p + b0
        if a >= 0 and b <= len(env500):
            out[i] = (np.argmax(env500[a:b]) - a0) / fs * 1000
    return out


def lag_gap_ms(env500, beats, s_dominant, keep=None):
    """Median S1 lag of S-dominant beats minus that of R-dominant beats (nan with fewer than min_s_beats)."""
    lags = s1_lags_ms(env500, beats)
    keep = np.ones(len(beats), bool) if keep is None else np.asarray(keep, bool)
    s, r = keep & s_dominant & np.isfinite(lags), keep & ~s_dominant & np.isfinite(lags)
    if s.sum() < BEATS.min_s_beats or r.sum() < BEATS.min_s_beats:
        return np.nan, int(s.sum())
    return float(np.median(lags[s]) - np.median(lags[r])), int(s.sum())
