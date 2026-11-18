"""Robust beat detector prototype (Panel H, 15 Oct 2026).

Uses: (a) teacher labels on real ECG; (b) beat read-out from a synthetic ECG (arm W) or an event trace (arm T).

Pipeline
  1. preprocess: NaN interpolation + mask, polyphase resampling to 500 Hz, SOS filters with a pole assertion,
     optional segment boundaries (generator windows) so no filter ever crosses a window join.
  2. QRS energy e = (5-25 Hz band-pass)^2, smoothed 20 ms (shape) and 60 ms (candidates).
  3. heart-rate prior: autocorrelation of the energy envelope in 10 s windows, highest local maximum in 40-220 bpm
     with a half-lag (double-rate) check.
  4. candidates: local maxima of the 60 ms energy, minimum spacing 120 ms, prominence >= 2% of the local beat level.
  5. features per candidate: amplitude ratio to the local beat level, template correlation (0.5-40 Hz, +/-100 ms,
     template per segment), high-frequency ratio (40-120 Hz / 5-25 Hz energy, spike detector), QRS share
     (5-25 Hz energy / 0.5-40 Hz energy, T-wave detector), half-energy width.
  6. emission score (fixed weights) + transition score log p(RR | expected RR) (log-normal sinus component,
     premature, long and missed-beat components), Viterbi over candidates, restart allowed with a penalty.
     Two passes: the second uses the running-median RR of the first.
  7. fiducial: polarity-free centroid of the raw 5-25 Hz energy within +/-60 ms (iterated twice), plus the
     record-polarity extremum (sign fixed by the template, never per beat) for comparison.
  8. confidence per beat and per recording; abstention mask for gaps; recording-level abstain flag.
"""
from math import gcd
import numpy as np
import scipy.signal as sg

FS = 500
P = dict(
    bp_qrs=(5.0, 25.0), bp_order=2, bp_hf=(40.0, 120.0), bb=(0.5, 40.0),
    e_shape_ms=20, e_cand_ms=60, cand_min_ms=120, cand_prom=0.02,
    hr_min=40.0, hr_max=220.0, acf_win_s=20.0, acf_hop_s=5.0, half_lag_ratio=0.6,
    tmpl_ms=100, ref_win_s=5.0,
    # emission weights
    w_amp=2.0, amp_clip=(-4.0, 1.0), w_corr=2.0, corr_mid=0.5, w_hf=3.0, hf_tol=3.0, w_share=3.0, share_tol=0.5,
    e_bias=1.0, w_wid=1.5, wid_ref=0.012,
    # transition model (log RR relative to the expected RR)
    sigma=0.15, eps_prem=0.10, prem_lo=0.40, eps_long=0.05, long_hi=1.8, eps_gap=0.01,
    rr_abs_min_s=0.20, rr_rel_min=0.35, max_gap_ratio=3.4, sigma1=0.25, max_gap1=3.5, restart=-12.0, join_ms=50,
    # fiducial
    cent_ms=60, cent_iter=2,
    # abstention
    gap_flag=1.8, rec_min_cov=0.80, rec_min_corr=0.60, rec_max_lowconf=0.20,
)


# ------------------------------------------------------------------ filters
def sos_design(kind, f, fs, order):
    sos = sg.butter(order, f, btype=kind, fs=fs, output='sos')
    r = float(np.abs(sg.sos2zpk(sos)[1]).max())
    assert r < 0.9999, 'unstable or near-unstable SOS design (pole radius %.6f)' % r
    return sos


_SOS = {}


def _sos(name, fs=FS):
    if (name, fs) not in _SOS:
        if name == 'qrs':
            _SOS[(name, fs)] = sos_design('bandpass', P['bp_qrs'], fs, P['bp_order'])
        elif name == 'hf':
            _SOS[(name, fs)] = sos_design('bandpass', P['bp_hf'], fs, 3)
        elif name == 'bb':
            _SOS[(name, fs)] = sos_design('bandpass', P['bb'], fs, 2)
    return _SOS[(name, fs)]


def resample_to(x, fs_in, fs_out=FS):
    fs_in, fs_out = int(round(fs_in)), int(round(fs_out))
    if fs_in == fs_out:
        return np.asarray(x, np.float64)
    g = gcd(fs_in, fs_out)
    return sg.resample_poly(np.asarray(x, np.float64), fs_out // g, fs_in // g)


def _filt_segments(x, sos, bounds):
    out = np.empty_like(x)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = x[a:b]
        pad = min(len(seg) - 1, 3 * 60) if len(seg) > 6 else 0
        out[a:b] = sg.sosfiltfilt(sos, seg, padlen=pad) if len(seg) > 12 else 0.0
    return out


def _movavg(x, n):
    n = max(1, int(n))
    return np.convolve(x, np.ones(n) / n, mode='same')


# ------------------------------------------------------------------ heart-rate prior
def acf_rr(env50, fs50=50):
    """Highest local ACF maximum within 40-220 bpm, with a half-lag check. Returns RR (s) and ACF strength."""
    x = env50 - np.mean(env50)
    n = len(x)
    if n < fs50 * 3 or np.std(x) == 0:
        return np.nan, 0.0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(60 / (P['hr_max'] * 1.1) * fs50), min(n - 2, int(60 / (P['hr_min'] * 0.75) * fs50))
    if hi <= lo + 2:
        return np.nan, 0.0
    seg = ac[lo:hi + 1]
    pk, _ = sg.find_peaks(seg)
    if len(pk) == 0:
        return np.nan, 0.0
    k = pk[np.argmax(seg[pk])]
    L = lo + k
    # half-lag check: a clear local maximum near L/2 means the strongest peak skipped every other beat
    h = L / 2.0
    cand = [lo + p for p in pk if abs(lo + p - h) <= 0.12 * h]
    if cand:
        c = max(cand, key=lambda q: ac[q])
        if ac[c] >= P['half_lag_ratio'] * ac[L]:
            L = c
    # parabolic refinement
    if 1 <= L < n - 1:
        y0, y1, y2 = ac[L - 1], ac[L], ac[L + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        d = 0.0
    return (L + np.clip(d, -0.5, 0.5)) / fs50, float(ac[L])


def rr_prior(e60, fs=FS):
    """Expected RR per sample. Global ACF over the record (first 120 s), then 20 s windows (5 s hop) accepted only
    within [0.55, 1.8] x the global value; the decoder's second pass replaces this prior by the decoded RR."""
    dec = fs // 50
    env50 = e60[: len(e60) // dec * dec].reshape(-1, dec).mean(1)
    env50 = np.sqrt(np.maximum(env50, 0))          # compress dynamic range: a single spike cannot dominate
    g_rr, g_st = acf_rr(env50[: 120 * 50])
    if not np.isfinite(g_rr):
        g_rr, g_st = 0.8, 0.0
    W, H = int(P['acf_win_s'] * 50), int(P['acf_hop_s'] * 50)
    cen, rr = [], []
    for a in range(0, max(1, len(env50) - W + 1), H):
        r, _ = acf_rr(env50[a:a + W])
        if not (np.isfinite(r) and 0.55 * g_rr <= r <= 1.8 * g_rr):
            r = g_rr
        cen.append(a + min(W, len(env50)) / 2); rr.append(r)
    cen, rr = np.array(cen), np.array(rr, float)
    if len(rr) >= 3:
        rr = sg.medfilt(rr, 3)
    t = np.arange(len(e60)) / fs * 50
    return np.interp(t, cen, rr), float(g_st)


# ------------------------------------------------------------------ despiking
def despike(x5, bounds, k=9):
    """Replace narrow impulsive artefacts (<= 16 ms) whose residual from an 18 ms running median exceeds
    max(8 MAD, 0.6 x the 5 s block's 99th-percentile QRS excursion). R waves leave a residual of 0.25-0.4 R."""
    x = x5.copy(); nsp = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = x[a:b]
        if len(seg) < 3 * k:
            continue
        med = sg.medfilt(seg, k)
        r = seg - med
        mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-12
        base = sg.medfilt(seg, 101)                 # 0.2 s running median: baseline + T waves
        thr = np.empty(len(seg))
        B = 5 * FS
        for c in range(0, len(seg), B):
            thr[c:c + B] = max(8 * mad, 0.6 * np.percentile(np.abs(seg[c:c + B] - base[c:c + B]), 99))
        flag = np.abs(r) > thr
        if not flag.any():
            continue
        lab = np.cumsum(np.r_[flag[0], np.diff(flag.astype(int)) == 1]) * flag
        for L in np.unique(lab[lab > 0]):
            idx = np.flatnonzero(lab == L)
            if len(idx) <= 8:
                lo, hi = max(0, idx[0] - 4), min(len(seg), idx[-1] + 5)
                seg[lo:hi] = med[lo:hi]; nsp += 1
        x[a:b] = seg
    return x, nsp


def width_est(bb, c, fs=FS):
    """Gaussian-equivalent width (s) of the dominant deflection near c: 0.607 x amplitude / max slope."""
    a0, b0 = max(0, c - int(0.03 * fs)), min(len(bb), c + int(0.03 * fs) + 1)
    a1, b1 = max(0, c - int(0.15 * fs)), min(len(bb), c + int(0.15 * fs) + 1)
    if b0 - a0 < 4:
        return np.nan
    seg = bb[a0:b0]
    amp = np.max(np.abs(seg - np.median(bb[a1:b1])))
    slope = np.max(np.abs(np.diff(seg))) * fs + 1e-12
    return 0.607 * amp / slope


# ------------------------------------------------------------------ main detector
def _centroid(eraw, p, half, n_iter):
    c = float(p)
    for _ in range(n_iter):
        a, b = max(0, int(round(c)) - half), min(len(eraw), int(round(c)) + half + 1)
        w = eraw[a:b]
        w = np.maximum(w - np.min(w), 0)
        s = w.sum()
        if s <= 0:
            break
        c = float(np.sum(np.arange(a, b) * w) / s)
    return c


def _rr_score(rr, rre, s):
    """log p(RR | expected RR), normalised to 0 at RR = expected RR; s = log-normal SD of the sinus component."""
    lr = np.log(rr / rre)
    pn = np.exp(-0.5 * (lr / s) ** 2)                                   # sinus, peak 1
    norm = 1.0 / (s * np.sqrt(2 * np.pi))
    dens_prem = P['eps_prem'] / np.log(1.0 / P['prem_lo']) / norm
    dens_long = P['eps_long'] / np.log(P['long_hi']) / norm
    pp = np.where((rr >= P['prem_lo'] * rre) & (rr < rre), dens_prem, 0.0)
    pl = np.where((rr > rre) & (rr <= P['long_hi'] * rre), dens_long, 0.0)
    pg = P['eps_gap'] * np.exp(-0.5 * ((lr - np.log(2.0)) / s) ** 2) + P['eps_gap'] * np.exp(-0.5 * ((lr - np.log(3.0)) / s) ** 2)
    return np.log(pn + pp + pl + pg + 1e-12)


def _viterbi(t, em, rre_at, fs, joins, n_samples, sigma, max_gap, rel_min=0.0):
    """Best chain over candidates. t: candidate samples (sorted), em: emission, rre_at: expected RR (s) at each
    candidate. A chain may restart at any candidate with penalty P['restart'] (that opens an abstained gap)."""
    n = len(t)
    best = np.full(n, -np.inf); prev = np.full(n, -1, int)
    rr_min = int(P['rr_abs_min_s'] * fs)
    jt = np.asarray(joins, int)
    pref_val = np.full(n, -np.inf); pref_arg = np.full(n, -1, int)   # running best over candidates 0..j
    for j in range(n):
        # a chain may start freely within ~1.5 RR of the signal start; otherwise it continues from the best earlier
        # beat across an abstained gap, paying the restart penalty (beats before the gap are kept)
        best[j] = em[j] + (0.0 if t[j] < 1.5 * rre_at[j] * fs else P['restart'])
        prev[j] = -1
        k0 = np.searchsorted(t, t[j] - rr_min, side='right') - 1
        if k0 >= 0 and pref_val[k0] + P['restart'] + em[j] > best[j]:
            best[j] = pref_val[k0] + P['restart'] + em[j]; prev[j] = pref_arg[k0]
        lo = np.searchsorted(t, t[j] - max_gap * rre_at[j] * fs, side='left')
        if lo < j:
            i = np.arange(lo, j)
            d = t[j] - t[i]
            ok = d >= np.maximum(rr_min, rel_min * rre_at[j] * fs)
            if ok.any():
                i, d = i[ok], d[ok]
                rre = 0.5 * (rre_at[i] + rre_at[j]) * fs
                sc = best[i] + _rr_score(d.astype(float), rre, sigma) + em[j]
                k = int(np.argmax(sc))
                if sc[k] > best[j]:
                    best[j] = sc[k]; prev[j] = i[k]
        if j == 0 or best[j] > pref_val[j - 1]:
            pref_val[j], pref_arg[j] = best[j], j
        else:
            pref_val[j], pref_arg[j] = pref_val[j - 1], pref_arg[j - 1]
    # end: last beat should be within one RR of the last candidate, else restart penalty applies
    end_pen = np.where(n_samples - t < 1.5 * rre_at * fs, 0.0, P['restart'])
    j = int(np.argmax(best + end_pen))
    path = []
    while j >= 0:
        path.append(j); j = prev[j]
    return np.array(path[::-1], int)


def detect(x, fs, segments=None, return_debug=False):
    """x: ECG-like trace at fs. segments: optional list of sample indices (at fs) of generator window joins.
    Returns dict with beats (centroid, 500 Hz samples), r_signed, beat confidence, abstain mask, record metrics."""
    x = np.asarray(x, np.float64)
    nanm = ~np.isfinite(x)
    if nanm.any():
        idx = np.arange(len(x))
        x = x.copy(); x[nanm] = np.interp(idx[nanm], idx[~nanm], x[~nanm]) if (~nanm).any() else 0.0
    x5 = resample_to(x, fs, FS)
    n = len(x5)
    bad = np.zeros(n, bool)
    if nanm.any():
        bad = resample_to(nanm.astype(float), fs, FS) > 0.5
    joins = [] if segments is None else sorted(int(round(s * FS / fs)) for s in segments if 0 < s < len(x))
    bounds = [0] + joins + [n]
    # robust scale per segment (z-scored, polarity-flipped generator windows need no special handling: energy is
    # polarity-free and templates are built per segment)
    x5, n_spk = despike(x5, bounds)
    qrs = _filt_segments(x5, _sos('qrs'), bounds)
    hf = _filt_segments(x5, _sos('hf'), bounds)
    bb = _filt_segments(x5, _sos('bb'), bounds)
    for a, b in zip(bounds[:-1], bounds[1:]):        # equalise segment scales (z-scored windows differ in gain)
        s = np.median(np.abs(qrs[a:b])) + 1e-12
        qrs[a:b] /= s; hf[a:b] /= s; bb[a:b] /= s
    eraw = qrs ** 2                                   # fiducial energy (5-25 Hz), polarity-free
    der = np.zeros_like(qrs)                          # detection energy: derivative of the band-passed trace
    der[2:-2] = (-qrs[4:] - 2 * qrs[3:-1] + 2 * qrs[1:-3] + qrs[:-4]) / 8.0   # Pan-Tompkins 5-point derivative
    for a, b in zip(bounds[:-1], bounds[1:]):
        der[a:min(b, a + 3)] = 0; der[max(a, b - 3):b] = 0
    edet = der ** 2
    e60 = _movavg(edet, P['e_cand_ms'] / 1000 * FS)
    ehf = _movavg(hf ** 2, 0.080 * FS)
    eqrs80 = _movavg(eraw, 0.080 * FS)
    eder80 = _movavg(edet, 0.080 * FS)
    ebb = _movavg(bb ** 2, 0.200 * FS)
    rre, acf_strength = rr_prior(e60)
    # candidates
    pk, pr = sg.find_peaks(e60, distance=int(P['cand_min_ms'] / 1000 * FS), prominence=0)
    if len(pk) < 3:
        return _empty(n, acf_strength)
    h = e60[pk]
    # local beat level: median of the top-k candidate heights within +/- ref_win_s, k = expected beats in window
    W = int(P['ref_win_s'] * FS)
    ref = np.empty(len(pk))
    for q, p in enumerate(pk):
        a, b = np.searchsorted(pk, p - W), np.searchsorted(pk, p + W)
        k = max(2, int(round(2 * P['ref_win_s'] / max(rre[p], 0.2))))
        hh = np.sort(h[a:b])[::-1][:k]
        ref[q] = np.median(hh)
    keep = pr['prominences'] >= P['cand_prom'] * ref
    pk, h, ref = pk[keep], h[keep], ref[keep]
    amp_d = h / (ref + 1e-12)
    hq = eqrs80[pk]
    refq = np.empty(len(pk))
    for q, p in enumerate(pk):
        a, b = np.searchsorted(pk, p - W), np.searchsorted(pk, p + W)
        k = max(2, int(round(2 * P['ref_win_s'] / max(rre[p], 0.2))))
        refq[q] = np.median(np.sort(hq[a:b])[::-1][:k])
    amp = np.maximum(amp_d, hq / (refq + 1e-12))     # wide premature beats carry band energy, not slope energy
    hfr = ehf[pk] / (eqrs80[pk] + 1e-12)
    share = eqrs80[pk] / (ebb[pk] + eqrs80[pk] + 1e-12)
    # centroid positions for templates
    half = int(P['cent_ms'] / 1000 * FS)
    cen = np.array([_centroid(eraw, p, half, P['cent_iter']) for p in pk])
    wid = np.array([width_est(bb, int(round(c))) for c in cen])
    wid = np.where(np.isfinite(wid), wid, 0.012)
    # templates per segment from confident candidates
    tw = int(P['tmpl_ms'] / 1000 * FS)
    corr = np.zeros(len(pk))
    seg_id = np.searchsorted(np.asarray(bounds[1:-1]), pk, side='right') if joins else np.zeros(len(pk), int)
    hf_ref = np.median(hfr[amp > 0.5]) if np.any(amp > 0.5) else np.median(hfr)
    sh_ref = np.median(share[amp > 0.5]) if np.any(amp > 0.5) else np.median(share)
    pol_votes = []
    for s in np.unique(seg_id):
        m = np.flatnonzero(seg_id == s)
        conf = m[(amp[m] > 0.5) & (hfr[m] < P['hf_tol'] * hf_ref) & (share[m] > P['share_tol'] * sh_ref) & (wid[m] < 0.02)]
        if len(conf) < 3:
            conf = m[np.argsort(-amp[m])[:max(3, len(m) // 3)]]
        segs = [bb[int(c) - tw:int(c) + tw + 1] for c in np.round(cen[conf]).astype(int) if c - tw >= 0 and c + tw + 1 <= n]
        if len(segs) < 2:
            continue
        T = np.median(np.array(segs), 0)
        Tn = (T - T.mean()) / (T.std() + 1e-12)
        pol_votes.append(np.sign(T[np.argmax(np.abs(T))]) * (len(m)))
        for q in m:
            c = int(round(cen[q]))
            if c - tw >= 0 and c + tw + 1 <= n:
                y = bb[c - tw:c + tw + 1]
                # allow +/-20 ms lag for the correlation
                best_r = -1.0
                for lag in range(-10, 11, 2):
                    a, b = c - tw + lag, c + tw + 1 + lag
                    if a < 0 or b > n:
                        continue
                    y = bb[a:b]
                    r = float(np.dot((y - y.mean()) / (y.std() + 1e-12), Tn) / len(Tn))
                    best_r = max(best_r, r)
                corr[q] = best_r
    em = (P['w_amp'] * np.clip(np.log2(amp + 1e-12), *P['amp_clip'])
          + P['w_corr'] * (corr - P['corr_mid'])
          - np.minimum(6.0, P['w_hf'] * np.maximum(0.0, np.log2(hfr / (P['hf_tol'] * hf_ref) + 1e-12)))
          - P['w_wid'] * np.clip(np.log2(wid / P['wid_ref']), 0.0, 3.0)
          - P['w_share'] * np.maximum(0.0, np.log2(P['share_tol'] * sh_ref / (share + 1e-12)))
          + P['e_bias'])
    near_join = np.zeros(len(pk), bool)
    for j in joins:
        near_join |= np.abs(pk - j) <= int(P['join_ms'] / 1000 * FS)
    em = em - 1.0 * near_join
    # pass 1
    tc = np.round(cen).astype(int)
    order = np.argsort(tc)
    tc, em_o = tc[order], em[order]
    path = _viterbi(tc, em_o, rre[np.clip(tc, 0, n - 1)], FS, joins, n, P['sigma1'], P['max_gap1'])
    # pass 2: expected RR from the decoded beats
    beats = tc[path]
    if len(beats) >= 5:
        rr = np.diff(beats) / FS
        medall = np.median(rr)
        okrr = rr < 1.7 * medall
        if okrr.sum() >= 3:
            mid = 0.5 * (beats[1:] + beats[:-1])
            k = min(9, int(okrr.sum()) | 1)
            rr_run = sg.medfilt(rr[okrr], k) if okrr.sum() >= k else np.full(okrr.sum(), np.median(rr[okrr]))
            rre2 = np.interp(np.arange(n), mid[okrr], rr_run)
            path = _viterbi(tc, em_o, rre2[np.clip(tc, 0, n - 1)], FS, joins, n, P['sigma'], P['max_gap_ratio'], P['rr_rel_min'])
            rre = rre2
    sel = order[path]
    beats = np.array([_centroid(eraw, pk[q], int(min(P['cent_ms'] / 1000, 0.15 * rre[min(pk[q], n - 1)]) * FS), P['cent_iter'])
                      for q in sel])
    beats = np.round(beats).astype(int)
    # fiducials
    pol = np.sign(np.sum(pol_votes)) if pol_votes else 1.0
    r_signed = []
    w40 = int(0.040 * FS)
    for c in beats:
        a, b = max(0, c - w40), min(n, c + w40 + 1)
        r_signed.append(a + int(np.argmax(pol * bb[a:b])))
    r_signed = np.array(r_signed, int)
    # confidence per beat
    rre_b = rre[np.clip(beats, 0, n - 1)] * FS
    rr_prev = np.r_[np.nan, np.diff(beats)]
    rr_next = np.r_[np.diff(beats), np.nan]
    def _reg(rr):
        v = np.exp(-0.5 * (np.log(np.where(np.isfinite(rr), rr, rre_b) / rre_b) / 0.2) ** 2)
        return v
    reg = np.sqrt(_reg(rr_prev) * _reg(rr_next))
    em_sel = em[sel]
    conf_b = (1 / (1 + np.exp(-em_sel))) * (0.5 + 0.5 * reg) * np.where(near_join[sel], 0.7, 1.0)
    # abstention mask: gaps longer than gap_flag x expected RR, NaN stretches
    abst = bad.copy()
    gaps = []
    for a, b, r in zip(beats[:-1], beats[1:], rre_b[:-1]):
        if b - a > P['gap_flag'] * r:
            abst[a:b] = True; gaps.append((int(a), int(b)))
    if len(beats):
        if beats[0] > P['gap_flag'] * rre_b[0]:
            abst[:beats[0]] = True
        if n - beats[-1] > P['gap_flag'] * rre_b[-1]:
            abst[beats[-1]:] = True
    cov = 1.0 - abst.mean()
    med_corr = float(np.median(corr[sel])) if len(sel) else 0.0
    lowconf = float(np.mean(conf_b < 0.5)) if len(sel) else 1.0
    rec = dict(n_beats=int(len(beats)), coverage=float(cov), median_corr=med_corr, frac_lowconf=lowconf,
               acf_strength=acf_strength, hr_median=float(60 * FS / np.median(np.diff(beats))) if len(beats) > 2 else np.nan,
               n_gaps=len(gaps), polarity=float(pol), n_candidates=int(len(pk)), n_despiked=int(n_spk))
    rec['abstain'] = bool(cov < P['rec_min_cov'] or med_corr < P['rec_min_corr'] or lowconf > P['rec_max_lowconf'])
    out = dict(beats=beats, r_signed=r_signed, conf=conf_b, abstain_mask=abst, rec=rec, fs=FS)
    if return_debug:
        out.update(dbg=dict(pk=pk, cen=cen, amp=amp, wid=wid, hfr=hfr, share=share, corr=corr, em=em, rre=rre, e60=e60, eraw=eraw,
                            bb=bb, qrs=qrs, sel=sel, hf_ref=hf_ref, sh_ref=sh_ref))
    return out


def _empty(n, acf_strength):
    return dict(beats=np.array([], int), r_signed=np.array([], int), conf=np.array([]), abstain_mask=np.ones(n, bool),
                rec=dict(n_beats=0, coverage=0.0, median_corr=0.0, frac_lowconf=1.0, acf_strength=acf_strength,
                         hr_median=np.nan, n_gaps=0, polarity=1.0, n_candidates=0, abstain=True), fs=FS)


# ------------------------------------------------------------------ scoring helpers
def match(ref, det, tol):
    """One-to-one matching (greedy by distance) of sorted event lists within tol samples -> (tp, fp, fn, offsets det-ref)."""
    ref, det = np.asarray(ref), np.asarray(det)
    if len(ref) == 0 or len(det) == 0:
        return 0, len(det), len(ref), np.array([])
    pairs = []
    for i, r in enumerate(ref):
        lo, hi = np.searchsorted(det, r - tol), np.searchsorted(det, r + tol, side='right')
        for j in range(lo, hi):
            pairs.append((abs(det[j] - r), i, j))
    pairs.sort()
    ur, ud, offs = set(), set(), []
    for d, i, j in pairs:
        if i in ur or j in ud:
            continue
        ur.add(i); ud.add(j); offs.append(det[j] - ref[i])
    tp = len(ur)
    return tp, len(det) - tp, len(ref) - tp, np.array(offs)


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)
