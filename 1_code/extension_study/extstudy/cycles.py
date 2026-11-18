"""Cycle export for the frozen T and W arms (Run 2): R1 cycle format at 4 kHz, boundaries at the 50% acoustic onset.

W is a synthetic ECG, so its R-peaks enter arm B's construction unchanged (`b_cycles`, the R1 notebook's arm_b_cycles
with the audio passed in): each R-peak searches the 20 to 150 Hz acoustic envelope for S1 and anchors the cycle at
the 50% onset; S2 is the strongest envelope maximum in [0.25, 0.55] RR. T predicts the heart sounds themselves, so its
cycles come from its own heads (`t_cycles`, 08 design section 2.7): the cycle anchor is the 50% acoustic onset of the
envelope maximum within 40 ms of each decoded S1 event; S2 is searched in the same heart-rate-dependent window as arm
B, as the maximum of T's S2 probability times a Gaussian (SD 40 ms) centred on the expected S1-to-S2 interval of the
paired law at the local rate, then moved to the envelope maximum within 40 ms and to its 50% onset. Phases follow arm
B (S1 for min(0.15 RR, 120 ms), systole to the S2 onset, S2 for min(0.10 RR, 100 ms), diastole to the next anchor);
malformed cycles are dropped and counted. FUSION, check_cycles, acoustic_envelope, onset_50 and b_cycles are copied
from the R1 notebook; tests hold them equal to the source. Constants are fixed in advance; nothing reads a label.
"""
import numpy as np
import scipy.signal as sig

TARGET_SR = 4000
PHASE_KEYS = ('s1', 'systole', 's2', 'diastole')

FUSION = dict(band_hz=(20, 150), env_smooth_ms=20, peak_dist_ms=250, peak_prom=0.10,
              search_lo_ms=-40, search_hi_ms=120, match_ms=150, gap_factor=1.6, support_min=0.5,
              s2_lo=0.25, s2_hi=0.55, s1_frac=0.15, s1_cap_ms=120, s2_frac=0.10, s2_cap_ms=100,
              s1_onset_back_ms=150, s2_onset_back_ms=100, rescue_height=0.50)
AC_SOS = sig.butter(4, list(FUSION['band_hz']), btype='bandpass', fs=TARGET_SR, output='sos')


def check_cycles(cyc, L, tag, strict=True):
    """Every boundary inside [0, L], every phase ordered (start <= end), phase starts non-decreasing.
    strict=True additionally requires the eight boundaries to be monotone; that holds for every arm
    built here. The dataset annotation (arm C) occasionally overlaps adjacent phases by a sample or
    two, so arm C and its jittered copies are checked with strict=False and phase_overlaps() counts them."""
    for c in cyc:
        seq = [v for k in PHASE_KEYS for v in c[k]]
        assert all(0 <= v <= L for v in seq), '%s: boundary outside the signal %s (L=%d)' % (tag, seq, L)
        assert all(c[k][0] <= c[k][1] for k in PHASE_KEYS), '%s: phase with end before start %s' % (tag, seq)
        starts = [c[k][0] for k in PHASE_KEYS]
        assert all(a <= b for a, b in zip(starts, starts[1:])), '%s: phase starts not ordered %s' % (tag, seq)
        if strict:
            assert all(a <= b for a, b in zip(seq, seq[1:])), '%s: boundaries not monotone %s' % (tag, seq)
    return cyc


def acoustic_envelope(x4, sr=TARGET_SR):
    f = sig.sosfiltfilt(AC_SOS, x4.astype(np.float64))
    w = int(FUSION['env_smooth_ms'] / 1000 * sr)
    env = np.convolve(np.abs(f), np.ones(w) / w, mode='same')
    return env / (np.percentile(env, 95) + 1e-12)


def onset_50(env, peak, back):
    """Last sample before `peak` where env crosses 50 % of the peak height, within `back` samples.
    Returns (onset, found)."""
    h = 0.5 * env[peak]
    a = max(0, int(peak) - int(back))
    below = np.flatnonzero(env[a:peak] < h)
    if len(below) == 0:
        return int(a), False
    return int(a + below[-1] + 1), True


def b_cycles(x4, pk4):
    """arm_b_cycles of the R1 notebook with the audio passed in (verbatim body; tests compare it with the source)."""
    L4 = len(x4)
    sr = TARGET_SR
    env = acoustic_envelope(x4)
    ac_pk, _ = sig.find_peaks(env, distance=int(FUSION['peak_dist_ms'] / 1000 * sr), prominence=FUSION['peak_prom'])
    ac_pk = np.asarray(ac_pk, int)
    raw_pk, _ = sig.find_peaks(env, prominence=FUSION['peak_prom'])
    strong = raw_pk[env[raw_pk] >= FUSION['rescue_height']]     # local maxima at >= 50 % of the p95 level, no spacing rule (rescue list)
    back1 = int(FUSION['s1_onset_back_ms'] / 1000 * sr)
    ac_on = [onset_50(env, p, back1) for p in ac_pk]          # (onset sample, crossing found) per thinned acoustic peak
    r4 = np.asarray(pk4, int)
    d = dict(n_syn=int(len(r4)), n_acoustic=int(len(ac_pk)), supported_frac=np.nan, mean_abs_syn_ac_offset_ms=np.nan,
             mean_syn_ac_offset_ms=np.nan, n_removed=0, n_inserted=0, n_rescued=0, fusion_fallback=False, n_onset_fallback=0,
             n_malformed=0, frac_s2_acoustic=np.nan)
    if len(r4) < 2:
        return [], d
    lo, hi = int(FUSION['search_lo_ms'] / 1000 * sr), int(FUSION['search_hi_ms'] / 1000 * sr)
    used = np.zeros(len(ac_pk), bool)
    supported = np.zeros(len(r4), bool)
    anchors, offs, onset_fb, rescued = [], [], 0, 0
    for i, r in enumerate(r4):
        cand = np.flatnonzero((ac_pk >= r + lo) & (ac_pk <= r + hi) & ~used)
        if len(cand):
            j = cand[int(np.argmax(env[ac_pk[cand]]))]
            used[j] = True
            on, found = ac_on[j]
        else:
            resc = strong[(strong >= r + lo) & (strong <= r + hi)]
            if len(resc) == 0:
                anchors.append(int(r))
                continue
            rescued += 1
            on, found = onset_50(env, int(resc[int(np.argmax(env[resc]))]), back1)
        supported[i] = True
        if found:
            anchors.append(int(on))
            offs.append((int(on) - int(r)) / sr * 1000.0)
        else:
            anchors.append(int(r))
            onset_fb += 1
    frac = float(supported.mean())
    d.update(supported_frac=frac, n_onset_fallback=int(onset_fb), n_rescued=int(rescued),
             mean_abs_syn_ac_offset_ms=float(np.mean(np.abs(offs))) if offs else np.nan,
             mean_syn_ac_offset_ms=float(np.mean(offs)) if offs else np.nan)
    if frac >= FUSION['support_min']:
        anc = np.array([a for a, s in zip(anchors, supported) if s], int)
        d['n_removed'] = int((~supported).sum())
        rr_sup = np.diff(anc)
        med = float(np.median(rr_sup)) if len(rr_sup) else np.nan
        ins = 0
        if np.isfinite(med) and med > 0:
            mtol = int(FUSION['match_ms'] / 1000 * sr)
            for j in range(len(ac_pk)):
                if used[j] or np.min(np.abs(r4 - ac_pk[j])) <= mtol:
                    continue
                o = int(ac_on[j][0]) if ac_on[j][1] else int(ac_pk[j])
                k = int(np.searchsorted(anc, o))
                if k == 0 or k == len(anc):
                    continue
                gap = anc[k] - anc[k - 1]
                if gap >= FUSION['gap_factor'] * med and (o - anc[k - 1]) >= 0.5 * med and (anc[k] - o) >= 0.5 * med:
                    anc = np.insert(anc, k, o)
                    ins += 1
        d['n_inserted'] = int(ins)
        anc = list(anc)
    else:
        anc = list(anchors)                  # fallback: the supported acoustic onsets stay; only removals and insertions are skipped
        d['fusion_fallback'] = True
    anc = sorted(set(int(a) for a in anc if 0 <= a <= L4))
    cyc, malformed, s2_ac = [], 0, 0
    s1_cap, s2_cap = int(FUSION['s1_cap_ms'] / 1000 * sr), int(FUSION['s2_cap_ms'] / 1000 * sr)
    back2 = int(FUSION['s2_onset_back_ms'] / 1000 * sr)
    for a, b in zip(anc, anc[1:]):
        rr = b - a
        if rr <= 0:
            malformed += 1
            continue
        s1e = a + min(int(FUSION['s1_frac'] * rr), s1_cap)
        w0, w1 = a + int(FUSION['s2_lo'] * rr), min(a + int(FUSION['s2_hi'] * rr), L4)
        s2on, found = a + int(0.50 * rr), False
        if w1 > w0 + 2:
            loc, _ = sig.find_peaks(env[w0:w1], prominence=FUSION['peak_prom'])
            if len(loc):
                p = w0 + int(loc[int(np.argmax(env[w0:w1][loc]))])
                cand_on, ok = onset_50(env, p, back2)
                if ok:
                    s2on, found = cand_on, True
        s2e = s2on + min(int(FUSION['s2_frac'] * rr), s2_cap)
        if not (a < s1e < s2on < s2e < b):
            malformed += 1
            continue
        s2_ac += int(found)
        cyc.append({'cycle': len(cyc), 's1': (a, s1e), 'systole': (s1e, s2on), 's2': (s2on, s2e), 'diastole': (s2e, b)})
    d.update(n_malformed=int(malformed), frac_s2_acoustic=(s2_ac / len(cyc)) if cyc else np.nan)
    return check_cycles(cyc, L4, 'b_cycles'), d


# ------------------------------------------------------------------ arm T (new; constants fixed before any target-cohort use)
# expected S1-peak-to-S2-peak interval (ms) at heart rate HR: mean over the six paired datasets of law_S2 - law_S1 in
# derived/labels_v3/laws.json (SHA-256 7ba1ce33...): 313 ms at 60 bpm, 251 ms at 100 bpm, 174 ms at 150 bpm
S2_LAW_MS = (406.2767, -1.549957)
T_EXPORT = dict(s1_refine_ms=40, s2_sigma_ms=40, s2_refine_ms=40, s2_law_ms=S2_LAW_MS, laws_sha256_prefix='7ba1ce33a789845e')


def t_cycles(x4, s1_s, p_s2, frame_t0=0.016, frame_s=0.004, s2_law=S2_LAW_MS):
    """Cycles from T's decoded S1 events (s1_s, seconds) and its seed-averaged S2 probability trace (p_s2, at 250 Hz,
    frame j at frame_t0 + j * frame_s). Returns (cycles, diagnostics)."""
    L4 = len(x4)
    sr = TARGET_SR
    env = acoustic_envelope(x4)
    back1 = int(FUSION['s1_onset_back_ms'] / 1000 * sr)
    back2 = int(FUSION['s2_onset_back_ms'] / 1000 * sr)
    r1 = int(T_EXPORT['s1_refine_ms'] / 1000 * sr)
    r2 = int(T_EXPORT['s2_refine_ms'] / 1000 * sr)
    sd = T_EXPORT['s2_sigma_ms'] / 1000 * sr
    s1_s = np.asarray(s1_s, float)
    d = dict(n_beats=int(len(s1_s)), n_anchors=0, n_onset_fallback=0, n_malformed=0, frac_s2_acoustic=np.nan,
             s2_head_at_choice_median=np.nan)
    anchors = {}
    for t in s1_s:
        p = int(round(t * sr))
        if not 0 <= p < L4:
            continue
        a, b = max(0, p - r1), min(L4, p + r1 + 1)
        q = a + int(np.argmax(env[a:b]))                      # S1 peak on the acoustic envelope
        on, found = onset_50(env, q, back1)
        if not found:
            d['n_onset_fallback'] += 1
        anchors[int(on) if found else int(q)] = int(q)          # anchor (50% onset) -> its S1 peak; duplicates collapse
    anc = sorted(a for a in anchors if 0 <= a <= L4)
    d['n_anchors'] = len(anc)
    p_s2 = np.asarray(p_s2, float)
    tf = frame_t0 + frame_s * np.arange(len(p_s2))
    s1_cap, s2_cap = int(FUSION['s1_cap_ms'] / 1000 * sr), int(FUSION['s2_cap_ms'] / 1000 * sr)
    cyc, malformed, s2_ac, head = [], 0, 0, []
    for a, b in zip(anc, anc[1:]):
        rr = b - a
        if rr <= 0:
            malformed += 1
            continue
        s1e = a + min(int(FUSION['s1_frac'] * rr), s1_cap)
        w0, w1 = a + int(FUSION['s2_lo'] * rr), min(a + int(FUSION['s2_hi'] * rr), L4)
        s2on, found = a + int(0.50 * rr), False
        if w1 > w0 + 2 and len(p_s2):
            hr = 60.0 * sr / rr
            mu = anchors[a] + (s2_law[0] + s2_law[1] * hr) / 1000 * sr
            grid = np.arange(w0, w1)
            ps2 = np.interp(grid / sr, tf, p_s2)
            t2 = int(grid[int(np.argmax(ps2 * np.exp(-0.5 * ((grid - mu) / sd) ** 2)))])
            head.append(float(np.interp(t2 / sr, tf, p_s2)))
            c0, c1 = max(w0, t2 - r2), min(w1, t2 + r2 + 1)
            q2 = c0 + int(np.argmax(env[c0:c1]))
            on2, ok = onset_50(env, q2, back2)
            if ok:
                s2on, found = int(on2), True
        s2e = s2on + min(int(FUSION['s2_frac'] * rr), s2_cap)
        if not (a < s1e < s2on < s2e < b):
            malformed += 1
            continue
        s2_ac += int(found)
        cyc.append({'cycle': len(cyc), 's1': (a, s1e), 'systole': (s1e, s2on), 's2': (s2on, s2e), 'diastole': (s2e, b)})
    d.update(n_malformed=int(malformed), frac_s2_acoustic=(s2_ac / len(cyc)) if cyc else np.nan,
             s2_head_at_choice_median=float(np.median(head)) if head else np.nan)
    return check_cycles(cyc, L4, 't_cycles'), d
