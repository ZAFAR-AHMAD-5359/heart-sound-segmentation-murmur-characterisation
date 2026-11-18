"""Simulated ECG with exact beat truth, for the R-peak acceptance tests (copied from Panel H's stress.py, 15 Oct 2026).

Sum of Gaussians per beat (P, Q, R, S, T) with rate-dependent PR and QT, respiratory HRV, baseline wander, mains
and white noise. Conditions: clean, spikes, bursts, tallT, deepS_inv, missing (10% of beats plus a 3 s pause),
premature (8% PVCs with compensatory pauses), joins (8 s generator windows, each z-scored with its own gain, offset,
and polarity), combined (tall T + spikes + bursts), biphasic (R and S of similar size, alternating).
"""
import numpy as np
import scipy.signal as sg

FS = 500
HRS = [40, 60, 80, 100, 120, 150, 180, 200, 220]
CONDS = ['clean', 'spikes', 'bursts', 'tallT', 'deepS_inv', 'missing', 'premature', 'joins', 'combined']
DUR = 60.0


def g(t, mu, sd, a):
    return a * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def simulate(hr, cond, seed):
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    rr0 = 60.0 / hr
    beats, kinds = [], []
    tb = 0.3 + rng.uniform(0, rr0)
    while tb < DUR - 0.3:
        beats.append(tb); kinds.append('N')
        rr = rr0 * (1 + 0.03 * np.sin(2 * np.pi * 0.25 * tb) + 0.015 * rng.standard_normal())
        if cond in ('premature',) and rng.random() < 0.08 and kinds[-1] == 'N':
            tp = tb + max(0.62 * rr0, 0.22)
            if tp < DUR - 0.3:
                beats.append(tp); kinds.append('V')
            tb = tb + 2.0 * rr0 * (1 + 0.01 * rng.standard_normal())     # compensatory pause
            continue
        tb += rr
    beats, kinds = np.array(beats), np.array(kinds)
    present = np.ones(len(beats), bool)
    if cond == 'missing':
        present &= rng.random(len(beats)) > 0.10
        d0 = rng.uniform(10, DUR - 14)
        present &= ~((beats > d0) & (beats < d0 + 3.0))
    x = np.zeros(n)
    child = hr >= 100
    r_sd = 0.008 if child else 0.010
    tallT = cond in ('tallT', 'combined')
    deepS = cond == 'deepS_inv'
    for b, k, pr in zip(beats, kinds, present):
        if not pr:
            continue
        rr = rr0
        qt = 0.40 * np.sqrt(rr)
        tpk = qt - 0.07 * np.sqrt(rr)
        prs = min(0.16, 0.10 + 0.06 * rr)
        lo, hi = max(0, int((b - 0.35) * FS)), min(n, int((b + 0.6) * FS))
        tt = t[lo:hi]
        if k == 'N':
            ra, sa = (0.5, -1.0) if deepS else (1.0, -0.25)
            if cond == 'biphasic':          # R and |S| comparable, the larger one alternating with respiration
                m = 0.35 * np.sin(2 * np.pi * 0.25 * b)
                ra, sa = 0.8 * (1 + m), -0.8 * (1 - m)
            ta = rng.uniform(1.6, 2.2) if tallT else 0.30
            y = (g(tt, b - prs, 0.022, 0.12) + g(tt, b - 0.022, 0.009, -0.10) + g(tt, b, r_sd, ra)
                 + g(tt, b + 0.024, 0.010, sa) + g(tt, b + tpk, 0.045 * np.sqrt(rr), ta))
        else:   # PVC: wide, larger, discordant T
            y = g(tt, b, 0.028, 1.4) + g(tt, b + 0.05, 0.03, -0.4) + g(tt, b + tpk + 0.03, 0.06, -0.5)
        x[lo:hi] += y
    x += 0.15 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6)) + 0.08 * np.sin(2 * np.pi * 0.06 * t)
    x += 0.02 * rng.standard_normal(n) + 0.02 * np.sin(2 * np.pi * 50 * t)
    if cond in ('spikes', 'combined'):
        for ts in rng.uniform(0.5, DUR - 0.5, int(DUR / 3)):
            sd = rng.uniform(0.001, 0.003)
            x += g(t, ts, sd, rng.choice([-1, 1]) * rng.uniform(1.5, 4.0))
    if cond in ('bursts', 'combined'):
        sos = sg.butter(3, [10, 60], btype='bandpass', fs=FS, output='sos')
        for ts in rng.uniform(0.5, DUR - 1.0, int(DUR / 5)):
            L = int(rng.uniform(0.15, 0.4) * FS); a = int(ts * FS)
            w = sg.sosfilt(sos, rng.standard_normal(L + 200))[200:]
            w = w / (w.std() + 1e-12) * rng.uniform(0.3, 0.6) * np.hanning(L)
            x[a:a + L] += w[:len(x[a:a + L])]
    if deepS:
        x = -x
    joins = []
    if cond == 'joins':
        W = 8 * FS
        for a in range(0, n, W):
            seg = x[a:a + W]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8) * rng.uniform(0.6, 1.6) + rng.uniform(-1, 1)
            if rng.random() < 0.5:
                seg = -seg
            if np.abs(seg.min()) > np.abs(seg.max()):    # per-window ensure_positive_polarity, as the generator stage does
                seg = -seg
            x[a:a + W] = seg
            if a > 0:
                joins.append(a)
    truth = np.round(beats[present] * FS).astype(int)
    kinds_p = kinds[present]
    return x, truth, kinds_p, joins


# ------------------------------------------------------------------ event traces (decoder tests; not from Panel H)
def simulate_events(hr, seed, fs=250, dur=60.0, sd_ms=12.0, p_miss=0.05, s2_amp=(0.1, 0.3), spur_per_min=6.0,
                    noise=0.05, premature=0.0, return_s2_head=False, return_shown=False):
    """Event-probability trace like a timing head's output: Gaussian bumps at true beats (amplitude 0.6 to 1.0),
    S2-like bumps one systole later (default 0.1 to 0.3: what remains after subtracting the S2 head), random spurious
    bumps, missed beats, premature beats, and a noise floor. Returns (trace at fs, true beat times in s), plus an S2-head
    trace (bumps 0.7 to 1.0 at every S2) when return_s2_head is set, plus the mask of beats that carry a bump when
    return_shown is set."""
    rng = np.random.default_rng(seed)
    rr0 = 60.0 / hr
    t, beats = 0.3 + rng.uniform(0, rr0), []
    while t < dur - 0.3:
        beats.append(t)
        if premature and rng.random() < premature:
            beats.append(t + 0.62 * rr0); t += 2.0 * rr0
            continue
        t += rr0 * (1 + 0.03 * np.sin(2 * np.pi * 0.25 * t) + 0.015 * rng.standard_normal())
    beats = np.array(beats)
    n = int(dur * fs)
    tt = np.arange(n) / fs
    x = np.abs(noise * rng.standard_normal(n))
    s2h = np.abs(noise * rng.standard_normal(n))
    sd = sd_ms / 1000
    shown = rng.random(len(beats)) > p_miss
    for b, s in zip(beats, shown):
        lo, hi = int((b - 5 * sd) * fs), int((b + 5 * sd) * fs) + 1
        lo, hi = max(0, lo), min(n, hi)
        if s:
            x[lo:hi] += rng.uniform(0.6, 1.0) * np.exp(-0.5 * ((tt[lo:hi] - b) / sd) ** 2)
        s2 = b + 0.30 * np.sqrt(rr0) + 0.02 * rng.standard_normal()          # S2 about one systole later
        lo, hi = max(0, int((s2 - 5 * sd) * fs)), min(n, int((s2 + 5 * sd) * fs) + 1)
        x[lo:hi] += rng.uniform(*s2_amp) * np.exp(-0.5 * ((tt[lo:hi] - s2) / sd) ** 2)
        s2h[lo:hi] += rng.uniform(0.7, 1.0) * np.exp(-0.5 * ((tt[lo:hi] - s2) / sd) ** 2)
    for ts in rng.uniform(0.2, dur - 0.2, int(spur_per_min * dur / 60)):
        lo, hi = max(0, int((ts - 5 * sd) * fs)), min(n, int((ts + 5 * sd) * fs) + 1)
        x[lo:hi] += rng.uniform(0.3, 0.9) * np.exp(-0.5 * ((tt[lo:hi] - ts) / sd) ** 2)
    out = (np.clip(x, 0, None), beats)
    if return_s2_head:
        out += (np.clip(s2h, 0, None),)
    if return_shown:
        out += (shown,)
    return out
