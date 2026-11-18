"""Controlled beat-level corruption of annotated cycles (Experiment 0, the beat-error budget; next-study concept,
section 3). Applied to arm C's cycles in the R1 format (4 kHz, phases s1, systole, s2, diastole).

Missed beats (`delete_starts`): round(rate x n) cycle starts are missed at random. A missed start merges its cycle into
the previous one when the two abut (the previous cycle's diastole runs to the end of the merged cycle, so a whole beat
lies inside "diastole"); a missed start with no abutting predecessor removes its cycle.
Spurious beats (`insert_starts`): round(rate x n) distinct cycles each receive one false start, uniform between 10% and
90% of the cycle's length; that cycle becomes two cycles with fixed-fraction phases (the caller passes arm D's
segment_phases_samples), so its annotated phases are lost.
Timing error: the R1 notebook's jitter_cycles (verbatim there; every cycle shifted rigidly by a Gaussian offset).
Draws are seeded per recording (condition seed, CRC-32 of the name), so they do not depend on processing order.
`boundary_stats` scores a cycle list against the reference starts (Hungarian matching within 150 ms).
"""
import zlib
import numpy as np
from scipy.optimize import linear_sum_assignment

ABUT = 2                   # samples: two annotated cycles abut when the next S1 onset is within 2 samples of the diastole end
INSERT_SPAN = (0.10, 0.90)


def rng_for(seed, name):
    return np.random.default_rng([int(seed), zlib.crc32(str(name).encode())])


def _renumber(cyc):
    return [dict(c, cycle=i) for i, c in enumerate(cyc)]


def delete_starts(cycles, rate, rng):
    n = len(cycles)
    k = int(round(rate * n))
    if n == 0 or k == 0:
        return _renumber(list(cycles))
    drop = set(rng.choice(n, size=min(k, n), replace=False).tolist())
    out = []
    for i, c in enumerate(cycles):
        if i not in drop:
            out.append(dict(c))
        elif out and abs(int(c['s1'][0]) - int(out[-1]['diastole'][1])) <= ABUT:
            out[-1] = dict(out[-1], diastole=(int(out[-1]['diastole'][0]), int(c['diastole'][1])))
    return _renumber(out)


def insert_starts(cycles, rate, rng, phases_fn, n_samples, span=INSERT_SPAN):
    n = len(cycles)
    k = int(round(rate * n))
    if n == 0 or k == 0:
        return _renumber(list(cycles))
    pick = set(rng.choice(n, size=min(k, n), replace=False).tolist())
    out = []
    for i, c in enumerate(cycles):
        if i not in pick:
            out.append(dict(c))
            continue
        a, b = int(c['s1'][0]), int(c['diastole'][1])
        t = int(round(a + rng.uniform(*span) * (b - a)))
        segs = phases_fn([a, t, b], n_samples)
        out.extend(dict(s) for s in segs)
    return _renumber(out)


def starts(cycles):
    return np.array(sorted(set(int(c['s1'][0]) for c in cycles)), int)


def boundary_stats(cycles, ref_cycles, sr=4000, tol_ms=150):
    """Cycle starts against the reference starts: matched pairs (Hungarian within tol), boundary F1, insertion and
    deletion rates, signed-offset SD of the matches (ms)."""
    a, b = starts(cycles), starts(ref_cycles)
    m, d = 0, np.array([])
    if len(a) and len(b):
        D = np.abs(a[:, None].astype(float) - b[None, :].astype(float))
        tol = tol_ms / 1000 * sr
        ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e9))
        ok = D[ri, ci] <= tol
        m, d = int(ok.sum()), (a[ri[ok]] - b[ci[ok]]) / sr * 1000.0
    return dict(n=int(len(a)), n_ref=int(len(b)), matched=m, f1=(2 * m / (len(a) + len(b))) if (len(a) + len(b)) else np.nan,
                insertion=((len(a) - m) / len(a)) if len(a) else np.nan, deletion=((len(b) - m) / len(b)) if len(b) else np.nan,
                offset_sd_ms=float(np.std(d)) if len(d) > 1 else np.nan)


# ------------------------------------------------------------------ Experiment 0b (concentrated against spread failure)
S2_MARGIN_MS = 10          # a displaced S2 keeps at least this much systole before it and diastole after it


def shift_s2(c, d, margin):
    """One cycle with its S2 window moved by d samples (same length; systole now ends and diastole now starts at the
    moved window), clipped to keep `margin` samples of systole and of diastole. Returns (cycle, moved, clipped); a
    cycle too short to move comes back unchanged with moved False."""
    s1e = int(c['s1'][1])
    s2on, s2e = int(c['s2'][0]), int(c['s2'][1])
    b = int(c['diastole'][1])
    L2 = s2e - s2on
    lo, hi = s1e + margin, b - margin - L2
    if hi < lo:
        return dict(c), False, False
    t = min(max(s2on + int(d), lo), hi)
    return dict(c, systole=(s1e, t), s2=(t, t + L2), diastole=(t + L2, b)), True, t != s2on + int(d)


def displace_s2(cycles, share, rng, sign, d_ms, sr=4000, margin_ms=S2_MARGIN_MS):
    """Move the S2 window of round(share x n) random cycles (share >= 1: every cycle) by sign x d_ms."""
    n = len(cycles)
    k = n if share >= 1 else int(round(share * n))
    if k >= n:
        pick = set(range(n))
    else:
        pick = set(rng.choice(n, size=k, replace=False).tolist()) if k > 0 else set()
    d = int(round(sign * d_ms / 1000 * sr))
    m = int(round(margin_ms / 1000 * sr))
    out, moved, clipped = [], 0, 0
    for i, c in enumerate(cycles):
        if i in pick:
            nc, mv, cl = shift_s2(c, d, m)
            moved += int(mv); clipped += int(cl)
            out.append(nc)
        else:
            out.append(dict(c))
    return _renumber(out), dict(n=n, picked=len(pick), moved=moved, clipped=clipped)


def select_recordings(names, share, seed):
    """round(share x N) recordings drawn with `seed` over the sorted names."""
    names = sorted(names)
    k = int(round(share * len(names)))
    perm = np.random.default_rng(int(seed)).permutation(len(names))
    return set(names[i] for i in perm[:k])


def s2_onset_offset_ms(cycles, ref_cycles, sr=4000, tol_ms=150):
    """(median signed S2 onset difference in ms, number of pairs) over cycles paired with the reference by start
    (Hungarian within tol); the statistic behind the S2 follow-up's 'recordings with S2 more than 60 ms off' (|median|
    above 60 ms, among recordings with at least one pair). (nan, 0) when nothing pairs."""
    a = np.array([c['s1'][0] for c in cycles], float)
    b = np.array([c['s1'][0] for c in ref_cycles], float)
    if not len(a) or not len(b):
        return np.nan, 0
    D = np.abs(a[:, None] - b[None, :])
    tol = tol_ms / 1000 * sr
    ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e9))
    e = [(cycles[i]['s2'][0] - ref_cycles[j]['s2'][0]) / sr * 1000 for i, j in zip(ri, ci) if D[i, j] <= tol]
    return (float(np.median(e)), len(e)) if e else (np.nan, 0)


EXP0B_D_MS = 100           # the S2 displacement (ms)
EXP0B_MISSED = 0.5         # the share of beats missed in a recording chosen for missed-beat failure


def exp0b_cycles(ref, name, typ, level, seed, selected, equal_time_fn, n_samples, sr=4000):
    """One recording's boundary set under an Experiment 0b condition, with its displacement counts.
    typ 's2c': every cycle's S2 displaced when the recording is selected; 's2s': a share `level` of its cycles displaced;
    'delc': half its beats missed when selected; 'fail': equal-time windows when selected. Draws come from
    rng_for(seed, name); for the S2 types the recording's direction is the first draw."""
    rng = rng_for(seed, name)
    info = dict(n=len(ref), picked=0, moved=0, clipped=0, sign=0)
    if typ in ('s2c', 's2s'):
        sign = 1 if int(rng.integers(2)) else -1
        info['sign'] = sign
        share = 1.0 if typ == 's2c' else float(level)
        if typ == 's2s' or name in selected:
            out, d = displace_s2(ref, share, rng, sign, EXP0B_D_MS, sr)
            info.update(d)
            return out, info
        return _renumber(list(ref)), info
    if name not in selected:
        return _renumber(list(ref)), info
    if typ == 'delc':
        return delete_starts(ref, EXP0B_MISSED, rng), info
    if typ == 'fail':
        return equal_time_fn(n_samples, sr), info
    raise ValueError('unknown Experiment 0b type %r' % typ)
