"""Experiment 0 corruption functions. Run:  python -m pytest tests/test_corrupt.py -q"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import corrupt as K


def seg(r_peaks, sig_len):
    """arm D's segment_phases_samples (R1 notebook, verbatim)."""
    phases = []
    for i in range(len(r_peaks) - 1):
        s, e = int(r_peaks[i]), int(r_peaks[i+1]); rr = e - s
        if rr <= 0: continue
        phases.append({'cycle': i, 's1': (s, s + int(0.15*rr)),
                       'systole': (s + int(0.15*rr), s + int(0.50*rr)),
                       's2': (s + int(0.50*rr), s + int(0.60*rr)),
                       'diastole': (s + int(0.60*rr), e)})
    return phases


def annotated(n=40, rr=3200, gap_after=None):
    s = 1000 + rr * np.arange(n + 1)
    if gap_after is not None:
        s[gap_after + 1:] += 2000                       # an annotation gap after cycle gap_after
    cyc = []
    for i in range(n):
        a, b = int(s[i]), int(s[i] + rr)
        cyc.append({'cycle': i, 's1': (a, a + 400), 'systole': (a + 400, a + 1200), 's2': (a + 1200, a + 1600), 'diastole': (a + 1600, b)})
    return cyc


def test_deletion_merges_into_the_previous_cycle():
    ref = annotated()
    out = K.delete_starts(ref, 0.2, K.rng_for(1, 'x'))
    k = 8
    assert len(out) == len(ref) - k or len(out) == len(ref) - k                      # first cycle may be removed
    st = K.boundary_stats(out, ref)
    assert st['matched'] == st['n'] and st['n'] == len(ref) - k
    assert abs(st['f1'] - 2 * (40 - k) / (80 - k)) < 1e-12
    merged = [c for c in out if c['diastole'][1] - c['s1'][0] > 3200]
    assert merged and all(c['diastole'][1] - c['diastole'][0] > 3200 for c in merged)   # the missed beat lies in diastole


def test_deletion_after_a_gap_removes_the_cycle():
    ref = annotated(n=10, gap_after=4)
    rng = np.random.default_rng(0)

    class Pick:
        def choice(self, n, size, replace):
            return np.array([5])                         # the cycle right after the gap
    out = K.delete_starts(ref, 0.1, Pick())
    assert len(out) == 9 and all(c['diastole'][1] - c['s1'][0] == 3200 for c in out)


def test_insertion_splits_with_fixed_fractions():
    ref = annotated()
    out = K.insert_starts(ref, 0.25, K.rng_for(2, 'y'), seg, 10 ** 7)
    st = K.boundary_stats(out, ref)
    assert st['n'] == 50 and st['matched'] == 40 and abs(st['f1'] - 80 / 90) < 1e-12
    for c in out:
        assert c['s1'][0] <= c['systole'][0] <= c['s2'][0] <= c['diastole'][0] < c['diastole'][1]


def test_draws_are_seeded_per_recording():
    ref = annotated()
    a = K.delete_starts(ref, 0.3, K.rng_for(7, 'rec_1'))
    b = K.delete_starts(ref, 0.3, K.rng_for(7, 'rec_1'))
    c = K.delete_starts(ref, 0.3, K.rng_for(7, 'rec_2'))
    assert a == b and a != c


def test_zero_rate_is_identity():
    ref = annotated()
    assert K.delete_starts(ref, 0.0, K.rng_for(1, 'z')) == ref and K.insert_starts(ref, 0.0, K.rng_for(1, 'z'), seg, 10 ** 7) == ref


def test_shift_s2_moves_the_window_and_keeps_order():
    c = annotated(n=1)[0]                                        # rr 3200, S2 at 1200..1600 after the start
    nc, mv, cl = K.shift_s2(c, 400, 40)
    assert mv and not cl and nc['s2'] == (c['s2'][0] + 400, c['s2'][1] + 400)
    assert nc['systole'] == (c['s1'][1], nc['s2'][0]) and nc['diastole'] == (nc['s2'][1], c['diastole'][1])
    nc, mv, cl = K.shift_s2(c, -2000, 40)                        # clipped: keeps 40 samples of systole
    assert mv and cl and nc['s2'][0] == c['s1'][1] + 40


def test_concentrated_and_spread_displace_the_same_share():
    ref = annotated(n=40)
    names = ['r%02d' % i for i in range(50)]
    sel = K.select_recordings(names, 0.2, 40101)
    assert len(sel) == 10 and sel == K.select_recordings(list(reversed(names)), 0.2, 40101)
    moved_conc = sum(K.displace_s2(ref, 1.0, K.rng_for(1, n), 1, 100)[1]['moved'] for n in names if n in sel)
    moved_spread = sum(K.displace_s2(ref, 0.2, K.rng_for(2, n), -1, 100)[1]['moved'] for n in names)
    assert moved_conc == moved_spread == 400


def test_s2_onset_offset_measures_the_shift():
    ref = annotated(n=20)
    out, info = K.displace_s2(ref, 1.0, K.rng_for(3, 'x'), -1, 100)
    assert K.s2_onset_offset_ms(out, ref) == (-100.0, 20) and K.s2_onset_offset_ms(ref, ref) == (0.0, 20)
    assert K.s2_onset_offset_ms(out[:0], ref)[1] == 0


def eqt(n_samples, sr=4000, bpm=75):
    """arm E's equal_time_cycles (R1 notebook, verbatim)."""
    cycle_len = int(sr * 60 / bpm); n_cycles = max(1, n_samples // cycle_len); out = []
    for i in range(n_cycles):
        s = i*cycle_len; e = min(n_samples, (i+1)*cycle_len); rr = e - s
        if rr <= 0: continue
        out.append({'cycle': i, 's1': (s, s+int(0.25*rr)), 'systole': (s+int(0.25*rr), s+int(0.50*rr)),
                    's2': (s+int(0.50*rr), s+int(0.75*rr)), 'diastole': (s+int(0.75*rr), e)})
    return out


def test_exp0b_conditions():
    ref = annotated(n=30)
    L = 10 ** 6
    a, ia = K.exp0b_cycles(ref, 'r1', 's2c', 0.2, 40101, {'r1'}, eqt, L)
    b, ib = K.exp0b_cycles(ref, 'r2', 's2c', 0.2, 40101, {'r1'}, eqt, L)
    assert ia['moved'] == 30 and abs(K.s2_onset_offset_ms(a, ref)[0]) == 100.0 and ib['moved'] == 0 and b == ref
    s, is_ = K.exp0b_cycles(ref, 'r2', 's2s', 0.3, 50201, set(), eqt, L)
    assert is_['moved'] == 9 and K.s2_onset_offset_ms(s, ref)[0] == 0.0
    assert K.exp0b_cycles(ref, 'r2', 's2s', 0.3, 50201, set(), eqt, L) == (s, is_)          # seeded per recording
    d, _ = K.exp0b_cycles(ref, 'r1', 'delc', 0.4, 60101, {'r1'}, eqt, L)
    assert len(d) == 15 and K.exp0b_cycles(ref, 'r2', 'delc', 0.4, 60101, {'r1'}, eqt, L)[0] == ref
    f, _ = K.exp0b_cycles(ref, 'r1', 'fail', 0.1, 70001, {'r1'}, eqt, L)
    assert f == eqt(L)
    signs = {K.exp0b_cycles(ref, 'r%d' % i, 's2s', 0.1, 50001, set(), eqt, L)[1]['sign'] for i in range(20)}
    assert signs == {-1, 1}
