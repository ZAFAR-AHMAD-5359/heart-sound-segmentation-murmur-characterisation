"""Reference check logic on synthetic annotations. Run:  python -m pytest tests/test_refcheck.py -q"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import refcheck as RC

FS = RC.FS_ANN


def cycle_rows(n=10, rr=0.8, s1=(0.0, 0.12), s2=(0.30, 0.40), noise=None):
    """State boundaries (samples) for n cycles: S1, systole, S2, diastole; optional noise markers (start_s, end_s)."""
    rows = []
    for k in range(n):
        t = 1.0 + k * rr
        rows += [(int((t + s1[0]) * FS), 'S1'), (int((t + s1[1]) * FS), 'systole'), (int((t + s2[0]) * FS), 'S2'),
                 (int((t + s2[1]) * FS), 'diastole')]
    rows.append((int((1.0 + n * rr) * FS), 'S1'))
    if noise:
        rows += [(int(noise[0] * FS), '(N'), (int(noise[1] * FS), 'N)')]
        rows.sort(key=lambda x: x[0])
    return rows


def test_parse_states_segments_noise_and_span():
    segs, noise, span = RC.parse_states(cycle_rows(n=3, noise=(1.5, 1.7)))
    assert list(segs.state[:4]) == ['S1', 'systole', 'S2', 'diastole']
    assert np.isclose(segs.start_s.iloc[0], 1.0) and np.isclose(segs.end_s.iloc[0], 1.12)
    assert noise == [(1.5, 1.7)] and np.isclose(span[0], 1.0) and np.isclose(span[1], 1.0 + 3 * 0.8)


def test_mark_inside_complex_with_20ms_tolerance():
    segs, _, _ = RC.parse_states(cycle_rows(n=3))
    assert RC.mark_scores(1.05, segs, 'S1')[0]                     # inside S1
    assert RC.mark_scores(1.135, segs, 'S1')[0]                    # 15 ms after the offset: within tolerance
    assert not RC.mark_scores(1.15, segs, 'S1')[0]                 # 30 ms after the offset
    ok, wide, off = RC.mark_scores(1.45, segs, 'S2')               # 50 ms after S2 offset: wrong, but within 100 ms of centre
    assert not ok and wide and np.isclose(off, 100.0 / 1000)


def test_score_record_skips_noise_and_out_of_span_beats():
    segs, noise, span = RC.parse_states(cycle_rows(n=5, noise=(2.7, 2.9)))
    r = 1.0 + 0.8 * np.arange(6) - 0.03
    ev = dict(r_s=r[:-1], r_next_s=r[1:], s1_s=r[:-1] + 0.08, s2_s=r[:-1] + 0.38, valid=np.ones(5, bool), hr=np.full(5, 75.0))
    rows = RC.score_record(ev, segs, noise, span)
    beats = [x['beat'] for x in rows]
    assert 0 not in beats                                            # starts before the first annotated boundary
    assert 2 not in beats                                            # its cycle overlaps the annotated noise
    assert all(x['s1_ok'] and x['s2_ok'] for x in rows)


def test_bootstrap_share_is_record_clustered():
    df = pd.DataFrame(dict(record=['a'] * 90 + ['b'] * 10, x=[True] * 90 + [False] * 10))
    est, lo, hi = RC.bootstrap_share(df, 'x', n=500)
    assert np.isclose(est, 0.9) and lo <= 0.1 + 1e-9 and hi >= 0.99   # two clusters: the interval spans both extremes


def test_unwrap_matlab_cells():
    nested = np.empty((1, 1), dtype=object); nested[0, 0] = np.array(['S2'])
    assert RC._unwrap(nested) == 'S2' and RC._unwrap(np.array([[679]], dtype=np.uint16)) == 679 and RC._unwrap('S1') == 'S1'
