"""run2_estimation.py: the masked analysis equals stats.analyse without a mask and scores a subset as macro-F1 over
its recordings. Run:  python -m pytest tests/test_run2_estimation.py -q"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'notebooks'))
from extstudy import stats
import run2_estimation as RE


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    data = {}
    names = np.array(['r%03d' % i for i in range(n)])
    groups = np.array(['c%02d' % (i // 3) for i in range(n)])
    for t in ('detection',) + stats.TASKS3:
        k = 2 if t == 'detection' else 3
        y = rng.integers(0, k, n)
        for tag in ('pcg_seg', 'equal_time', 'timing_net'):
            p = np.where(rng.random(n) < (0.8 if tag == 'pcg_seg' else 0.5), y, rng.integers(0, k, n))
            data[(t, tag)] = dict(y=y, groups=groups, names=names, mlp=p)
    return data, ['detection'] + list(stats.TASKS3), ['equal_time', 'pcg_seg', 'timing_net'], names


def _f1(y, p, k):
    f = []
    for c in range(k):
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return 100 * np.mean(f)


def test_no_mask_equals_stats_analyse():
    data, tasks, tags, _ = _data()
    a, b = RE.masked_analyse(data, tasks, tags), stats.analyse(data, tasks, tags)
    for k in b['comp_point']:
        assert a['comp_point'][k] == b['comp_point'][k] and np.array_equal(a['comp_draws'][k], b['comp_draws'][k])


def test_subset_scores_its_recordings_with_the_full_draws():
    data, tasks, tags, names = _data(1)
    keep = set(names[:50])
    r = RE.masked_analyse(data, tasks, tags, keep)
    m = np.isin(names, sorted(keep))
    d = data[('timing', 'pcg_seg')]
    assert abs(r['point'][('timing', 'pcg_seg')] - _f1(d['y'][m], d['mlp'][m], 3)) < 1e-9 and r['n']['timing'] == 50
    full = RE.masked_analyse(data, tasks, tags)
    assert np.array_equal(r['children'], full['children'])                # same children, so the same draws
