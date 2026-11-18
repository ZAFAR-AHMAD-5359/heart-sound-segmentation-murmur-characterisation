"""Tests of revision_checks2.py on synthetic arrays (no package data needed).
Run:  python -m pytest test_revision_checks2.py -q
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
import revision_checks2 as R


def _synthetic(seed=0, n=240, n_children=60):
    """Four tasks, seven arms with graded accuracy, 60 children of 4 recordings each."""
    rng = np.random.default_rng(seed)
    names = np.array(['r%03d' % i for i in range(n)])
    groups = np.repeat(np.arange(n_children), n // n_children)
    acc = {'C': 0.9, 'D': 0.8, 'T': 0.7, 'W': 0.7, 'S': 0.6, 'S_pub': 0.6, 'E': 0.4}
    data = {}
    for t in R.TASKS:
        k = R.NCLS[t]
        y = rng.integers(0, k, n)
        for a, tag in R.ARMS.items():
            wrong = (y + rng.integers(1, k, n)) % k
            data[(t, tag)] = dict(y=y, groups=groups, names=names, pred=np.where(rng.random(n) < acc[a], y, wrong))
    return data, names, groups


def test_macro_f1_weights_equal_explicit_resampling_and_absent_class_scores_zero():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert R.macro_f1(y, y, 3, np.ones((1, 6)))[0] == 100.0
    pred = np.array([0, 1, 1, 1, 2, 0])
    w = np.array([2, 0, 1, 1, 3, 1], float)
    idx = np.repeat(np.arange(6), w.astype(int))
    assert abs(R.macro_f1(y, pred, 3, w[None]) - R.macro_f1(y[idx], pred[idx], 3, np.ones((1, len(idx)))))[0] < 1e-9
    # class 2 absent from truth and prediction: F1 0 for that class, so the macro over 3 classes is 2/3 of the perfect score
    y2, p2 = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    assert abs(R.macro_f1(y2, p2, 3, np.ones((1, 4)))[0] - 100.0 * 2 / 3) < 1e-9


def test_child_draws_reproducible_per_seed_and_sum_to_the_number_of_children():
    ch = np.arange(50)
    a, b, c = R.child_draws(ch, 100, 7), R.child_draws(ch, 100, 7), R.child_draws(ch, 100, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (100, 50) and np.all(a.sum(1) == 50)


def test_seed_sensitivity_identical_arms_give_zero_and_bounds_hold():
    data, names, groups = _synthetic()
    children = np.unique(groups)
    for t in R.TASKS:                                   # S identical to T: every T - S percentile is exactly 0
        data[(t, 'springer_wpk')] = dict(data[(t, 'timing_net')])
    out = R.seed_sensitivity(data, children, seeds=(1, 2, 3, 4), n_boot=300, n_big=3000)
    assert all(v['p1_25'] == 0.0 and v['p2_5'] == 0.0 for v in out['T-S']['per_seed'].values())
    te = out['T-E']
    vals = [v['p1_25'] for v in te['per_seed'].values()]
    assert te['min_p1_25'] == min(vals) and te['max_p1_25'] == max(vals) and te['n_seeds'] == 4
    assert len(set(vals)) > 1                           # the percentile moves with the seed
    assert all(v['p2_5'] >= v['p1_25'] for v in te['per_seed'].values())
    # the large-draw percentile lies within the seed range widened by half a point (Monte Carlo error at 300 draws)
    assert te['min_p1_25'] - 0.5 <= te['seed42_20000']['p1_25'] <= te['max_p1_25'] + 0.5
    assert te['n_seeds_p1_25_above_0'] == sum(v > 0 for v in vals)
    # T clearly above E in the synthetic data: every percentile positive
    assert te['min_p1_25'] > 0 and out['S-E']['seed42_20000']['p2_5'] > 0


def test_miss_shares_none_all_and_by_class():
    data, names, groups = _synthetic()
    det = {'none': {n: 1 for n in names}, 'all': {n: 0 for n in names}}
    out = R.miss_shares(data, det)
    for t in R.TASKS3:
        assert out['none'][t]['all']['share_missed'] == 0.0 and out['all'][t]['all']['share_missed'] == 1.0
        assert sum(out['none'][t][c]['n'] for c in R.CLASS_NAMES[t]) == out['none'][t]['all']['n'] == len(names)
    y = data[('grading', 'pcg_seg')]['y']
    o = R.miss_shares(data, {'z': {n: int(c != 0) for n, c in zip(names, y)}})['z']['grading']    # miss every I/VI recording
    assert o['I/VI']['share_missed'] == 1.0 and o['II/VI']['share_missed'] == 0.0 and o['III/VI or louder']['n_missed'] == 0
    assert o['all']['n_missed'] == int((y == 0).sum())


def test_fallback_counts_restrict_to_the_analysed_set_and_union(tmp_path):
    for run, arms, names in (('run1_a', ('springer_wpk_annot', 'equal_time_annot'), ['a', 'b', 'c', 'z']),
                             ('run2_a', ('timing_net_annot', 'waveform_net_annot'), ['a', 'b', 'c'])):
        rows = [dict(recording=n, arm=a, arm_fallback=bool(n in ('a', 'z') and a != 'equal_time_annot')) for a in arms for n in names]
        d = tmp_path / run
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / 'diagnostics.csv', index=False)
    out = R.fallback_counts(str(tmp_path), analysed={'a', 'b', 'c'})
    assert out['springer_wpk_annot']['n_fallback_all_rows'] == 2 and out['springer_wpk_annot']['n_fallback_analysed'] == 1
    assert out['equal_time_annot']['n_fallback_analysed'] == 0 and out['timing_net_annot']['recordings_analysed'] == ['a']
    assert out['union']['analysed'] == 1 and out['union']['all_rows'] == 2 and abs(out['springer_wpk_annot']['share_analysed_pct'] - 100 / 3) < 1e-9
