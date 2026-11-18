"""Pre-registered statistics (extstudy/stats.py) on synthetic predictions. Run:  python -m pytest tests/test_stats.py -q"""
import os, sys
import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats as ST


def test_macro_f1_matches_sklearn_with_weights():
    rng = np.random.default_rng(0)
    for ncls in (2, 3):
        y, p = rng.integers(0, ncls, 300), rng.integers(0, ncls, 300)
        w = rng.integers(0, 4, 300).astype(float)
        mine = ST.macro_f1_from_confusion(ST.weighted_confusions(y, p, ncls, w[None, :]))[0]
        ref = 100 * f1_score(y, p, average='macro', labels=list(range(ncls)), zero_division=0, sample_weight=w)
        assert abs(mine - ref) < 1e-9
    # a class absent from truth and prediction scores 0, as in sklearn with labels = all classes
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    assert abs(ST.macro_f1_from_confusion(ST.weighted_confusions(y, p, 3, np.ones((1, 4))))[0]
               - 100 * f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0)) < 1e-9


def test_bootstrap_draw_equals_explicit_resample():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 3); y = rng.integers(0, 3, 120); p = rng.integers(0, 3, 120)
    children = np.unique(groups)
    D = ST.child_draws(children, 5, 42)
    for b in range(5):
        w = D[b, np.searchsorted(children, groups)].astype(float)
        idx = np.repeat(np.arange(120), w.astype(int))
        ref = 100 * f1_score(y[idx], p[idx], average='macro', labels=[0, 1, 2], zero_division=0)
        assert abs(ST.macro_f1_from_confusion(ST.weighted_confusions(y, p, 3, w[None, :]))[0] - ref) < 1e-9
    assert np.array_equal(ST.child_draws(children, 5, 42), D)          # same seed, same draws


def _res(diffs):
    """Fake analysis result with composite draws for E (zeros) and the given arm offsets."""
    rng = np.random.default_rng(3)
    base = rng.normal(50, 1, 2000)
    out = dict(comp_draws={'equal_time': base}, comp_point={'equal_time': 50.0})
    for k, (mu, sd) in diffs.items():
        d = base + rng.normal(mu, sd, 2000)
        out['comp_draws'][k] = d; out['comp_point'][k] = 50.0 + mu
    return out


def test_h1_alone_without_t():
    h = ST.hypotheses(_res({'springer_wpk': (3.0, 1.0)}), t_frozen=False)
    assert h['H1']['level'] == 0.025 and h['H1']['rejected'] and set(h) == {'H1'}
    h = ST.hypotheses(_res({'springer_wpk': (0.5, 1.0)}), t_frozen=False)
    assert not h['H1']['rejected']


def test_graphical_procedure_and_h3_verdicts():
    # both clear: H1 and H2 rejected, H3 non-inferior (T about equal to S)
    h = ST.hypotheses(_res({'springer_wpk': (6.0, 0.5), 'timing_net': (6.0, 0.5)}), t_frozen=True)   # T - S spread about 0.7
    assert h['H1']['rejected'] and h['H2']['rejected'] and h['H3']['verdict'] == 'non-inferior' and h['H3']['level'] == 0.025
    # wide T - S spread around 0: inconclusive, not a trend
    h = ST.hypotheses(_res({'springer_wpk': (6.0, 1.5), 'timing_net': (6.0, 1.5)}), t_frozen=True)
    assert h['H3']['verdict'] == 'inconclusive'
    # H2 fails: no H3
    h = ST.hypotheses(_res({'springer_wpk': (6.0, 1.0), 'timing_net': (0.0, 1.0)}), t_frozen=True)
    assert h['H1']['rejected'] and not h['H2']['rejected'] and 'H3' not in h
    # T clearly below S: inferior
    h = ST.hypotheses(_res({'springer_wpk': (12.0, 0.5), 'timing_net': (5.0, 0.5)}), t_frozen=True)
    assert h['H3']['verdict'] == 'inferior'


def test_alignment_check_catches_misordered_arms(tmp_path):
    names = np.array(['a', 'b', 'c']); arr = {}
    for tag, nm in (('equal_time', names), ('springer_wpk', names[::-1])):
        pfx = 'grading__%s__' % tag
        arr.update({pfx + 'y': np.array([0, 1, 2]), pfx + 'groups': np.array([1, 2, 3]), pfx + 'names': nm, pfx + 'mlp__pred': np.array([0, 1, 2])})
    f = tmp_path / 'oof.npz'; np.savez(f, **arr)
    data, tasks, tags = ST.load_oof(str(f))
    with pytest.raises(AssertionError):
        ST.check_alignment(data, tasks, tags)
