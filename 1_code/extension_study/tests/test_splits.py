"""Folds and splits: run with  python -m pytest tests -q  from extension_study/."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import splits
from extstudy.config import LOCAL, circor_paths

R1_FOLDS = os.path.join(LOCAL['r1_run'], 'folds')
META = circor_paths()['metadata']
TASKS = ['detection', 'grading', 'timing', 'shape']


@pytest.fixture(scope='module')
def child_of():
    return splits.circor_children(META)


def test_children_count(child_of):
    assert len(child_of) == 942
    assert len(set(child_of.values())) == 942 - 70          # 70 children enrolled twice


@pytest.mark.parametrize('task', TASKS)
def test_child_folds_on_r1_sets(task, child_of):
    old = np.load(os.path.join(R1_FOLDS, task + '.npz'), allow_pickle=True)
    names, y = old['names'].astype(str), old['y']
    new = splits.draw_folds(names, y, child_of)
    s = splits.check_folds(new, child_of, META)
    assert list(new['names']) == list(names) and np.array_equal(new['y'], y)    # same recordings and labels as R1
    assert s['linked_pairs_split'] == 0
    if task == 'detection':
        assert s['children'] == 816 and s['n'] == 3000
    # the old folds did split linked children (regression check on the leak itself)
    old_fold = dict(names=names, patient_ids=old['patient_ids'].astype(int), patient_ids_original=old['patient_ids'].astype(int),
                    y=y, fold_id=old['fold_id'], n_splits=old['n_splits'])
    assert splits.check_folds(old_fold, child_of, META)['linked_pairs_split'] > 0


def test_fold_hash_stable(child_of):
    old = np.load(os.path.join(R1_FOLDS, 'detection.npz'), allow_pickle=True)
    a = splits.draw_folds(old['names'].astype(str), old['y'], child_of)
    b = splits.draw_folds(old['names'].astype(str), old['y'], child_of)
    assert splits.fold_hash(a) == splits.fold_hash(b)


def test_paired_splits():
    T = splits.paired_splits(LOCAL['audit_table'])
    assert T.groupby('group')['role'].nunique().max() == 1
    assert set(T[T.dataset == 'cinc2016a']['role']) == {'train'}
    assert not ((T.dataset == 'ephnogram') & T.record.isin(['ECGPCG0001', 'ECGPCG0002'])).any()
    h = T[T.dataset == 'hfref']
    assert set(h['role']) == {'train', 'val', 'test'}
