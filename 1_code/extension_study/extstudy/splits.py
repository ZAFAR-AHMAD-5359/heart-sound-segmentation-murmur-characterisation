"""Grouping and folds.

CirCor: children are connected components of Patient ID and Additional ID (70 children were enrolled in both
campaigns under two IDs; the R1 folds split 54 of 58 such pairs). Folds reproduce the R1 rule exactly
(StratifiedGroupKFold(10, shuffle=True, random_state=42), drawn per task on the labelled set) except that groups
are children. Every fold file carries the recording names and a SHA-256 of its content.

Paired data: (dataset, subject) groups; HFrEF keeps its published folds (0-3 train, 4 validation) and its
community test set; PhysioNet 2016 training-a (no subject map) is training only.
"""
import hashlib, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from extstudy.config import FOLDS


# ------------------------------------------------------------------ CirCor children
def circor_children(metadata_csv):
    """Patient ID -> child ID (the smallest Patient ID in its connected component)."""
    M = pd.read_csv(metadata_csv)
    parent = {int(p): int(p) for p in M['Patient ID']}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, a in zip(M['Patient ID'], M['Additional ID']):
        if pd.notna(a) and int(a) in parent:
            ra, rb = find(int(p)), find(int(a))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {p: find(p) for p in parent}


def patient_of(recording_name):
    return int(str(recording_name).split('_')[0])


def draw_folds(names, y, child_of, n_splits=FOLDS.n_splits, seed=FOLDS.seed):
    """R1 rule on one labelled set: y == -1 excluded; groups = child IDs. Returns a dict ready to save."""
    names, y = np.asarray(names), np.asarray(y)
    mask = y != -1
    nm, ym = names[mask], y[mask]
    pid = np.array([patient_of(n) for n in nm])
    gm = np.array([child_of[p] for p in pid])
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(ym, ym, groups=gm))
    fold_id = np.full(len(ym), -1)
    for i, (tr, te) in enumerate(splits):
        assert not set(gm[tr]) & set(gm[te]), 'child overlap between train and test'
        fold_id[te] = i
    assert (fold_id >= 0).all()
    out = dict(names=nm, patient_ids=gm, patient_ids_original=pid, y=ym, fold_id=fold_id, n_splits=np.array(n_splits),
               keep_names=names, mask_over_keep=mask, seed=np.array(seed), grouping=np.array('child = Patient ID union Additional ID'))
    for i, (tr, te) in enumerate(splits):
        out['train_%02d' % i] = tr
        out['test_%02d' % i] = te
    return out


def fold_hash(fold):
    h = hashlib.sha256()
    for k in ('names', 'patient_ids', 'y', 'fold_id'):
        h.update(np.ascontiguousarray(fold[k]).astype(str).tobytes() if fold[k].dtype.kind in 'US' else np.ascontiguousarray(fold[k]).tobytes())
    return h.hexdigest()


def check_folds(fold, child_of, metadata_csv=None):
    """Assertions every fold file must pass; returns a summary dict."""
    gm, fid, pid = fold['patient_ids'], fold['fold_id'], fold['patient_ids_original']
    for k in range(int(fold['n_splits'])):
        assert not set(gm[fid == k]) & set(gm[fid != k]), 'child overlap in fold %d' % k
    pairs_split = 0
    if metadata_csv is not None:
        M = pd.read_csv(metadata_csv)
        f_of = pd.Series(fid, index=pid).groupby(level=0).agg(lambda s: set(s))
        for p, a in zip(M['Patient ID'], M['Additional ID']):
            if pd.notna(a) and int(p) in f_of.index and int(a) in f_of.index and not (f_of[int(p)] & f_of[int(a)]):
                pairs_split += 1
    return dict(n=len(gm), children=len(np.unique(gm)), patients=len(np.unique(pid)), linked_pairs_split=pairs_split,
                sha256=fold_hash(fold))


def save_folds(fold, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **fold)
    os.replace(tmp, path)
    return fold_hash(fold)


# ------------------------------------------------------------------ paired data
def paired_splits(audit_csv=None, seed=FOLDS.seed, frac=(0.70, 0.15, 0.15)):
    """Split table: one row per usable record with dataset, subject, group, role in {train, val, test}.
    Default source: the teacher-label table (derived/beats_v2/audit_v2.csv, column usable_v2), with the site taken
    from the first audit; a first-audit table (column usable) is still accepted."""
    from extstudy.config import EXCLUDE_RECORDS, LOCAL
    audit_csv = audit_csv or os.path.join(LOCAL['beats'], 'audit_v2.csv')
    A = pd.read_csv(audit_csv, low_memory=False)
    A = A[A['error'].isna()] if 'error' in A else A
    if 'usable_v2' in A:
        A = A[A['usable_v2'].astype(bool)].copy()
        first = pd.read_csv(LOCAL['audit_table'], low_memory=False)[['dataset', 'record', 'site']]
        A = A.merge(first, on=['dataset', 'record'], how='left')
        A['n_rpeaks'] = A['n_beats']
    else:
        A = A[A['usable'].astype(bool)]
    A['subject'] = A['subject'].astype(str)
    A = A[~A.apply(lambda r: (r['dataset'], r['record']) in EXCLUDE_RECORDS, axis=1)].copy()
    A['group'] = A['dataset'] + ':' + A['subject'].astype(str)
    role = {}
    rng = np.random.default_rng(seed)
    for d, g in A.groupby('dataset'):
        if d == 'cinc2016a':                                   # no subject map: training only
            role.update({k: 'train' for k in g['group']})
            continue
        if d == 'hfref':                                       # published folds and community test set
            for k, s in zip(g['group'], g['meta_split'].astype(str)):
                role[k] = 'test' if s == 'community_test' else ('val' if s == 'fold_4' else 'train')
            continue
        groups = np.array(sorted(g['group'].unique()))
        rng.shuffle(groups)
        n = len(groups); n_tr = int(round(frac[0] * n)); n_va = int(round(frac[1] * n))
        for i, k in enumerate(groups):
            role[k] = 'train' if i < n_tr else ('val' if i < n_tr + n_va else 'test')
    A['role'] = A['group'].map(role)
    # every group sits in exactly one role
    assert A.groupby('group')['role'].nunique().max() == 1
    return A[['dataset', 'subject', 'group', 'record', 'site', 'role', 'duration_s', 'n_rpeaks', 'hr_median']]


def table_hash(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def build_circor_folds(r1_folds_dir, out_dir, metadata_csv):
    """Draw child-grouped folds on exactly the R1 labelled sets and write them with a manifest of hashes."""
    child_of = circor_children(metadata_csv)
    manifest = {}
    for task in ('detection', 'grading', 'timing', 'shape'):
        old = np.load(os.path.join(r1_folds_dir, task + '.npz'), allow_pickle=True)
        fold = draw_folds(old['names'].astype(str), old['y'], child_of)
        summary = check_folds(fold, child_of, metadata_csv)
        assert summary['linked_pairs_split'] == 0
        sha = save_folds(fold, os.path.join(out_dir, task + '.npz'))
        manifest[task] = dict(summary, file=task + '.npz', sha256=sha)
    json.dump(manifest, open(os.path.join(out_dir, 'folds_manifest.json'), 'w'), indent=1)
    return manifest


if __name__ == '__main__':
    from extstudy.config import LOCAL, circor_paths
    m = build_circor_folds(os.path.join(LOCAL['r1_run'], 'folds'), os.path.join(LOCAL['runs'], 'folds_child'),
                           circor_paths()['metadata'])
    for t, v in m.items():
        print('%-10s n=%d children=%d patients=%d linked_pairs_split=%d sha256=%s' % (
            t, v['n'], v['children'], v['patients'], v['linked_pairs_split'], v['sha256'][:16]))
