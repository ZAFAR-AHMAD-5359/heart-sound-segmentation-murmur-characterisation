"""
Cross-validation and uncertainty.

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

The default protocol matches the article: recording-level stratified 10-fold
cross-validation, with paired t-tests across the folds for between-arm
comparisons. A stricter variant is available through `--split patient`, which
groups folds by participant so that no participant's recordings appear on both
sides of a split; recordings from one participant are correlated, and the
grouped protocol treats them as one unit. Bootstrap intervals resample at the
same unit as the split: recordings under the recording-level protocol,
participants under the grouped one.

Both a paired t-test across folds (the article's test) and a paired bootstrap
interval on the pooled out-of-fold predictions are reported for between-arm
comparisons, so the reader can inspect either.
"""
import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from .model import fit_mlp, fit_rf

SEED = 42
N_SPLITS = 10
N_BOOT = 2000


def make_folds(y, groups, n_splits=N_SPLITS, seed=SEED, split='recording'):
    """Draw folds once from the labels and identifiers alone.

    The folds do not depend on the features, so every arm of the study reuses
    exactly the same splits. That makes all between-arm comparisons paired and
    removes fold assignment as a source of difference.

    split='recording' stratifies over recordings (the article's protocol);
    split='patient' additionally groups by participant.
    """
    mask = y != -1                      # -1 marks a label that does not apply
    y_m, g_m = y[mask], groups[mask]
    if split == 'patient':
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(y_m, y_m, groups=g_m))
    else:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(y_m, y_m))
    return mask, y_m, g_m, int(y_m.max()) + 1, splits


def enough_for_folds(y, groups, n_splits=N_SPLITS, split='recording'):
    """True when every class has at least n_splits members (and, under the
    grouped protocol, at least n_splits participants exist), so the fold draw
    cannot fail. Small --limit runs skip a task rather than crash."""
    y_m = y[y != -1]
    if len(y_m) == 0:
        return False
    counts = np.bincount(y_m)
    counts = counts[counts > 0]
    if counts.min() < n_splits:
        return False
    if split == 'patient' and len(np.unique(groups[y != -1])) < n_splits:
        return False
    return True


def cross_validate(X, y, groups, device, folds=None, seed=SEED,
                   classifier='mlp', split='recording'):
    """Return pooled out-of-fold predictions, per-fold scores and the headline
    metrics."""
    mask, y_m, g_m, n_classes, splits = folds or make_folds(y, groups, split=split)
    X_m = X[mask]
    labels = list(range(n_classes))
    oof = np.empty_like(y_m)
    fold_f1, fold_acc = [], []

    for train_idx, test_idx in splits:
        # standardisation is fitted on the training fold only
        scaler = StandardScaler().fit(X_m[train_idx])
        if classifier == 'rf':
            model = fit_rf(scaler.transform(X_m[train_idx]), y_m[train_idx],
                           seed=seed)
            pred = model.predict(scaler.transform(X_m[test_idx]))
        else:
            model = fit_mlp(scaler.transform(X_m[train_idx]), y_m[train_idx],
                            g_m[train_idx], n_classes, device, seed=seed)
            with torch.no_grad():
                xb = torch.tensor(scaler.transform(X_m[test_idx]),
                                  dtype=torch.float32, device=device)
                pred = model(xb).argmax(1).cpu().numpy()
        oof[test_idx] = pred
        fold_f1.append(f1_score(y_m[test_idx], pred, average='macro',
                                labels=labels, zero_division=0) * 100)
        fold_acc.append(accuracy_score(y_m[test_idx], pred) * 100)

    return {
        'macro_f1': f1_score(y_m, oof, average='macro', labels=labels,
                             zero_division=0) * 100,
        'accuracy': accuracy_score(y_m, oof) * 100,
        'fold_f1': fold_f1, 'fold_acc': fold_acc,
        'y_true': y_m, 'y_pred': oof, 'groups': g_m, 'mask': mask,
        'n_classes': n_classes,
    }


def majority_baseline(y, n_classes):
    """Score of always predicting the most common class.

    Reported for every task because accuracy is misleading under this
    imbalance: on detection a majority-class classifier scores roughly 79%
    accuracy while contributing nothing clinically. Macro-F1 against this
    baseline is the honest comparison.
    """
    y_m = y[y != -1]
    top = np.bincount(y_m).argmax()
    return f1_score(y_m, np.full(len(y_m), top), average='macro',
                    labels=list(range(n_classes)), zero_division=0) * 100


def paired_fold_ttest(fold_scores_a, fold_scores_b):
    """Paired t-test across folds, the article's between-arm test. Both arms
    are scored on identical folds, so the pairing is exact. Returns None for t
    and p when the test is undefined (all fold differences identical)."""
    t, p = stats.ttest_rel(fold_scores_a, fold_scores_b)
    if not (np.isfinite(t) and np.isfinite(p)):
        return {'t': None, 'p': None}
    return {'t': float(t), 'p': float(p)}


def _unit_index(units):
    values, inverse = np.unique(units, return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(len(values))]


def bootstrap_ci(y_true, y_pred, units, n_classes, n_boot=N_BOOT, seed=SEED):
    """95% percentile interval, resampling the given units with replacement.

    `units` is the resampling unit: participant identifiers under the grouped
    protocol, one id per recording under the recording-level protocol.
    """
    rng = np.random.default_rng(seed)
    by_unit = _unit_index(units)
    labels = list(range(n_classes))
    scores = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_unit), len(by_unit))
        sel = np.concatenate([by_unit[k] for k in pick])
        scores[b] = f1_score(y_true[sel], y_pred[sel], average='macro',
                             labels=labels, zero_division=0) * 100
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_difference(y_true, pred_a, pred_b, units, n_classes,
                         n_boot=N_BOOT, seed=SEED):
    """Paired interval on (a - b).

    The same resample scores both arms, so the interval reflects the
    difference rather than the sum of two independent uncertainties. `delta`
    is the observed difference on the full sample; the interval and
    `boot_mean` come from the resamples.
    """
    rng = np.random.default_rng(seed)
    by_unit = _unit_index(units)
    labels = list(range(n_classes))

    def diff(sel):
        return (f1_score(y_true[sel], pred_a[sel], average='macro',
                         labels=labels, zero_division=0)
                - f1_score(y_true[sel], pred_b[sel], average='macro',
                           labels=labels, zero_division=0)) * 100

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_unit), len(by_unit))
        diffs[b] = diff(np.concatenate([by_unit[k] for k in pick]))
    return {'delta': float(diff(np.arange(len(y_true)))),
            'boot_mean': float(diffs.mean()),
            'lo': float(np.percentile(diffs, 2.5)),
            'hi': float(np.percentile(diffs, 97.5)),
            'p_gt_zero': float((diffs > 0).mean())}
