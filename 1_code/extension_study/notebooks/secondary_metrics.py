"""Secondary metrics requested by Reviewer 3 (R3-M5), defined in prereg/CHANGES_AFTER_FREEZE.md before computing:
balanced accuracy, class-wise sensitivity, macro one-vs-rest AUROC and macro average precision (AUPRC; detection: the
present class), from the out-of-fold MLP predictions and five-seed mean probabilities of every arm, and from the CNN's
recorded probabilities on the same rows. Intervals: the shared child bootstrap of stats.py (2,000 draws, seed 42).
Descriptive only.
Usage:  python notebooks/secondary_metrics.py <merged oof> <cnn oof> <output folder>
"""
import hashlib, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

CNN_SHA = 'd921b7c99e6587bfdb0c88c802a8f7e40d0ae96d88ee8eb0363b7652ddf38689'
CNN_TASK = {'detection': 'murmur', 'grading': 'grading', 'timing': 'timing', 'shape': 'shape'}
ARMS = ('E', 'A', 'B', 'S', 'S_pub', 'T', 'W', 'D', 'C')
TASKS = ('detection', 'grading', 'timing', 'shape')


def load(merged, cnn_path):
    z = np.load(merged, allow_pickle=False)
    rows = {}
    for t in TASKS:
        for a in ARMS:
            p = '%s__%s__' % (t, stats.ARM_KEYS[a])
            rows[(t, a)] = dict(names=z[p + 'names'].astype(str), groups=z[p + 'groups'], y=z[p + 'y'].astype(int),
                                pred=z[p + 'mlp__pred'].astype(int), proba=z[p + 'mlp__proba'].astype(float))
    assert hashlib.sha256(open(cnn_path, 'rb').read()).hexdigest() == CNN_SHA, 'CNN predictions differ from the ledger'
    c = np.load(cnn_path, allow_pickle=True)                         # our own CNN output; names are object arrays
    for t, tc in CNN_TASK.items():
        ref = rows[(t, 'C')]
        o = {n: i for i, n in enumerate(c['%s__rec_names' % tc].astype(str))}
        ix = np.array([o[n] for n in ref['names']])
        assert np.array_equal(c['%s__y_true' % tc][ix].astype(int), ref['y'])
        rows[(t, 'CNN')] = dict(names=ref['names'], groups=ref['groups'], y=ref['y'], pred=c['%s__pred' % tc][ix].astype(int),
                                proba=c['%s__probs' % tc][ix].astype(float))
    for (t, a), r in rows.items():
        assert np.array_equal(r['names'], rows[(t, 'C')]['names']) and np.array_equal(r['y'], rows[(t, 'C')]['y'])
    return rows


def metrics(y, pred, proba, k, w=None):
    w = np.ones(len(y)) if w is None else w
    rec = []
    for c in range(k):
        m = y == c
        rec.append(float(np.sum(w[m] * (pred[m] == c)) / np.sum(w[m])) if np.sum(w[m]) > 0 else np.nan)
    out = dict(balanced_accuracy=float(np.nanmean(rec)), sensitivity=rec)
    keep = w > 0
    if k == 2:
        out['auroc'] = float(roc_auc_score(y[keep], proba[keep, 1], sample_weight=w[keep]))
        out['auprc'] = float(average_precision_score(y[keep], proba[keep, 1], sample_weight=w[keep]))
    else:
        au, ap = [], []
        for c in range(k):
            yc = (y[keep] == c).astype(int)
            au.append(roc_auc_score(yc, proba[keep, c], sample_weight=w[keep]))
            ap.append(average_precision_score(yc, proba[keep, c], sample_weight=w[keep]))
        out['auroc'], out['auprc'] = float(np.mean(au)), float(np.mean(ap))
    return out


def main(merged, cnn_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = load(merged, cnn_path)
    children = np.unique(np.concatenate([rows[(t, 'C')]['groups'] for t in TASKS]))
    D = stats.child_draws(children)
    out = dict(merged_oof_sha256=hashlib.sha256(open(merged, 'rb').read()).hexdigest(), n_boot=int(D.shape[0]), results={})
    for t in TASKS:
        g = rows[(t, 'C')]['groups']
        W = D[:, np.searchsorted(children, g)].astype(float)
        k = int(rows[(t, 'C')]['y'].max() + 1)
        for a in ARMS + ('CNN',):
            r = rows[(t, a)]
            pt = metrics(r['y'], r['pred'], r['proba'], k)
            dr = [metrics(r['y'], r['pred'], r['proba'], k, W[b]) for b in range(W.shape[0])]
            ci = lambda key: stats.interval(np.array([d[key] for d in dr]))
            out['results']['%s/%s' % (t, a)] = dict(
                n=int(len(r['y'])), balanced_accuracy=dict(point=pt['balanced_accuracy'], ci95=ci('balanced_accuracy')),
                auroc=dict(point=pt['auroc'], ci95=ci('auroc')), auprc=dict(point=pt['auprc'], ci95=ci('auprc')),
                sensitivity=[dict(point=pt['sensitivity'][c], ci95=stats.interval(np.array([d['sensitivity'][c] for d in dr]))) for c in range(k)])
            print('%-9s %-5s BA %.3f [%.3f, %.3f]  AUROC %.3f [%.3f, %.3f]  AUPRC %.3f  sens %s' % (
                t, a, pt['balanced_accuracy'], *ci('balanced_accuracy'), pt['auroc'], *ci('auroc'), pt['auprc'],
                ' '.join('%.2f' % s for s in pt['sensitivity'])), flush=True)
    json.dump(out, open(os.path.join(out_dir, 'secondary_metrics.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    main(*sys.argv[1:4])
