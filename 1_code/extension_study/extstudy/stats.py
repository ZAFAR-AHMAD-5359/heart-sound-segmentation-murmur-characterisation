"""Pre-registered analysis of the extension runs (prereg/PREREG_EXT.md, section 3), from oof_predictions.npz.

Primary endpoint: composite macro-F1 = mean of the MLP macro-F1 on grading, timing and shape, from the out-of-fold
prediction (argmax of the mean probability over the five seed fits). Macro-F1 over all classes with zero_division 0,
as in the notebook.

Inference: 2,000 bootstrap draws of the children (seed 42), one set of draws shared by every task and arm. In a draw,
each recording carries the multiplicity of its child; each task's macro-F1 and the composite are recomputed; arm
differences use the same draws. Percentile intervals.

Hypotheses (one-sided, total alpha 0.025, graphical): H1 S - E > 0 and H2 T - E > 0 at 0.0125 each (rejected when the
1.25th percentile of the difference lies above 0); a rejected hypothesis passes its alpha to the other (2.5th
percentile); H3 (only after H2) T non-inferior to S with margin 2.5 points at the level H2 passed; H4 (only after H3)
T - S > 0. Without a frozen T, H1 is tested alone at 0.025.

Usage:  python -m extstudy.stats <run folder> [--out <folder>]
"""
import argparse, hashlib, json, os
import numpy as np

TASKS3 = ('grading', 'timing', 'shape')
N_BOOT, SEED = 2000, 42
MARGIN = 2.5
ARM_KEYS = dict(S='springer_wpk', E='equal_time', C='pcg_seg', D='pcg_seg_fixed', E2='equal_time_matched', A='syn_fixed',
                B='syn_fusion', S_pub='springer_pub', S_annot='springer_wpk_annot', E_annot='equal_time_annot',
                T='timing_net', W='waveform_net', T_annot='timing_net_annot', W_annot='waveform_net_annot')


# ------------------------------------------------------------------ scoring
def macro_f1_from_confusion(M):
    """M: (..., k, k) confusion counts (rows = truth). Macro-F1 over all k classes; a class with no truth and no
    prediction scores 0 (sklearn zero_division=0, labels = all classes)."""
    tp = np.diagonal(M, axis1=-2, axis2=-1)
    fp = M.sum(-2) - tp
    fn = M.sum(-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return 100.0 * f1.mean(-1)


def weighted_confusions(y, pred, ncls, W):
    """W: (B, n) weights -> (B, ncls, ncls) weighted confusion matrices."""
    onehot = np.zeros((len(y), ncls * ncls))
    onehot[np.arange(len(y)), np.asarray(y) * ncls + np.asarray(pred)] = 1.0
    return (W @ onehot).reshape(len(W), ncls, ncls)


def child_draws(children, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_children) multiplicity of every child; children sorted."""
    rng = np.random.default_rng(seed)
    k = len(children)
    D = np.zeros((n_boot, k), np.int32)
    for b in range(n_boot):
        D[b] = np.bincount(rng.integers(0, k, size=k), minlength=k)
    return D


# ------------------------------------------------------------------ loading
def load_oof(path):
    z = np.load(path, allow_pickle=False)
    keys = z.files
    tasks = sorted({k.split('__')[0] for k in keys if k.count('__') >= 2})
    tags = sorted({k.split('__')[1] for k in keys if k.count('__') >= 2})
    data = {}
    for t in tasks:
        for tag in tags:
            pfx = '%s__%s__' % (t, tag)
            if pfx + 'mlp__pred' not in keys:
                continue
            d = dict(y=z[pfx + 'y'].astype(int), groups=z[pfx + 'groups'], names=z[pfx + 'names'].astype(str),
                     mlp=z[pfx + 'mlp__pred'].astype(int))
            if pfx + 'rf__pred' in keys:
                d['rf'] = z[pfx + 'rf__pred'].astype(int)
            if pfx + 'mlp__proba_seeds' in keys:
                d['proba_seeds'] = z[pfx + 'mlp__proba_seeds']
            data[(t, tag)] = d
    return data, tasks, tags


def check_alignment(data, tasks, tags):
    """Every arm of a task scores the same recordings in the same order with the same labels and children."""
    for t in tasks:
        ref = None
        for tag in tags:
            if (t, tag) not in data:
                continue
            d = data[(t, tag)]
            if ref is None:
                ref = d
                continue
            assert list(d['names']) == list(ref['names']) and np.array_equal(d['y'], ref['y']) and np.array_equal(d['groups'], ref['groups']), \
                'arm %s is not aligned with the other arms on %s' % (tag, t)


# ------------------------------------------------------------------ analysis
def analyse(data, tasks, tags, clf='mlp', n_boot=N_BOOT, seed=SEED):
    check_alignment(data, tasks, tags)
    children = np.unique(np.concatenate([data[(t, tags[0])]['groups'] for t in tasks]))
    D = child_draws(children, n_boot, seed)
    point, draws = {}, {}
    for t in tasks:
        g = data[(t, tags[0])]['groups']
        W = D[:, np.searchsorted(children, g)].astype(float)            # recording weight = multiplicity of its child
        ncls = int(max(data[(t, tag)]['y'].max() for tag in tags) + 1)
        for tag in tags:
            d = data[(t, tag)]
            if clf not in d:
                continue
            ones = np.ones((1, len(d['y'])))
            point[(t, tag)] = float(macro_f1_from_confusion(weighted_confusions(d['y'], d[clf], ncls, ones))[0])
            draws[(t, tag)] = macro_f1_from_confusion(weighted_confusions(d['y'], d[clf], ncls, W))
    comp_point, comp_draws = {}, {}
    if all(t in tasks for t in TASKS3):
        for tag in tags:
            if all((t, tag) in point for t in TASKS3):
                comp_point[tag] = float(np.mean([point[(t, tag)] for t in TASKS3]))
                comp_draws[tag] = np.mean([draws[(t, tag)] for t in TASKS3], 0)
    return dict(children=children, point=point, draws=draws, comp_point=comp_point, comp_draws=comp_draws, n_boot=n_boot, seed=seed)


def interval(x, lo=2.5, hi=97.5):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def difference(res, a, b):
    d = res['comp_draws'][a] - res['comp_draws'][b]
    return dict(point=res['comp_point'][a] - res['comp_point'][b], ci95=interval(d), p1_25=float(np.percentile(d, 1.25)),
                p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5)), p98_75=float(np.percentile(d, 98.75)),
                share_above_0=float(np.mean(d > 0)))


def hypotheses(res, t_frozen):
    """Graphical procedure of prereg section 3. Returns the verdicts with the levels used."""
    have = lambda arm: ARM_KEYS[arm] in res['comp_draws']
    out = {}
    if not t_frozen or not have('T'):
        h1 = difference(res, ARM_KEYS['S'], ARM_KEYS['E'])
        out['H1'] = dict(h1, level=0.025, rejected=bool(h1['p2_5'] > 0), note='T not frozen: H1 tested alone at 0.025')
        return out
    h1 = difference(res, ARM_KEYS['S'], ARM_KEYS['E'])
    h2 = difference(res, ARM_KEYS['T'], ARM_KEYS['E'])
    r1, r2 = h1['p1_25'] > 0, h2['p1_25'] > 0
    lev1, lev2 = 0.0125, 0.0125
    if r1 and not r2:
        r2, lev2 = h2['p2_5'] > 0, 0.025
    if r2 and not r1:
        r1, lev1 = h1['p2_5'] > 0, 0.025
    if r1 and r2:
        lev1 = lev2 = 0.025
    out['H1'] = dict(h1, level=lev1, rejected=bool(r1))
    out['H2'] = dict(h2, level=lev2, rejected=bool(r2))
    if r2:
        h3 = difference(res, ARM_KEYS['T'], ARM_KEYS['S'])
        lo = h3['p1_25'] if lev2 == 0.0125 else h3['p2_5']
        hi = h3['p98_75'] if lev2 == 0.0125 else h3['p97_5']
        verdict = 'non-inferior' if lo > -MARGIN else ('inferior' if hi < 0 else 'inconclusive')
        out['H3'] = dict(h3, level=lev2, lower=lo, upper=hi, margin=MARGIN, verdict=verdict)
        if verdict == 'non-inferior':
            out['H4'] = dict(h3, level=lev2, rejected=bool(lo > 0))
    return out


def recovery_fraction(res, x_annot, e_annot='equal_time_annot', c='pcg_seg'):
    """(X - E) / (C - E) at matched coverage, per draw; draws with C - E <= 0 are reported, not dropped."""
    num = res['comp_draws'][x_annot] - res['comp_draws'][e_annot]
    den = res['comp_draws'][c] - res['comp_draws'][e_annot]
    ok = den > 0
    pt = (res['comp_point'][x_annot] - res['comp_point'][e_annot]) / (res['comp_point'][c] - res['comp_point'][e_annot])
    return dict(point=float(pt), ci95=interval(num[ok] / den[ok]) if ok.sum() > 10 else None, n_draws_denominator_not_positive=int((~ok).sum()))


def run(run_dir, out_dir=None, t_frozen=False):
    data, tasks, tags = load_oof(os.path.join(run_dir, 'oof_predictions.npz'))
    out = dict(run_dir=run_dir, n_boot=N_BOOT, seed=SEED, stats_py_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
               oof_sha256=hashlib.sha256(open(os.path.join(run_dir, 'oof_predictions.npz'), 'rb').read()).hexdigest())
    for clf in ('mlp', 'rf'):
        res = analyse(data, tasks, tags, clf)
        blk = dict(n_children=int(len(res['children'])),
                   per_task={'%s/%s' % k: dict(point=v, ci95=interval(res['draws'][k])) for k, v in res['point'].items()},
                   composite={tag: dict(point=v, ci95=interval(res['comp_draws'][tag])) for tag, v in res['comp_point'].items()},
                   vs_E={tag: difference(res, tag, ARM_KEYS['E']) for tag in res['comp_point'] if tag != ARM_KEYS['E']})
        if clf == 'mlp':
            blk['hypotheses'] = hypotheses(res, t_frozen)
            for x in ('springer_wpk_annot', 'timing_net_annot', 'waveform_net_annot'):
                if x in res['comp_draws'] and 'equal_time_annot' in res['comp_draws']:
                    blk.setdefault('recovery_fraction', {})[x] = recovery_fraction(res, x)
        out[clf] = blk
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'stats_results.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--t-frozen', action='store_true')
    a = ap.parse_args()
    r = run(a.run_dir, a.out, a.t_frozen)
    print(json.dumps(r['mlp']['hypotheses'], indent=1))
