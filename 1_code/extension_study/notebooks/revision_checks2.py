"""Revision checks 2 (independent code review, 29 Oct 2026). Descriptive analyses of the saved out-of-fold predictions,
written so the authors can run and log them. Reads only; no CirCor audio, feature or classifier is touched.
(a) Bootstrap-seed sensitivity of the one-sided percentiles: for seeds 1 to 10, 123 and 2024 (2,000 child draws each,
    the draw rule of extstudy/stats.py: default_rng(seed), bincount(integers(0, k, k)) over the sorted children), the
    1.25th and 2.5th percentiles of T - S, T - E and S - E on the composite macro-F1 (mean of grading, timing, shape),
    their min and max over seeds, the seed-42 2,000-draw values (the pre-registered analysis) and one 20,000-draw
    bootstrap at seed 42.
(b) Detection misses by characterisation class: for E, S, S_pub, T, W, D, C and the CNN, the share of murmur-present
    recordings with a grading, timing or shape label that the arm's own detection model called absent, by class.
(c) Matched-coverage fallback counts among the analysed recordings (S_annot, T_annot, W_annot, E_annot, union).
Inputs under <results root>: run2_scoring/oof_predictions.npz, cnn_baseline/oof_predictions.npz, folds_child/*.npz,
run1_*/diagnostics.csv, run2_*/diagnostics.csv; optionally the CirCor training_data.csv to verify the label coding.
Usage:  python revision_checks2.py <results root> [--meta <training_data.csv>] [--out <folder>]
Deterministic; numpy and pandas only; no import of extstudy.
"""
import argparse, glob, hashlib, json, os
import numpy as np
import pandas as pd

TASKS3 = ('grading', 'timing', 'shape')
TASKS = ('detection',) + TASKS3
NCLS = {'detection': 2, 'grading': 3, 'timing': 3, 'shape': 3}
ARMS = dict(E='equal_time', S='springer_wpk', S_pub='springer_pub', T='timing_net', W='waveform_net', D='pcg_seg_fixed', C='pcg_seg')
CLASS_NAMES = {'grading': ('I/VI', 'II/VI', 'III/VI or louder'), 'timing': ('holosystolic', 'early systolic', 'mid systolic'),
               'shape': ('plateau', 'decrescendo', 'diamond')}
CNN_TASK = {'detection': 'murmur', 'grading': 'grading', 'timing': 'timing', 'shape': 'shape'}
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 123, 2024)
N_BOOT, N_BIG, SEED = 2000, 20000, 42
CONTRASTS = (('T', 'S'), ('T', 'E'), ('S', 'E'))
ANNOT_ARMS = {'run1': ('springer_wpk_annot', 'equal_time_annot'), 'run2': ('timing_net_annot', 'waveform_net_annot')}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ------------------------------------------------------------------ loading
def load_oof(path, tags):
    """{(task, tag): dict(y, groups, names, pred)}; every arm of a task must hold the same rows, labels and children."""
    z = np.load(path, allow_pickle=False)
    out = {}
    for t in TASKS:
        for tag in tags:
            p = '%s__%s__' % (t, tag)
            out[(t, tag)] = dict(y=z[p + 'y'].astype(int), groups=z[p + 'groups'].astype(int), names=z[p + 'names'].astype(str),
                                 pred=z[p + 'mlp__pred'].astype(int))
        ref = out[(t, tags[0])]
        for tag in tags[1:]:
            d = out[(t, tag)]
            assert np.array_equal(d['names'], ref['names']) and np.array_equal(d['y'], ref['y']) and np.array_equal(d['groups'], ref['groups']), (t, tag)
    return out


def load_cnn(path, data):
    """The CNN's recording-level predictions aligned by name to the rows of arm C; labels asserted equal."""
    z = np.load(path, allow_pickle=True)                          # the authors' own file; names are object arrays
    out = {}
    for t, tc in CNN_TASK.items():
        ref = data[(t, 'pcg_seg')]
        o = {n: i for i, n in enumerate(z['%s__rec_names' % tc].astype(str))}
        ix = np.array([o[n] for n in ref['names']])
        assert np.array_equal(z['%s__y_true' % tc][ix].astype(int), ref['y']), 'CNN labels differ on %s' % t
        out[t] = dict(ref, pred=z['%s__pred' % tc][ix].astype(int))
    return out


def check_fold_files(root, data):
    """The fold files define the analysed set and the folds; they must agree with the oof file."""
    ok = {}
    for t in TASKS:
        f = np.load(os.path.join(root, 'folds_child', t + '.npz'), allow_pickle=False)
        d = data[(t, 'pcg_seg')]
        ok[t] = bool(np.array_equal(f['names'].astype(str), d['names']) and np.array_equal(f['y'].astype(int), d['y'])
                     and np.array_equal(f['patient_ids'].astype(int), d['groups']))
    analysed = set(np.load(os.path.join(root, 'folds_child', 'detection.npz'), allow_pickle=False)['keep_names'].astype(str))
    assert analysed == set(data[('detection', 'pcg_seg')]['names']), 'kept set differs between the fold file and the oof file'
    return ok, analysed


# ------------------------------------------------------------------ (a) bootstrap
def child_draws(children, n_boot, seed):
    """(n_boot, k) multiplicity of every child, the rule of stats.child_draws."""
    rng = np.random.default_rng(seed)
    k = len(children)
    return np.stack([np.bincount(rng.integers(0, k, size=k), minlength=k) for _ in range(n_boot)]).astype(float)


def macro_f1(y, pred, ncls, W):
    """Weighted macro-F1 (per cent) over all ncls classes with zero_division 0; W: (B, n) recording weights."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    onehot = np.zeros((len(y), ncls * ncls))
    onehot[np.arange(len(y)), y * ncls + pred] = 1.0
    M = (np.asarray(W, float) @ onehot).reshape(len(W), ncls, ncls)       # rows = truth, columns = prediction
    tp = np.diagonal(M, axis1=1, axis2=2)
    fp, fn = M.sum(1) - tp, M.sum(2) - tp
    den = 2 * tp + fp + fn
    return 100.0 * np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0).mean(1)


def composite_draws(data, tag, children, D):
    """Composite macro-F1 per draw: each recording carries the multiplicity of its child in that draw."""
    return np.mean([macro_f1(data[(t, tag)]['y'], data[(t, tag)]['pred'], NCLS[t], D[:, np.searchsorted(children, data[(t, tag)]['groups'])])
                    for t in TASKS3], 0)


def contrast_percentiles(data, children, seeds, n_boot, contrasts=CONTRASTS):
    """{'A-B': {str(seed): {p1_25, p2_5}}} of the composite difference A - B over the given seeds."""
    out = {}
    arms = sorted({x for c in contrasts for x in c})
    for seed in seeds:
        D = child_draws(children, n_boot, seed)
        comp = {a: composite_draws(data, ARMS[a], children, D) for a in arms}
        for a, b in contrasts:
            d = comp[a] - comp[b]
            out.setdefault('%s-%s' % (a, b), {})[str(seed)] = dict(p1_25=float(np.percentile(d, 1.25)), p2_5=float(np.percentile(d, 2.5)),
                                                                   point_boot_mean=float(d.mean()))
    return out


def seed_sensitivity(data, children, seeds=SEEDS, n_boot=N_BOOT, n_big=N_BIG, ref_seed=SEED):
    per_seed = contrast_percentiles(data, children, seeds, n_boot)
    ref = contrast_percentiles(data, children, (ref_seed,), n_boot)
    big = contrast_percentiles(data, children, (ref_seed,), n_big)
    out = {}
    for c, vals in per_seed.items():
        r = dict(per_seed=vals, seed42_2000=ref[c][str(ref_seed)], seed42_20000=big[c][str(ref_seed)], n_seeds=len(vals), n_boot=n_boot, n_big=n_big)
        for q in ('p1_25', 'p2_5'):
            v = [x[q] for x in vals.values()]
            r['min_' + q], r['max_' + q], r['n_seeds_%s_above_0' % q] = float(min(v)), float(max(v)), int(sum(x > 0 for x in v))
        out[c] = r
    return out


# ------------------------------------------------------------------ (b) detection misses by class
def miss_shares(data, det_pred_by_arm):
    """det_pred_by_arm: {arm label: {recording: detection prediction, 0 = absent}}. Per arm, task and class: n, n_missed,
    share_missed over the murmur-present recordings labelled for that task (labels shared by every arm)."""
    out = {}
    for arm, det in det_pred_by_arm.items():
        out[arm] = {}
        for t in TASKS3:
            d = data[(t, 'pcg_seg')]
            missed = np.array([det[n] == 0 for n in d['names']], bool)
            rows = {}
            for c, name in enumerate(CLASS_NAMES[t]):
                m = d['y'] == c
                rows[name] = dict(n=int(m.sum()), n_missed=int(missed[m].sum()), share_missed=float(missed[m].mean()) if m.any() else float('nan'))
            rows['all'] = dict(n=int(len(missed)), n_missed=int(missed.sum()), share_missed=float(missed.mean()))
            out[arm][t] = rows
    return out


# ------------------------------------------------------------------ (c) matched-coverage fallback
def fallback_counts(root, analysed):
    """Fallback recordings of the matched-coverage arms from run1_*/ and run2_*/diagnostics.csv, over all rows and
    among the analysed recordings; union over the four arms."""
    out, union_all, union_kept = {}, set(), set()
    for run, arms in ANNOT_ARMS.items():
        paths = sorted(glob.glob(os.path.join(root, run + '_*', 'diagnostics.csv')))
        assert len(paths) == 1, 'expected one %s_*/diagnostics.csv under %s, found %s' % (run, root, paths)
        d = pd.read_csv(paths[0], low_memory=False)
        for a in arms:
            s = d[d.arm == a]
            fb = s['arm_fallback'].astype(str).str.strip().str.lower().isin(('true', '1', '1.0'))
            names_all, names_kept = set(s.recording[fb]), set(s.recording[fb & s.recording.isin(analysed)])
            union_all |= names_all
            union_kept |= names_kept
            out[a] = dict(source=os.path.relpath(paths[0], root), n_rows=int(len(s)), n_fallback_all_rows=len(names_all),
                          n_fallback_analysed=len(names_kept), share_analysed_pct=100.0 * len(names_kept) / len(analysed), recordings_analysed=sorted(names_kept))
    out['union'] = dict(all_rows=len(union_all), analysed=len(union_kept), n_analysed=len(analysed), recordings_analysed=sorted(union_kept))
    return out


# ------------------------------------------------------------------ optional label check against the metadata
def labels_from_metadata(meta_csv):
    """The run notebooks' coding (cell v_labels): grading I/II/III -> 0/1/2, timing Holo/Early/Mid -> 0/1/2, shape
    Plateau/Decresc/Diamond -> 0/1/2, -1 otherwise; murmur Present -> 1, Absent -> 0, Unknown dropped."""
    M = pd.read_csv(meta_csv)
    lab = {}
    for _, r in M.iterrows():
        if r['Murmur'] == 'Unknown':
            continue
        p = r['Murmur'] == 'Present'
        g, t, s = str(r.get('Systolic murmur grading', '')), str(r.get('Systolic murmur timing', '')), str(r.get('Systolic murmur shape', ''))
        lab[int(r['Patient ID'])] = dict(
            murmur=int(p), grading=(0 if g in ('I/VI', 'I') else 1 if g in ('II/VI', 'II') else 2 if g in ('III/VI', 'III') else -1) if p else -1,
            timing=(0 if 'Holosystolic' in t else 1 if 'Early' in t else 2 if 'Mid' in t else -1) if p else -1,
            shape=(0 if 'Plateau' in s else 1 if 'Decresc' in s else 2 if 'Diamond' in s else -1) if p else -1)
    return lab


def verify_labels(data, lab):
    return {t: bool(all(lab[int(n.split('_')[0])][k] == y for n, y in zip(data[(t, 'pcg_seg')]['names'], data[(t, 'pcg_seg')]['y'])))
            for t, k in (('detection', 'murmur'), ('grading', 'grading'), ('timing', 'timing'), ('shape', 'shape'))}


# ------------------------------------------------------------------ main
def main(root, meta=None, out_dir='.'):
    oof, cnn = os.path.join(root, 'run2_scoring', 'oof_predictions.npz'), os.path.join(root, 'cnn_baseline', 'oof_predictions.npz')
    data = load_oof(oof, [ARMS[a] for a in ARMS])
    cnn_data = load_cnn(cnn, data)
    folds_ok, analysed = check_fold_files(root, data)
    children = np.unique(np.concatenate([data[(t, 'pcg_seg')]['groups'] for t in TASKS]))
    rep = dict(inputs=dict(merged_oof_sha256=sha256(oof), cnn_oof_sha256=sha256(cnn), fold_files_match_oof=folds_ok,
                           n_analysed=len(analysed), n_children=int(len(children)), n_per_task={t: int(len(data[(t, 'pcg_seg')]['y'])) for t in TASKS},
                           script_sha256=sha256(os.path.abspath(__file__))))
    rep['seed_sensitivity'] = seed_sensitivity(data, children)
    det = {a: dict(zip(data[('detection', ARMS[a])]['names'], data[('detection', ARMS[a])]['pred'])) for a in ARMS}
    det['CNN'] = dict(zip(cnn_data['detection']['names'], cnn_data['detection']['pred']))
    rep['detection_misses_by_class'] = miss_shares(data, det)
    rep['matched_coverage_fallback'] = fallback_counts(root, analysed)
    if meta:
        rep['labels_verified_against_metadata'] = verify_labels(data, labels_from_metadata(meta))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(out_dir, 'revision_checks2.json'), 'w'), indent=1, default=float)
    # ---- summary ----
    print('SUMMARY revision_checks2  (merged oof %s, cnn oof %s, %d analysed recordings, %d children)' % (
        rep['inputs']['merged_oof_sha256'][:12], rep['inputs']['cnn_oof_sha256'][:12], len(analysed), len(children)))
    print('(a) one-sided percentiles of the composite difference; 2,000 draws per seed; min/max over %d seeds; 20,000 draws at seed 42' % len(SEEDS))
    print('    %-5s %-22s %-22s %-22s %-22s %s' % ('', 'seed 42, 2,000', 'seed 42, 20,000', 'p1.25 min / max', 'p2.5 min / max', 'seeds with p1.25>0 / p2.5>0'))
    for c, r in rep['seed_sensitivity'].items():
        print('    %-5s %+6.2f / %+6.2f        %+6.2f / %+6.2f        %+6.2f / %+6.2f        %+6.2f / %+6.2f        %d / %d of %d' % (
            c, r['seed42_2000']['p1_25'], r['seed42_2000']['p2_5'], r['seed42_20000']['p1_25'], r['seed42_20000']['p2_5'],
            r['min_p1_25'], r['max_p1_25'], r['min_p2_5'], r['max_p2_5'], r['n_seeds_p1_25_above_0'], r['n_seeds_p2_5_above_0'], r['n_seeds']))
    print('(b) share of murmur-present recordings called absent by the arm\'s own detection model, by class (n in brackets)')
    for t in TASKS3:
        print('    %-8s %s' % (t, '  '.join('%s (%d)' % (n, rep['detection_misses_by_class']['C'][t][n]['n']) for n in CLASS_NAMES[t] + ('all',))))
        for arm in list(ARMS) + ['CNN']:
            r = rep['detection_misses_by_class'][arm][t]
            print('      %-6s %s' % (arm, '  '.join('%.2f' % r[n]['share_missed'] for n in CLASS_NAMES[t] + ('all',))))
    fb = rep['matched_coverage_fallback']
    print('(c) matched-coverage fallback among the %d analysed recordings: %s; union %d (all diagnostics rows: %d)' % (
        len(analysed), ', '.join('%s %d (%.2f%%)' % (a, fb[a]['n_fallback_analysed'], fb[a]['share_analysed_pct']) for a in fb if a != 'union'),
        fb['union']['analysed'], fb['union']['all_rows']))
    if meta:
        print('    labels verified against training_data.csv:', rep['labels_verified_against_metadata'])
    print('written', os.path.join(out_dir, 'revision_checks2.json'))
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='results root (06_results)')
    ap.add_argument('--meta', default=None, help='CirCor training_data.csv (optional label check)')
    ap.add_argument('--out', default='.', help='output folder for revision_checks2.json')
    a = ap.parse_args()
    main(a.root, a.meta, a.out)
