"""
End-to-end pipeline.

Reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

Runs the whole method in six stages:

  1. index        pair each recording with its patient identifier and labels
  2. reconstruct  synthetic ECG from the phonocardiogram (conditional
                  adversarial network, 8 s windows at 8 kHz)
  3. gate         R-peaks on the synthetic ECG divide each cycle into
                  S1, systole, S2 and diastole
  4. features     464 phase-aligned acoustic features per recording, at 4 kHz
  5. classify     stratified ten-fold cross-validation, four tasks
  6. report       macro-F1, fold-wise paired t-tests and bootstrap intervals

Usage
-----
    python run_pipeline.py --data     /path/to/circor/training_data \
                           --metadata /path/to/circor/training_data.csv \
                           --weights  weights/generator.pt \
                           --out      results/

The comparison the study rests on is between two boundary sources, run through an
otherwise identical pipeline:

    --boundaries synthetic   cardiac phases from R-peaks of the synthetic ECG
    --boundaries fixed       equal-duration windows, ignoring the cardiac cycle

Passing `both` runs each in turn and reports the paired difference, which
isolates the contribution of cycle alignment. The fixed arm needs no generator
weights. Both arms analyse identical audio; only the boundaries differ.

`--split recording` (default) is the article's protocol: recording-level
stratified folds. `--split patient` groups folds by participant instead, so no
participant's recordings appear on both sides of a split. `--classifier`
selects the multilayer perceptron (default) or the random forest.

The R-peak detector (src/gating.py) is modular on purpose: it consumes the
synthetic trace and returns peak indices, and the rest of the pipeline consumes
nothing but those indices. Replacing it with a stronger detector is the most
direct route to better phase alignment and is an open invitation to other
researchers.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample as sp_resample

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_record_index, load_audio, TARGET_SR
from src.generator import load_generator, reconstruct_ecg, analysed_portion, GAN_SR
from src.gating import (ensure_positive_polarity, detect_r_peaks,
                        synthetic_cycles, equal_time_cycles)
from src.features import extract_all_v4_features
from src.evaluate import (make_folds, enough_for_folds, cross_validate,
                          majority_baseline, paired_fold_ttest,
                          bootstrap_ci, bootstrap_difference)

TASKS = {'detection': 'murmur', 'grading': 'grading',
         'timing': 'timing', 'shape': 'shape'}


def to_4k(pcg_8k):
    """The analysis audio both arms share: 8 kHz analysed portion resampled to
    4 kHz and max-abs normalised (matching training)."""
    pcg_4k = sp_resample(pcg_8k, int(len(pcg_8k) * TARGET_SR / GAN_SR))
    return pcg_4k / (np.max(np.abs(pcg_4k)) + 1e-10)


def gate_recording(wav_path, boundaries, generator, device):
    """Stages 2 to 3 for one recording.

    Returns (pcg_4k, cycles). Both arms derive pcg_4k through the identical
    path (native rate -> 8 kHz -> analysed portion -> 4 kHz -> max-abs), so the
    arms differ in their boundaries and in nothing else.
    """
    pcg_8k = analysed_portion(load_audio(wav_path, GAN_SR))
    if boundaries == 'synthetic':
        ecg, pcg_8k = reconstruct_ecg(generator, pcg_8k, GAN_SR, device)
        ecg = ensure_positive_polarity(ecg)
        r_peaks = detect_r_peaks(ecg, fs=GAN_SR)
        cycles = synthetic_cycles(r_peaks, out_sr=TARGET_SR, ecg_sr=GAN_SR)
    elif boundaries == 'fixed':
        n4 = int(len(pcg_8k) * TARGET_SR / GAN_SR)
        cycles = equal_time_cycles(n4, TARGET_SR)
    else:
        raise ValueError('boundaries must be "synthetic" or "fixed"')
    return to_4k(pcg_8k), cycles


def build_matrix(records, boundaries, generator, device, verbose=True):
    """Stages 2 to 4: one feature vector per recording."""
    rows, kept, n_beats = [], [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        name, pid, labels, wav_path = rec
        try:
            pcg, cycles = gate_recording(wav_path, boundaries, generator, device)
        except Exception as exc:
            print('  skipped %s (%s)' % (name, type(exc).__name__))
            continue
        if len(cycles) < 2 or len(pcg) < TARGET_SR:
            continue                       # too short to characterise
        rows.append(extract_all_v4_features(pcg, cycles, TARGET_SR))
        kept.append(rec)
        n_beats.append(len(cycles))
        if verbose and (i + 1) % 250 == 0:
            print('  %4d/%d  (%.1f min)' % (i + 1, len(records),
                                            (time.time() - t0) / 60))

    if not rows:
        sys.exit('no recording survived stages 2-4; check --data points at the '
                 'CirCor .wav directory and --metadata at training_data.csv')
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    X = np.nan_to_num(np.array([[float(r.get(k, 0.0)) for k in keys] for r in rows],
                               dtype=np.float32))
    if verbose:
        print('  feature matrix: %d recordings x %d features' % X.shape)
        print('  cycles per recording: median %d, IQR %d to %d'
              % tuple(np.percentile(n_beats, [50, 25, 75]).astype(int)))
    return X, keys, kept


def run(args):
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('device:', device)

    print('\n[1/6] indexing recordings')
    records = build_record_index(args.data, args.metadata)
    if args.limit:
        records = records[:args.limit]
    print('  recordings with audio and labels: %d' % len(records))
    if not records:
        sys.exit('no recordings indexed; check --data and --metadata paths')

    arms = ['synthetic', 'fixed'] if args.boundaries == 'both' else [args.boundaries]

    generator = None
    if 'synthetic' in arms:
        if not args.weights:
            sys.exit('--weights is required for the synthetic arm '
                     '(see weights/README.md)')
        print('\nloading generator from', args.weights)
        generator = load_generator(args.weights, device)

    matrices = {}
    for arm in arms:
        print('\n[2-4/6] boundaries = %s' % arm)
        matrices[arm] = build_matrix(records, arm, generator, device)

    # When both arms run, restrict each to the recordings both kept. The folds
    # are drawn from the labels of the kept set, so without this step a
    # recording dropped by one arm would desynchronise the folds and the
    # between-arm comparison would no longer be paired.
    if len(arms) == 2:
        names = [set(r[0] for r in matrices[a][2]) for a in arms]
        common = names[0] & names[1]
        for a in arms:
            X, keys, kept = matrices[a]
            keep = np.array([r[0] in common for r in kept])
            matrices[a] = (X[keep], keys, [r for r, k in zip(kept, keep) if k])
        dropped = (len(names[0]) - len(common)) + (len(names[1]) - len(common))
        if dropped:
            print('\n  paired analysis restricted to %d recordings kept by both '
                  'arms (%d dropped)' % (len(common), dropped))

    results, oof = {}, {}
    skipped_tasks = set()
    for arm in arms:
        X, keys, kept = matrices[arm]
        groups = np.array([r[1] for r in kept])
        y = {t: np.array([r[2].get(k, -1) for r in kept]) for t, k in TASKS.items()}

        print('\n[5/6] cross-validation (%s, %s split, %s)'
              % (arm, args.split, args.classifier))
        results[arm], oof[arm] = {}, {}
        for task in TASKS:
            if not enough_for_folds(y[task], groups, split=args.split):
                print('  %-10s skipped: too few labelled recordings for '
                      '10-fold cross-validation (rerun without --limit)' % task)
                skipped_tasks.add(task)
                continue
            folds = make_folds(y[task], groups, split=args.split)
            out = cross_validate(X, y[task], groups, device, folds=folds,
                                 classifier=args.classifier)
            units = out['groups'] if args.split == 'patient' \
                else np.arange(len(out['y_true']))
            lo, hi = bootstrap_ci(out['y_true'], out['y_pred'],
                                  units, out['n_classes'])
            results[arm][task] = {'macro_f1': out['macro_f1'], 'ci': [lo, hi],
                                  'accuracy': out['accuracy'],
                                  'fold_f1': out['fold_f1'],
                                  'fold_acc': out['fold_acc'],
                                  'majority': majority_baseline(y[task], out['n_classes'])}
            oof[arm][task] = out
            print('  %-10s macro-F1 %5.1f [%4.1f, %4.1f]   majority %4.1f'
                  % (task, out['macro_f1'], lo, hi, results[arm][task]['majority']))

    print('\n[6/6] report')
    paired = {}
    if len(arms) == 2:
        print('\n  contribution of cycle alignment (synthetic minus fixed):')
        for task in TASKS:
            if task in skipped_tasks:
                continue
            a, b = oof['synthetic'][task], oof['fixed'][task]
            units = a['groups'] if args.split == 'patient' \
                else np.arange(len(a['y_true']))
            d = bootstrap_difference(a['y_true'], a['y_pred'], b['y_pred'],
                                     units, a['n_classes'])
            d['ttest_f1'] = paired_fold_ttest(a['fold_f1'], b['fold_f1'])
            d['ttest_acc'] = paired_fold_ttest(a['fold_acc'], b['fold_acc'])
            paired[task] = d
            p_txt = ('%.4f' % d['ttest_f1']['p']
                     if d['ttest_f1']['p'] is not None else 'undefined')
            print('    %-10s %+5.1f points  [%+5.1f, %+5.1f]  '
                  'fold t-test p = %s' % (task, d['delta'], d['lo'], d['hi'], p_txt))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump({'results': results, 'paired_difference': paired,
                   'split': args.split, 'classifier': args.classifier,
                   'n_recordings_indexed': len(records),
                   'n_recordings_analysed': len(matrices[arms[0]][2]),
                   'seed': 42, 'n_splits': 10, 'n_bootstrap': 2000}, fh, indent=2)
    np.savez_compressed(os.path.join(args.out, 'out_of_fold_predictions.npz'),
                        **{'%s__%s__%s' % (arm, task, field): oof[arm][task][field]
                           for arm in oof for task in oof[arm]
                           for field in ('y_true', 'y_pred', 'groups')})
    print('\nwritten to', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='directory of CirCor .wav files')
    p.add_argument('--metadata', required=True, help='training_data.csv')
    p.add_argument('--weights', default='',
                   help='trained generator checkpoint (synthetic arm)')
    p.add_argument('--out', default='results', help='output directory')
    p.add_argument('--boundaries', default='both',
                   choices=['synthetic', 'fixed', 'both'])
    p.add_argument('--split', default='recording',
                   choices=['recording', 'patient'],
                   help='fold protocol: recording-level stratified (the '
                        'article) or grouped by participant')
    p.add_argument('--classifier', default='mlp', choices=['mlp', 'rf'])
    p.add_argument('--device', default='auto', help='auto, cuda or cpu')
    p.add_argument('--limit', type=int, default=0,
                   help='process only the first N recordings (smoke test)')
    run(p.parse_args())
