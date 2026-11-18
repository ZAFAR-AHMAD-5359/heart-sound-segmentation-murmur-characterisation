#!/usr/bin/env python
"""Build R1_CNN_Baseline.ipynb (reviewer comment R1-m5).

The builder reads the ORIGINAL Milestone-3 notebooks at build time, extracts the
preprocessing, MFCC, augmentation, balancing-scorer, label-coding and model
classes VERBATIM (by def/class name, through the ast module), verifies that the
procedural snippets it re-uses are present verbatim in the originals, and then
writes the notebook with nbformat. Every cell is a plain Python string.

Run:  python build_cnn_baseline.py
"""
import ast
import hashlib
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT_NB = HERE / 'R1_CNN_Baseline.ipynb'

SRC = {
    'dataprep': Path(os.environ.get(
        'M3_DATAPREP_NB',
        r'E:\NUST MS Thesis\Milestone-3\Data-Preparation\MILESTONE3_DATA_PREPARATION.ipynb')),
    'aug': Path(os.environ.get(
        'M3_AUG_NB',
        r'E:\NUST MS Thesis\Milestone-3\Data-Preparation\MILESTONE3_DATA_AUGMENTATION_V5.ipynb')),
    'train': Path(os.environ.get(
        'M3_TRAIN_NB',
        r'E:\NUST MS Thesis\Milestone-3\Final PCG Only Results\MILESTONE3_MODEL_TRAINING.ipynb')),
    'm4': Path(os.environ.get(
        'M4_TEMPORAL_NB',
        r'E:\NUST MS Thesis\Milestone-4\Temporal V4\MILESTONE4_TEMPORAL_V4.ipynb')),
}


# ----------------------------------------------------------------------------
# verbatim extraction helpers
# ----------------------------------------------------------------------------
def load_code_cells(path):
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def find_cell(cells, needle, label):
    hits = [c for c in cells if needle in c]
    if not hits:
        raise SystemExit(f'[{label}] needle not found in any code cell: {needle!r}')
    return hits[0]


def extract_named(source, names, label):
    """Return the exact source lines of the named defs / classes / assignments."""
    tree = ast.parse(source)
    lines = source.splitlines()
    found = {}
    for node in tree.body:
        key = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
        if key in names and key not in found:
            found[key] = '\n'.join(lines[node.lineno - 1: node.end_lineno])
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f'[{label}] names not found: {missing}')
    return '\n\n\n'.join(found[n] for n in names)


VERBATIM_BLOCKS = {}


def verbatim(key, nb_key, needle, names):
    cells = load_code_cells(SRC[nb_key])
    cell = find_cell(cells, needle, key)
    body = extract_named(cell, names, key)
    sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
    VERBATIM_BLOCKS[key] = {'notebook': SRC[nb_key].name, 'names': names, 'sha256': sha}
    header = (f"# ---- verbatim from {SRC[nb_key].name} "
              f"(code cell containing {needle!r}); names: {', '.join(names)} ----")
    footer = f"# ---- end verbatim block '{key}' (sha256 {sha[:16]}) ----"
    return header + '\n' + body + '\n' + footer


def assert_snippet(nb_key, snippet, label):
    """Verify that a procedural snippet re-used in the notebook exists verbatim in the original."""
    cells = load_code_cells(SRC[nb_key])
    norm = lambda s: '\n'.join(l.rstrip() for l in s.strip().splitlines())
    target = norm(snippet)
    if not any(target in norm(c) for c in cells):
        raise SystemExit(f'[{label}] snippet not found verbatim in {SRC[nb_key].name}:\n{snippet}')
    VERBATIM_BLOCKS[f'snippet:{label}'] = {
        'notebook': SRC[nb_key].name,
        'sha256': hashlib.sha256(target.encode('utf-8')).hexdigest()}
    print(f'  verified snippet: {label}')


for k, p in SRC.items():
    if not p.exists():
        raise SystemExit(f'source notebook missing: {k} -> {p}')

print('Extracting verbatim blocks ...')
VB_DATAPREP_CONFIG = verbatim('dataprep_config', 'dataprep', "'chunk_duration': 4", ['CONFIG'])
VB_DATAPREP_FUNCS = verbatim('dataprep_functions', 'dataprep', 'def create_chunks_circular',
                             ['load_audio', 'bandpass_filter', 'normalize_zscore', 'create_chunks_circular'])
VB_LOCATION_LABELS = verbatim('location_labels', 'dataprep', 'def is_murmur_audible_at_location',
                              ['is_murmur_audible_at_location', 'get_murmur_characteristics'])
VB_TRAIN_CONSTANTS = verbatim('training_constants', 'train', 'HOP_LENGTH = 160',
                              ['SAMPLE_RATE', 'DURATION', 'N_SAMPLES', 'N_MFCC', 'N_FFT', 'HOP_LENGTH',
                               'N_MELS', 'FMAX', 'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY',
                               'PATIENCE', 'VAL_SPLIT', 'DROPOUT'])
VB_MFCC = verbatim('mfcc_extractor', 'train', 'def extract_mfcc_from_array', ['extract_mfcc_from_array'])
VB_AUG = verbatim('augmentation', 'aug', 'def augment_audio',
                  ['time_shift', 'amplitude_scale', 'add_noise', 'time_stretch', 'AUGMENTATION_COMBOS',
                   'augment_audio'])
VB_JOINT = verbatim('joint_balancing_scorers', 'aug', 'def score_candidate',
                    ['calculate_stage_distributions', 'calculate_imbalance_scores', 'score_candidate',
                     'is_balanced'])
VB_BLOCKS = verbatim('conv_attention_blocks', 'train', 'class AttentionBlock', ['AttentionBlock', 'ConvBlock'])
VB_MURMUR = verbatim('murmur_classifier', 'train', 'class MurmurClassifier', ['MurmurClassifier'])
VB_MULTI = verbatim('multilabel_classifier', 'train', 'class MultiLabelClassifier', ['MultiLabelClassifier'])
VB_EARLY = verbatim('early_stopping', 'train', 'class EarlyStopping', ['EarlyStopping'])
VB_PATIENT_LABELS = verbatim('patient_labels_m4', 'm4', 'def get_recording_labels', ['get_recording_labels'])

print('Verifying procedural snippets ...')
assert_snippet('dataprep', """
train_patients, temp_patients, train_labels, temp_labels = train_test_split(
    patient_ids, patient_labels,
    train_size=CONFIG['train_ratio'],
    stratify=patient_labels,
    random_state=CONFIG['random_state']
)
""", 'split_first')
assert_snippet('dataprep', """
val_ratio_adjusted = CONFIG['val_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio'])
val_patients, test_patients = train_test_split(
    temp_patients,
    train_size=val_ratio_adjusted,
    stratify=temp_labels,
    random_state=CONFIG['random_state']
)
""", 'split_second')
assert_snippet('dataprep', "df_valid_patients = df_recordings[df_recordings['Murmur'] != 'Unknown'].drop_duplicates('patient_id')", 'split_patients')
assert_snippet('dataprep', "CONFIG['chunk_samples'] = CONFIG['sample_rate'] * CONFIG['chunk_duration']", 'config_chunk_samples')
assert_snippet('dataprep', "CONFIG['min_remainder_samples'] = CONFIG['sample_rate'] * CONFIG['min_remainder']", 'config_min_remainder')
assert_snippet('dataprep', "parts = wav_file.replace('.wav', '').split('_')", 'wav_parse_1')
assert_snippet('dataprep', "location = parts[-1] if len(parts) >= 2 else None", 'wav_parse_2')
assert_snippet('aug', """
target_count = len(absent_indices)
num_to_augment = target_count - len(present_indices)
""", 'murmur_target')
assert_snippet('aug', """
    source_idx = np.random.choice(present_indices)

    # Augment audio
    aug_audio = augment_audio(chunks[source_idx], combo_idx=combo_idx)
    combo_idx += 1
""", 'murmur_oversampling_loop')
assert_snippet('aug', "shuffle_idx = np.random.permutation(len(murmur_train_chunks))", 'murmur_shuffle')
assert_snippet('aug', "MAX_ITERATIONS = 10000  # Safety limit", 'joint_max_iter')
assert_snippet('aug', "TOLERANCE = 2.0", 'joint_tolerance')
assert_snippet('aug', """
        best_idx = scores.idxmax()
        source_row = working_metadata.loc[best_idx]

        # Augment this sample
        aug_audio = augment_audio(working_chunks[best_idx], combo_idx=combo_idx)
        combo_idx += 1
""", 'joint_loop_core')
assert_snippet('aug', "shuffle_idx = np.random.permutation(len(multilabel_train_chunks))", 'multilabel_shuffle')
assert_snippet('train', "features = (features - features.mean()) / (features.std() + 1e-8)", 'per_sample_standardisation')
assert_snippet('train', "optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)", 'optimizer')
assert_snippet('train', "scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)", 'scheduler')
assert_snippet('train', "early_stopping = EarlyStopping(patience=PATIENCE, mode='max')", 'early_stopping_cfg')
assert_snippet('train', "scheduler.step(val_auc)", 'monitor_murmur')
assert_snippet('train', "scheduler.step(val_aucs['average'])", 'monitor_multilabel')
assert_snippet('train', "loss_total = loss_grading + loss_timing + loss_shape", 'multilabel_loss')
assert_snippet('train', "criterion = nn.CrossEntropyLoss()", 'criterion')
assert_snippet('train', """
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=stratify,
        random_state=SEED
    )
""", 'chunk_level_es_split')
assert_snippet('train', """
            if np.random.random() < 0.3:
                audio = audio * np.random.uniform(0.9, 1.1)
""", 'online_gain')
assert_snippet('train', """
            if np.random.random() < 0.3:
                shift = int(N_SAMPLES * np.random.uniform(-0.05, 0.05))
                audio = np.roll(audio, shift)
""", 'online_shift')

VERBATIM_JSON = json.dumps(VERBATIM_BLOCKS, indent=1, sort_keys=True)

# ----------------------------------------------------------------------------
# notebook cells
# ----------------------------------------------------------------------------
CELLS = []


def md(text):
    CELLS.append(new_markdown_cell(text.strip('\n')))


def code(text):
    CELLS.append(new_code_cell(text.strip('\n') + '\n'))


md("""
# R1 CNN baseline: the PCG-only CNN retrained under the shared patient-grouped folds

Reviewer comment R1-m5 asks why the PCG-only baseline row of Table 2 has no standard deviation while the gated methods do. The original baseline (Milestone-3, `MILESTONE3_MODEL_TRAINING.ipynb`) was scored once, at chunk level, on a single 70/15/15 patient split, and its early-stopping set was a chunk-level re-split of the pooled training and validation chunks, so it was not patient-disjoint.

This notebook retrains the same model (same preprocessing, MFCC input, architecture, optimiser, scheduler, batch size, epochs, augmentation and class balancing) under the ten patient-grouped folds exported by `R1_Gating_Campaign.ipynb`, with a patient-disjoint inner validation split for early stopping, and scores it at recording level (mean of chunk softmax probabilities, argmax). Recording-level metrics are primary; chunk-level metrics are reported alongside. Every pooled metric carries a patient-level bootstrap 95% confidence interval, and per-fold accuracy and macro-F1 are saved for the fold-wise paired tests.

Section 11 re-runs the original single split so the article can state exactly why 93.4 / 67.4 / 68.4 / 58.8 differ from the new baseline row (chunk versus recording level, single split versus folds, location-aware versus patient-level labels, leaky versus patient-disjoint early stopping).

All model, preprocessing and augmentation code is copied verbatim from the original notebooks by the builder script (`build_cnn_baseline.py`); the SHA-256 of each copied block is written to the run manifest. Deviations from the original are listed in the manifest under `deviations` and in the README.

Runtime: Google Colab Pro, GPU (A100 or L4), high-RAM runtime. Estimated wall time for the full run on an A100: about 3 to 4 hours (chunk cache 10 to 20 minutes, murmur folds about 90 minutes, characterisation folds about 60 to 90 minutes, continuity section about 30 minutes). The estimate was made with mixed precision on and TF32 at its default; the article run uses full fp32 (mixed precision off, TF32 off) and may take longer.
""")

md("""
## 0. Runtime setup

Mounts Drive, installs `librosa`, imports everything, and reports the hardware. Nothing here depends on the data.
""")

code(r'''
# Stage 0. Runtime setup: Drive, packages, hardware report, seed helper
import os, sys, subprocess, time, json, math, random, re, hashlib, glob, platform, warnings, copy
from pathlib import Path
from datetime import datetime

STAGE_SECONDS = {}
NOTEBOOK_T0 = time.time()
RUN_STARTED = datetime.now().isoformat(timespec='seconds')


class StageTimer:
    """Context manager that prints and accumulates wall-clock seconds per stage."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] >> {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        STAGE_SECONDS[self.name] = STAGE_SECONDS.get(self.name, 0.0) + dt
        status = 'done' if exc_type is None else f'FAILED ({exc_type.__name__})'
        print(f"[{datetime.now().strftime('%H:%M:%S')}] << {self.name} {status} in {dt:,.1f} s")
        return False


try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    pip_result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'librosa'],
                                capture_output=True, text=True)
    if pip_result.returncode != 0:
        print(pip_result.stdout[-2000:])
        print(pip_result.stderr[-2000:])
        raise RuntimeError('pip install failed')

import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy import stats
from scipy.io import wavfile
from scipy.ndimage import zoom
import librosa
import sklearn
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             f1_score, accuracy_score, balanced_accuracy_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from joblib import Parallel, delayed
import matplotlib
if not IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import psutil
except ImportError:
    psutil = None

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
GPU_MEM_GB = (torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else 0.0
RAM_GB = (psutil.virtual_memory().total / 2**30) if psutil is not None else float('nan')
N_CPU = os.cpu_count() or 1
VERSIONS = {
    'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
    'scipy': scipy.__version__, 'librosa': librosa.__version__, 'sklearn': sklearn.__version__,
    'torch': torch.__version__, 'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__,
    'cuda': torch.version.cuda if torch.cuda.is_available() else None,
    'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False      # full fp32 on Ampere and newer GPUs, as the gating notebook
    torch.backends.cudnn.allow_tf32 = False


print(f"Colab: {IN_COLAB} | device: {DEVICE} | GPU: {GPU_NAME} ({GPU_MEM_GB:.1f} GB) | CPUs: {N_CPU} | RAM: {RAM_GB:.1f} GB")
print(json.dumps(VERSIONS, indent=1))
''')

md("""
## 1. Configuration (MANUAL OVERRIDE cell)

Edit only this cell. `DRY_RUN` processes 40 recordings, 3 epochs and 2 folds end to end, with the joint balancing capped at 300 iterations (the verbatim safety limit of 10,000 applies otherwise), and writes to its own output folder. `LABEL_MODE = 'patient'` is the coding used by the gated arms (every recording of a Murmur = Present patient is Present; Unknown dropped; grading I/II/III = 0/1/2 with higher grades excluded; timing Holosystolic/Early/Mid = 0/1/2 with Late-systolic excluded; shape Plateau/Decrescendo/Diamond = 0/1/2 with Crescendo excluded). `LABEL_MODE = 'location'` is the original Milestone-3 coding (murmur Present only at the annotated murmur locations; higher grades, Late-systolic and Crescendo merged into class 2). Both distributions are printed in Section 4.

The constants that follow the flags are the training notebook's own constants, copied verbatim.
""")

code(r'''
# MANUAL OVERRIDE: run flags
DRY_RUN = False                 # True: 40 recordings, 3 epochs, 2 folds, AMP off, 200 bootstrap resamples
LIMIT_RECORDINGS = None         # e.g. 300: recording subset with full epochs and folds (smoke test only)
LABEL_MODE = 'patient'          # 'patient' = coding of the gated arms (default); 'location' = original M3 coding
RUN_DIR = None                  # pin a gating-campaign run folder; None = newest run under GATING_ROOT
DRIVE_ROOT = '/content/drive/MyDrive'
FOLD_SOURCE = 'gating'          # 'gating' = folds exported by R1_Gating_Campaign (required for the article)
                                # 'self'   = StratifiedGroupKFold drawn in this notebook (smoke tests only)
RUN_CONTINUITY = True           # Section 11: original 70/15/15 split re-run
USE_AMP = False                 # mixed precision off: the original trained in fp32. True is recorded under deviations
CACHE_AUG_FEATURES = True       # cache the augmented training features of every fold on Drive (resumable)
SEED = 42
N_BOOT = 2000                   # patient-level bootstrap resamples
ALLOW_LABEL_MISMATCH = False    # fold files may carry labels; stop on disagreement (False) or warn (True)
GATED_KEY_MAP = {}              # optional manual key map for the merge cell (Section 13); see README

# Environment overrides (used by the local self-test harness; harmless on Colab)
DRY_RUN = bool(int(os.environ.get('R1_DRY_RUN', '1' if DRY_RUN else '0')))
DRIVE_ROOT = os.environ.get('R1_DRIVE_ROOT', DRIVE_ROOT)
RUN_DIR = os.environ.get('R1_RUN_DIR', RUN_DIR) or None
FOLD_SOURCE = os.environ.get('R1_FOLD_SOURCE', FOLD_SOURCE)
LABEL_MODE = os.environ.get('R1_LABEL_MODE', LABEL_MODE)
if os.environ.get('R1_LIMIT_RECORDINGS'):
    LIMIT_RECORDINGS = int(os.environ['R1_LIMIT_RECORDINGS'])
if os.environ.get('R1_RUN_CONTINUITY'):
    RUN_CONTINUITY = bool(int(os.environ['R1_RUN_CONTINUITY']))
assert LABEL_MODE in ('patient', 'location'), LABEL_MODE
assert FOLD_SOURCE in ('gating', 'self'), FOLD_SOURCE

''' + VB_TRAIN_CONSTANTS + r'''

''' + VB_DATAPREP_CONFIG + r'''
CONFIG['chunk_samples'] = CONFIG['sample_rate'] * CONFIG['chunk_duration']
CONFIG['min_remainder_samples'] = CONFIG['sample_rate'] * CONFIG['min_remainder']
assert CONFIG['sample_rate'] == SAMPLE_RATE and CONFIG['chunk_samples'] == N_SAMPLES

# Derived run parameters
EPOCHS_RUN = 3 if DRY_RUN else EPOCHS
SUBSET_N = 40 if DRY_RUN else LIMIT_RECORDINGS
DRY_RUN_FOLDS = 2
N_BOOT_RUN = 200 if DRY_RUN else N_BOOT
JOINT_MAX_ITERATIONS = 300 if DRY_RUN else 10000   # verbatim MAX_ITERATIONS of the joint balancing; capped under DRY_RUN only
AMP_ENABLED = bool(USE_AMP and torch.cuda.is_available() and not DRY_RUN)
INNER_VAL_SIZE = 0.15
INNER_VAL_SEED = 42
TASKS = ['murmur', 'grading', 'timing', 'shape']
GATING_TASK = {'murmur': 'detection', 'grading': 'grading', 'timing': 'timing', 'shape': 'shape'}   # names used by R1_Gating_Campaign
CHAR_TASKS = ['grading', 'timing', 'shape']
CLASS_NAMES = {                     # integer coding of the gated arms (V5 order)
    'murmur': ['Absent', 'Present'],
    'grading': ['Grade I', 'Grade II', 'Grade III'],
    'timing': ['Holosystolic', 'Early systolic', 'Mid systolic'],
    'shape': ['Plateau', 'Decrescendo', 'Diamond'],
}
N_CLASSES = {t: len(v) for t, v in CLASS_NAMES.items()}
PUBLISHED_ORIGINAL = {'murmur_accuracy': 93.375, 'murmur_auroc': 0.9243, 'grading_accuracy': 67.412,
                      'timing_accuracy': 68.371, 'shape_accuracy': 58.786, 'murmur_test_chunks': 2551,
                      'multilabel_test_chunks': 313}
DEVIATIONS = []                     # appended as the run proceeds; written to the manifest
CACHE_EVENTS = []                   # fold caches rebuilt or refused during this run; written to the manifest


def note_deviation(item, detail):
    DEVIATIONS.append({'item': item, 'detail': detail})
    print(f"  [deviation] {item}: {detail}")


VERBATIM_BLOCKS = json.loads(r"""''' + VERBATIM_JSON + r'''""")

print(f"DRY_RUN={DRY_RUN} SUBSET_N={SUBSET_N} EPOCHS_RUN={EPOCHS_RUN} LABEL_MODE={LABEL_MODE} "
      f"FOLD_SOURCE={FOLD_SOURCE} AMP={AMP_ENABLED} N_BOOT={N_BOOT_RUN}")
print(f"MFCC {N_MFCC} x3 | n_fft {N_FFT} hop {HOP_LENGTH} n_mels {N_MELS} fmax {FMAX} | batch {BATCH_SIZE} "
      f"lr {LEARNING_RATE} wd {WEIGHT_DECAY} patience {PATIENCE} dropout {DROPOUT}")
seed_everything(SEED)
''')

md("""
## 2. Paths

The dataset is resolved exactly as the gating notebook resolves it: the complete copy (3163 wav + tsv pairs) is expected at `JBHI-Revision/October Thesis Dataset/CirCor DigiScope/.../training_data`; if it is not there, every `training_data` folder under Drive is counted and the one with the maximum number of paired files is used. `training_data.csv` is its sibling.

The fold files come from a run folder under `JBHI-Revision/R1_Gating_Campaign/` unless `RUN_DIR` is pinned. The gating notebook names its folders `<stamp>` for full runs, `<stamp>_dry` for dry runs and `<stamp>_limitN` for `LIMIT_RECORDINGS = N` runs: a full CNN run binds the newest full folder and skips every `_dry` and `_limitN` folder; a `DRY_RUN` prefers the newest `_dry` folder and falls back to the newest other one. The cell prints which folder was bound and why. A full run pinned to a `_dry` or `_limitN` folder stops, and so does a full run whose gating folder says `dry_run` or `limit_recordings` in its own `run_flags.json`.

Outputs go to `<RUN_DIR>/cnn_baseline<suffix>/`, where the suffix is empty for the article run and otherwise `_selffolds`, `_dryrun`, `_limit<N>`, `_location` (for `LABEL_MODE = 'location'`) or their combination, so a dry run, a subset run or a location-label run never touches the caches or results of the article run. `run_flags.json` (dry_run, limit_recordings, fold_source, seed, epochs, label_mode, use_amp) is written on a fresh run and compared on resume; a folder made under different flags is refused.
""")

code(r'''
with StageTimer('paths'):
    EXPECTED_TRAINING_DATA = os.path.join(
        DRIVE_ROOT, 'JBHI-Revision', 'October Thesis Dataset', 'CirCor DigiScope',
        'the-circor-digiscope-phonocardiogram-dataset-1.0.3',
        'the-circor-digiscope-phonocardiogram-dataset-1.0.3', 'training_data')
    COMPLETE_PAIRS = 3163

    def count_pairs(d):
        try:
            names = os.listdir(d)
        except OSError:
            return 0
        w = {n[:-4] for n in names if n.endswith('.wav')}
        t = {n[:-4] for n in names if n.endswith('.tsv')}
        return len(w & t)

    def resolve_training_data(root):
        if os.path.isdir(EXPECTED_TRAINING_DATA) and count_pairs(EXPECTED_TRAINING_DATA) == COMPLETE_PAIRS:
            return EXPECTED_TRAINING_DATA, COMPLETE_PAIRS, 'expected path (complete copy)'
        cands = [p for p in glob.glob(os.path.join(root, '**', 'training_data'), recursive=True) if os.path.isdir(p)]
        if not cands:
            raise FileNotFoundError(f'no training_data folder under {root}')
        scored = sorted(((count_pairs(p), p) for p in cands), key=lambda t: (-t[0], t[1]))
        for n, p in scored[:5]:
            print(f'    candidate {n:5d} pairs  {p}')
        return scored[0][1], scored[0][0], f'glob over {len(cands)} candidates, maximum paired count'

    TRAINING_DATA_PATH, N_PAIRS, DATASET_RESOLUTION = resolve_training_data(DRIVE_ROOT)
    CSV_PATH = os.path.join(os.path.dirname(TRAINING_DATA_PATH), 'training_data.csv')
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f'training_data.csv not found next to {TRAINING_DATA_PATH}')
    if N_PAIRS != COMPLETE_PAIRS:
        print(f'  WARNING: dataset copy has {N_PAIRS} paired files, the complete copy has {COMPLETE_PAIRS}')

    GATING_ROOT = os.path.join(DRIVE_ROOT, 'JBHI-Revision', 'R1_Gating_Campaign')

    def find_gating_runs(root):
        """Gating run folders under root that hold fold files: (mtime, path, is_dry). The gating notebook
        writes '<stamp>' for full runs, '<stamp>_dry' for dry runs and '<stamp>_limitN' for LIMIT_RECORDINGS
        runs. A full CNN run skips '_dry' and '_limitN' folders; a DRY_RUN ranks '_dry' folders first.
        Within each group the newest fold files win."""
        runs = []
        if not os.path.isdir(root):
            return runs
        for d in sorted(os.listdir(root)):
            fd = os.path.join(root, d, 'folds')
            files = glob.glob(os.path.join(fd, '*.npz')) + glob.glob(os.path.join(fd, '*.json'))
            if not files:
                continue
            is_dry = d.endswith('_dry')
            is_limit = re.search(r'_limit\d+$', d) is not None
            if is_dry and not DRY_RUN:
                print(f'    skipped (gating DRY_RUN folder, this is not a DRY_RUN): {os.path.join(root, d)}')
                continue
            if is_limit and not (DRY_RUN or LIMIT_RECORDINGS):
                print(f'    skipped (gating LIMIT_RECORDINGS folder, this is a full run): {os.path.join(root, d)}')
                continue
            runs.append((max(os.path.getmtime(f) for f in files), os.path.join(root, d), is_dry))
        return sorted(runs, key=lambda t: (0 if (DRY_RUN and t[2]) else 1, -t[0], t[1]))

    RUN_DIR_SOURCE = 'pinned' if RUN_DIR else 'auto (newest folds/ under GATING_ROOT)'
    if RUN_DIR is None:
        runs = find_gating_runs(GATING_ROOT)
        for mt, d, is_dry in runs[:5]:
            print(f'    gating run {datetime.fromtimestamp(mt).isoformat(timespec="minutes")}  {d}' + ('  [_dry]' if is_dry else ''))
        if runs:
            RUN_DIR = runs[0][1]
            if runs[0][2]:
                RUN_DIR_SOURCE = 'auto: newest gating DRY_RUN folder (_dry), preferred because DRY_RUN=True'
            elif DRY_RUN:
                RUN_DIR_SOURCE = 'auto: newest full gating run (no _dry folder found under GATING_ROOT)'
            else:
                RUN_DIR_SOURCE = 'auto: newest full gating run (_dry and _limitN folders skipped)'
            print(f'    bound: {RUN_DIR}  ({RUN_DIR_SOURCE})')
        elif FOLD_SOURCE == 'self' or DRY_RUN:
            RUN_DIR = os.path.join(GATING_ROOT, 'cnn_selftest_' + datetime.now().strftime('%Y%m%d_%H%M'))
            if FOLD_SOURCE != 'self':
                FOLD_SOURCE = 'self'
                note_deviation('fold source', 'no gating run with folds/ found; DRY_RUN fell back to self-drawn folds '
                               '(StratifiedGroupKFold). Not valid for the article.')
        else:
            raise FileNotFoundError(
                f'No gating-campaign run with a folds/ folder under {GATING_ROOT}. Run R1_Gating_Campaign first, '
                f'pin RUN_DIR, or set FOLD_SOURCE="self" for a smoke test.')
    FOLDS_DIR = os.path.join(RUN_DIR, 'folds')
    _run_name = os.path.basename(os.path.normpath(RUN_DIR))
    FULL_CNN_RUN = not (DRY_RUN or LIMIT_RECORDINGS)
    if _run_name.endswith('_dry') and not DRY_RUN:
        raise RuntimeError(f'RUN_DIR {RUN_DIR} is a gating DRY_RUN folder (a reduced recording set, about 120 recordings). '
                           f'A full CNN run needs a full gating run: pin RUN_DIR to a folder without the _dry suffix, '
                           f'or set DRY_RUN = True.')
    if re.search(r'_limit\d+$', _run_name) and FULL_CNN_RUN:
        raise RuntimeError(f'RUN_DIR {RUN_DIR} is a gating LIMIT_RECORDINGS folder. A full CNN run needs a full gating run: '
                           f'pin RUN_DIR to a folder without the _limitN suffix.')
    # the gating run's own flags: a full CNN run must sit on a full gating run, whatever the folder is called
    GATING_RUN_FLAGS = None
    if FOLD_SOURCE == 'gating':
        _gating_flags_path = os.path.join(RUN_DIR, 'run_flags.json')
        if os.path.isfile(_gating_flags_path):
            with open(_gating_flags_path) as fh:
                GATING_RUN_FLAGS = json.load(fh)
            print(f'  gating run_flags.json: dry_run={GATING_RUN_FLAGS.get("dry_run")} '
                  f'limit_recordings={GATING_RUN_FLAGS.get("limit_recordings")} seed={GATING_RUN_FLAGS.get("seed")} '
                  f'n_splits={GATING_RUN_FLAGS.get("n_splits")}')
            if FULL_CNN_RUN and (GATING_RUN_FLAGS.get('dry_run') or GATING_RUN_FLAGS.get('limit_recordings')):
                raise RuntimeError(f'RUN_DIR {RUN_DIR} is not a full gating run (its run_flags.json holds '
                                   f'dry_run={GATING_RUN_FLAGS.get("dry_run")}, '
                                   f'limit_recordings={GATING_RUN_FLAGS.get("limit_recordings")}). '
                                   f'Pin RUN_DIR to the full gating run.')
        else:
            print(f'  note: no run_flags.json in {RUN_DIR}; the gating run kind is judged from the folder name '
                  f'and from the kept-set sizes checked in Section 5')
    # Separate output folders per run kind, so a dry run, a subset run or a location-label run never shares caches or
    # results with the article run
    _suffix = (('' if FOLD_SOURCE == 'gating' else '_selffolds') + ('_dryrun' if DRY_RUN else '')
               + (f'_limit{LIMIT_RECORDINGS}' if LIMIT_RECORDINGS else '')
               + ('' if LABEL_MODE == 'patient' else '_location'))
    OUT_DIR = os.path.join(RUN_DIR, 'cnn_baseline' + _suffix)
    CACHE_DIR = os.path.join(OUT_DIR, 'cache')
    MODELS_DIR = os.path.join(OUT_DIR, 'models')
    FIG_DIR = os.path.join(OUT_DIR, 'figures')
    for d in (OUT_DIR, CACHE_DIR, MODELS_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)

    # run flags: written on a fresh run, compared on resume; a folder produced under other flags is refused
    RUN_FLAGS = {'dry_run': bool(DRY_RUN), 'limit_recordings': LIMIT_RECORDINGS, 'fold_source': FOLD_SOURCE,
                 'seed': int(SEED), 'epochs': int(EPOCHS_RUN), 'label_mode': LABEL_MODE, 'use_amp': bool(USE_AMP)}
    RUN_FLAGS_PATH = os.path.join(OUT_DIR, 'run_flags.json')
    if os.path.isfile(RUN_FLAGS_PATH):
        with open(RUN_FLAGS_PATH) as fh:
            _prev_flags = json.load(fh)
        _flag_diff = {k: {'stored': _prev_flags.get(k), 'current': v} for k, v in RUN_FLAGS.items() if _prev_flags.get(k) != v}
        if _flag_diff:
            raise RuntimeError(f'{OUT_DIR} holds a run made under different flags: {_flag_diff}. Resuming would mix two '
                               f'runs. Move or delete that folder, or restore the stored flags.')
        RUN_FLAGS_STATUS = 'resume: run_flags.json equals the current flags'
    else:
        with open(RUN_FLAGS_PATH, 'w') as fh:
            json.dump(RUN_FLAGS, fh, indent=1)
        RUN_FLAGS_STATUS = 'fresh run: run_flags.json written'
    print(f'  {RUN_FLAGS_STATUS}: {RUN_FLAGS}')

    ORIGINAL_SPLITS_JSON = os.environ.get('R1_ORIGINAL_SPLITS_JSON', os.path.join(
        DRIVE_ROOT, 'October Thesis Dataset', 'CirCor DigiScope', 'Processed-Data', 'patient_splits.json'))
    ORIGINAL_CHUNK_META = os.environ.get('R1_ORIGINAL_CHUNK_META', os.path.join(
        DRIVE_ROOT, 'October Thesis Dataset', 'CirCor DigiScope', 'Processed-Data', 'chunk_metadata.csv'))

    print(f'TRAINING_DATA_PATH = {TRAINING_DATA_PATH}  ({N_PAIRS} pairs; {DATASET_RESOLUTION})')
    print(f'CSV_PATH           = {CSV_PATH}')
    print(f'RUN_DIR            = {RUN_DIR}  ({RUN_DIR_SOURCE})')
    print(f'FOLDS_DIR          = {FOLDS_DIR}  (source: {FOLD_SOURCE})')
    print(f'OUT_DIR            = {OUT_DIR}')
    print(f'original split file: {ORIGINAL_SPLITS_JSON}  exists={os.path.isfile(ORIGINAL_SPLITS_JSON)}')
''')

md("""
## 3. Verbatim preprocessing, MFCC extractor and augmentation

The four data-preparation functions (`load_audio`, `bandpass_filter`, `normalize_zscore`, `create_chunks_circular`), the MFCC extractor (`extract_mfcc_from_array`), the four augmentations with `augment_audio`, and the joint-balancing scorers are copied verbatim by the builder. The bandpass design is checked: if any pole of the `ba` design has radius at or above 1 the equivalent second-order-sections design is used instead and the manifest records it. Any recording whose `ba` output is not finite also falls back to `sos`.
""")

code(VB_DATAPREP_FUNCS + r'''


# Filter stability check (design brief, section 1.3)
def butter_pole_check(sr, low_freq, high_freq, order):
    nyquist = 0.5 * sr
    b, a = signal.butter(order, [low_freq / nyquist, high_freq / nyquist], btype='band')
    sos = signal.butter(order, [low_freq / nyquist, high_freq / nyquist], btype='band', output='sos')
    radii = np.abs(np.roots(a))
    return b, a, sos, float(radii.max()), radii


BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER = CONFIG['bandpass_low'], CONFIG['bandpass_high'], CONFIG['filter_order']
FILTER_B, FILTER_A, FILTER_SOS, FILTER_MAX_POLE_RADIUS, FILTER_POLE_RADII = butter_pole_check(
    SAMPLE_RATE, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
FILTER_DESIGN = 'ba' if FILTER_MAX_POLE_RADIUS < 1.0 else 'sos'
if FILTER_DESIGN == 'sos':
    note_deviation('bandpass design', f'ba design unstable (max pole radius {FILTER_MAX_POLE_RADIUS:.6f}); '
                   f'equivalent sos design used for every recording')
print(f'Butterworth order {FILTER_ORDER}, {BANDPASS_LOW}-{BANDPASS_HIGH} Hz at {SAMPLE_RATE} Hz: '
      f'max pole radius {FILTER_MAX_POLE_RADIUS:.6f} -> design used: {FILTER_DESIGN}')


def bandpass_filter_checked(audio, sr, low_freq, high_freq, order=4):
    """Verbatim ba path when stable; sos otherwise. Returns (filtered, design_used)."""
    if FILTER_DESIGN == 'ba':
        out = bandpass_filter(audio, sr, low_freq, high_freq, order)
        if np.all(np.isfinite(out)):
            return out, 'ba'
    out = signal.sosfiltfilt(FILTER_SOS, audio).astype(np.float32)
    return out, ('sos' if FILTER_DESIGN == 'sos' else 'sos_fallback_nonfinite')


# Unit checks of the chunker on synthetic signals
_t = np.random.default_rng(0).standard_normal(int(10.0 * SAMPLE_RATE)).astype(np.float32)
_c = create_chunks_circular(_t, CONFIG['chunk_samples'], CONFIG['min_remainder_samples'])
assert len(_c) == 3 and all(len(x) == N_SAMPLES for x in _c), 'chunker: 10 s must give 2 full + 1 padded chunk'
assert np.array_equal(_c[2][:int(2.0 * SAMPLE_RATE)], _t[int(8.0 * SAMPLE_RATE):]) and \
       np.array_equal(_c[2][int(2.0 * SAMPLE_RATE):], _t[:N_SAMPLES - int(2.0 * SAMPLE_RATE)]), 'circular padding'
_c = create_chunks_circular(_t[:int(9.9 * SAMPLE_RATE)], CONFIG['chunk_samples'], CONFIG['min_remainder_samples'])
assert len(_c) == 2, 'chunker: remainder below 2 s must be dropped'
_f, _d = bandpass_filter_checked(_t, SAMPLE_RATE, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER)
assert _f.dtype == np.float32 and np.all(np.isfinite(_f))
print('preprocessing functions ready; chunker and filter checks passed')
''')

code(VB_MFCC + r'''


def f16_roundtrip(feats):
    """Round-trip a batch of feature maps through the on-disk representation so a fresh run and a
    resumed run see bit-identical inputs. Maps are stored standardised in float16 plus per-map mean
    and std in float32 (about 5x smaller quantisation error than raw float16)."""
    feats = np.asarray(feats, dtype=np.float32)
    mean = feats.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    std = feats.std(axis=(1, 2), dtype=np.float64).astype(np.float32) + np.float32(1e-8)
    z16 = ((feats - mean[:, None, None]) / std[:, None, None]).astype(np.float16)
    return z16, mean, std


def f16_restore(z16, mean, std):
    return (z16.astype(np.float32) * std[:, None, None] + mean[:, None, None]).astype(np.float32)


def mfcc_list(chunks):
    out = []
    for c in chunks:
        f = extract_mfcc_from_array(c)
        if f is None or f.shape != (N_MFCC * 3, N_FRAMES):
            raise RuntimeError(f'MFCC extraction failed or wrong shape: {None if f is None else f.shape}')
        out.append(f.astype(np.float32))
    return np.stack(out)


def mfcc_parallel(chunks, n_jobs=None, batch=256):
    """MFCC maps for a list/array of 4 s chunks, parallel across CPU cores (deterministic: no RNG)."""
    n_jobs = n_jobs or N_CPU
    chunks = list(chunks)
    if len(chunks) == 0:
        return np.zeros((0, N_MFCC * 3, N_FRAMES), dtype=np.float32)
    if len(chunks) <= batch or n_jobs == 1:
        return mfcc_list(chunks)
    parts = Parallel(n_jobs=n_jobs)(delayed(mfcc_list)(chunks[i:i + batch]) for i in range(0, len(chunks), batch))
    return np.concatenate(parts, axis=0)


_probe = extract_mfcc_from_array(np.random.default_rng(1).standard_normal(N_SAMPLES).astype(np.float32))
N_FRAMES = _probe.shape[1]
assert _probe.shape == (N_MFCC * 3, 1 + N_SAMPLES // HOP_LENGTH), _probe.shape
_z, _m, _s = f16_roundtrip(_probe[None])
F16_ROUNDTRIP_ERR_STD_UNITS = float(np.abs(f16_restore(_z, _m, _s)[0] - _probe).max() / _probe.std())
print(f'MFCC map per chunk: {_probe.shape} (N_FRAMES={N_FRAMES}); float16 round-trip max error '
      f'{F16_ROUNDTRIP_ERR_STD_UNITS:.2e} map-std units')
''')

code(VB_AUG + r'''


''' + VB_JOINT + r'''


# Sanity: every augmentation preserves the chunk length
_a = np.random.default_rng(2).standard_normal(N_SAMPLES).astype(np.float32)
for _ci in range(len(AUGMENTATION_COMBOS)):
    _o = augment_audio(_a, combo_idx=_ci)
    assert _o.shape == _a.shape and _o.dtype == np.float32, (_ci, _o.shape, _o.dtype)
print(f'augmentation functions ready: {len(AUGMENTATION_COMBOS)} combos; joint-balancing scorers ready')
''')

md("""
## 4. Recording index and labels

Recordings are listed from the wav files (sorted), parsed with the data-preparation notebook's rule (`patient_id = parts[0]`, `location = parts[-1]`), and merged with `training_data.csv`. Two label codings are computed for every recording: the patient-level coding of the gated arms (verbatim `get_recording_labels` from the Milestone-4 feature notebook that fed V5) and the original location-aware coding (verbatim functions from the Milestone-3 data preparation). Both distributions are printed; `LABEL_MODE` selects the coding used for training and scoring.
""")

code(VB_PATIENT_LABELS + r'''


''' + VB_LOCATION_LABELS + r'''


with StageTimer('labels'):
    df_patients = pd.read_csv(CSV_PATH)
    wav_files = sorted(f for f in os.listdir(TRAINING_DATA_PATH) if f.endswith('.wav'))
    tsv_stems = {f[:-4] for f in os.listdir(TRAINING_DATA_PATH) if f.endswith('.tsv')}
    rows = []
    for wav_file in wav_files:
        parts = wav_file.replace('.wav', '').split('_')
        patient_id = int(parts[0])
        location = parts[-1] if len(parts) >= 2 else None
        rows.append({'rec': wav_file[:-4], 'patient_id': patient_id, 'filename': wav_file,
                     'filepath': os.path.join(TRAINING_DATA_PATH, wav_file), 'location': location,
                     'has_tsv': wav_file[:-4] in tsv_stems})
    df_rec = pd.DataFrame(rows).merge(df_patients, left_on='patient_id', right_on='Patient ID', how='left')
    assert df_rec['Patient ID'].notna().all(), 'every recording must map to a patient row'
    assert df_rec['rec'].is_unique

    # patient-level coding (V5 / gated arms)
    _lab = [get_recording_labels(r, df_patients) for r in df_rec['rec']]
    for t in TASKS:
        df_rec[f'{t}_patient'] = np.array([(-1 if d is None else d[t]) for d in _lab], dtype=int)

    # location-aware coding (original Milestone-3)
    _m, _g, _ti, _s = [], [], [], []
    for _, row in df_rec.iterrows():
        m = is_murmur_audible_at_location(row['Murmur'], row['Murmur locations'], row['location'])
        chars = get_murmur_characteristics(row, row['location'])
        _m.append(-1 if m is None else int(m))
        _g.append(-1 if chars['grading'] is None else int(chars['grading']))
        _ti.append(-1 if chars['timing'] is None else int(chars['timing']))
        _s.append(-1 if chars['shape'] is None else int(chars['shape']))
    df_rec['murmur_location'] = np.array(_m, dtype=int)
    df_rec['grading_location'] = np.array(_g, dtype=int)
    df_rec['timing_location'] = np.array(_ti, dtype=int)
    df_rec['shape_location'] = np.array(_s, dtype=int)

    odd_loc = df_rec[~df_rec['location'].isin(['AV', 'PV', 'TV', 'MV', 'Phc'])]
    print(f'{len(df_rec)} recordings from {df_rec["patient_id"].nunique()} patients; '
          f'{int(df_rec["has_tsv"].sum())} with a tsv annotation')
    if len(odd_loc):
        print(f'  note: {len(odd_loc)} recordings have a parsed location outside AV/PV/TV/MV/Phc '
              f'(numbered repeats such as _MV_1); the original location-aware rule treats them as not audible: '
              f'{sorted(odd_loc["rec"].tolist())[:8]} ...')

    LABEL_DISTRIBUTIONS = {}
    for mode in ('patient', 'location'):
        LABEL_DISTRIBUTIONS[mode] = {}
        print(f'\n--- label coding: {mode} ---')
        for t in TASKS:
            y = df_rec[f'{t}_{mode}'].values
            valid = y >= 0
            rec_counts = {CLASS_NAMES[t][c]: int((y == c).sum()) for c in range(N_CLASSES[t])}
            pat_counts = {CLASS_NAMES[t][c]: int(df_rec.loc[y == c, 'patient_id'].nunique()) for c in range(N_CLASSES[t])}
            LABEL_DISTRIBUTIONS[mode][t] = {'recordings': rec_counts, 'patients': pat_counts,
                                            'n_recordings_valid': int(valid.sum()),
                                            'n_patients_valid': int(df_rec.loc[valid, 'patient_id'].nunique())}
            print(f'  {t:8s} valid recordings {int(valid.sum()):5d} | recordings {rec_counts} | patients {pat_counts}')

    LABELS = {t: df_rec[f'{t}_{LABEL_MODE}'].values.astype(int) for t in TASKS}
    REC_PATIENT = dict(zip(df_rec['rec'], df_rec['patient_id'].astype(int)))
    REC_ROW = {r: i for i, r in enumerate(df_rec['rec'])}
    REC_LABEL = {mode: {t: dict(zip(df_rec['rec'], df_rec[f'{t}_{mode}'].astype(int))) for t in TASKS}
                 for mode in ('patient', 'location')}
    print(f'\nactive LABEL_MODE = {LABEL_MODE}')
''')

md("""
## 5. Folds from the gating campaign

The gating notebook writes four fold files, `folds/detection.npz`, `folds/grading.npz`, `folds/timing.npz` and `folds/shape.npz` (it calls the murmur task `detection`; `GATING_TASK` maps the names). Each holds `names`, `patient_ids`, `y`, `fold_id`, `n_splits` and the `train_NN` / `test_NN` index arrays into `names`. The join is by recording name, never by position. Assertions: every test recording exists in this notebook's index and has a valid label under the active coding; no patient appears in both train and test of any fold; test folds are pairwise disjoint; if the fold file carries labels they must equal ours. Under `DRY_RUN` the ten folds are merged into two super-folds (test = union of five test folds, train = the rest) on a 40-recording subset drawn from the fold file's kept set, so the loader and the join are exercised.
""")

code(r'''
def _norm_rec(s):
    s = os.path.basename(str(s))
    return s[:-4] if s.lower().endswith('.wav') else s


def _first_key(keys, candidates):
    low = {k.lower(): k for k in keys}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def load_fold_file(path):
    """Tolerant reader for one task's fold file. Returns dict(rec, pid, labels, folds=[(train_recs, test_recs)])."""
    if path.endswith('.json'):
        with open(path) as fh:
            d = json.load(fh)
        arrays = {k: np.asarray(v) for k, v in d.items() if not isinstance(v, (dict, list)) or k != 'folds'}
        if 'folds' in d and isinstance(d['folds'], list):
            for k, fd in enumerate(d['folds']):
                if isinstance(fd, dict):
                    arrays[f'train_idx_{k}'] = np.asarray(fd.get('train', fd.get('train_idx')))
                    arrays[f'test_idx_{k}'] = np.asarray(fd.get('test', fd.get('test_idx')))
                else:
                    arrays[f'train_idx_{k}'] = np.asarray(fd[0])
                    arrays[f'test_idx_{k}'] = np.asarray(fd[1])
    else:
        z = np.load(path, allow_pickle=True)
        arrays = {k: z[k] for k in z.keys()}
    keys = list(arrays.keys())

    name_key = _first_key(keys, ['recording_names', 'rec_names', 'recordings', 'recording_ids', 'record_names',
                                 'names', 'recording', 'rec', 'ids', 'recording_name'])
    if name_key is None:
        for k in keys:
            a = np.asarray(arrays[k])
            if a.dtype.kind in ('U', 'S', 'O') and a.ndim == 1 and 'patient' not in k.lower():
                name_key = k
                break
    if name_key is None:
        raise ValueError(f'{path}: no recording-name array found; keys={keys}')
    rec = np.array([_norm_rec(x) for x in np.asarray(arrays[name_key]).ravel()])

    pid_key = next((k for k in keys if 'patient' in k.lower() and 'fold' not in k.lower()), None)
    pid = np.asarray(arrays[pid_key]).ravel() if pid_key else None
    lab_key = _first_key(keys, ['labels', 'y', 'y_true', 'label', 'targets', 'target'])
    labels = np.asarray(arrays[lab_key]).ravel().astype(int) if lab_key else None

    folds = {}
    pat = re.compile(r'^(?:fold[_-]?(\d+)[_-]?(train|test|val)(?:[_-]?(?:idx|indices|index|ids))?'
                     r'|(train|test|val)(?:[_-]?(?:idx|indices|index|ids))?[_-]?(\d+))$', re.I)
    for k in keys:
        m = pat.match(k)
        if m:
            fold_no = int(m.group(1) or m.group(4))
            part = (m.group(2) or m.group(3)).lower()
            folds.setdefault(fold_no, {})[part] = np.asarray(arrays[k]).ravel()
    if not folds:
        tr_key = _first_key(keys, ['train_idx', 'train_indices', 'train'])
        te_key = _first_key(keys, ['test_idx', 'test_indices', 'test'])
        if tr_key and te_key:
            for k, (a, b) in enumerate(zip(arrays[tr_key], arrays[te_key])):
                folds[k] = {'train': np.asarray(a).ravel(), 'test': np.asarray(b).ravel()}
        elif 'folds' in arrays:
            for k, fd in enumerate(arrays['folds']):
                if isinstance(fd, dict):
                    folds[k] = {'train': np.asarray(fd['train']).ravel(), 'test': np.asarray(fd['test']).ravel()}
                else:
                    folds[k] = {'train': np.asarray(fd[0]).ravel(), 'test': np.asarray(fd[1]).ravel()}
    if not folds:
        raise ValueError(f'{path}: no per-fold train/test arrays found; keys={keys}')

    out = []
    for k in sorted(folds):
        fd = folds[k]
        if 'test' not in fd and 'val' in fd:
            fd['test'] = fd['val']
            print(f'  note: fold {k} has no test array; using its val array as the held-out fold')
        if 'train' not in fd or 'test' not in fd:
            raise ValueError(f'{path}: fold {k} incomplete: {list(fd)}')

        def to_names(a):
            """Recording names of one fold part: names as given, a boolean mask over rec, or integer indices into rec
            (the gating schema). No guessing between masks and indices from the values."""
            a = np.asarray(a)
            if a.dtype.kind in ('U', 'S', 'O'):
                return np.array([_norm_rec(x) for x in a])
            if a.dtype.kind == 'b':
                assert len(a) == len(rec), f'{path}: boolean fold mask of length {len(a)} for {len(rec)} recordings'
                return rec[a]
            a = a.astype(int)
            assert a.min() >= 0 and a.max() < len(rec), f'{path}: fold index out of range for {len(rec)} recordings'
            return rec[a]
        out.append((to_names(fd['train']), to_names(fd['test'])))
    return {'rec': rec, 'pid': pid, 'labels': labels, 'folds': out, 'path': path,
            'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}


def draw_self_folds(task, n_splits):
    mask = LABELS[task] >= 0
    recs = df_rec['rec'].values[mask]
    y = LABELS[task][mask]
    groups = df_rec['patient_id'].values[mask]
    n_splits = int(min(n_splits, len(np.unique(groups))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    self_dir = os.path.join(OUT_DIR, 'folds_self')
    os.makedirs(self_dir, exist_ok=True)
    path = os.path.join(self_dir, f'{task}.npz')
    payload = {'recording_names': recs, 'patient_ids': groups, 'labels': y, 'n_folds': np.array(n_splits),
               'task': np.array(task), 'label_mode': np.array(LABEL_MODE)}
    for k, (tr, te) in enumerate(folds):
        payload[f'train_idx_{k}'] = tr
        payload[f'test_idx_{k}'] = te
    np.savez(path, **payload)
    return path


def check_fold_plan(task, plan, eval_recs):
    eval_set = set(eval_recs)
    seen_test = {}
    for fp in plan:
        tr, te = fp['train_recs'], fp['test_recs']
        assert len(te) > 0, f'{task} fold {fp["fold"]}: empty test fold'
        missing = [r for r in te if r not in REC_ROW]
        assert not missing, f'{task} fold {fp["fold"]}: {len(missing)} test recordings not in this index, e.g. {missing[:5]}'
        assert set(te) <= eval_set, f'{task} fold {fp["fold"]}: test recordings outside the evaluation set'
        p_tr = {REC_PATIENT[r] for r in tr}
        p_te = {REC_PATIENT[r] for r in te}
        overlap = p_tr & p_te
        assert not overlap, f'{task} fold {fp["fold"]}: {len(overlap)} patients in both train and test, e.g. {sorted(overlap)[:5]}'
        assert not (set(tr) & set(te)), f'{task} fold {fp["fold"]}: recordings in both train and test'
        for r in te:
            assert r not in seen_test, f'{task}: recording {r} in test folds {seen_test[r]} and {fp["fold"]}'
            seen_test[r] = fp['fold']
    return len(seen_test)


FOLD_FILES = {}
FOLD_PLAN = {}
EVAL_RECS = {}
FOLD_INTEGRITY = {}
with StageTimer('folds'):
    for task in TASKS:
        if FOLD_SOURCE == 'self':
            path = draw_self_folds(task, 2 if DRY_RUN else 10)
        else:
            gt = GATING_TASK[task]          # the gating notebook names the murmur task 'detection'
            cands = [os.path.join(FOLDS_DIR, f) for f in (f'{gt}.npz', f'{gt}.json', f'{task}.npz', f'folds_{gt}.npz')]
            path = next((p for p in cands if os.path.isfile(p)), None)
            if path is None:
                raise FileNotFoundError(f'no fold file for task {task} (gating name {gt}) in {FOLDS_DIR} (tried {cands})')
        ff = load_fold_file(path)
        FOLD_FILES[task] = ff
        kept = ff['rec']
        in_index = np.array([r in REC_ROW for r in kept])
        if not in_index.all():
            bad = kept[~in_index]
            raise AssertionError(f'{task}: {len(bad)} kept recordings of the fold file are not in this index, e.g. {bad[:5]}')
        ours = np.array([REC_LABEL[LABEL_MODE][task][r] for r in kept])
        if ff['pid'] is not None:
            pid_ok = np.array([str(REC_PATIENT[r]) == str(p).split('_')[0] for r, p in zip(kept, ff['pid'])])
            assert pid_ok.all(), f'{task}: patient ids of the fold file disagree with ours for {int((~pid_ok).sum())} recordings'
        label_note = 'no labels in fold file'
        if ff['labels'] is not None:
            mism = int((ff['labels'] != ours).sum())
            label_note = f'{mism} label disagreements out of {len(kept)}'
            if mism and LABEL_MODE == 'location':
                print(f'  {task}: {label_note} (expected: fold file uses the patient coding, LABEL_MODE=location)')
                note_deviation('label coding', f'{task}: LABEL_MODE=location differs from the fold file labels for {mism} recordings')
            elif mism and not ALLOW_LABEL_MISMATCH:
                raise AssertionError(f'{task}: {label_note}; set ALLOW_LABEL_MISMATCH=True to proceed')
            elif mism:
                note_deviation('label coding', f'{task}: {label_note} (ALLOW_LABEL_MISMATCH)')
        valid = ours >= 0
        dropped_invalid = kept[~valid]
        if len(dropped_invalid):
            print(f'  {task}: {len(dropped_invalid)} kept recordings have no valid label under LABEL_MODE={LABEL_MODE}; '
                  f'excluded from training and scoring')
        eval_recs = kept[valid]
        plan = []
        for k, (tr, te) in enumerate(ff['folds']):
            plan.append({'fold': k, 'train_recs': np.array([r for r in tr if REC_LABEL[LABEL_MODE][task].get(r, -1) >= 0]),
                         'test_recs': np.array([r for r in te if REC_LABEL[LABEL_MODE][task].get(r, -1) >= 0])})
        FOLD_PLAN[task] = plan
        EVAL_RECS[task] = eval_recs
        FOLD_INTEGRITY[task] = {'path': path, 'sha256': ff['sha256'], 'n_kept_in_file': int(len(kept)),
                                'n_eval': int(len(eval_recs)), 'n_folds': len(plan), 'labels': label_note,
                                'extra_in_index_not_in_file': int((LABELS[task] >= 0).sum() - len(kept) + (~valid).sum())}
        if not (DRY_RUN or LIMIT_RECORDINGS or FOLD_SOURCE == 'self'):
            # a full run must see the full kept sets (about 3,006 detection; 612 / 608 / 605 characterisation)
            min_kept = 2500 if task == 'murmur' else 500
            assert len(kept) >= min_kept, (f'{task}: fold file holds only {len(kept)} recordings (expected at least {min_kept}); '
                                           f'is RUN_DIR a _dry gating run?')
        print(f'{task:8s}: {os.path.basename(path)} kept {len(kept)} recordings, {len(plan)} folds, {label_note}')

    # Subset for DRY_RUN / LIMIT_RECORDINGS: drawn from the murmur kept set, both classes, patient-wise
    SUBSET = None
    if SUBSET_N:
        rng = np.random.default_rng(SEED)
        base = EVAL_RECS['murmur']
        lab = np.array([REC_LABEL[LABEL_MODE]['murmur'][r] for r in base])
        allchar = np.array([all(REC_LABEL[LABEL_MODE][t][r] >= 0 for t in CHAR_TASKS) for r in base])
        pos_pats = np.unique([REC_PATIENT[r] for r in base[(lab == 1) & allchar]])
        neg_pats = np.unique([REC_PATIENT[r] for r in base[lab == 0]])
        rng.shuffle(pos_pats)
        rng.shuffle(neg_pats)
        chosen = []
        for pats, quota in ((pos_pats, SUBSET_N // 2), (neg_pats, SUBSET_N - SUBSET_N // 2)):
            got = []
            for p in pats:
                got.extend([r for r in base if REC_PATIENT[r] == p and r not in got])
                if len(got) >= quota:
                    break
            chosen.extend(got[:quota])
        SUBSET = set(chosen)
        print(f'\nSUBSET of {len(SUBSET)} recordings from {len({REC_PATIENT[r] for r in SUBSET})} patients '
              f'(Present {sum(REC_LABEL[LABEL_MODE]["murmur"][r] == 1 for r in SUBSET)})')
        for task in TASKS:
            EVAL_RECS[task] = np.array([r for r in EVAL_RECS[task] if r in SUBSET])
            plan = FOLD_PLAN[task]
            if DRY_RUN and len(plan) > DRY_RUN_FOLDS:
                merged = []
                per = int(math.ceil(len(plan) / DRY_RUN_FOLDS))
                eval_set = set(EVAL_RECS[task])
                for g in range(DRY_RUN_FOLDS):
                    te = [r for fp in plan[g * per:(g + 1) * per] for r in fp['test_recs'] if r in eval_set]
                    te_set = set(te)
                    tr = [r for r in EVAL_RECS[task] if r not in te_set]
                    merged.append({'fold': g, 'train_recs': np.array(tr), 'test_recs': np.array(te)})
                plan = merged
                print(f'  {task}: {len(FOLD_PLAN[task])} folds merged into {DRY_RUN_FOLDS} super-folds for DRY_RUN')
            else:
                plan = [{'fold': fp['fold'], 'train_recs': np.array([r for r in fp['train_recs'] if r in SUBSET]),
                         'test_recs': np.array([r for r in fp['test_recs'] if r in SUBSET])} for fp in plan]
            plan = [fp for fp in plan if len(fp['test_recs']) > 0 and len(fp['train_recs']) > 0]
            FOLD_PLAN[task] = plan

    for task in TASKS:
        n_tested = check_fold_plan(task, FOLD_PLAN[task], EVAL_RECS[task])
        FOLD_INTEGRITY[task]['n_recordings_in_test_folds'] = int(n_tested)
        FOLD_INTEGRITY[task]['n_folds_run'] = len(FOLD_PLAN[task])
        if n_tested != len(EVAL_RECS[task]):
            print(f'  WARNING {task}: {n_tested} recordings appear in test folds but the evaluation set has {len(EVAL_RECS[task])}')
        sizes = [(len(fp['train_recs']), len(fp['test_recs'])) for fp in FOLD_PLAN[task]]
        print(f'{task:8s}: {len(FOLD_PLAN[task])} folds; (train, test) recordings per fold: {sizes}')
    print('fold integrity checks passed: patient-disjoint, test folds disjoint, all test recordings present')
''')

md("""
## 6. Chunk cache

Every kept recording is processed once with the verbatim pipeline: 4 kHz, Butterworth 25 to 400 Hz zero-phase, per-recording z-score, non-overlapping 4 s chunks with the trailing remainder kept and circularly padded when at least 2 s, then the 120 x 101 MFCC + delta + delta-delta map per chunk. Maps are stored standardised in float16 with per-map mean and std in float32 (Drive) and restored to float32 in RAM, so a fresh run and a resumed run see identical inputs. The z-scored audio of every chunk from a Murmur = Present patient is kept in float32 for the augmentation stage. Work is parallel across CPU cores and resumable through a content hash.
""")

code(r'''
def process_one_recording(rec, filepath, keep_audio):
    """Verbatim preprocessing chain of MILESTONE3_DATA_PREPARATION cell 26 for one recording."""
    audio, sr = load_audio(filepath, CONFIG['sample_rate'])
    if audio is None:
        return {'rec': rec, 'error': 'load failed'}
    audio, design = bandpass_filter_checked(audio, sr, CONFIG['bandpass_low'], CONFIG['bandpass_high'],
                                            CONFIG['filter_order'])
    audio = normalize_zscore(audio)
    chunks = create_chunks_circular(audio, CONFIG['chunk_samples'], CONFIG['min_remainder_samples'])
    if len(chunks) == 0:
        return {'rec': rec, 'error': f'no chunk (length {len(audio) / sr:.2f} s)'}
    feats = mfcc_list(chunks)
    z16, mean, std = f16_roundtrip(feats)
    return {'rec': rec, 'n_samples': int(len(audio)), 'n_chunks': len(chunks), 'z16': z16, 'mean': mean,
            'std': std, 'audio': (np.stack(chunks).astype(np.float32) if keep_audio else None),
            'filter_design': design}


with StageTimer('chunk_cache'):
    KEPT_ALL = sorted(set().union(*[set(EVAL_RECS[t]) for t in TASKS]))
    PRESENT_PATIENT_RECS = set(df_rec.loc[df_rec['Murmur'] == 'Present', 'rec'])
    cache_key_src = json.dumps({'recs': KEPT_ALL, 'sr': SAMPLE_RATE, 'n_fft': N_FFT, 'hop': HOP_LENGTH,
                                'n_mels': N_MELS, 'fmax': FMAX, 'n_mfcc': N_MFCC, 'chunk': CONFIG['chunk_samples'],
                                'min_rem': CONFIG['min_remainder_samples'], 'band': [BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER],
                                'design': FILTER_DESIGN, 'librosa': librosa.__version__, 'v': 2})
    CACHE_KEY = hashlib.sha256(cache_key_src.encode()).hexdigest()[:16]
    cp = lambda name: os.path.join(CACHE_DIR, f'chunks_{CACHE_KEY}_{name}')
    meta_path, z_path, stats_path, audio_path, table_path = (cp('meta.json'), cp('mfcc_z_f16.npy'),
                                                             cp('mfcc_stats_f32.npy'), cp('audio_present_f32.npy'),
                                                             cp('table.csv'))
    if all(os.path.isfile(p) for p in (meta_path, z_path, stats_path, audio_path, table_path)):
        with open(meta_path) as fh:
            CHUNK_META = json.load(fh)
        CT = pd.read_csv(table_path)
        z16 = np.load(z_path)
        st = np.load(stats_path)
        MFCC = f16_restore(z16, st[:, 0], st[:, 1])
        AUDIO_PRESENT = np.load(audio_path)
        del z16
        print(f'chunk cache {CACHE_KEY} loaded from Drive: {len(CT)} chunks, {len(KEPT_ALL)} recordings')
    else:
        print(f'building chunk cache {CACHE_KEY} for {len(KEPT_ALL)} recordings on {N_CPU} cores ...')
        t0 = time.time()
        results = Parallel(n_jobs=N_CPU, batch_size=4, verbose=5)(
            delayed(process_one_recording)(rec, df_rec.loc[REC_ROW[rec], 'filepath'], rec in PRESENT_PATIENT_RECS)
            for rec in KEPT_ALL)
        errors = [r for r in results if 'error' in r]
        if errors:
            raise RuntimeError(f'{len(errors)} recordings failed: {errors[:5]}')
        rows, z_parts, m_parts, s_parts, a_parts = [], [], [], [], []
        a_row = 0
        designs = {}
        for r in results:
            designs[r['filter_design']] = designs.get(r['filter_design'], 0) + 1
            for ci in range(r['n_chunks']):
                rows.append({'rec': r['rec'], 'patient_id': REC_PATIENT[r['rec']], 'chunk_idx': ci,
                             'n_chunks': r['n_chunks'], 'audio_row': (a_row + ci) if r['audio'] is not None else -1,
                             'n_samples_rec': r['n_samples']})
            z_parts.append(r['z16']); m_parts.append(r['mean']); s_parts.append(r['std'])
            if r['audio'] is not None:
                a_parts.append(r['audio']); a_row += r['n_chunks']
        CT = pd.DataFrame(rows)
        z16 = np.concatenate(z_parts); mean = np.concatenate(m_parts); std = np.concatenate(s_parts)
        AUDIO_PRESENT = np.concatenate(a_parts) if a_parts else np.zeros((0, N_SAMPLES), dtype=np.float32)
        for mode in ('patient', 'location'):
            for t in TASKS:
                CT[f'{t}_{mode}'] = CT['rec'].map(REC_LABEL[mode][t]).astype(int)
        CHUNK_META = {'cache_key': CACHE_KEY, 'n_recordings': len(KEPT_ALL), 'n_chunks': int(len(CT)),
                      'n_audio_chunks_kept': int(len(AUDIO_PRESENT)), 'filter_designs': designs,
                      'f16_roundtrip_err_std_units': F16_ROUNDTRIP_ERR_STD_UNITS,
                      'seconds': time.time() - t0, 'built': datetime.now().isoformat(timespec='seconds')}
        np.save(z_path, z16)
        np.save(stats_path, np.stack([mean, std], axis=1).astype(np.float32))
        np.save(audio_path, AUDIO_PRESENT)
        CT.to_csv(table_path, index=False)
        with open(meta_path, 'w') as fh:
            json.dump(CHUNK_META, fh, indent=1)
        MFCC = f16_restore(z16, mean, std)
        del z16, results, z_parts, a_parts
        print(f'chunk cache built in {time.time() - t0:,.0f} s: {len(CT)} chunks; filter designs {designs}')

    assert MFCC.shape == (len(CT), N_MFCC * 3, N_FRAMES), MFCC.shape
    assert (CT['audio_row'].values[CT['audio_row'].values >= 0] < len(AUDIO_PRESENT)).all()
    assert CT.groupby('rec')['chunk_idx'].count().eq(CT.groupby('rec')['n_chunks'].first()).all(), 'chunk/recording alignment'
    REC2CHUNKS = {r: np.asarray(v) for r, v in CT.groupby('rec').indices.items()}
    AUDIO_ROW = CT['audio_row'].values
    CT_LABELS = {t: CT[f'{t}_{LABEL_MODE}'].values.astype(int) for t in TASKS}
    CT_LABELS_BY_MODE = {mode: {t: CT[f'{t}_{mode}'].values.astype(int) for t in TASKS} for mode in ('patient', 'location')}
    print(f'MFCC in RAM: {MFCC.shape} float32 ({MFCC.nbytes / 2**30:.2f} GB); audio for augmentation: '
          f'{AUDIO_PRESENT.shape} ({AUDIO_PRESENT.nbytes / 2**30:.2f} GB); chunks per recording: '
          f'median {CT["n_chunks"].median():.0f}, max {CT["n_chunks"].max()}')

    # Optional cross-check against the original chunk_metadata.csv (17,176 chunks on the full set)
    CHUNK_COUNT_CHECK = None
    if os.path.isfile(ORIGINAL_CHUNK_META):
        om = pd.read_csv(ORIGINAL_CHUNK_META)
        om['rec'] = om['filename'].str.replace('.wav', '', regex=False)
        oc = om.groupby('rec').size()
        ours = CT.groupby('rec').size()
        common = oc.index.intersection(ours.index)
        mism = int((oc.loc[common] != ours.loc[common]).sum())
        CHUNK_COUNT_CHECK = {'n_common_recordings': int(len(common)), 'n_count_mismatches': mism,
                             'original_total_chunks': int(len(om))}
        print(f'chunk-count cross-check vs original chunk_metadata.csv: {len(common)} common recordings, '
              f'{mism} mismatches (original total {len(om)} chunks)')
''')

md("""
## 7. Models and training utilities (verbatim architecture and configuration)

`AttentionBlock`, `ConvBlock`, `MurmurClassifier`, `MultiLabelClassifier` and `EarlyStopping` are the original classes. The training loop keeps the original configuration: `nn.CrossEntropyLoss` (sum of three for the multilabel model), `AdamW(lr=1e-3, weight_decay=1e-4)`, batch 64, up to 100 epochs, `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)` and early stopping with patience 10, both on the inner-validation AUC (murmur) or the mean of the three validation AUCs (multilabel), best state restored.

Two corrections to the original loop are applied and recorded as deviations: the original guard `if early_stopping(score): if early_stopping.early_stop: break` could never fire (the call returns True only on improvement), and `best_model_state = model.state_dict().copy()` was a shallow copy that tracked the live weights. Here early stopping stops training when patience is exhausted and the best state is a detached copy.

Online augmentation is applied to the cached maps in the feature domain, as an approximation of the original audio-domain operations. The gain part is exact: a gain g on the audio adds `sqrt(n_mels) * 20 * log10(g)` to MFCC coefficient 0 in every frame and leaves the deltas unchanged (asserted at runtime to 1e-3). The shift part is not: a circular shift of s samples becomes a circular roll of `round(s / hop)` frames. Measured against an audio-domain roll followed by re-extraction, a whole-frame shift (s a multiple of 160 samples) differs in 6 to 8 frames at each edge of the map (librosa's centred reflect padding and the delta interpolation), a sub-frame shift (the general case) differs in every frame by a small amount (rms about 0.02 to 0.07 standardised units, up to 3.3 at the edges), and |s| below 80 samples (about 10% of the draws) rounds to zero frames and does nothing. The per-sample global standardisation `(x - mean) / (std + 1e-8)` follows, as in `HeartSoundDataset.__getitem__` (population standard deviation).
""")

code(VB_BLOCKS + r'''


''' + VB_MURMUR + r'''


''' + VB_MULTI + r'''


''' + VB_EARLY + r'''


def make_model(kind):
    if kind == 'murmur':
        return MurmurClassifier(num_classes=2, dropout=DROPOUT)
    return MultiLabelClassifier(num_grading=3, num_timing=3, num_shape=3, dropout=DROPOUT)


_n_m = sum(p.numel() for p in make_model('murmur').parameters())
_n_ml = sum(p.numel() for p in make_model('multilabel').parameters())
assert _n_m == 455042 and _n_ml == 521737, (_n_m, _n_ml)
print(f'MurmurClassifier {_n_m:,} parameters; MultiLabelClassifier {_n_ml:,} parameters (both equal the original counts)')
''')

code(r'''
GAIN_C0_FACTOR = float(np.sqrt(N_MELS)) * 20.0     # change of MFCC c0 per unit log10(gain), DCT-II ortho over n_mels

# Runtime check of the feature-domain gain: audio * 1.1 must move c0 by GAIN_C0_FACTOR * log10(1.1) in every frame
# and leave every other row (c1..c39 and both delta blocks) unchanged
_a = np.random.default_rng(2).standard_normal(N_SAMPLES).astype(np.float32)
_f0, _f1 = extract_mfcc_from_array(_a), extract_mfcc_from_array(_a * 1.1)
assert np.allclose(_f1[0] - _f0[0], GAIN_C0_FACTOR * np.log10(1.1), atol=1e-3), \
    f'c0 gain offset check failed: max |diff| {np.abs(_f1[0] - _f0[0] - GAIN_C0_FACTOR * np.log10(1.1)).max():.2e}'
assert np.allclose(_f1[1:], _f0[1:], atol=1e-3), 'a gain must not change c1..c39 or the delta blocks'
print(f'c0 gain offset check passed: sqrt({N_MELS}) * 20 * log10(1.1) = {GAIN_C0_FACTOR * np.log10(1.1):.4f}')


def standardise_batch(x):
    """Per-sample global standardisation, the verbatim formula of HeartSoundDataset.__getitem__ (population std)."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - m) / (s + 1e-8)


def online_augment_batch(x, gen):
    """Feature-domain equivalent of the original online augmentation (p=0.3 gain U(0.9,1.1); p=0.3 circular
    shift of int(N_SAMPLES*U(-0.05,0.05)) samples). Draws come from a seeded CPU generator."""
    B = x.shape[0]
    u = torch.rand(4, B, generator=gen)
    apply_gain = u[0] < 0.3
    gain = 0.9 + 0.2 * u[1]
    apply_shift = u[2] < 0.3
    shift_samples = torch.trunc(N_SAMPLES * (u[3] * 0.10 - 0.05))
    frames = torch.round(shift_samples / HOP_LENGTH).to(torch.long)
    frames = torch.where(apply_shift, frames, torch.zeros_like(frames))
    offset = torch.where(apply_gain, GAIN_C0_FACTOR * torch.log10(gain), torch.zeros_like(gain)).to(x.device, x.dtype)
    x[:, 0, 0, :] = x[:, 0, 0, :] + offset[:, None]
    for f in torch.unique(frames).tolist():
        if f == 0:
            continue
        m = (frames == f).to(x.device)
        x[m] = torch.roll(x[m], shifts=int(f), dims=-1)
    return x


def _auroc_ovr(y, P):
    C = P.shape[1]
    if C == 2:
        return float(roc_auc_score(y, P[:, 1])) if len(np.unique(y)) == 2 else float('nan')
    vals = []
    for c in range(C):
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            vals.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(vals)) if vals else float('nan')


@torch.no_grad()
def predict_probs(model, kind, X, batch=512):
    """Chunk-level softmax probabilities from float32 maps [N,120,T] (clean, standardised per sample)."""
    model.eval()
    outs = {'grading': [], 'timing': [], 'shape': []} if kind != 'murmur' else []
    for s in range(0, len(X), batch):
        xb = torch.as_tensor(np.ascontiguousarray(X[s:s + batch])).unsqueeze(1).to(DEVICE)
        xb = standardise_batch(xb)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP_ENABLED):
            out = model(xb)
        if kind == 'murmur':
            outs.append(F.softmax(out.float(), dim=1).cpu().numpy())
        else:
            for t in CHAR_TASKS:
                outs[t].append(F.softmax(out[t].float(), dim=1).cpu().numpy())
    if kind == 'murmur':
        return np.concatenate(outs) if outs else np.zeros((0, 2), dtype=np.float32)
    return {t: (np.concatenate(v) if v else np.zeros((0, 3), dtype=np.float32)) for t, v in outs.items()}


def monitor_score(kind, P, Y):
    """Murmur: validation AUC. Multilabel: mean of the three OvR-macro validation AUCs (verbatim monitors)."""
    if kind == 'murmur':
        auc = _auroc_ovr(Y, P)
        if np.isnan(auc):
            return float((P.argmax(1) == Y).mean()), {'val_auc': float('nan'), 'monitor_fallback': 'accuracy'}
        return auc, {'val_auc': auc}
    aucs = {t: _auroc_ovr(Y[:, i], P[t]) for i, t in enumerate(CHAR_TASKS)}
    usable = [v for v in aucs.values() if not np.isnan(v)]
    if not usable:
        acc = float(np.mean([(P[t].argmax(1) == Y[:, i]).mean() for i, t in enumerate(CHAR_TASKS)]))
        return acc, {**{f'val_auc_{t}': v for t, v in aucs.items()}, 'monitor_fallback': 'accuracy'}
    return float(np.mean(usable)), {f'val_auc_{t}': v for t, v in aucs.items()}


def gpu_can_hold(n_bytes):
    if DEVICE.type != 'cuda':
        return False
    free, total = torch.cuda.mem_get_info()
    return n_bytes < 0.45 * free


def train_model(kind, X_tr, Y_tr, X_va, Y_va, epochs, seed, tag, use_amp=None, verbose_every=5):
    """Original configuration: CrossEntropy, AdamW(1e-3, 1e-4), batch 64, ReduceLROnPlateau(max, 0.5, 5),
    EarlyStopping(patience 10, max) on the validation monitor, best state restored."""
    use_amp = AMP_ENABLED if use_amp is None else use_amp
    assert len(X_tr) > 1 and len(X_va) > 0, (f'{tag}: training needs at least 2 training and 1 validation samples '
                                             f'(got {len(X_tr)} and {len(X_va)})')
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_model(kind).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    Xtr = torch.as_tensor(np.ascontiguousarray(X_tr))
    Ytr = torch.as_tensor(np.asarray(Y_tr), dtype=torch.long)
    on_gpu = gpu_can_hold(Xtr.numel() * 4 + Ytr.numel() * 8)
    if on_gpu:
        Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    N = len(Xtr)
    history, best_score, best_state, best_epoch = [], -float('inf'), None, -1
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, generator=gen)
        if on_gpu:
            perm = perm.to(DEVICE)
        running, nseen, n_skipped = 0.0, 0, 0
        for s in range(0, N, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            if len(idx) == 1:          # BatchNorm cannot train on a single sample (same constraint as the original loader)
                n_skipped += 1
                continue
            xb = Xtr[idx].unsqueeze(1).to(DEVICE, non_blocking=True).clone()
            yb = Ytr[idx].to(DEVICE, non_blocking=True)
            xb = standardise_batch(online_augment_batch(xb, gen))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                if kind == 'murmur':
                    loss = criterion(out, yb)
                else:
                    loss_grading = criterion(out['grading'], yb[:, 0])
                    loss_timing = criterion(out['timing'], yb[:, 1])
                    loss_shape = criterion(out['shape'], yb[:, 2])
                    loss = loss_grading + loss_timing + loss_shape
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(idx)
            nseen += len(idx)
        P = predict_probs(model, kind, X_va)
        score, detail = monitor_score(kind, P, Y_va)
        if 'monitor_fallback' in detail and epoch == 0:
            note_deviation('validation monitor', f'{tag}: AUC undefined on the inner validation set; accuracy used')
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(score)
        new_lr = optimizer.param_groups[0]['lr']
        early_stopping(score)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({'epoch': epoch + 1, 'train_loss': running / max(nseen, 1), 'val_score': float(score),
                        'lr': new_lr, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in detail.items()}})
        if (epoch + 1) % verbose_every == 0 or epoch == 0 or new_lr != old_lr or early_stopping.early_stop:
            print(f'    {tag} epoch {epoch + 1:3d}/{epochs} loss {running / max(nseen, 1):.4f} val {score:.4f} '
                  f'(best {best_score:.4f} @ {best_epoch + 1}) lr {new_lr:.2e} {time.time() - t_start:,.0f} s'
                  + (' LR reduced' if new_lr != old_lr else ''))
        if early_stopping.early_stop:
            print(f'    {tag}: early stopping after {epoch + 1} epochs (patience {PATIENCE})')
            break
    model.load_state_dict(best_state)
    info = {'best_score': float(best_score), 'best_epoch': int(best_epoch + 1), 'epochs_run': len(history),
            'seconds': time.time() - t_start, 'n_train': int(N), 'n_val': int(len(X_va)), 'seed': seed,
            'amp': bool(use_amp), 'train_on_gpu': bool(on_gpu), 'history': history}
    return model, info


print('training utilities ready')
''')

md("""
## 8. Class balancing per fold (original procedures)

Murmur model: the training chunks of the fold are balanced by oversampling Present chunks with `augment_audio` (random combo per augmented copy, source chunk drawn with `np.random.choice`) until Present equals Absent, exactly as `MILESTONE3_DATA_AUGMENTATION_V5` did on its single training split. The MFCC map of each augmented chunk is computed with the same extractor.

Multilabel model: the original iterative joint balancing (score every original Present chunk by the summed deficits of its grading, timing and shape classes below 33.33%, augment the best-scoring chunk, repeat until every class is within 2 points or 10,000 iterations) is ported as a vectorised planner with the same selection rule and tie-breaking; a reference check runs the verbatim pandas scorers on the first iterations and asserts the same picks. Augmented chunks are then produced by `augment_audio` in the planned order, which reproduces the original random stream.

Balancing is applied to the inner-training chunks only; the inner-validation and test chunks stay clean.
""")

code(r'''
def murmur_balance(train_chunk_ids, y, seed=None):
    """Verbatim oversampling rule of MILESTONE3_DATA_AUGMENTATION_V5 cells 9-10. Returns aug audio, source ids, info."""
    if seed is not None:
        np.random.seed(seed)
    absent_indices = [int(i) for i in train_chunk_ids if y[i] == 0]
    present_indices = [int(i) for i in train_chunk_ids if y[i] == 1]
    target_count = len(absent_indices)
    num_to_augment = target_count - len(present_indices)
    augmented, sources = [], []
    combo_idx = 0
    for i in range(max(num_to_augment, 0)):
        source_idx = np.random.choice(present_indices)
        arow = AUDIO_ROW[source_idx]
        assert arow >= 0, f'no cached audio for Present chunk {source_idx}'
        aug_audio = augment_audio(AUDIO_PRESENT[arow], combo_idx=combo_idx)
        combo_idx += 1
        augmented.append(aug_audio)
        sources.append(int(source_idx))
    info = {'n_absent': len(absent_indices), 'n_present': len(present_indices), 'n_augmented': len(sources)}
    return augmented, np.array(sources, dtype=int), info


def joint_balance_plan(L3, max_iterations=10000, tolerance=2.0, target_pct=33.33):
    """Vectorised port of the verbatim iterative joint balancing (cells 13-15). L3: (n,3) labels of the ORIGINAL
    Present chunks, all valid. Returns the ordered list of source rows to augment plus a summary."""
    L3 = np.asarray(L3, dtype=int)
    counts = np.stack([np.bincount(L3[:, t], minlength=3) for t in range(3)]).astype(float)
    order = []
    balanced = False
    for iteration in range(max_iterations):
        present = counts > 0
        pct = np.where(present, counts / counts.sum(axis=1, keepdims=True) * 100.0, target_pct)
        if np.all(np.abs(pct[present] - target_pct) <= tolerance):
            balanced = True
            break
        deficit = np.where(present, target_pct - pct, 0.0)
        pos = np.where(deficit > 0, deficit, 0.0)
        scores = pos[0, L3[:, 0]] + pos[1, L3[:, 1]] + pos[2, L3[:, 2]]
        if scores.max() <= 0:
            break
        best = int(np.argmax(scores))         # first maximum, as Series.idxmax in the original
        order.append(best)
        for t in range(3):
            counts[t, L3[best, t]] += 1
    final_pct = {CHAR_TASKS[t]: {CLASS_NAMES[CHAR_TASKS[t]][c]: float(counts[t, c] / counts[t].sum() * 100)
                                 for c in range(3)} for t in range(3)}
    return order, {'iterations': len(order), 'balanced': balanced, 'final_pct': final_pct}


def joint_balance_reference(L3, n_iter):
    """Runs the ORIGINAL scoring loop (verbatim scorers, pandas) for n_iter picks, for the equivalence check."""
    meta = pd.DataFrame({f'{t}_label': L3[:, i].astype(float) for i, t in enumerate(CHAR_TASKS)})
    meta['is_augmented'] = False
    working = meta.copy()
    picks = []
    for _ in range(n_iter):
        current_dist = calculate_stage_distributions(working)
        if is_balanced(current_dist, tolerance=2.0):
            break
        imbalances = calculate_imbalance_scores(current_dist)
        original_samples = working[working['is_augmented'] == False].copy()
        scores = original_samples.apply(lambda row: score_candidate(row, imbalances), axis=1)
        if scores.max() <= 0:
            break
        best_idx = scores.idxmax()
        picks.append(int(best_idx))
        aug_meta = working.loc[best_idx].to_dict()
        aug_meta['is_augmented'] = True
        working = pd.concat([working, pd.DataFrame([aug_meta])], ignore_index=True)
    return picks


def joint_balance_augment(pool_chunk_ids, L3, seed=None, max_iterations=10000):
    """Plan with the vectorised port, then augment in the planned order with augment_audio (verbatim calls)."""
    if seed is not None:
        np.random.seed(seed)
    order, info = joint_balance_plan(L3, max_iterations=max_iterations)
    augmented, sources = [], []
    combo_idx = 0
    for best in order:
        chunk_id = int(pool_chunk_ids[best])
        arow = AUDIO_ROW[chunk_id]
        assert arow >= 0, f'no cached audio for Present chunk {chunk_id}'
        aug_audio = augment_audio(AUDIO_PRESENT[arow], combo_idx=combo_idx)
        combo_idx += 1
        augmented.append(aug_audio)
        sources.append(chunk_id)
    info['n_original'] = int(len(pool_chunk_ids))
    info['n_augmented'] = len(sources)
    return augmented, np.array(sources, dtype=int), info


# Equivalence check of the planner against the verbatim scoring loop on a small synthetic pool
_rng = np.random.default_rng(3)
_L = np.stack([_rng.choice(3, 120, p=[0.6, 0.25, 0.15]), _rng.choice(3, 120, p=[0.5, 0.4, 0.1]),
               _rng.choice(3, 120, p=[0.7, 0.2, 0.1])], axis=1)
_ref = joint_balance_reference(_L, 40)
_plan, _ = joint_balance_plan(_L)
assert _plan[:len(_ref)] == _ref, 'vectorised joint-balancing planner disagrees with the verbatim loop'
print(f'joint-balancing planner matches the verbatim loop on {len(_ref)} reference picks; balancing utilities ready')
''')

md("""
## 9. Cross-validation runs

For every task and fold: the fold's training recordings are split by patient into inner-training and inner-validation recordings (`GroupShuffleSplit(test_size=0.15, random_state=42)`; if a class of the monitored task is absent from the inner-validation recordings the next seed is tried and the seed used is recorded), the inner-training chunks are balanced, the model is trained, and chunk-level probabilities are produced for the test fold. For the three characterisation tasks the multilabel model (shared trunk, three heads) is trained once per task on that task's folds so that its test predictions for the task are always out of fold; only the head of the task is scored. Every fold is cached and the loop resumes from the caches.
""")

code(r'''
def inner_split(train_recs, task, fold=None):
    """Patient-disjoint inner validation split of the training recordings (GroupShuffleSplit 0.15, seed 42).
    Seeds 42..61 are tried until every class of the monitored task(s) is present in the inner validation set;
    if none qualifies, the first seed with at least two classes per task is used and a deviation is recorded."""
    recs = np.asarray(train_recs)
    pats = np.array([REC_PATIENT[r] for r in recs])
    if task == 'murmur':
        need = {'murmur': {0, 1}}
    else:
        need = {t: {0, 1, 2} for t in CHAR_TASKS}
    fallback = None
    for rs in range(INNER_VAL_SEED, INNER_VAL_SEED + 20):
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=rs)
        tr, va = next(gss.split(recs, groups=pats))
        va_recs = recs[va]
        if task != 'murmur':
            va_recs = np.array([r for r in va_recs if all(REC_LABEL[LABEL_MODE][t][r] >= 0 for t in CHAR_TASKS)])
        present = {t: {REC_LABEL[LABEL_MODE][t][r] for r in va_recs} for t in need}
        if all(need[t] <= present[t] for t in need):
            return recs[tr], recs[va], rs
        if fallback is None and all(len(present[t]) >= 2 for t in need):
            fallback = (recs[tr], recs[va], rs)
    if fallback is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=INNER_VAL_SEED)
        tr, va = next(gss.split(recs, groups=pats))
        fallback = (recs[tr], recs[va], INNER_VAL_SEED)
    va_recs = fallback[1]
    if task != 'murmur':
        va_recs = np.array([r for r in va_recs if all(REC_LABEL[LABEL_MODE][t][r] >= 0 for t in CHAR_TASKS)])
    present = {t: sorted(int(v) for v in {REC_LABEL[LABEL_MODE][t][r] for r in va_recs}) for t in need}
    note_deviation('inner validation split',
                   f'{task} fold {fold}: no seed in {INNER_VAL_SEED}..{INNER_VAL_SEED + 19} gave every class of the monitored '
                   f'task(s) in the inner validation recordings ({len(recs)} training recordings, '
                   f'{len(np.unique(pats))} patients); seed {fallback[2]} used with classes present {present}')
    return fallback


def chunks_of(recs):
    recs = list(recs)
    if not recs:
        return np.zeros(0, dtype=int)
    return np.concatenate([REC2CHUNKS[r] for r in recs])


def build_train_set(kind, tr_chunk_ids, seed, cache_path=None, reseed=True):
    """Balanced training set: clean maps of the inner-training chunks plus augmented maps. Cached per fold."""
    if kind == 'murmur':
        y = CT_LABELS['murmur']
        ids = tr_chunk_ids
        Y_clean = y[ids]
    else:
        L3_all = np.stack([CT_LABELS[t] for t in CHAR_TASKS], axis=1)
        ids = np.array([i for i in tr_chunk_ids if (L3_all[i] >= 0).all()], dtype=int)
        Y_clean = L3_all[ids]
    # the cache is valid only for the same train chunk set (sorted ids hash) built on the same chunk cache
    ids_sha1 = hashlib.sha1(np.sort(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()
    cached = None
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if 'ids_sha1' in z.files and str(z['ids_sha1']) == ids_sha1 and str(z['chunk_cache_key']) == CACHE_KEY:
            cached = z
        else:
            why = ('no ids_sha1 stored (older cache)' if 'ids_sha1' not in z.files else
                   'train chunk ids differ' if str(z['ids_sha1']) != ids_sha1 else 'chunk cache key differs')
            print(f'    aug cache {os.path.basename(cache_path)} not reused: {why}; rebuilding')
            CACHE_EVENTS.append({'kind': 'aug_cache_rebuilt', 'file': os.path.basename(cache_path), 'reason': why})
    if cached is not None:
        z = cached
        feats_aug = f16_restore(z['z16'], z['mean'], z['std'])
        Y_aug = z['labels']
        info = json.loads(str(z['info']))
        info['from_cache'] = True
    else:
        t0 = time.time()
        if kind == 'murmur':
            aug_audio, src, info = murmur_balance(ids, y, seed=(seed if reseed else None))
            Y_aug = y[src] if len(src) else np.zeros(0, dtype=int)
        else:
            aug_audio, src, info = joint_balance_augment(ids, Y_clean, seed=(seed if reseed else None),
                                                         max_iterations=JOINT_MAX_ITERATIONS)
            Y_aug = L3_all[src] if len(src) else np.zeros((0, 3), dtype=int)
        feats = mfcc_parallel(aug_audio)
        z16, mean, std = f16_roundtrip(feats) if len(feats) else (np.zeros((0, N_MFCC * 3, N_FRAMES), np.float16),
                                                                  np.zeros(0, np.float32), np.zeros(0, np.float32))
        feats_aug = f16_restore(z16, mean, std) if len(feats) else feats
        info['aug_seconds'] = time.time() - t0
        info['from_cache'] = False
        if cache_path and CACHE_AUG_FEATURES:
            np.savez(cache_path, z16=z16, mean=mean, std=std, labels=Y_aug, sources=src, info=json.dumps(info),
                     ids_sha1=np.array(ids_sha1), chunk_cache_key=np.array(CACHE_KEY))
    X = np.concatenate([MFCC[ids], feats_aug], axis=0) if len(feats_aug) else MFCC[ids]
    Y = np.concatenate([Y_clean, Y_aug], axis=0) if len(Y_aug) else Y_clean
    info['n_clean'] = int(len(ids))
    info['n_total'] = int(len(Y))
    return X, Y, ids, info


CV_PREDS = {}
CV_TRAIN_INFO = {}
with StageTimer('cross_validation'):
    n_jobs_total = sum(len(FOLD_PLAN[t]) for t in TASKS)
    done = 0
    t_cv0 = time.time()
    for task in TASKS:
        kind = 'murmur' if task == 'murmur' else 'multilabel'
        CV_PREDS[task] = []
        CV_TRAIN_INFO[task] = []
        for fp in FOLD_PLAN[task]:
            k = fp['fold']
            tag = f'{task} fold {k}'
            pred_path = os.path.join(CACHE_DIR, f'preds_{LABEL_MODE}_{task}_fold{k}.npz')
            model_path = os.path.join(MODELS_DIR, f'{LABEL_MODE}_{task}_fold{k}.pt')
            info_path = os.path.join(CACHE_DIR, f'train_{LABEL_MODE}_{task}_fold{k}.json')
            # resume only when the cached fold was made for the same fold plan (test and train recordings) and
            # the same epoch budget; otherwise recompute and say why
            resume_ok, why_not = False, None
            have = [os.path.isfile(p) for p in (pred_path, model_path, info_path)]
            if all(have):
                z = np.load(pred_path, allow_pickle=True)
                if not all(kk in z.files for kk in ('test_recs', 'train_recs', 'epochs_max')):
                    why_not = 'cached predictions carry no fold plan (older cache format)'
                elif not np.array_equal(np.sort(z['test_recs'].astype(str)), np.sort(np.asarray(fp['test_recs']).astype(str))):
                    why_not = f'test recordings differ ({len(z["test_recs"])} cached vs {len(fp["test_recs"])} planned)'
                elif not np.array_equal(np.sort(z['train_recs'].astype(str)), np.sort(np.asarray(fp['train_recs']).astype(str))):
                    why_not = f'train recordings differ ({len(z["train_recs"])} cached vs {len(fp["train_recs"])} planned)'
                elif int(z['epochs_max']) != EPOCHS_RUN:
                    why_not = f'epoch budget differs ({int(z["epochs_max"])} cached vs {EPOCHS_RUN} now)'
                else:
                    resume_ok = True
            elif any(have):
                why_not = 'incomplete fold cache (predictions, model or training record missing)'
            if resume_ok:
                CV_PREDS[task].append({'fold': k, 'chunk_ids': z['chunk_ids'], 'probs': z['probs'], 'y': z['y'],
                                       'rec': z['rec']})
                with open(info_path) as fh:
                    finfo = json.load(fh)
                CV_TRAIN_INFO[task].append(finfo)
                # deviations noted when the fold was trained (inner split fallback, monitor fallback) belong to
                # this run as well: restore them so a resumed run writes the same manifest as an uninterrupted one
                restored = [d for d in finfo.get('deviations', []) if d not in DEVIATIONS]
                DEVIATIONS.extend(restored)
                done += 1
                print(f'{tag}: resumed from cache (fold plan and epoch budget verified)'
                      + (f'; {len(restored)} deviation(s) of the fold restored' if restored else ''))
                continue
            if why_not:
                print(f'{tag}: cached fold not reused: {why_not}; recomputing')
                CACHE_EVENTS.append({'kind': 'fold_recomputed', 'task': task, 'fold': int(k), 'reason': why_not})
            t_fold = time.time()
            n_dev0 = len(DEVIATIONS)
            seed_k = SEED + k
            inner_tr, inner_va, rs_used = inner_split(fp['train_recs'], task, fold=k)
            tr_ids = chunks_of(inner_tr)
            va_ids = chunks_of(inner_va)
            te_ids = chunks_of(fp['test_recs'])
            if kind == 'murmur':
                Y_va = CT_LABELS['murmur'][va_ids]
            else:
                L3_all = np.stack([CT_LABELS[t] for t in CHAR_TASKS], axis=1)
                va_ids = np.array([i for i in va_ids if (L3_all[i] >= 0).all()], dtype=int)
                Y_va = L3_all[va_ids]
            aug_cache = os.path.join(CACHE_DIR, f'aug_{LABEL_MODE}_{task}_fold{k}.npz')
            X_tr, Y_tr, clean_ids, bal = build_train_set(kind, tr_ids, seed_k, cache_path=aug_cache)
            print(f'{tag}: train recs {len(inner_tr)} (chunks {len(clean_ids)} clean + {bal["n_augmented"]} augmented) | '
                  f'inner val recs {len(inner_va)} (chunks {len(va_ids)}, seed {rs_used}) | test recs {len(fp["test_recs"])} '
                  f'(chunks {len(te_ids)})' + (f' | joint balance: {bal["iterations"]} it, balanced={bal["balanced"]}'
                                               if kind != 'murmur' else ''))
            model, tinfo = train_model(kind, X_tr, Y_tr, MFCC[va_ids], Y_va, EPOCHS_RUN, seed_k, tag)
            P = predict_probs(model, kind, MFCC[te_ids])
            if kind != 'murmur':
                P = P[task]
            y_te = CT_LABELS[task][te_ids]
            assert (y_te >= 0).all() and len(P) == len(te_ids)
            rec_te = CT['rec'].values[te_ids]
            np.savez(pred_path, chunk_ids=te_ids, probs=P.astype(np.float32), y=y_te, rec=rec_te, fold=k,
                     test_recs=np.asarray(fp['test_recs']).astype(str), train_recs=np.asarray(fp['train_recs']).astype(str),
                     epochs_max=np.array(EPOCHS_RUN))
            torch.save({'model_state_dict': model.state_dict(), 'kind': kind, 'task': task, 'fold': k,
                        'label_mode': LABEL_MODE, 'best_score': tinfo['best_score'],
                        'config': {'n_mfcc': N_MFCC, 'sample_rate': SAMPLE_RATE, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH}},
                       model_path)
            finfo = {'task': task, 'fold': k, 'seed': seed_k, 'inner_val_seed': rs_used,
                     'n_train_recs': int(len(inner_tr)), 'n_inner_val_recs': int(len(inner_va)),
                     'n_test_recs': int(len(fp['test_recs'])), 'n_test_chunks': int(len(te_ids)),
                     'balancing': bal, 'training': {kk: v for kk, v in tinfo.items() if kk != 'history'},
                     'history': tinfo['history'], 'fold_seconds': time.time() - t_fold,
                     'deviations': list(DEVIATIONS[n_dev0:])}
            with open(info_path, 'w') as fh:
                json.dump(finfo, fh, indent=1)
            CV_PREDS[task].append({'fold': k, 'chunk_ids': te_ids, 'probs': P, 'y': y_te, 'rec': rec_te})
            CV_TRAIN_INFO[task].append(finfo)
            done += 1
            el = time.time() - t_cv0
            print(f'{tag}: done in {time.time() - t_fold:,.0f} s (best val {tinfo["best_score"]:.4f} at epoch '
                  f'{tinfo["best_epoch"]}, {tinfo["epochs_run"]} epochs) | {done}/{n_jobs_total} folds, elapsed {el / 60:.1f} min, '
                  f'ETA {el / done * (n_jobs_total - done) / 60:.1f} min')
            del model, X_tr
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
''')

md("""
## 10. Scoring

Chunk-level predictions of every test fold are pooled (out of fold). Recording-level prediction = mean of the chunk softmax probabilities per recording, argmax (for detection this equals a 0.5 threshold on the mean probability). Metrics per task, on the gating notebook's definitions: accuracy, macro-F1 over the fixed label set of the task (an absent or never-predicted class counts as F1 = 0, as `f1_score(labels=range(C), zero_division=0)`), balanced accuracy, per-class recall and specificity, AUROC (binary; one-vs-rest macro for three classes, undefined when a class is absent), AUPRC (one-vs-rest macro; positive-class AP for detection), confusion matrix, majority-class baseline, per-fold accuracy and macro-F1 with their mean and population SD (`fold_macro_f1_sd`, ddof 0, the gating notebook's name; the sample SD is kept under `_sd_sample`), and patient-level bootstrap 95% CIs (2000 resamples of patients with replacement) on every pooled metric. Recording level is primary; chunk level is reported alongside. Besides the native layout under `tasks`, `results.json` carries the same numbers under `arms.<gating task>.cnn_pcg_only` in the gating notebook's per-classifier layout (rates in percent), so one table script reads both files.
""")

code(r'''
def _auroc_fixed(y, P):
    """AUROC on the fixed label set, as the gating notebook's score_metrics: binary AUROC for two classes;
    one-vs-rest macro for three classes; NaN when fewer than two classes are present or, for three classes,
    when any class is absent (the training monitor keeps _auroc_ovr, which averages the classes present)."""
    C = P.shape[1]
    present = np.unique(y)
    if len(present) < 2 or (C > 2 and len(present) < C):
        return float('nan')
    if C == 2:
        return float(roc_auc_score(y, P[:, 1]))
    return float(roc_auc_score(y, P, multi_class='ovr', average='macro'))


def metrics_from_preds(y, P, light=False):
    """All metrics from labels and class probabilities, on the gating notebook's definitions (fixed label set
    0..C-1). light=True skips AUROC/AUPRC (paired bootstrap loops)."""
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=np.float64)
    C = P.shape[1]
    pred = P.argmax(1)
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    fp_ = predicted - tp
    tn = float(len(y)) - support - predicted + tp
    recall = np.divide(tp, support, out=np.full(C, np.nan), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted > 0)
    specificity = np.divide(tn, tn + fp_, out=np.zeros(C), where=(tn + fp_) > 0)
    rec0 = np.nan_to_num(recall, nan=0.0)
    f1 = np.divide(2 * precision * rec0, precision + rec0, out=np.zeros(C), where=(precision + rec0) > 0)
    out = {'n': int(len(y)), 'accuracy': float(tp.sum() / max(len(y), 1)),
           'macro_f1': float(f1.mean()),                 # over all C classes: absent or never-predicted class = 0
           'balanced_accuracy': float(np.nanmean(recall)) if np.isfinite(recall).any() else float('nan'),
           'per_class_recall': [float(v) for v in recall], 'per_class_precision': [float(v) for v in precision],
           'per_class_specificity': [float(v) for v in specificity],
           'per_class_f1': [float(v) for v in f1], 'confusion_matrix': cm.tolist()}
    if C == 2:
        out['sensitivity'] = out['per_class_recall'][1]
        out['specificity'] = out['per_class_recall'][0]
    if light:
        return out
    ap = []
    for c in range(C):
        yc = (y == c).astype(int)
        if yc.sum() > 0:
            ap.append(average_precision_score(yc, P[:, c]))
    out['auroc'] = _auroc_fixed(y, P)
    out['auprc_macro'] = float(np.mean(ap)) if ap else float('nan')
    if C == 2:
        out['auprc_positive'] = float(average_precision_score(y, P[:, 1])) if (y == 1).any() else float('nan')
    return out


def sklearn_crosscheck(y, P):
    """The numpy metrics must equal sklearn on the gating definitions (fixed label set, zero_division=0)."""
    pred = P.argmax(1)
    C = P.shape[1]
    m = metrics_from_preds(y, P)
    assert abs(m['accuracy'] - accuracy_score(y, pred)) < 1e-12
    assert abs(m['macro_f1'] - f1_score(y, pred, average='macro', labels=list(range(C)), zero_division=0)) < 1e-9
    assert abs(m['balanced_accuracy'] - balanced_accuracy_score(y, pred)) < 1e-9
    if C > 2 and len(np.unique(y)) == C:
        assert abs(m['auroc'] - roc_auc_score(y, P, multi_class='ovr', average='macro')) < 1e-9
    elif C == 2 and len(np.unique(y)) == 2:
        assert abs(m['auroc'] - roc_auc_score(y, P[:, 1])) < 1e-9
    return True


def patient_resamples(patients, n_boot, seed):
    pats, inv = np.unique(np.asarray(patients), return_inverse=True)
    idx_by = [np.flatnonzero(inv == i) for i in range(len(pats))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        samp = rng.integers(0, len(pats), len(pats))
        yield np.concatenate([idx_by[i] for i in samp])


BOOT_KEYS = ['accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'auprc_macro']


def bootstrap_ci(y, P, patients, n_boot, seed):
    C = P.shape[1]
    store = {k: [] for k in BOOT_KEYS + ['auprc_positive', 'sensitivity', 'specificity']}
    recalls, specs = [], []
    for idx in patient_resamples(patients, n_boot, seed):
        m = metrics_from_preds(y[idx], P[idx])
        for k in store:
            if k in m:
                store[k].append(m[k])
        recalls.append(m['per_class_recall'])
        specs.append(m['per_class_specificity'])
    def pct(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan'), float('nan')]

    ci = {k: pct(v) for k, v in store.items() if v}
    recalls = np.array(recalls, dtype=float)
    specs = np.array(specs, dtype=float)
    ci['per_class_recall'] = [pct(recalls[:, c]) for c in range(C)]
    ci['per_class_specificity'] = [pct(specs[:, c]) for c in range(C)]
    ci['n_boot'] = n_boot
    # resamples in which the AUROC is undefined (a class absent) are left out of its interval, as in the gating notebook
    ci['auroc_n_resamples_defined'] = int(np.isfinite(np.asarray(store['auroc'], dtype=float)).sum())
    return ci


def aggregate_recordings(chunk_rec, chunk_probs, chunk_y, chunk_fold):
    recs, inv = np.unique(chunk_rec, return_inverse=True)
    C = chunk_probs.shape[1]
    P = np.zeros((len(recs), C))
    np.add.at(P, inv, chunk_probs)
    counts = np.bincount(inv, minlength=len(recs))
    P /= counts[:, None]
    y = np.full(len(recs), -1)
    fold = np.full(len(recs), -1)
    for i in range(len(recs)):
        rows = np.flatnonzero(inv == i)
        assert len(np.unique(chunk_y[rows])) == 1 and len(np.unique(chunk_fold[rows])) == 1, recs[i]
        y[i] = chunk_y[rows[0]]
        fold[i] = chunk_fold[rows[0]]
    pats = np.array([REC_PATIENT[r] for r in recs])
    return recs, P, y, fold, counts, pats


def fold_table(y, P, folds, patients):
    """Per-fold metrics and their summary under the gating notebook's names: fold_<metric> (list),
    fold_<metric>_mean, fold_<metric>_sd (population SD, ddof 0, as np.std in the gating notebook) and
    fold_<metric>_sd_sample (ddof 1). The per-fold AUROC is NaN when a class is absent from the fold."""
    rows = []
    for k in sorted(np.unique(folds)):
        m = folds == k
        mm = metrics_from_preds(y[m], P[m])
        rows.append({'fold': int(k), 'n': int(m.sum()), 'n_patients': int(len(np.unique(patients[m]))),
                     'n_classes_present': int(len(np.unique(y[m]))),
                     'accuracy': mm['accuracy'], 'macro_f1': mm['macro_f1'],
                     'balanced_accuracy': mm['balanced_accuracy'], 'auroc': mm['auroc']})
    summ = {'n_folds': len(rows)}
    for key in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc'):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        summ[f'fold_{key}'] = [r[key] for r in rows]
        summ[f'fold_{key}_mean'] = float(v.mean()) if len(v) else float('nan')
        summ[f'fold_{key}_sd'] = float(v.std(ddof=0)) if len(v) else float('nan')
        summ[f'fold_{key}_sd_sample'] = float(v.std(ddof=1)) if len(v) > 1 else float('nan')
        summ[f'fold_{key}_n_defined'] = int(len(v))
    summ['fold_auroc_n_undefined'] = int(len(rows) - summ['fold_auroc_n_defined'])
    summ['sd_definition'] = 'fold_*_sd: population SD (ddof 0, gating notebook); fold_*_sd_sample: sample SD (ddof 1)'
    return rows, summ


def score_level(y, P, patients, folds, n_boot, seed):
    pooled = metrics_from_preds(y, P)
    sklearn_crosscheck(y, P)
    pooled['ci95'] = bootstrap_ci(y, P, patients, n_boot, seed)
    per_fold, summ = fold_table(y, P, folds, patients)
    C = P.shape[1]
    counts = np.bincount(y, minlength=C)
    Pm = np.zeros((len(y), C)); Pm[:, int(counts.argmax())] = 1.0
    mb = metrics_from_preds(y, Pm)
    return {'pooled': pooled, 'per_fold': per_fold, 'fold_summary': summ,
            'majority_baseline': {'majority_class': int(counts.argmax()), 'accuracy': mb['accuracy'],
                                  'macro_f1': mb['macro_f1'], 'balanced_accuracy': mb['balanced_accuracy']},
            'class_counts': counts.tolist(), 'n_patients': int(len(np.unique(patients)))}


def gating_style_entry(level):
    """The CNN result of one level in the layout of arms.<task>.<arm>.<clf> of the gating results.json:
    rates and their CIs in percent, AUROC / AUPRC in [0, 1], fold_* lists in percent, fold_*_sd = population SD."""
    p, ci, fs = level['pooled'], level['pooled']['ci95'], level['fold_summary']
    pc = lambda v: None if v is None or not np.isfinite(v) else 100.0 * float(v)
    pci = lambda c: [pc(c[0]), pc(c[1])]
    return {'n': p['n'], 'n_patients': level['n_patients'],
            'accuracy': pc(p['accuracy']), 'accuracy_ci': pci(ci['accuracy']),
            'macro_f1': pc(p['macro_f1']), 'macro_f1_ci': pci(ci['macro_f1']),
            'balanced_accuracy': pc(p['balanced_accuracy']), 'balanced_accuracy_ci': pci(ci['balanced_accuracy']),
            'sensitivity': [pc(v) for v in p['per_class_recall']], 'sensitivity_ci': [pci(c) for c in ci['per_class_recall']],
            'specificity': [pc(v) for v in p['per_class_specificity']], 'specificity_ci': [pci(c) for c in ci['per_class_specificity']],
            'confusion_matrix': p['confusion_matrix'],
            'auroc': (float(p['auroc']) if np.isfinite(p['auroc']) else None), 'auprc': (float(p['auprc_macro']) if np.isfinite(p['auprc_macro']) else None),
            'auroc_ci': [(float(v) if np.isfinite(v) else None) for v in ci.get('auroc', [float('nan')] * 2)],
            'auprc_ci': [(float(v) if np.isfinite(v) else None) for v in ci.get('auprc_macro', [float('nan')] * 2)],
            'score_note': '' if np.isfinite(p['auroc']) else 'AUROC undefined: a class is absent',
            'fold_macro_f1': [pc(v) for v in fs['fold_macro_f1']], 'fold_macro_f1_mean': pc(fs['fold_macro_f1_mean']),
            'fold_macro_f1_sd': pc(fs['fold_macro_f1_sd']),
            'fold_accuracy': [pc(v) for v in fs['fold_accuracy']], 'fold_accuracy_mean': pc(fs['fold_accuracy_mean']),
            'fold_accuracy_sd': pc(fs['fold_accuracy_sd'])}


RESULTS = {'meta': {}, 'tasks': {}, 'label_distributions': LABEL_DISTRIBUTIONS, 'arms': {},
           'arms_schema': ('arms.<gating task>.cnn_pcg_only mirrors arms.<task>.<arm>.<clf> of the gating results.json: '
                           'n, n_patients, majority (majority_class, accuracy, macro_f1, class_counts) and the metric '
                           'entries in percent (accuracy, macro_f1, balanced_accuracy, sensitivity, specificity with *_ci; '
                           'auroc, auprc in [0,1]; confusion_matrix; fold_macro_f1, fold_accuracy lists with _mean and _sd, '
                           'population SD). Recording level; chunk_level holds the same entries at chunk level. '
                           'Task names follow the gating notebook (detection = murmur).')}
OOF = {}
with StageTimer('scoring'):
    for task in TASKS:
        preds = CV_PREDS[task]
        if not preds:
            print(f'{task}: no predictions')
            continue
        c_rec = np.concatenate([p['rec'] for p in preds])
        c_ids = np.concatenate([p['chunk_ids'] for p in preds])
        c_P = np.concatenate([p['probs'] for p in preds]).astype(np.float64)
        c_y = np.concatenate([p['y'] for p in preds]).astype(int)
        c_fold = np.concatenate([np.full(len(p['y']), p['fold']) for p in preds])
        c_pat = np.array([REC_PATIENT[r] for r in c_rec])
        recs, P, y, fold, n_chunks, pats = aggregate_recordings(c_rec, c_P, c_y, c_fold)
        assert set(recs) == set(EVAL_RECS[task]) or SUBSET is not None or FOLD_INTEGRITY[task]['n_recordings_in_test_folds'] != len(EVAL_RECS[task]), \
            f'{task}: recording set of the OOF predictions differs from the evaluation set'
        rec_level = score_level(y, P, pats, fold, N_BOOT_RUN, SEED)
        chunk_level = score_level(c_y, c_P, c_pat, c_fold, N_BOOT_RUN, SEED)
        RESULTS['tasks'][task] = {
            'class_names': CLASS_NAMES[task], 'label_mode': LABEL_MODE, 'n_recordings': int(len(recs)),
            'n_patients': int(len(np.unique(pats))), 'n_chunks': int(len(c_y)), 'n_folds': int(len(preds)),
            'recording_level': rec_level, 'chunk_level': chunk_level,
            'training': [{k: v for k, v in fi.items() if k != 'history'} for fi in CV_TRAIN_INFO[task]],
        }
        # the same numbers in the gating notebook's layout (C-11): arms.<gating task>.cnn_pcg_only
        mbl = rec_level['majority_baseline']
        RESULTS['arms'][GATING_TASK[task]] = {'cnn_pcg_only': {
            'n': int(len(recs)), 'n_patients': int(len(np.unique(pats))),
            'majority': {'majority_class': mbl['majority_class'], 'accuracy': 100.0 * mbl['accuracy'],
                         'macro_f1': 100.0 * mbl['macro_f1'], 'class_counts': rec_level['class_counts']},
            **gating_style_entry(rec_level),
            'chunk_level': gating_style_entry(chunk_level),
            'level': 'recording (mean chunk softmax, argmax)', 'label_mode': LABEL_MODE, 'n_folds': int(len(preds))}}
        OOF.update({f'{task}__rec_names': recs, f'{task}__patient_ids': pats, f'{task}__y_true': y,
                    f'{task}__probs': P.astype(np.float32), f'{task}__pred': P.argmax(1), f'{task}__fold': fold,
                    f'{task}__n_chunks': n_chunks, f'{task}__chunk_rec': c_rec, f'{task}__chunk_ids': c_ids,
                    f'{task}__chunk_y': c_y, f'{task}__chunk_probs': c_P.astype(np.float32), f'{task}__chunk_fold': c_fold,
                    f'{task}__class_names': np.array(CLASS_NAMES[task])})
        pr, ch = rec_level['pooled'], chunk_level['pooled']
        fs = rec_level['fold_summary']
        print(f"\n{task.upper()} (n={len(recs)} recordings, {len(np.unique(pats))} patients, {len(c_y)} chunks, {len(preds)} folds)")
        print(f"  recording level: acc {pr['accuracy'] * 100:5.1f} [{pr['ci95']['accuracy'][0] * 100:.1f}, {pr['ci95']['accuracy'][1] * 100:.1f}]"
              f" | macro-F1 {pr['macro_f1'] * 100:5.1f} [{pr['ci95']['macro_f1'][0] * 100:.1f}, {pr['ci95']['macro_f1'][1] * 100:.1f}]"
              f" | bal acc {pr['balanced_accuracy'] * 100:5.1f} | AUROC {pr['auroc']:.3f} | AUPRC {pr['auprc_macro']:.3f}")
        print(f"  per fold ({fs['n_folds']}): acc {fs['fold_accuracy_mean'] * 100:.1f} +/- {fs['fold_accuracy_sd'] * 100:.1f} | "
              f"macro-F1 {fs['fold_macro_f1_mean'] * 100:.1f} +/- {fs['fold_macro_f1_sd'] * 100:.1f} (population SD; sample SD "
              f"{fs['fold_macro_f1_sd_sample'] * 100:.1f}) | AUROC undefined in {fs['fold_auroc_n_undefined']} folds")
        print(f"  sklearn cross-check passed (accuracy, macro-F1 on labels 0..{N_CLASSES[task] - 1}, balanced accuracy, AUROC)")
        print(f"  chunk level    : acc {ch['accuracy'] * 100:5.1f} | macro-F1 {ch['macro_f1'] * 100:5.1f} | AUROC {ch['auroc']:.3f}")
        print(f"  per-class recall (recording): " + ', '.join(f'{n} {v * 100:.1f}' for n, v in zip(CLASS_NAMES[task], pr['per_class_recall'])))
        print(f"  majority baseline: acc {rec_level['majority_baseline']['accuracy'] * 100:.1f}, macro-F1 {rec_level['majority_baseline']['macro_f1'] * 100:.1f}")
        print(f"  confusion (rows true, cols predicted): {pr['confusion_matrix']}")

    RESULTS['meta'] = {'notebook': 'R1_CNN_Baseline.ipynb', 'baseline': 'PCG-only CNN+attention (Milestone-3 architecture)',
                       'run_dir': RUN_DIR, 'label_mode': LABEL_MODE, 'fold_source': FOLD_SOURCE, 'dry_run': DRY_RUN,
                       'subset_n': SUBSET_N, 'epochs': EPOCHS_RUN, 'n_boot': N_BOOT_RUN, 'seed': SEED,
                       'decision_rule': 'recording = argmax of mean chunk softmax; chunk = argmax of chunk softmax',
                       'timestamp': datetime.now().isoformat(timespec='seconds')}

    def to_jsonable(o):
        if isinstance(o, dict):
            return {str(k): to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_jsonable(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and math.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else float(o)
        if isinstance(o, np.ndarray):
            return to_jsonable(o.tolist())
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as fh:
        json.dump(to_jsonable(RESULTS), fh, indent=1)
    OOF['meta'] = np.array(json.dumps(to_jsonable(RESULTS['meta'])))
    np.savez(os.path.join(OUT_DIR, 'oof_predictions.npz'), **OOF)
    print(f"\nresults.json and oof_predictions.npz written to {OUT_DIR}")
''')

md("""
## 11. Continuity: the original 70/15/15 patient split

This section exists so the article can state exactly why the published 93.4 / 67.4 / 68.4 / 58.8 differ from the new baseline row. The original patient split is loaded from `Processed-Data/patient_splits.json` when present and recreated with the verbatim `train_test_split` calls otherwise (both are compared when both exist). Two runs are made on that split:

* C1, original protocol: location-aware labels; balanced training chunks pooled with the validation chunks and re-split 80/20 at chunk level for early stopping (not patient-disjoint, as in the original); test on the test split. Reported at chunk level (the published numbers) and at recording level, under both label codings at test time.
* C2, corrected protocol on the same split: patient-level labels; early stopping on the original validation split (patient-disjoint); test on the test split; chunk and recording level.

Together with Section 10 this separates the four factors: chunk versus recording level (C1 chunk vs C1 recording), label coding (C1 recording under both codings, and C1 vs C2), early-stopping leakage (C1 vs C2), and single split versus ten folds (C2 vs Section 10). Not used for Table 2.
""")

code(r'''
CONTINUITY = {}


def recreate_original_split():
    """Verbatim split logic of MILESTONE3_DATA_PREPARATION cell 19 on the recording table (sorted wav order)."""
    df_recordings = df_rec
    df_valid_patients = df_recordings[df_recordings['Murmur'] != 'Unknown'].drop_duplicates('patient_id')
    patient_ids = df_valid_patients['patient_id'].tolist()
    patient_labels = df_valid_patients['Murmur'].tolist()  # Stratify by murmur status
    train_patients, temp_patients, train_labels, temp_labels = train_test_split(
        patient_ids, patient_labels,
        train_size=CONFIG['train_ratio'],
        stratify=patient_labels,
        random_state=CONFIG['random_state']
    )
    val_ratio_adjusted = CONFIG['val_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio'])
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=CONFIG['random_state']
    )
    return {'train': sorted(int(p) for p in train_patients), 'val': sorted(int(p) for p in val_patients),
            'test': sorted(int(p) for p in test_patients)}


def score_split_predictions(chunk_ids, P, y_chunk, tag):
    """Chunk-level and recording-level metrics for one split evaluation."""
    if len(chunk_ids) == 0:
        print(f'    {tag}: no test chunks on this recording set; not scored')
        return {'n_chunks': 0, 'n_recordings': 0, 'note': 'empty test part'}
    recs = CT['rec'].values[chunk_ids]
    pats = CT['patient_id'].values[chunk_ids]
    fold0 = np.zeros(len(chunk_ids), dtype=int)
    ch = metrics_from_preds(y_chunk, P)
    ch['ci95'] = bootstrap_ci(y_chunk, P, pats, N_BOOT_RUN, SEED)
    r_recs, rP, ry, _, n_ch, rpats = aggregate_recordings(recs, P, y_chunk, fold0)
    rl = metrics_from_preds(ry, rP)
    rl['ci95'] = bootstrap_ci(ry, rP, rpats, N_BOOT_RUN, SEED)
    print(f"    {tag}: chunk acc {ch['accuracy'] * 100:.2f} (n={len(y_chunk)}), macro-F1 {ch['macro_f1'] * 100:.1f}, AUROC {ch['auroc']:.4f}"
          f" | recording acc {rl['accuracy'] * 100:.2f} (n={len(ry)}), macro-F1 {rl['macro_f1'] * 100:.1f}, AUROC {rl['auroc']:.4f}")
    return {'chunk_level': ch, 'recording_level': rl, 'n_chunks': int(len(y_chunk)), 'n_recordings': int(len(ry))}


def chunk_level_es_split(X, labels, stratify):
    """Verbatim early-stopping split of get_data_loaders_v5: chunk-level 80/20, stratified, random_state=SEED."""
    val_split = VAL_SPLIT
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=stratify,
        random_state=SEED
    )
    return train_idx, val_idx


if RUN_CONTINUITY:
    with StageTimer('continuity_original_split'):
        recreated = recreate_original_split()
        split_from_file = None
        if os.path.isfile(ORIGINAL_SPLITS_JSON):
            with open(ORIGINAL_SPLITS_JSON) as fh:
                split_from_file = {k: sorted(int(p) for p in v) for k, v in json.load(fh).items()}
            identical = all(split_from_file[k] == recreated[k] for k in ('train', 'val', 'test'))
            print(f'original patient_splits.json found: train/val/test patients '
                  f'{[len(split_from_file[k]) for k in ("train", "val", "test")]}; recreated split identical: {identical}')
            SPLIT = split_from_file
            split_source = 'drive_file'
        else:
            identical = None
            SPLIT = recreated
            split_source = 'recreated_verbatim'
            print(f'original patient_splits.json not found; recreated split: train/val/test patients '
                  f'{[len(SPLIT[k]) for k in ("train", "val", "test")]}')
        n_index_non_unknown = int((df_rec['Murmur'] != 'Unknown').sum())
        CONTINUITY['split'] = {'source': split_source, 'identical_to_recreated': identical,
                               'n_patients': {k: len(v) for k, v in SPLIT.items()},
                               'file': ORIGINAL_SPLITS_JSON if split_from_file else None,
                               'recording_set': (f'gating kept set: the {len(KEPT_ALL)} recordings of the fold files'
                                                 + (f' (subset of {SUBSET_N} active)' if SUBSET_N else '')
                                                 + f', not the {n_index_non_unknown} non-Unknown recordings of the original index'),
                               'n_recordings_kept_set': int(len(KEPT_ALL)),
                               'n_recordings_index_non_unknown': n_index_non_unknown}
        print(f'recording set for the continuity runs: gating kept set, {len(KEPT_ALL)} recordings '
              f'(original non-Unknown index: {n_index_non_unknown})')
        pat_split = {}
        for k, v in SPLIT.items():
            for p in v:
                pat_split[int(p)] = k
        CT_SPLIT = CT['patient_id'].map(pat_split).fillna('none').values
        n_recs = {k: int(CT.loc[CT_SPLIT == k, 'rec'].nunique()) for k in ('train', 'val', 'test')}
        n_chunks = {k: int((CT_SPLIT == k).sum()) for k in ('train', 'val', 'test')}
        print(f'recordings per split (kept set): {n_recs}; chunks per split: {n_chunks}')
        CONTINUITY['split']['n_recordings'] = n_recs
        CONTINUITY['split']['n_chunks'] = n_chunks

        LOC = CT_LABELS_BY_MODE['location']
        PAT = CT_LABELS_BY_MODE['patient']
        L3_loc = np.stack([LOC[t] for t in CHAR_TASKS], axis=1)
        L3_pat = np.stack([PAT[t] for t in CHAR_TASKS], axis=1)
        tr_ids = np.flatnonzero(CT_SPLIT == 'train')
        va_ids = np.flatnonzero(CT_SPLIT == 'val')
        te_ids = np.flatnonzero(CT_SPLIT == 'test')
        cont_epochs = EPOCHS_RUN

        def parts_ok(tag, strat=None, **parts):
            """A continuity run needs non-empty train, validation and test parts and, for the chunk-level
            stratified early-stopping split, at least two members per class. On a DRY_RUN / LIMIT subset
            the original 70/15/15 split can leave a part empty; the run is then skipped and recorded."""
            empty = [k for k, v in parts.items() if len(v) == 0]
            why = f'empty {", ".join(empty)} part' if empty else None
            if why is None and strat is not None:
                cnt = np.bincount(np.asarray(strat, dtype=int))
                if cnt[cnt > 0].min() < 2 or len(strat) < 5:
                    why = 'too few chunks per class for the stratified chunk-level early-stopping split'
            if why:
                msg = f'{tag}: skipped, {why} on this recording set ({len(KEPT_ALL)} recordings{", subset active" if SUBSET_N else ""})'
                print('  ' + msg)
                CONTINUITY.setdefault('skipped', []).append(msg)
                return False
            return True

        # ---------------- C1: original protocol (location labels, chunk-level pooled ES split) ----------------
        print('\nC1: original protocol')
        CONTINUITY['C1_original_protocol'] = {}
        np.random.seed(SEED)                      # the augmentation notebook seeded 42 once, then ran murmur then joint balancing
        m_tr = tr_ids[LOC['murmur'][tr_ids] >= 0]
        m_va = va_ids[LOC['murmur'][va_ids] >= 0]
        m_te = te_ids[LOC['murmur'][te_ids] >= 0]
        pool_ml = np.array([i for i in tr_ids if LOC['murmur'][i] == 1 and (L3_loc[i] >= 0).all()], dtype=int)
        ml_va = np.array([i for i in va_ids if LOC['murmur'][i] == 1 and (L3_loc[i] >= 0).all()], dtype=int)
        ml_te = np.array([i for i in te_ids if LOC['murmur'][i] == 1 and (L3_loc[i] >= 0).all()], dtype=int)
        if parts_ok('C1 murmur', train=m_tr, val=m_va, test=m_te,
                    strat=np.concatenate([LOC['murmur'][m_tr], LOC['murmur'][m_va]])):
            aug_audio, src, bal1 = murmur_balance(m_tr, LOC['murmur'], seed=None)
            X_bal = np.concatenate([MFCC[m_tr], mfcc_parallel(aug_audio)]) if len(src) else MFCC[m_tr]
            y_bal = np.concatenate([LOC['murmur'][m_tr], LOC['murmur'][src]]) if len(src) else LOC['murmur'][m_tr]
            shuffle_idx = np.random.permutation(len(X_bal))
            X_bal, y_bal = X_bal[shuffle_idx], y_bal[shuffle_idx]
            print(f'  murmur train chunks after balancing {len(y_bal)} ({bal1})')
            X_pool = np.concatenate([X_bal, MFCC[m_va]]); y_pool = np.concatenate([y_bal, LOC['murmur'][m_va]])
            es_tr, es_va = chunk_level_es_split(X_pool, y_pool, y_pool)
            model, tinfo1 = train_model('murmur', X_pool[es_tr], y_pool[es_tr], X_pool[es_va], y_pool[es_va], cont_epochs,
                                        SEED, 'C1 murmur')
            P_te = predict_probs(model, 'murmur', MFCC[m_te]).astype(np.float64)
            CONTINUITY['C1_original_protocol']['murmur'] = {
                'test_labels_location': score_split_predictions(m_te, P_te, LOC['murmur'][m_te], 'C1 murmur, location labels'),
                'training': {k: v for k, v in tinfo1.items() if k != 'history'}, 'balancing': bal1,
                'es_split': 'chunk-level 80/20 of pooled balanced-train + val chunks (original, not patient-disjoint)'}
            m_te_pat = m_te[PAT['murmur'][m_te] >= 0]
            P_te_pat = P_te[np.isin(m_te, m_te_pat)]
            CONTINUITY['C1_original_protocol']['murmur']['test_labels_patient'] = score_split_predictions(
                m_te_pat, P_te_pat, PAT['murmur'][m_te_pat], 'C1 murmur, patient labels')
            del model

        if parts_ok('C1 multilabel', train=pool_ml, val=ml_va, test=ml_te,
                    strat=np.concatenate([L3_loc[pool_ml, 0], L3_loc[ml_va, 0]])):
            aug_ml, src_ml, bal2 = joint_balance_augment(pool_ml, L3_loc[pool_ml], seed=None, max_iterations=JOINT_MAX_ITERATIONS)
            X_ml = np.concatenate([MFCC[pool_ml], mfcc_parallel(aug_ml)]) if len(src_ml) else MFCC[pool_ml]
            Y_ml = np.concatenate([L3_loc[pool_ml], L3_loc[src_ml]]) if len(src_ml) else L3_loc[pool_ml]
            shuffle_idx = np.random.permutation(len(X_ml))
            X_ml, Y_ml = X_ml[shuffle_idx], Y_ml[shuffle_idx]
            print(f'  multilabel train chunks after joint balancing {len(Y_ml)} ({bal2["iterations"]} iterations, '
                  f'balanced={bal2["balanced"]}, limit {JOINT_MAX_ITERATIONS})')
            X_pool = np.concatenate([X_ml, MFCC[ml_va]]); Y_pool = np.concatenate([Y_ml, L3_loc[ml_va]])
            es_tr, es_va = chunk_level_es_split(X_pool, Y_pool, Y_pool[:, 0])
            model, tinfo2 = train_model('multilabel', X_pool[es_tr], Y_pool[es_tr], X_pool[es_va], Y_pool[es_va], cont_epochs,
                                        SEED, 'C1 multilabel')
            P_ml = predict_probs(model, 'multilabel', MFCC[ml_te])
            for i, t in enumerate(CHAR_TASKS):
                CONTINUITY['C1_original_protocol'][t] = {
                    'test_labels_location': score_split_predictions(ml_te, P_ml[t].astype(np.float64), L3_loc[ml_te, i], f'C1 {t}, location labels')}
                ok = (L3_pat[ml_te, i] >= 0)
                if ok.any():
                    CONTINUITY['C1_original_protocol'][t]['test_labels_patient'] = score_split_predictions(
                        ml_te[ok], P_ml[t][ok].astype(np.float64), L3_pat[ml_te[ok], i], f'C1 {t}, patient labels')
            CONTINUITY['C1_original_protocol']['multilabel_training'] = {k: v for k, v in tinfo2.items() if k != 'history'}
            CONTINUITY['C1_original_protocol']['multilabel_balancing'] = bal2
            del model
        # test chunk counts against the published single-split evaluation (2551 murmur, 313 multilabel chunks)
        CONTINUITY['C1_original_protocol']['test_chunk_counts'] = {
            'murmur': {'this_run': int(len(m_te)), 'published': PUBLISHED_ORIGINAL['murmur_test_chunks']},
            'multilabel': {'this_run': int(len(ml_te)), 'published': PUBLISHED_ORIGINAL['multilabel_test_chunks']},
            'note': 'this run evaluates on the gating kept set (and on the DRY_RUN / LIMIT subset when active); '
                    'the published counts come from the full non-Unknown set'}
        print(f'  C1 test chunks: murmur {len(m_te)} (published {PUBLISHED_ORIGINAL["murmur_test_chunks"]}); '
              f'multilabel {len(ml_te)} (published {PUBLISHED_ORIGINAL["multilabel_test_chunks"]})')

        # ---------------- C2: corrected protocol on the same split (patient labels, ES on the val split) ----------------
        print('\nC2: corrected protocol on the same split')
        CONTINUITY['C2_corrected_single_split'] = {}
        m_tr = tr_ids[PAT['murmur'][tr_ids] >= 0]
        m_va = va_ids[PAT['murmur'][va_ids] >= 0]
        m_te = te_ids[PAT['murmur'][te_ids] >= 0]
        if parts_ok('C2 murmur', train=m_tr, val=m_va, test=m_te):
            aug_audio, src, bal3 = murmur_balance(m_tr, PAT['murmur'], seed=SEED)
            X_bal = np.concatenate([MFCC[m_tr], mfcc_parallel(aug_audio)]) if len(src) else MFCC[m_tr]
            y_bal = np.concatenate([PAT['murmur'][m_tr], PAT['murmur'][src]]) if len(src) else PAT['murmur'][m_tr]
            model, tinfo3 = train_model('murmur', X_bal, y_bal, MFCC[m_va], PAT['murmur'][m_va], cont_epochs, SEED, 'C2 murmur')
            P_te = predict_probs(model, 'murmur', MFCC[m_te]).astype(np.float64)
            CONTINUITY['C2_corrected_single_split']['murmur'] = {
                'test_labels_patient': score_split_predictions(m_te, P_te, PAT['murmur'][m_te], 'C2 murmur, patient labels'),
                'training': {k: v for k, v in tinfo3.items() if k != 'history'}, 'balancing': bal3,
                'es_split': 'original val split (patient-disjoint)'}
            del model
        pool_ml = np.array([i for i in tr_ids if PAT['murmur'][i] == 1 and (L3_pat[i] >= 0).all()], dtype=int)
        ml_va = np.array([i for i in va_ids if PAT['murmur'][i] == 1 and (L3_pat[i] >= 0).all()], dtype=int)
        ml_te_all = np.array([i for i in te_ids if PAT['murmur'][i] == 1], dtype=int)
        if parts_ok('C2 multilabel', train=pool_ml, val=ml_va, test=ml_te_all):
            aug_ml, src_ml, bal4 = joint_balance_augment(pool_ml, L3_pat[pool_ml], seed=SEED, max_iterations=JOINT_MAX_ITERATIONS)
            X_ml = np.concatenate([MFCC[pool_ml], mfcc_parallel(aug_ml)]) if len(src_ml) else MFCC[pool_ml]
            Y_ml = np.concatenate([L3_pat[pool_ml], L3_pat[src_ml]]) if len(src_ml) else L3_pat[pool_ml]
            model, tinfo4 = train_model('multilabel', X_ml, Y_ml, MFCC[ml_va], L3_pat[ml_va], cont_epochs, SEED, 'C2 multilabel')
            P_ml = predict_probs(model, 'multilabel', MFCC[ml_te_all])
            for i, t in enumerate(CHAR_TASKS):
                ok = L3_pat[ml_te_all, i] >= 0
                CONTINUITY['C2_corrected_single_split'][t] = {'test_labels_patient': score_split_predictions(
                    ml_te_all[ok], P_ml[t][ok].astype(np.float64), L3_pat[ml_te_all[ok], i], f'C2 {t}, patient labels')}
            CONTINUITY['C2_corrected_single_split']['multilabel_training'] = {k: v for k, v in tinfo4.items() if k != 'history'}
            CONTINUITY['C2_corrected_single_split']['multilabel_balancing'] = bal4
            del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        CONTINUITY['published_original'] = PUBLISHED_ORIGINAL
        RESULTS['continuity'] = CONTINUITY
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as fh:
            json.dump(to_jsonable(RESULTS), fh, indent=1)

        print('\nSummary (accuracy %, chunk level -> recording level):')
        for t in TASKS:
            c1 = CONTINUITY['C1_original_protocol'].get(t, {}).get('test_labels_location')
            c1p = CONTINUITY['C1_original_protocol'].get(t, {}).get('test_labels_patient')
            c2 = CONTINUITY['C2_corrected_single_split'].get(t, {}).get('test_labels_patient')
            cv = RESULTS['tasks'].get(t)
            pub = PUBLISHED_ORIGINAL.get(f'{t}_accuracy')
            line = f'  {t:8s} published {pub:5.1f}'
            if c1 and 'chunk_level' in c1:
                line += f" | C1 loc {c1['chunk_level']['accuracy'] * 100:5.1f} -> {c1['recording_level']['accuracy'] * 100:5.1f}"
            if c1p and 'recording_level' in c1p:
                line += f" | C1 pat-labels rec {c1p['recording_level']['accuracy'] * 100:5.1f}"
            if c2 and 'chunk_level' in c2:
                line += f" | C2 {c2['chunk_level']['accuracy'] * 100:5.1f} -> {c2['recording_level']['accuracy'] * 100:5.1f}"
            if cv:
                line += f" | {cv['n_folds']}-fold rec {cv['recording_level']['pooled']['accuracy'] * 100:5.1f} (macro-F1 {cv['recording_level']['pooled']['macro_f1'] * 100:.1f})"
            print(line)
else:
    print('RUN_CONTINUITY = False: section skipped')
''')

md("""
## 12. Figures (European Heart Journal standard)

Vector PDF plus 600 dpi PNG, 88 mm (single column) or 180 mm (double column), sans-serif, minimum 8 pt, palette blue #184f95 / amber #eda100 / greys, fixed shared y ranges for percentages, CIs as error bars, panel letters.

* B1: recording-level macro-F1 per task with 95% CI (baseline alone).
* B3: chunk-level versus recording-level macro-F1 and accuracy per task.
* B4: recording-level confusion matrices (supplement).
* B2 (Section 13): baseline versus gated arms forest plot. Upper row: pooled macro-F1 with 95% CI; an arm's point and interval come from the gating `results.json` when every recording of the arm is aligned to the baseline (the article run) and otherwise from the patient bootstrap on the aligned recordings (a DRY_RUN or LIMIT subset), so point and interval always refer to one recording set. Lower row: paired differences with one shared x range. The rows are the arms the gating file declares (A, B, C, D, E, E2; MLP and RF), at the same height in every panel; ablation groups stored in the same file (jitter, window length, scaling, redundancy) are compared in `paired_vs_gated.json` and left out of the figure.
""")

code(r'''
MM = 1 / 25.4
BLUE, AMBER, GREY, LIGHT, DARK = '#184f95', '#eda100', '#7f7f7f', '#c8c8c8', '#333333'
plt.rcParams.update({
    'font.family': 'DejaVu Sans',      # explicit, as in the gating notebook: same output on Colab and on a local machine
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
TASK_LABELS = {'murmur': 'Detection', 'grading': 'Grading', 'timing': 'Timing', 'shape': 'Shape'}


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=600)
    plt.close(fig)
    print(f'  saved {name}.pdf / .png')


def panel_letter(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')


with StageTimer('figures'):
    tasks_done = [t for t in TASKS if t in RESULTS['tasks']]
    # B1
    fig, ax = plt.subplots(figsize=(88 * MM, 60 * MM))
    xs = np.arange(len(tasks_done))
    vals = np.array([RESULTS['tasks'][t]['recording_level']['pooled']['macro_f1'] * 100 for t in tasks_done])
    lo = np.array([RESULTS['tasks'][t]['recording_level']['pooled']['ci95']['macro_f1'][0] * 100 for t in tasks_done])
    hi = np.array([RESULTS['tasks'][t]['recording_level']['pooled']['ci95']['macro_f1'][1] * 100 for t in tasks_done])
    maj = np.array([RESULTS['tasks'][t]['recording_level']['majority_baseline']['macro_f1'] * 100 for t in tasks_done])
    ax.bar(xs, vals, width=0.6, color=BLUE, yerr=[vals - lo, hi - vals], capsize=2, error_kw={'lw': 0.8, 'color': DARK})
    ax.scatter(xs, maj, marker='_', s=120, color=AMBER, linewidths=1.2, zorder=3, label='Majority-class macro-F1')
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{TASK_LABELS[t]}\n(n={RESULTS['tasks'][t]['n_recordings']})" for t in tasks_done])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Macro-F1 (%), recording level')
    ax.legend(loc='upper right')
    save_fig(fig, 'B1_cnn_baseline_macro_f1_recording')

    # B3
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 65 * MM), sharey=True)
    for ax, key, letter, ylab in ((axes[0], 'macro_f1', 'A', 'Macro-F1 (%)'), (axes[1], 'accuracy', 'B', 'Accuracy (%)')):
        w = 0.36
        for j, (level, colour, lab) in enumerate((('chunk_level', LIGHT, 'Chunk level'), ('recording_level', BLUE, 'Recording level'))):
            v = np.array([RESULTS['tasks'][t][level]['pooled'][key] * 100 for t in tasks_done])
            l = np.array([RESULTS['tasks'][t][level]['pooled']['ci95'][key][0] * 100 for t in tasks_done])
            h = np.array([RESULTS['tasks'][t][level]['pooled']['ci95'][key][1] * 100 for t in tasks_done])
            ax.bar(xs + (j - 0.5) * w, v, width=w, color=colour, label=lab, yerr=[v - l, h - v], capsize=2,
                   error_kw={'lw': 0.8, 'color': DARK})
        ax.set_xticks(xs)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks_done])
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylab)
        ax.tick_params(labelleft=True)        # shared y range, numbers printed on both panels (each has its own axis label)
        panel_letter(ax, letter)
    axes[0].legend(loc='upper right')
    save_fig(fig, 'B3_chunk_vs_recording_level')

    # B4 confusion matrices (recording level, row-normalised, counts annotated)
    # two panels per row: at 180 mm a single row of four leaves the 8 pt cell annotations no room in DejaVu Sans
    n = len(tasks_done)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(180 * MM, 72 * nrows * MM), constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for k_ax, (ax, t, letter) in enumerate(zip(axes.ravel(), tasks_done, 'ABCD')):
        cm = np.array(RESULTS['tasks'][t]['recording_level']['pooled']['confusion_matrix'], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#ffffff', BLUE]), vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{int(cm[i, j])}\n{norm[i, j] * 100:.0f}%', ha='center', va='center', fontsize=8,
                        color='white' if norm[i, j] > 0.6 else DARK)
        names = CLASS_NAMES[t]
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels([nm.replace(' ', '\n') if len(nm) > 10 else nm for nm in names])   # 'Early systolic' on two lines
        ax.set_yticklabels(names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True' if k_ax % ncols == 0 else '')
        ax.set_title(TASK_LABELS[t])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
        panel_letter(ax, letter)
    save_fig(fig, 'B4_confusion_matrices_recording')
''')

md("""
## 13. Merge with the gating campaign (paired comparisons on identical folds)

Loads `<RUN_DIR>/oof_predictions.npz` and `<RUN_DIR>/results.json` written by `R1_Gating_Campaign.ipynb` (arrays `<task>__y`, `<task>__names`, `<task>__patient_ids`, `<task>__fold_id`, `<task>__<arm>__<clf>__proba`; task names detection / grading / timing / shape), discovers the arm / task / classifier groups, aligns every arm to this baseline by recording name and asserts that the gating labels equal the baseline labels on every aligned recording (the run stops on a disagreement). For each arm and task it computes the paired patient-level bootstrap difference in macro-F1, accuracy and balanced accuracy (arm minus baseline, same patient resamples, 95% CI and P(difference > 0)) and the fold-wise paired t-test on per-fold macro-F1 and accuracy over the shared folds. Writes `paired_vs_gated.json` and figure B2. If the gating outputs are absent or their arrays cannot be discovered, the cell reports that and writes a skipped status; `GATED_KEY_MAP` in the configuration cell can name the arrays by hand (see README).
""")

code(r'''
def discover_gated_keys(keys):
    """Group npz keys into (arm, task, classifier) -> {kind: key}. Tolerant to several naming schemes."""
    TASK_RE = {'murmur': r'murmur|detect', 'grading': r'grad', 'timing': r'timing|\btim\b', 'shape': r'shape'}
    KIND_RE = [('probs', r'prob'), ('pred', r'pred'), ('patient', r'patient|(^|_)pid|group'), ('fold', r'fold'),
               ('rec', r'rec_?names?|recording|record|names?$|^rec$|_rec$|(^|_)ids$'),
               ('y', r'y_true|label|target|true|(^|_)y$')]
    groups = {}
    for k in keys:
        low = k.lower()
        task = next((t for t, p in TASK_RE.items() if re.search(p, low)), None)
        kind = next((kd for kd, p in KIND_RE if re.search(p, low)), None)
        if kind is None or task is None:
            continue
        clf = 'rf' if re.search(r'(^|[^a-z])rf([^a-z]|$)', low) else ('mlp' if 'mlp' in low else None)
        tokens = [t for t in re.split(r'__|/|\|', k) if t]
        if len(tokens) == 1:
            tokens = [t for t in re.split(r'_', k) if t]
        arm_tokens = []
        for t in tokens:
            tl = t.lower()
            if re.search(TASK_RE[task], tl) or tl in ('mlp', 'rf') or re.search(dict(KIND_RE)[kind], tl):
                continue
            arm_tokens.append(t)
        arm = '_'.join(arm_tokens) or 'global'
        groups.setdefault((arm, task, clf), {})[kind] = k
    return groups


def per_fold_metric(y, P, folds, key):
    vals = []
    for k in sorted(np.unique(folds)):
        m = folds == k
        vals.append(metrics_from_preds(y[m], P[m])[key])
    return np.array(vals, dtype=float)


PAIRED = {'baseline': 'cnn_pcg_only', 'label_mode': LABEL_MODE, 'run_dir': RUN_DIR, 'arms': {}, 'status': 'ok'}
with StageTimer('merge_with_gated'):
    gated_oof_path = os.path.join(RUN_DIR, 'oof_predictions.npz')
    gated_res_path = os.path.join(RUN_DIR, 'results.json')
    if not os.path.isfile(gated_oof_path):
        PAIRED['status'] = f'skipped: {gated_oof_path} not found (run the gating campaign first)'
        print(PAIRED['status'])
    else:
        g = np.load(gated_oof_path, allow_pickle=True)
        gkeys = list(g.keys())
        print(f'gating oof_predictions.npz: {len(gkeys)} keys, e.g. {gkeys[:8]}')
        groups = discover_gated_keys(gkeys)
        table = []
        for (arm, task, clf), kinds in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], str(kv[0][2]))):
            table.append(f'  arm={arm!r:28s} task={task:8s} clf={clf!s:5s} kinds={sorted(kinds)}')
        print('\n'.join(table) if table else '  no (arm, task) groups discovered')
        if GATED_KEY_MAP:
            groups = {}
            for arm, per_task in GATED_KEY_MAP.items():
                for task, kinds in per_task.items():
                    groups[(arm, task, kinds.get('clf'))] = kinds
            print(f'GATED_KEY_MAP override in use: {list(GATED_KEY_MAP)}')
        # the gating file keeps names and labels once per task (<task>__names, <task>__y): the 'global' group
        global_rec = {t: next((kinds['rec'] for (a, tt, c), kinds in groups.items() if tt == t and 'rec' in kinds and 'probs' not in kinds), None)
                      for t in TASKS}
        global_y = {t: next((kinds['y'] for (a, tt, c), kinds in groups.items() if tt == t and 'y' in kinds and 'probs' not in kinds), None)
                    for t in TASKS}
        print(f'  per-task name arrays: {global_rec}\n  per-task label arrays: {global_y}')
        gated_results = None
        if os.path.isfile(gated_res_path):
            with open(gated_res_path) as fh:
                gated_results = json.load(fh)
        ARM_LETTERS = ((gated_results or {}).get('run', {}) or {}).get('arm_letters', {}) or {}
        KNOWN_ARM_LABELS = {'syn_fixed': 'A synthetic-fixed', 'syn_fusion': 'B synthetic-fusion',
                            'pcg_seg': 'C PCG segmentation', 'pcg_seg_fixed': 'D PCG-seg + fixed',
                            'equal_time': 'E equal-time 75 bpm', 'equal_time_matched': 'E2 equal-time matched'}

        def arm_display(arm, clf):
            name = KNOWN_ARM_LABELS.get(arm) or (f'{ARM_LETTERS[arm]} {arm}' if arm in ARM_LETTERS else arm)
            return name + (f' ({clf.upper()})' if clf else '')      # (MLP), (RF), as the manuscript writes them

        BASE_BOOT = {}
        n_arms = 0
        for (arm, task, clf), kinds in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], str(kv[0][2]))):
            if 'probs' not in kinds or task not in RESULTS['tasks'] or arm == 'global':
                continue
            rec_key = kinds.get('rec') or global_rec.get(task)
            if rec_key is None:
                print(f'  {arm}/{task}/{clf}: no recording-name array; skipped (set GATED_KEY_MAP)')
                continue
            arm_rec = np.array([_norm_rec(x) for x in np.asarray(g[rec_key]).ravel()])
            arm_P = np.asarray(g[kinds['probs']], dtype=np.float64)
            if arm_P.ndim == 1:
                arm_P = np.stack([1 - arm_P, arm_P], axis=1)
            if len(arm_rec) != len(arm_P):
                print(f'  {arm}/{task}/{clf}: {len(arm_rec)} names vs {len(arm_P)} probability rows; skipped')
                continue
            base_rec = OOF[f'{task}__rec_names']
            pos = {r: i for i, r in enumerate(arm_rec)}
            keep = np.array([r in pos for r in base_rec])
            if keep.sum() == 0:
                print(f'  {arm}/{task}/{clf}: no recording names in common; skipped')
                continue
            bi = np.flatnonzero(keep)
            ai = np.array([pos[r] for r in base_rec[keep]])
            y = OOF[f'{task}__y_true'][bi]
            # label cross-check: the gating labels must equal the baseline labels on every aligned recording
            y_key = kinds.get('y') or global_y.get(task)
            if y_key is None:
                raise AssertionError(f'{arm}/{task}/{clf}: no label array for the task in the gating file (expected '
                                     f'{GATING_TASK[task]}__y); the paired comparison cannot be verified')
            ay = np.asarray(g[y_key]).ravel().astype(int)
            assert len(ay) == len(arm_rec), f'{arm}/{task}: label array {y_key} has {len(ay)} rows for {len(arm_rec)} names'
            n_dis = int((ay[ai] != y).sum())
            if n_dis:
                msg = (f'{arm}/{task}/{clf}: {n_dis} of {len(y)} aligned recordings carry a different label in the gating '
                       f'file ({y_key}) than in this baseline (LABEL_MODE={LABEL_MODE})')
                if LABEL_MODE == 'location':
                    note_deviation('paired comparison', msg + '; expected under LABEL_MODE=location, arm skipped')
                    continue
                raise AssertionError(msg)
            Pb = OOF[f'{task}__probs'][bi].astype(np.float64)
            Pa = arm_P[ai]
            pats = OOF[f'{task}__patient_ids'][bi]
            folds = OOF[f'{task}__fold'][bi]
            label = arm_display(arm, clf)
            entry = {'task': task, 'gating_task': GATING_TASK[task], 'classifier': clf, 'arm': arm, 'n_aligned': int(len(bi)),
                     'n_baseline': int(len(base_rec)), 'n_arm': int(len(arm_rec)), 'arm_keys': kinds,
                     'label_check': f'{len(y)} aligned recordings, 0 label disagreements ({y_key})'}
            mb, ma = metrics_from_preds(y, Pb), metrics_from_preds(y, Pa)
            diff_keys = ('macro_f1', 'accuracy', 'balanced_accuracy')
            cache_key = (task, hashlib.sha1(bi.tobytes()).hexdigest())
            if cache_key not in BASE_BOOT:      # baseline resample metrics are shared by every arm aligned to the same recordings
                BASE_BOOT[cache_key] = [metrics_from_preds(y[idx], Pb[idx], light=True)
                                        for idx in patient_resamples(pats, N_BOOT_RUN, SEED)]
            diffs = {k: [] for k in diff_keys}
            arm_boot = {k: [] for k in diff_keys}
            for m2, idx in zip(BASE_BOOT[cache_key], patient_resamples(pats, N_BOOT_RUN, SEED)):
                m1 = metrics_from_preds(y[idx], Pa[idx], light=True)
                for k in diff_keys:
                    diffs[k].append(m1[k] - m2[k])
                    arm_boot[k].append(m1[k])
            pct95 = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k in diffs:
                d = np.array(diffs[k], dtype=float)
                # arm and baseline CIs on the ALIGNED recordings (same patient resamples as the difference); the gating
                # results.json CI refers to the arm's full kept set, which equals the aligned set only in a full run
                entry[k] = {'arm': ma[k], 'baseline': mb[k], 'diff': ma[k] - mb[k],
                            'arm_ci95_aligned': pct95(np.array(arm_boot[k], dtype=float)),
                            'baseline_ci95_aligned': pct95(np.array([m[k] for m in BASE_BOOT[cache_key]], dtype=float)),
                            'diff_ci95': pct95(d), 'p_diff_gt_0': float(np.mean(d > 0))}
                fa, fb = per_fold_metric(y, Pa, folds, k), per_fold_metric(y, Pb, folds, k)
                ok = np.isfinite(fa) & np.isfinite(fb)
                if ok.sum() >= 2:
                    tt = stats.ttest_rel(fa[ok], fb[ok])
                    entry[k]['fold_paired_t'] = {'t': float(tt.statistic), 'p': float(tt.pvalue), 'df': int(ok.sum() - 1),
                                                 'per_fold_arm': fa.tolist(), 'per_fold_baseline': fb.tolist(),
                                                 'mean_diff': float(np.mean(fa[ok] - fb[ok]))}
            PAIRED['arms'].setdefault(label, {})[task] = entry
            n_arms += 1
            e = entry['macro_f1']
            print(f"  {label:34s} {task:8s} n={len(bi):4d} macro-F1 arm {e['arm'] * 100:5.1f} vs CNN {e['baseline'] * 100:5.1f} "
                  f"diff {e['diff'] * 100:+5.1f} [{e['diff_ci95'][0] * 100:+.1f}, {e['diff_ci95'][1] * 100:+.1f}] "
                  f"P(>0)={e['p_diff_gt_0']:.3f}" + (f" fold t p={e['fold_paired_t']['p']:.3f}" if 'fold_paired_t' in e else ''))
        if n_arms > 0:
            print(f'label cross-check: gating labels equal the baseline labels on every aligned recording in all {n_arms} '
                  f'arm x task groups (0 disagreements)')
            PAIRED['label_cross_check'] = f'0 disagreements in all {n_arms} arm x task groups'
        if n_arms == 0:
            PAIRED['status'] = 'skipped: no gated arm could be aligned (see printed key table; use GATED_KEY_MAP)'
            print(PAIRED['status'])
    with open(os.path.join(OUT_DIR, 'paired_vs_gated.json'), 'w') as fh:
        json.dump(to_jsonable(PAIRED), fh, indent=1)

    # B2 forest plot
    if PAIRED['arms']:
        # Rows of the figure: the arms the gating file declares in its top-level 'arms' array, in that order
        # (A, B, C, D, E, E2), MLP before RF. Ablation groups found in the same file (jitter_*, window_*, ...)
        # stay in paired_vs_gated.json and are left out of the figure. Without a declared list every group is drawn.
        declared = [str(a) for a in np.asarray(g['arms']).ravel()] if 'arms' in gkeys else []
        _arm_of = {lab: next(iter(per_task.values())) for lab, per_task in PAIRED['arms'].items()}
        arms = [lab for lab in PAIRED['arms'] if _arm_of[lab]['arm'] in declared]
        if arms:
            arms.sort(key=lambda lab: (declared.index(_arm_of[lab]['arm']), {'mlp': 0, 'rf': 1}.get(_arm_of[lab]['classifier'], 2)))
            left_out = sorted(set(PAIRED['arms']) - set(arms))
            b2_rule = 'arms declared in the gating oof_predictions.npz (top-level arms array)'
        else:
            arms, left_out, b2_rule = list(PAIRED['arms']), [], 'every discovered group (no declared arms array)'
        PAIRED['b2'] = {'rows': arms, 'rule': b2_rule, 'left_out_of_figure': left_out}
        print(f'B2 rows: {len(arms)} ({b2_rule})' + (f'; {len(left_out)} other groups kept in paired_vs_gated.json only' if left_out else ''))
        tasks_done = [t for t in TASKS if t in RESULTS['tasks']]
        fig, axes = plt.subplots(2, len(tasks_done), figsize=(180 * MM, (50 + 7 * len(arms)) * MM), squeeze=False,
                                 constrained_layout=True)
        # fixed row positions shared by all panels: the labels are printed on the first panel only, so a row
        # must sit at the same height in every panel even when an arm has no result for a task
        top_pos = {lab: len(arms) - i for i, lab in enumerate(['CNN baseline'] + arms)}
        low_pos = {lab: len(arms) - 1 - i for i, lab in enumerate(arms)}
        # one shared x range for the difference panels, so the tasks are comparable by eye
        _all_ci = [abs(PAIRED['arms'][a][t]['macro_f1']['diff_ci95'][i]) * 100
                   for a in arms for t in tasks_done if t in PAIRED['arms'][a] for i in (0, 1)]
        diff_lim = max(5.0, max(_all_ci, default=5.0)) * 1.1
        n_clamped = 0
        for j, t in enumerate(tasks_done):
            ax = axes[0, j]
            rows = [('CNN baseline', RESULTS['tasks'][t]['recording_level']['pooled']['macro_f1'],
                     RESULTS['tasks'][t]['recording_level']['pooled']['ci95']['macro_f1'], BLUE)]
            for a in arms:
                e = PAIRED['arms'][a].get(t)
                if e:
                    # point and interval must come from the same recording set: the gating results.json values
                    # (arm's full kept set, percent) when every arm recording is aligned, else the aligned-set bootstrap
                    v, ci, src = e['macro_f1']['arm'], e['macro_f1']['arm_ci95_aligned'], 'patient bootstrap on the aligned recordings'
                    if e['n_aligned'] == e['n_arm'] and gated_results is not None:
                        try:
                            ge = gated_results['arms'][GATING_TASK[t]][e['arm']][e['classifier']]
                            if ge.get('macro_f1') is not None and ge.get('macro_f1_ci') and all(x is not None for x in ge['macro_f1_ci']):
                                v, ci, src = ge['macro_f1'] / 100.0, [ge['macro_f1_ci'][0] / 100.0, ge['macro_f1_ci'][1] / 100.0], 'gating results.json'
                        except (KeyError, TypeError):
                            pass
                    e['macro_f1']['b2_point_and_ci_source'] = src
                    rows.append((a, v, ci, AMBER))
            for name, v, ci, col in rows:
                yv = top_pos[name]
                if ci and all(np.isfinite(ci)):
                    lo_err, hi_err = (v - ci[0]) * 100, (ci[1] - v) * 100
                    if lo_err < 0 or hi_err < 0:
                        n_clamped += 1
                    ax.errorbar(v * 100, yv, xerr=[[max(lo_err, 0.0)], [max(hi_err, 0.0)]], fmt='o', color=col, ms=4, capsize=2, lw=0.8)
                else:
                    ax.plot(v * 100, yv, 'o', color=col, ms=4)
            ax.set_yticks([top_pos[lab] for lab in ['CNN baseline'] + arms])
            ax.set_yticklabels((['CNN baseline'] + arms) if j == 0 else [])
            ax.set_ylim(-0.6, len(arms) + 0.6)
            ax.set_xlim(0, 100); ax.set_xlabel('Macro-F1 (%)'); ax.set_title(TASK_LABELS[t])
            panel_letter(ax, 'ABCD'[j])
            ax = axes[1, j]
            panel_letter(ax, 'EFGH'[j])
            drows = [(a, PAIRED['arms'][a][t]['macro_f1']) for a in arms if t in PAIRED['arms'][a]]
            for name, e in drows:
                yv = low_pos[name]
                d, lo, hi = e['diff'] * 100, e['diff_ci95'][0] * 100, e['diff_ci95'][1] * 100
                ax.errorbar(d, yv, xerr=[[max(d - lo, 0.0)], [max(hi - d, 0.0)]], fmt='s', color=AMBER if lo > 0 or hi < 0 else GREY, ms=4, capsize=2, lw=0.8)
            ax.axvline(0, color=DARK, lw=0.6)
            ax.set_yticks([low_pos[lab] for lab in arms]); ax.set_yticklabels(arms if j == 0 else [])
            ax.set_ylim(-0.6, len(arms) - 0.4)
            ax.set_xlabel('Arm minus CNN,\nmacro-F1 (points)')
            ax.set_xlim(-diff_lim, diff_lim)
        if n_clamped:
            print(f'  WARNING B2: {n_clamped} interval(s) did not contain their point estimate and were clamped at the point; '
                  f'check paired_vs_gated.json b2_point_and_ci_source')
        with open(os.path.join(OUT_DIR, 'paired_vs_gated.json'), 'w') as fh:     # rewrite with the B2 source notes
            json.dump(to_jsonable(PAIRED), fh, indent=1)
        # legend below the lower row: what the markers and the two colours of the difference row mean
        _h = lambda marker, colour, text: matplotlib.lines.Line2D([], [], marker=marker, color=colour, ms=4, lw=0.8, label=text)
        fig.legend(handles=[_h('o', BLUE, 'CNN baseline, macro-F1 (95% CI)'),
                            _h('o', AMBER, 'Gated arm, macro-F1 (95% CI)'),
                            _h('s', AMBER, 'Difference, interval excludes zero'),
                            _h('s', GREY, 'Difference, interval includes zero')],
                   loc='outside lower center', ncol=2, fontsize=8, handletextpad=0.4, columnspacing=2.0)   # two columns: fits 180 mm
        save_fig(fig, 'B2_baseline_vs_gated_forest')
    else:
        print('B2 skipped (no aligned gated arms)')
''')

md("""
## 14. Run manifest

Versions, hardware, seeds, dataset resolution, filter design, label mode and distributions, chunk cache facts, fold files (paths, hashes, integrity counts), per-stage seconds, the deviation list, and the SHA-256 of every verbatim block.
""")

code(r'''
with StageTimer('manifest'):
    manifest = {
        'notebook': 'R1_CNN_Baseline.ipynb', 'built_by': 'build_cnn_baseline.py', 'run_started': RUN_STARTED,
        'run_finished': datetime.now().isoformat(timespec='seconds'),
        'total_seconds': time.time() - NOTEBOOK_T0, 'stage_seconds': STAGE_SECONDS,
        'environment': {'colab': IN_COLAB, 'device': str(DEVICE), 'gpu': GPU_NAME, 'gpu_memory_gb': GPU_MEM_GB,
                        'cpu_count': N_CPU, 'ram_gb': RAM_GB, 'versions': VERSIONS,
                        'cudnn_deterministic': torch.backends.cudnn.deterministic, 'amp': AMP_ENABLED,
                        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32},
        'seeds': {'global': SEED, 'per_fold_model_seed': 'SEED + fold', 'inner_validation_seed': INNER_VAL_SEED,
                  'inner_validation_seeds_used': {t: [fi['inner_val_seed'] for fi in CV_TRAIN_INFO.get(t, [])] for t in TASKS},
                  'bootstrap': SEED, 'balancing_rng': 'np.random.seed(SEED + fold) before each fold'},
        'config': {'DRY_RUN': DRY_RUN, 'LIMIT_RECORDINGS': LIMIT_RECORDINGS, 'SUBSET_N': SUBSET_N, 'LABEL_MODE': LABEL_MODE,
                   'FOLD_SOURCE': FOLD_SOURCE, 'EPOCHS_RUN': EPOCHS_RUN, 'N_BOOT': N_BOOT_RUN, 'RUN_CONTINUITY': RUN_CONTINUITY,
                   'CACHE_AUG_FEATURES': CACHE_AUG_FEATURES, 'INNER_VAL_SIZE': INNER_VAL_SIZE,
                   'JOINT_MAX_ITERATIONS': JOINT_MAX_ITERATIONS,
                   'training': {'batch_size': BATCH_SIZE, 'epochs_max': EPOCHS, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                                'patience': PATIENCE, 'dropout': DROPOUT, 'scheduler': 'ReduceLROnPlateau(max, 0.5, 5)',
                                'loss': 'CrossEntropyLoss (sum of three for multilabel)', 'monitor': 'inner-val AUC / mean of three AUCs'},
                   'mfcc': {'n_mfcc': N_MFCC, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'n_mels': N_MELS, 'fmax': FMAX,
                            'frames': N_FRAMES, 'sample_rate': SAMPLE_RATE, 'chunk_samples': CONFIG['chunk_samples'],
                            'min_remainder_samples': CONFIG['min_remainder_samples']}},
        'paths': {'DRIVE_ROOT': DRIVE_ROOT, 'TRAINING_DATA_PATH': TRAINING_DATA_PATH, 'CSV_PATH': CSV_PATH, 'RUN_DIR': RUN_DIR,
                  'RUN_DIR_SOURCE': RUN_DIR_SOURCE, 'FOLDS_DIR': FOLDS_DIR, 'OUT_DIR': OUT_DIR},
        'run_flags': {**RUN_FLAGS, 'status': RUN_FLAGS_STATUS, 'file': RUN_FLAGS_PATH},
        'gating_run_flags': GATING_RUN_FLAGS,
        'cache_events': CACHE_EVENTS,
        'gating_task_names': GATING_TASK,
        'dataset': {'n_pairs': N_PAIRS, 'complete_pairs': COMPLETE_PAIRS, 'resolution': DATASET_RESOLUTION,
                    'n_recordings_index': int(len(df_rec)), 'n_patients_index': int(df_rec['patient_id'].nunique())},
        'filter': {'design_used': FILTER_DESIGN, 'max_pole_radius_ba': FILTER_MAX_POLE_RADIUS,
                   'pole_radii_ba': [float(r) for r in sorted(FILTER_POLE_RADII)],
                   'per_recording_designs': CHUNK_META.get('filter_designs')},
        'labels': {'mode': LABEL_MODE, 'distributions': LABEL_DISTRIBUTIONS, 'class_names': CLASS_NAMES},
        'chunk_cache': {**CHUNK_META, 'chunk_count_check_vs_original': CHUNK_COUNT_CHECK},
        'folds': FOLD_INTEGRITY,
        'subset': sorted(SUBSET) if SUBSET is not None else None,
        'deviations': DEVIATIONS + [
            {'item': 'early stopping', 'detail': 'the original guard (if early_stopping(score): if early_stopping.early_stop: break) '
                                                 'could never trigger; here training stops when patience (10) is exhausted'},
            {'item': 'best state', 'detail': 'the original best_model_state was a shallow copy of the live state_dict; here a detached copy is restored'},
            {'item': 'early-stopping set', 'detail': 'patient-disjoint inner validation split (GroupShuffleSplit 0.15, seed 42) instead of '
                                                     'the chunk-level 80/20 re-split of pooled train+val chunks'},
            {'item': 'online augmentation', 'detail': 'applied to cached MFCC maps in the feature domain. Gain: exact, c0 offset '
                                                      'sqrt(n_mels)*20*log10(g) in every frame, deltas unchanged (asserted at runtime). '
                                                      'Shift: approximate, circular roll of round(shift/hop) frames. Against an '
                                                      'audio-domain roll with re-extraction, a whole-frame shift differs in 6 to 8 frames '
                                                      'at each edge (centred reflect padding, delta interpolation); a sub-frame shift '
                                                      '(the general case) differs in every frame by a small amount (rms about 0.02 to '
                                                      '0.07 standardised units, up to 3.3 at the edges); |shift| below 80 samples '
                                                      '(about 10% of the draws) rounds to zero frames and does nothing'},
        ] + ([{'item': 'mixed precision', 'detail': 'USE_AMP=True: training and inference ran under torch.autocast float16 '
                                                    'on CUDA; the original trained in fp32'}] if AMP_ENABLED else []) + [
            {'item': 'feature cache', 'detail': 'MFCC maps stored standardised in float16 with float32 mean/std; fresh and resumed runs '
                                                f'use the same round-tripped values (max error {F16_ROUNDTRIP_ERR_STD_UNITS:.1e} map-std units)'},
            {'item': 'joint balancing', 'detail': 'vectorised planner with the verbatim selection rule (checked against the pandas loop); '
                                                  'pool restricted to chunks with all three labels valid before balancing (the original dropped '
                                                  'them after balancing, at load time)'},
            {'item': 'batching', 'detail': 'shuffled GPU-resident tensors instead of a DataLoader (shuffle=True, batch 64, single-sample '
                                           'last batch skipped as BatchNorm requires)'},
            {'item': 'scoring', 'detail': 'recording level (mean chunk softmax, argmax) is primary; chunk level reported alongside'},
        ],
        'verbatim_blocks': VERBATIM_BLOCKS,
        'outputs': sorted(os.path.relpath(os.path.join(dp, f), OUT_DIR) for dp, _, fs in os.walk(OUT_DIR) for f in fs
                          if not f.startswith('chunks_') and not f.startswith('aug_')),
        'paired_vs_gated_status': PAIRED['status'],
    }
    with open(os.path.join(OUT_DIR, 'run_manifest.json'), 'w') as fh:
        json.dump(to_jsonable(manifest), fh, indent=1)
    print(json.dumps({k: manifest[k] for k in ('stage_seconds', 'total_seconds')}, indent=1))
    print(f"manifest written: {os.path.join(OUT_DIR, 'run_manifest.json')}")
    print(f"deviations recorded: {len(manifest['deviations'])}")
''')

md("""
## How to read the outputs

* `results.json`: `tasks.<task>.recording_level.pooled` holds the Table 2 baseline row (accuracy, macro-F1, balanced accuracy, per-class recall and specificity, AUROC, AUPRC, confusion matrix) with `ci95`; `fold_summary` gives `fold_macro_f1`, `fold_macro_f1_mean`, `fold_macro_f1_sd` (population SD) and `fold_macro_f1_sd_sample` (sample SD), and the same for accuracy, balanced accuracy and AUROC; `chunk_level` mirrors the same structure; `arms.<gating task>.cnn_pcg_only` repeats the recording-level numbers in the gating notebook's per-classifier layout (percent); `continuity` holds Section 11.
* `oof_predictions.npz`: per task `<task>__rec_names`, `<task>__patient_ids`, `<task>__y_true`, `<task>__probs`, `<task>__pred`, `<task>__fold`, `<task>__n_chunks`, plus the chunk-level arrays `<task>__chunk_*`.
* `paired_vs_gated.json`: paired bootstrap differences and fold-wise paired t-tests of each gated arm against this baseline on identical recordings and folds, with the label cross-check per arm.
* `figures/`: B1, B2, B3, B4 as PDF and 600 dpi PNG.
* `run_manifest.json`: everything needed to cite the run (versions, GPU, seeds, filter design, label mode, fold-file hashes, timings, run flags, cache events, deviations, verbatim-block hashes).
* `run_flags.json`: the flags the folder was made under (dry_run, limit_recordings, fold_source, seed, epochs, label_mode, use_amp); a resume under other flags is refused. A `LABEL_MODE = 'location'` run writes to its own `..._location` folder.
* `models/`: one checkpoint per task and fold; `cache/`: chunk cache, per-fold augmented features (tagged with the hash of their train chunk ids) and predictions (tagged with the fold's test and train recordings and the epoch budget; a fold whose tags differ from the current plan is recomputed). Delete a fold's files to force recomputation.
""")

# ----------------------------------------------------------------------------
# assemble, validate, compile
# ----------------------------------------------------------------------------
nb = new_notebook(cells=CELLS)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'colab': {'provenance': [], 'gpuType': 'A100', 'machine_shape': 'hm', 'name': 'R1_CNN_Baseline.ipynb'},
    'accelerator': 'GPU',
}
nbformat.validate(nb)
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')

# Checks run BEFORE the notebook file is written: a failed check leaves the file on disk untouched.
# py_compile every code cell
tmpdir = Path(tempfile.mkdtemp())
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    p = tmpdir / f'cell_{i:02d}.py'
    p.write_text(c.source, encoding='utf-8')
    py_compile.compile(str(p), doraise=True)
print(f'\npy_compile: all {n_code} code cells compile')
import re as _re
for i, c in enumerate(nb.cells):
    if '\u2014' in c.source or '\u2013' in c.source:
        raise SystemExit(f'{c.cell_type} cell {i} contains an em or en dash, which is not allowed; notebook NOT written')
print('dash check: no em/en dashes in any cell')
BANNED_MD = [r'\brobust(ly|ness)?\b', r'\bkeys?\b', r'\bpriors?\b', r'\byet\b', r'\bbeyond\b', r'\bnovel(ty)?\b',
             r'\bframeworks?\b', r'\butili[sz](e|es|ed|ing)\b', r'\bleverag(e|es|ed|ing)\b', r'\bachieves?\b',
             r'\bprovides?\b', r'\benables?\b', r'\bdemonstrates?\b', r'\bsignificant(ly)?\b', r'\bcomprehensive(ly)?\b',
             r'\btherefore\b', r'\bindicative\b', r'\baddresses\b', r'\bserves\b', r'\bfiltering\b']
hits = []
for i, c in enumerate(nb.cells):
    if c.cell_type != 'markdown':
        continue
    for pat in BANNED_MD:
        for m in _re.finditer(pat, c.source, flags=_re.I):
            hits.append(f'markdown cell {i}: {m.group(0)!r} in ...{c.source[max(0, m.start() - 40):m.end() + 40]!r}')
if hits:
    raise SystemExit('banned words in markdown; notebook NOT written:\n' + '\n'.join(hits))
print(f'markdown check: none of the {len(BANNED_MD)} banned word patterns')

with open(OUT_NB, 'w', encoding='utf-8') as fh:
    nbformat.write(nb, fh)
print(f'wrote {OUT_NB} ({len(nb.cells)} cells: {n_code} code, {n_md} markdown)')
print(f'verbatim blocks: {len([k for k in VERBATIM_BLOCKS if not k.startswith("snippet:")])}, '
      f'verified snippets: {len([k for k in VERBATIM_BLOCKS if k.startswith("snippet:")])}')
