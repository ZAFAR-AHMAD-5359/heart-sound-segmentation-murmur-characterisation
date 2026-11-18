#!/usr/bin/env python
"""
build_gating_campaign.py

Assembles R1_Gating_Campaign.ipynb for the EHJ-DH R1 revision.

Reusable code is NOT retyped here. At build time this script slices the verbatim blocks
out of the analysis source by line range (feature extractor, generator classes, TSV parser,
loader, label coding, MLP class) and extracts named functions from the reference notebooks
with the ast module (SOS detector, phase clamp, rhythm statistics, one-to-one matching,
jitter, bootstrap oracles, the published equal-time control). Every block is checked against
an anchor string, hashed, and the hashes are embedded in the notebook so the run manifest
can prove what was executed.

Usage:  python build_gating_campaign.py            (writes R1_Gating_Campaign.ipynb beside it)
        GATING_SRC_DIR=<dir> python build_gating_campaign.py   (override the source folder)
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
SRC_DIR = Path(os.environ.get('GATING_SRC_DIR', HERE.parents[2] / 'Synthetic-Gating-Experiment'))
SRC_PY = SRC_DIR / 'Synthetic_vs_Expert_Gating_CV.py'
NB_MECH = SRC_DIR / 'Mechanism-2' / 'Mechanism_Rhythm_Integrity.ipynb'
NB_PEAK = SRC_DIR / 'Peak_Detection_Diagnostic.ipynb'
NB_JIT = SRC_DIR / 'Jitter_Timing_Tolerance.ipynb'
NB_FAIR = SRC_DIR / 'Fair_MLP_Rerun_Bootstrap.ipynb'
NB_ABL = Path(os.environ.get('GATING_ABLATION_NB',
                             HERE.parents[3] / 'Milestone-3' / 'New Pipeline (Critic)' /
                             'ABLATION_GATING_VALIDATION.ipynb'))
OUT_NB = HERE / 'R1_Gating_Campaign.ipynb'

for p in (SRC_PY, NB_MECH, NB_PEAK, NB_JIT, NB_FAIR, NB_ABL):
    if not p.exists():
        sys.exit(f'missing source: {p}')

PY_LINES = SRC_PY.read_text(encoding='utf-8').split('\n')
PROVENANCE = []


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def py_lines(a, b, anchor):
    """Lines a..b (1-based, inclusive) of the analysis source, anchored on the first line."""
    block = PY_LINES[a - 1:b]
    if not block[0].startswith(anchor):
        sys.exit(f'anchor mismatch at line {a}: expected {anchor!r}, found {block[0]!r}')
    text = '\n'.join(block)
    PROVENANCE.append({'source': SRC_PY.name, 'lines': f'{a}-{b}', 'anchor': anchor, 'sha256': _sha(text)})
    return text


def nb_funcs(path, names):
    """Named function definitions extracted verbatim from a notebook's code cells."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    found = {}
    for ci, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # cells with shell magics are not the ones holding these functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                seg = ast.get_source_segment(src, node)
                if node.name in found:
                    sys.exit(f'{path.name}: {node.name} defined more than once')
                found[node.name] = (ci, seg)
    missing = [n for n in names if n not in found]
    if missing:
        sys.exit(f'{path.name}: functions not found: {missing}')
    parts = []
    for n in names:
        ci, seg = found[n]
        PROVENANCE.append({'source': path.name, 'cell': ci, 'function': n, 'sha256': _sha(seg)})
        parts.append(seg)
    return '\n\n\n'.join(parts)


# ----------------------------------------------------------------------------------------------
# Verbatim blocks
# ----------------------------------------------------------------------------------------------
V_GENERATOR = py_lines(160, 231, 'class MultiHeadSelfAttention')
V_CKPT_LOAD = py_lines(233, 239, 'gen = ImprovedGeneratorNoTanh')
V_CONSTANTS = py_lines(247, 261, 'GAN_SR = 8000')            # constants + ensure_positive_polarity
V_SEGMENT = py_lines(282, 291, 'def segment_phases_samples')
V_FEATURES = py_lines(336, 602, 'def _fast_mfcc')             # every helper through extract_all_v4_features
V_TSV_EQ_LOAD = py_lines(604, 646, 'def parse_tsv_annotations')  # parser, equal_time_cycles, fast_load_audio
V_LABELS = py_lines(656, 668, 'metadata_df = pd.read_csv')
V_MLP = py_lines(750, 759, 'class LightweightMLP')

V_MECH_RHYTHM = nb_funcs(NB_MECH, ['cycles_to_peaks', 'rr_intervals', 'rhythm'])
V_MECH_CLAMP = nb_funcs(NB_MECH, ['phases_from'])
V_MECH_SOS = nb_funcs(NB_MECH, ['_sos_env', '_detect_sos'])
V_PEAK_MATCH = nb_funcs(NB_PEAK, ['match_one_to_one', 'match_stats', 'best_offset'])
V_JITTER = nb_funcs(NB_JIT, ['jitter_cycles', 'equivalent_sigma'])
V_FAIR_BOOT = nb_funcs(NB_FAIR, ['boot_ci', 'boot_diff_ci'])
V_NONGATED = nb_funcs(NB_ABL, ['get_nongated_cycles'])

PROV_JSON = json.dumps(PROVENANCE, indent=1)

# ----------------------------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------------------------
CELLS = []


def md(text):
    CELLS.append(('markdown', None, text.strip('\n')))


def code(name, text):
    CELLS.append(('code', name, text.strip('\n')))


md(r'''
# R1 gating campaign: reconstruction, six boundary arms, patient-grouped CV, diagnostics, ablations, figures

This notebook produces every number and figure for the Milestone-3 rerun of the EHJ-DH revision
(items R3-M1, R3-M5, R1-m5, R1-M1, R1-M2, R3-M4, R1-m8, R1-m3, R3-m3). It runs top to bottom on a
Colab Pro high-RAM GPU runtime and writes everything to one dated folder on Drive.

**What is held fixed.** One analysed portion of audio per recording, shared by every arm. One
464-feature extractor, byte-identical to the analysis source. One set of patient-grouped folds per
task, drawn once and saved. One classifier configuration. The arms differ only in where the cardiac
phase boundaries come from.

**The arms.**

| arm | key | boundaries |
|---|---|---|
| A | `syn_fixed` | R-peaks from the reconstructed ECG (corrected SOS detector), fixed 15/35/10/40 %RR phases |
| B | `syn_fusion` | synthetic R-peaks used as candidates, validated and refined by the acoustic S1/S2 envelope |
| C | `pcg_seg` | the dataset's per-beat S1/systole/S2/diastole annotation |
| D | `pcg_seg_fixed` | annotated S1 onsets as anchors, fixed-fraction phases |
| E | `equal_time` | 75 bpm pseudo-cycles in four equal quarters, no cardiac information |
| E2 | `equal_time_matched` | the analysed portion cut into N equal pseudo-cycles, N = the arm C cycles inside that portion, four equal quarters (the published control's definition: it inherits the annotated count, not the annotated rate, so it runs slow wherever annotation covers only part of the portion) |

**Reading order.** Rate-fidelity diagnostics (Stage 4) are printed before any classification
number, because they are what explains a weak arm. The validation gate (Stage 8) must pass before
the figures are trusted.

**Stages.** 0 paths, 1 index and labels, 2 shared audio, 3 generator, 4 boundaries and diagnostics,
5 features, 6 folds, 7 classifiers and evaluation, 8 validation gate, 9 ablations, 10 figures,
11 manifest. Every stage caches to Drive and skips work that is already done, so a crashed session
can be resumed by setting `RESUME_DIR` (the folder's `run_flags.json` must match the current flags). A
dry run (`DRY_RUN = True`) takes a class-aware subset of about 120 recordings with 5 folds and 100
bootstrap replicates; a task whose subset is too small for grouped folds is skipped and listed.
''')

code('flags', r'''
# ================================ RUN FLAGS ================================
DRY_RUN = False           # True: class-aware subset (about 60 murmur-present + 60 murmur-absent recordings), 5 folds, 100 bootstrap replicates, own '_dry' run folder
LIMIT_RECORDINGS = 0      # 0 = all recordings; N > 0 = an evenly spaced subset of N (not for the paper; own '_limitN' run folder)
N_BOOT = 2000             # patient-level bootstrap replicates
RESUME_DIR = None         # e.g. '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20260930_1200'; its run_flags.json must equal the flags below
FORCE_FIGURES = False     # draw figures even if the validation gate failed (diagnostic use only)
SEED = 42
N_SPLITS = 10
GATE_TOL = 3.0            # macro-F1 points allowed between this run and the reference values
CHUNK = 400               # recordings per cache file
N_MIN_CYCLES = 2          # a recording is kept only when every arm gives at least this many cycles
DRY_RUN_SUBSET = dict(present=60, absent=60)   # DRY_RUN sample: murmur-present recordings drawn round-robin over the grading / timing / shape classes, plus murmur-absent ones

if DRY_RUN:
    N_BOOT = 100
    N_SPLITS = 5
VALIDATION_APPLICABLE = (not DRY_RUN) and LIMIT_RECORDINGS == 0    # the reference values exist for the full set only
print('DRY_RUN', DRY_RUN, '| LIMIT_RECORDINGS', LIMIT_RECORDINGS, '| N_BOOT', N_BOOT, '| N_SPLITS', N_SPLITS,
      '| N_MIN_CYCLES', N_MIN_CYCLES, '| RESUME_DIR', RESUME_DIR, '| SEED', SEED, '| validation gate applicable', VALIDATION_APPLICABLE)
''')

md(r'''
## Environment

BLAS thread counts are pinned to one **before** numpy is imported (BLAS is the linear-algebra
library behind numpy; unpinned, every forked feature worker would open its own thread pool). The
feature workers are forked from this process where the platform has the fork start method (Linux,
so Colab) and inherit that setting; where fork does not exist the same code runs serially through
an executor with the same `.map` interface, and the manifest records which path was used. cuDNN
(the GPU convolution library) is held to its deterministic kernels, TF32 (the reduced-precision
matrix arithmetic of Ampere GPUs) is switched off, and `torch.use_deterministic_algorithms` is on,
so the generator trace is reproducible bit for bit on the same GPU class.
''')

code('env', r'''
import os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_v] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'      # what torch.use_deterministic_algorithms needs from cuBLAS

import subprocess
IN_COLAB = False
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    print('google.colab is not importable: not running in Colab, Drive not mounted')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'PyWavelets', 'nbformat'], check=True)

import json, time, pickle, gc, collections, platform, hashlib, random, math, datetime, zipfile
import contextlib, warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.io import wavfile
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq, dct
from scipy.stats import skew, kurtosis, spearmanr, ttest_rel
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
from tqdm.auto import tqdm
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the verbatim feature functions emit RuntimeWarnings on short segments by design; keep the rest
warnings.simplefilter('ignore', RuntimeWarning)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)   # warn, not raise, on an op without a deterministic kernel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEV = device
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32
N_PROC = max(1, os.cpu_count() or 1)                       # the design brief: every core
_vm = psutil.virtual_memory()
print('python', platform.python_version(), '| torch', torch.__version__, '| numpy', np.__version__,
      '| scipy', scipy.__version__, '| sklearn', sklearn.__version__)
print('GPU', GPU_NAME, '| CPUs', os.cpu_count(), '| RAM %.1f GB total, %.1f GB free' % (_vm.total / 1e9, _vm.available / 1e9))
print('generator batch', GAN_BATCH, '| feature workers', N_PROC)

# ---- worker pool: forked processes where the platform has them, else a serial executor with the same interface ----
class _SerialExecutor:
    """Same .map interface as ProcessPoolExecutor, run in-process. Used where the fork start method is unavailable
    (the workers read inherited globals, so the spawn method cannot stand in for fork)."""
    def __init__(self, initializer=None):
        if initializer is not None:
            initializer()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def map(self, fn, it, chunksize=1):
        return map(fn, it)

POOL_KIND = 'fork' if 'fork' in mp.get_all_start_methods() else 'sequential'
POOL_USES = []

def _pool(max_workers, initializer=None):
    if not POOL_USES:
        print('   worker pool: %s' % (POOL_KIND if POOL_KIND == 'fork' else 'sequential (fork start method unavailable on %s)' % sys.platform))
    POOL_USES.append(POOL_KIND)
    if POOL_KIND == 'fork':
        if torch.cuda.is_available():
            torch.cuda.synchronize()                          # no pending GPU work crosses the fork
        kw = dict(max_workers=max_workers, mp_context=mp.get_context('fork'))
        if initializer is not None:
            kw['initializer'] = initializer
        return ProcessPoolExecutor(**kw)
    return _SerialExecutor(initializer)

# ---- stage timer, cumulative across sessions of the same run folder ----
STAGE_SECONDS = {}                 # this session
STAGE_SECONDS_PREV = {}            # cumulative seconds of earlier sessions on a resumed folder (loaded in Stage 0)
STAGE_SESSIONS_PREV = []
SESSION_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def stage_seconds_cumulative():
    cum = dict(STAGE_SECONDS_PREV)
    for k_, v_ in STAGE_SECONDS.items():
        cum[k_] = cum.get(k_, 0.0) + v_
    return cum

def _persist_stage_seconds():
    if 'RUN_DIR' not in globals():
        return
    tmp = str(RUN_DIR / 'stage_seconds.json') + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(dict(sessions=STAGE_SESSIONS_PREV + [SESSION_ID], this_session=STAGE_SECONDS, cumulative=stage_seconds_cumulative()), fh, indent=1)
    os.replace(tmp, str(RUN_DIR / 'stage_seconds.json'))

@contextlib.contextmanager
def stage(name):
    t0 = time.time()
    print('\n' + '=' * 78 + '\n' + name + '\n' + '=' * 78)
    yield
    dt = time.time() - t0
    STAGE_SECONDS[name] = STAGE_SECONDS.get(name, 0.0) + dt
    _persist_stage_seconds()
    print('--- %s: %.1f s (%.1f min)' % (name, dt, dt / 60))

def sha256_file(path, block=1 << 23):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(block), b''):
            h.update(b)
    return h.hexdigest()

def git_blob_sha1(path, block=1 << 23):
    """The hash git would assign to this file as a blob."""
    h = hashlib.sha1()
    h.update(('blob %d\0' % os.path.getsize(path)).encode())
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(block), b''):
            h.update(b)
    return h.hexdigest()

# ---- cache writes: atomic (temporary file, then os.replace) and hashed at write time, so Stage 11 re-reads nothing ----
CACHE_HASHES = {}      # {path relative to RUN_DIR: {git_blob_sha1, bytes}}; persisted to cache_hashes.json after every write
CACHE_REBUILDS = []    # (file, reason) for every cache file found unreadable or stale and rebuilt

def _record_cache_hash(path):
    if 'RUN_DIR' not in globals():
        return
    path = Path(path)
    try:
        rel = str(path.relative_to(RUN_DIR))
    except ValueError:
        return
    CACHE_HASHES[rel] = dict(git_blob_sha1=git_blob_sha1(path), bytes=os.path.getsize(path))
    tmp = str(RUN_DIR / 'cache_hashes.json') + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(CACHE_HASHES, fh)
    os.replace(tmp, str(RUN_DIR / 'cache_hashes.json'))

def atomic_savez(path, **arrays):
    tmp = str(path) + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, str(path))
    _record_cache_hash(path)

def atomic_savez_compressed(path, **arrays):
    tmp = str(path) + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, str(path))
    _record_cache_hash(path)

def atomic_pickle(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(tmp, str(path))
    _record_cache_hash(path)

def atomic_np_save(path, arr):
    tmp = str(path) + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, str(path))
    _record_cache_hash(path)

def read_cache(fp, fn):
    """fn(z) on the opened .npz -> value, or None when the file is missing, unreadable or stale (fn returned None).
    An unreadable or stale file is removed and listed in CACHE_REBUILDS; the caller rebuilds it."""
    fp = Path(fp)
    if not fp.exists():
        return None
    try:
        with np.load(fp, allow_pickle=False) as z:
            out = fn(z)
        why = None if out is not None else 'stale'
    except Exception as e:
        out, why = None, type(e).__name__
    if why is not None:
        CACHE_REBUILDS.append((fp.name, why))
        print('   %s cache %s: rebuilding' % ('stale' if why == 'stale' else 'unreadable (%s)' % why, fp.name))
        try:
            fp.unlink()
        except OSError:
            pass
    return out

def jsonable(o):
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (str, bool, int, float)) or o is None:
        return o
    return str(o)
''')

md(r'''
## Manual override

Leave every value as `None` and the next cell searches Drive. Set a value to pin that path and skip
the search for it. Paths are printed and asserted either way.
''')

code('manual_override', r'''
# ====================== MANUAL OVERRIDE (optional) ======================
MANUAL_TRAINING_DATA = None
# e.g. '/content/drive/MyDrive/JBHI-Revision/October Thesis Dataset/CirCor DigiScope/the-circor-digiscope-phonocardiogram-dataset-1.0.3/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'
MANUAL_CHECKPOINT = None
# e.g. '/content/drive/MyDrive/October Thesis Dataset/trail4_checkpoints/trail4_best.pt'
MANUAL_OUTPUT_ROOT = None
# e.g. '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign'
''')

md(r'''
## Stage 0: resolve Drive paths

One walk of Drive finds both targets. The dataset folder is the `training_data` directory with the
**largest** number of paired `.wav` + `.tsv` files (the complete copy has 3163 pairs; a 3155-pair
copy and a 14-file stale folder also exist and lose on that count). The generator checkpoint is
every `trail4_best.pt` on Drive, hashed with SHA-256; the `trail4_checkpoints` copy is used when
present and a loud warning is printed if the copies differ. A pinned `MANUAL_CHECKPOINT` must exist
and be a torch archive, or the cell stops with a clear message. Outputs go to a dated run folder
(suffix `_dry` for a dry run, `_limitN` for `LIMIT_RECORDINGS = N`) unless `RESUME_DIR` points at
an existing one; the folder's `run_flags.json` is written on a fresh run and must equal the current
flags on a resume, so a full run's caches can never be rebuilt by a dry run by mistake.
''')

code('resolve', r'''
DRIVE_ROOT = '/content/drive/MyDrive'
SKIP_DIRS = {'.Trash', '.ipynb_checkpoints', '__pycache__', 'node_modules'}

def walk_find_multi(root, targets, maxdepth=9):
    """One walk of Drive for several targets. targets: {name: 'dir' | 'file'} -> {name: sorted hits}."""
    hits = {t: [] for t in targets}
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return hits
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dp, dns, fns in os.walk(root):
        depth = dp.rstrip(os.sep).count(os.sep) - base_depth
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith('.')]
        for t, kind in targets.items():
            hits[t] += [os.path.join(dp, x) for x in (dns if kind == 'dir' else fns) if x == t]
        if depth >= maxdepth - 1:
            dns[:] = []
    return {t: sorted(set(v)) for t, v in hits.items()}

def paired_count(td):
    td = Path(td)
    if not td.is_dir():
        return 0
    return len({p.stem for p in td.glob('*.wav')} & {p.stem for p in td.glob('*.tsv')})

RESUMED_FROM = None
with stage('S0 resolve Drive paths'):
    _targets = {}
    if not MANUAL_TRAINING_DATA:
        _targets['training_data'] = 'dir'
    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
    _found = {}
    if _targets:
        print('searching %s once for %s (this can take 1-3 min on Drive)...' % (DRIVE_ROOT, ' and '.join(_targets)))
        _found = walk_find_multi(DRIVE_ROOT, _targets, 9)

    # ---- dataset ----
    cands = [Path(MANUAL_TRAINING_DATA)] if MANUAL_TRAINING_DATA else [Path(p) for p in _found.get('training_data', [])]
    rows = sorted(((paired_count(p), p) for p in cands), key=lambda r: r[0], reverse=True)
    print('training_data candidates:')
    for n_, p_ in rows:
        print('   paired=%5d  %s' % (n_, p_))
    if not rows or rows[0][0] == 0:
        raise FileNotFoundError('no training_data folder with paired wav+tsv files; use the MANUAL OVERRIDE cell')
    N_PAIRS, AUDIO_PATH = rows[0]
    if MANUAL_TRAINING_DATA is None and N_PAIRS < 2500:
        raise RuntimeError('largest copy has only %d pairs; expected ~3163 (complete copy)' % N_PAIRS)
    BASE_PATH = AUDIO_PATH.parent
    METADATA_PATH = BASE_PATH / 'training_data.csv'

    # ---- generator checkpoint ----
    if MANUAL_CHECKPOINT:
        if not os.path.isfile(MANUAL_CHECKPOINT):
            raise FileNotFoundError('MANUAL_CHECKPOINT is not a file: %s' % MANUAL_CHECKPOINT)
        if not zipfile.is_zipfile(MANUAL_CHECKPOINT):
            raise ValueError('MANUAL_CHECKPOINT is not a torch checkpoint (torch.save writes a zip archive): %s' % MANUAL_CHECKPOINT)
        hits = [MANUAL_CHECKPOINT]
    else:
        hits = _found.get('trail4_best.pt', [])
    if not hits:
        raise FileNotFoundError('trail4_best.pt not found on Drive; use the MANUAL OVERRIDE cell')
    CKPT_HASHES = {}
    for h_ in hits:
        CKPT_HASHES[str(Path(h_))] = sha256_file(h_)                # keyed by the normalised path, read back the same way
        print('   %s  sha256=%s  (%.1f MB)' % (h_, CKPT_HASHES[str(Path(h_))], os.path.getsize(h_) / 1e6))
    if len(set(CKPT_HASHES.values())) > 1:
        print('\n' + '!' * 78 + '\n!!! WARNING: the trail4_best.pt copies DIFFER. Using the trail4_checkpoints copy.\n' + '!' * 78)
    _pref = [h_ for h_ in hits if 'trail4_checkpoints' in h_]
    GAN_CHECKPOINT = Path(_pref[0] if _pref else hits[0])
    CKPT_SHA256 = CKPT_HASHES[str(GAN_CHECKPOINT)]

    # ---- output folder and run flags ----
    OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/R1_Gating_Campaign'))
    if RESUME_DIR:
        RUN_DIR = Path(RESUME_DIR)
        if not RUN_DIR.is_dir():
            raise FileNotFoundError('RESUME_DIR does not exist: %s' % RUN_DIR)
        RESUMED_FROM = str(RUN_DIR)
    else:
        _suffix = '_dry' if DRY_RUN else ('_limit%d' % LIMIT_RECORDINGS if LIMIT_RECORDINGS > 0 else '')
        RUN_DIR = OUT_ROOT / (datetime.datetime.now().strftime('%Y%m%d_%H%M') + _suffix)
    # the folder suffix must agree with the flags before anything is created or written in the folder
    assert RUN_DIR.name.endswith('_dry') == DRY_RUN, 'DRY_RUN and the run folder suffix disagree: %s' % RUN_DIR.name
    if LIMIT_RECORDINGS > 0 and not DRY_RUN:
        assert RUN_DIR.name.endswith('_limit%d' % LIMIT_RECORDINGS), 'LIMIT_RECORDINGS and the run folder suffix disagree: %s' % RUN_DIR.name
    FOLD_DIR, CACHE_DIR, FIG_DIR = RUN_DIR / 'folds', RUN_DIR / 'cache', RUN_DIR / 'figures'
    CV_DIR = CACHE_DIR / 'cv'
    for d_ in (RUN_DIR, FOLD_DIR, CACHE_DIR, FIG_DIR, CV_DIR):
        d_.mkdir(parents=True, exist_ok=True)
    RUN_FLAGS = dict(dry_run=DRY_RUN, limit_recordings=LIMIT_RECORDINGS, seed=SEED, n_splits=N_SPLITS, chunk=CHUNK,
                     n_min_cycles=N_MIN_CYCLES, dry_run_subset=DRY_RUN_SUBSET if DRY_RUN else None)
    _flags_fp = RUN_DIR / 'run_flags.json'
    if _flags_fp.exists():
        with open(_flags_fp) as fh:
            _prev = json.load(fh)
        assert _prev == RUN_FLAGS, 'RESUME_DIR was produced with different flags %s; refusing to resume with %s' % (_prev, RUN_FLAGS)
        print('run_flags.json matches this session:', RUN_FLAGS)
    else:
        with open(_flags_fp, 'w') as fh:
            json.dump(RUN_FLAGS, fh, indent=1)
    if (RUN_DIR / 'stage_seconds.json').exists():
        with open(RUN_DIR / 'stage_seconds.json') as fh:
            _ss = json.load(fh)
        STAGE_SECONDS_PREV.update(_ss.get('cumulative', {}))
        STAGE_SESSIONS_PREV.extend(_ss.get('sessions', []))
    if (RUN_DIR / 'cache_hashes.json').exists():
        with open(RUN_DIR / 'cache_hashes.json') as fh:
            CACHE_HASHES.update(json.load(fh))
    if RESUMED_FROM:
        print('resuming %s: %d earlier session(s), %d cache files already hashed at write time' % (RUN_DIR, len(STAGE_SESSIONS_PREV), len(CACHE_HASHES)))

    print('\n--- resolved paths ---')
    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),
                   ('RUN_DIR', RUN_DIR), ('FOLD_DIR', FOLD_DIR), ('CACHE_DIR', CACHE_DIR), ('FIG_DIR', FIG_DIR)]:
        print('%-15s %s' % (k_, v_))
        assert Path(v_).exists(), '%s does not exist: %s' % (k_, v_)
    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))
    print('run flags:', RUN_FLAGS)
''')

md(r'''
## Verbatim code from the analysis source

The next cells are copied at build time, by line range, from
`Synthetic_vs_Expert_Gating_CV.py` (the jupytext export of the analysis notebook). They are not
edited here in any way, so the generator architecture, the checkpoint loader, the constants, the
fixed-fraction phase model and the 464-feature extractor are byte-identical to what produced the
reference numbers. The SHA-256 of each block is listed in the provenance cell near the end and
written to the run manifest. These blocks keep the source's bare `except:` clauses; the code
written for this notebook does not use them.
''')

code('v_generator', V_GENERATOR)

code('v_ckpt_load', V_CKPT_LOAD)

code('v_constants_segment', V_CONSTANTS + '\n\n' + V_SEGMENT)

code('v_features', V_FEATURES)

code('v_tsv_equal_load', V_TSV_EQ_LOAD)

md(r'''
## Reused machinery from the diagnostic notebooks

Extracted verbatim (by function name, with the `ast` module) from the mechanism study, the
peak-detection diagnostic, the jitter-tolerance sweep, the fair-MLP bootstrap rerun and the
published gating ablation:

* `_sos_env`, `_detect_sos`: the second-order-section R-peak detector with a finite check.
* `phases_from`: clamps a peak train to the signal before fixed-fraction segmentation, so no
  boundary can exceed the signal length.
* `cycles_to_peaks`, `rr_intervals`, `rhythm`: RR statistics over contiguous cycles only.
* `match_one_to_one`, `match_stats`, `best_offset`: one-to-one peak matching and the offset search.
* `jitter_cycles`, `equivalent_sigma`: the timing-budget perturbation and its inverse read-off.
* `boot_ci`, `boot_diff_ci`: the original patient-level bootstrap, kept as an oracle. The
  vectorised bootstrap used below draws the same patient resamples and is checked against it.
* `get_nongated_cycles`: the published equal-time control (arm E2), boundaries in seconds.
''')

code('reused_machinery',
     "SOS = sig.butter(4, [5, 30], btype='bandpass', fs=GAN_SR, output='sos')\n"
     "PHASE_KEYS = ('s1', 'systole', 's2', 'diastole')\n"
     "np.seterr(over='ignore', invalid='ignore')\n\n"
     + V_MECH_SOS + '\n\n\n' + V_MECH_CLAMP + '\n\n\n' + V_MECH_RHYTHM + '\n\n\n'
     + V_PEAK_MATCH + '\n\n\n' + V_JITTER + '\n\n\n' + V_FAIR_BOOT + '\n\n\n' + V_NONGATED + '\n\n'
     "_pole_sos = float(max(np.abs(np.roots(s[3:])).max() for s in SOS))\n"
     "print('SOS bandpass 5-30 Hz at %d Hz: max pole radius %.6f (%s)' % (GAN_SR, _pole_sos, 'stable' if _pole_sos < 1 else 'UNSTABLE'))\n"
     "assert _pole_sos < 1.0, 'the SOS design must be stable'\n"
     "print('reused functions defined')")

md(r'''
## Stage 1: recording index and labels

Labels come from `training_data.csv` with the source's coding, verbatim: murmur `Unknown` is
dropped; grading I / II / III+ is 0 / 1 / 2; timing Holosystolic / Early / Mid is 0 / 1 / 2 with
Late-systolic coded -1; shape Plateau / Decrescendo / Diamond is 0 / 1 / 2 with Crescendo coded -1;
murmur-negative recordings carry -1 on the three characterisation tasks. A recording is indexed only
if it has a WAV, a TSV annotation and a label. Everything excluded is listed with its reason and
saved to `index.json`.
''')

code('v_labels', V_LABELS)

code('s1_index', r'''
def dry_run_subset(records_all, n_present, n_absent, seed):
    """Class-aware DRY_RUN sample. Murmur-present recordings are drawn round-robin over the nine
    (task, class) buckets of grading / timing / shape, one recording per patient while distinct
    patients last; murmur-absent recordings come from further distinct patients. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)

    def _perm(rows):
        return [rows[i] for i in rng.permutation(len(rows))]

    present = [r for r in records_all if r[2]['murmur'] == 1]
    absent = [r for r in records_all if r[2]['murmur'] == 0]
    buckets = {}
    for r in present:
        for key_ in ('grading', 'timing', 'shape'):
            if r[2][key_] >= 0:
                buckets.setdefault((key_, int(r[2][key_])), []).append(r)
    buckets = {k: _perm(v) for k, v in sorted(buckets.items())}
    chosen, chosen_names, used_pat = [], set(), set()
    for allow_repeat_patient in (False, True):
        progress = True
        while len(chosen) < n_present and progress:
            progress = False
            for key_, rows in buckets.items():
                if len(chosen) >= n_present:
                    break
                for r in rows:
                    if r[0] in chosen_names or (not allow_repeat_patient and r[1] in used_pat):
                        continue
                    chosen.append(r)
                    chosen_names.add(r[0])
                    used_pat.add(r[1])
                    progress = True
                    break
    picked_absent = []
    absent_p = _perm(absent)
    for allow_repeat_patient in (False, True):
        for r in absent_p:
            if len(picked_absent) >= n_absent:
                break
            if r[0] in chosen_names or (not allow_repeat_patient and r[1] in used_pat):
                continue
            picked_absent.append(r)
            chosen_names.add(r[0])
            used_pat.add(r[1])
    sub = sorted(chosen + picked_absent, key=lambda r: r[0])
    print('DRY_RUN class-aware subset: %d murmur-present (round-robin over %d (task, class) buckets) + %d murmur-absent = %d recordings from %d patients' % (
        len(chosen), len(buckets), len(picked_absent), len(sub), len({r[1] for r in sub})))
    for key_ in ('grading', 'timing', 'shape'):
        cnt = collections.Counter(int(r[2][key_]) for r in chosen if r[2][key_] >= 0)
        print('   %-8s classes in the subset: %s | patients per class: %s' % (
            key_, dict(sorted(cnt.items())), {c: len({r[1] for r in chosen if r[2][key_] == c}) for c in sorted(cnt)}))
    return sub

with stage('S1 index and labels'):
    UNKNOWN_PIDS = set(int(p) for p in metadata_df.loc[metadata_df['Murmur'] == 'Unknown', 'Patient ID'])
    AGE_OF = {}
    for _, _row in metadata_df.iterrows():
        _a = _row.get('Age', None)
        AGE_OF[int(_row['Patient ID'])] = _a if isinstance(_a, str) and _a.strip() else 'Unknown'

    EXCLUDED = []
    records_all = []
    # sorted by file name as a plain string: the same order on every platform (Path objects sort case-insensitively on
    # Windows), and the order of the reference run, so the folds drawn from it are the same everywhere
    for wav in sorted(AUDIO_PATH.glob('*.wav'), key=lambda p: p.name):
        name = wav.stem
        try:
            pid = int(name.split('_')[0])
        except ValueError:
            EXCLUDED.append((name, 'filename does not start with a patient id'))
            continue
        if pid not in label_lookup:
            EXCLUDED.append((name, 'murmur label Unknown' if pid in UNKNOWN_PIDS else 'patient absent from training_data.csv'))
            continue
        tsv = AUDIO_PATH / (name + '.tsv')
        if not tsv.exists():
            EXCLUDED.append((name, 'no tsv annotation'))
            continue
        records_all.append((name, pid, label_lookup[pid], wav, tsv))

    print('wav files: %d | indexed (wav + tsv + label): %d | excluded: %d' % (
        len(list(AUDIO_PATH.glob('*.wav'))), len(records_all), len(EXCLUDED)))
    for reason, cnt in collections.Counter(r for _, r in EXCLUDED).most_common():
        print('   excluded %4d  %s' % (cnt, reason))

    if DRY_RUN:
        records = dry_run_subset(records_all, DRY_RUN_SUBSET['present'], DRY_RUN_SUBSET['absent'], SEED)
    elif LIMIT_RECORDINGS and LIMIT_RECORDINGS < len(records_all):
        _pick = np.unique(np.linspace(0, len(records_all) - 1, LIMIT_RECORDINGS).astype(int))
        records = [records_all[i] for i in _pick]
        print('LIMIT_RECORDINGS: using an evenly spaced subset of %d recordings' % len(records))
    else:
        records = list(records_all)

    NAMES_ALL = [r[0] for r in records]
    REC = {r[0]: r for r in records}
    PID_OF = {r[0]: r[1] for r in records}
    LAB_OF = {r[0]: r[2] for r in records}
    TASKS = {'detection': 'murmur', 'grading': 'grading', 'timing': 'timing', 'shape': 'shape'}
    NCLS = {'detection': 2, 'grading': 3, 'timing': 3, 'shape': 3}
    print('recordings in this run: %d from %d patients' % (len(NAMES_ALL), len(set(PID_OF.values()))))
    for t_, key_ in TASKS.items():
        _y = np.array([LAB_OF[n][key_] for n in NAMES_ALL])
        _c = collections.Counter(_y[_y >= 0].tolist())
        print('   %-10s labelled %5d  classes %s' % (t_, int((_y >= 0).sum()), dict(sorted(_c.items()))))

    with open(RUN_DIR / 'index.json', 'w') as fh:
        json.dump({'names': NAMES_ALL, 'patient_ids': [PID_OF[n] for n in NAMES_ALL],
                   'labels': {n: LAB_OF[n] for n in NAMES_ALL},
                   'age_band': {n: AGE_OF.get(PID_OF[n], 'Unknown') for n in NAMES_ALL},
                   'excluded': EXCLUDED, 'n_wav_files': len(list(AUDIO_PATH.glob('*.wav'))),
                   'limit_recordings': LIMIT_RECORDINGS, 'dry_run': DRY_RUN,
                   'dry_run_subset': DRY_RUN_SUBSET if DRY_RUN else None}, fh, indent=1)
''')

md(r'''
## Stage 2: shared audio

Each WAV is read at its native rate and resampled **once** to 8 kHz (the generator's rate). The
analysed portion is every whole 8 s window plus a trailing window when at least 2 s remain; a
shorter tail is dropped, and a recording shorter than 2 s is kept whole and padded to one window at
generation time. The 4 kHz array used by the feature extractor is the native 4 kHz samples of the
analysed portion (every CirCor file is 4 kHz; a file at another rate is resampled once from the
8 kHz portion), max-abs normalised over the analysed portion as the source's loader normalises a
whole file, so on an untrimmed recording it is sample-identical to that loader. Both arrays are
cached per chunk of 400 recordings. Every arm reads only these arrays, so the arms cannot differ in
audio. The 8 kHz arrays are loaded only for the chunks whose generator cache is missing.
''')

code('s2_audio', r'''
def chunk_slices(names):
    return [(k, names[k * CHUNK:(k + 1) * CHUNK]) for k in range(int(math.ceil(len(names) / CHUNK)))]

def analysed_length_8k(n8):
    """(analysed samples at 8 kHz, number of generator windows)"""
    n_full, tail = divmod(int(n8), SIGNAL_LENGTH)
    if n_full == 0:
        return int(n8), 1                            # short recording: whole thing, one padded window
    if tail >= 2 * GAN_SR:
        return int(n8), n_full + 1                   # trailing window kept (>= 2 s)
    return n_full * SIGNAL_LENGTH, n_full            # tail under 2 s dropped

def load_shared_audio(name):
    sr_raw, audio = wavfile.read(str(REC[name][3]))
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr_raw == GAN_SR:
        x8 = audio
    else:
        x8 = sig.resample(audio, int(len(audio) * GAN_SR / sr_raw)).astype(np.float32)
    n8_total = len(x8)
    L8, n_win = analysed_length_8k(n8_total)
    x8 = np.ascontiguousarray(x8[:L8], dtype=np.float32)
    L4 = int(round(L8 * TARGET_SR / GAN_SR))
    if sr_raw == TARGET_SR:
        # native 4 kHz (every CirCor file): the reference loader's own samples cut to the analysed portion, which equals
        # x8[::2] up to float32 rounding; no second resampling of a truncated array
        x4 = audio[:L4].astype(np.float64)
    else:
        x4 = sig.resample(x8.astype(np.float64), L4)
    assert len(x4) == L4, (name, len(x4), L4)
    x4 = (x4 / (np.max(np.abs(x4)) + 1e-10)).astype(np.float32)   # max-abs normaliser over the analysed portion (the reference normalised the whole file)
    meta = dict(sr_raw=int(sr_raw), n_raw=int(len(audio)), n8_total=int(n8_total), L8=int(L8), L4=int(L4),
                n_win=int(n_win), dropped_tail_s=float((n8_total - L8) / GAN_SR), padded_short=bool(L8 < SIGNAL_LENGTH))
    return x8, x4, meta

# per chunk: is the synthetic-ECG cache usable (same recordings, same generator checkpoint)? The 8 kHz audio is loaded
# only for the chunks that must be generated.
def ecg_chunk_matches(z, names):
    return list(z['names']) == list(names) and str(z['ckpt_sha256']) == CKPT_SHA256

ECG_CHUNK_OK = {}
for k, names in chunk_slices(NAMES_ALL):
    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True
ECG_CACHE_COMPLETE = all(ECG_CHUNK_OK.values())
print('generator cache: %d of %d chunks complete; 8 kHz audio is loaded only for the incomplete chunks' % (sum(ECG_CHUNK_OK.values()), len(ECG_CHUNK_OK)))

AUDIO8, AUDIO4, AUDIO_META = {}, {}, {}
with stage('S2 shared audio'):
    for k, names in chunk_slices(NAMES_ALL):
        fp = CACHE_DIR / ('audio_%03d.npz' % k)
        need8 = not ECG_CHUNK_OK[k]

        def _load(z, names=names, need8=need8):
            if list(z['names']) != list(names):
                return None
            metas = json.loads(str(z['meta']))
            return {n: (z[n + '__8k'] if need8 else None, z[n + '__4k'], metas[n]) for n in names}

        got = read_cache(fp, _load)
        if got is not None:
            for n, (x8, x4, meta) in got.items():
                AUDIO4[n], AUDIO_META[n] = x4, meta
                if need8:
                    AUDIO8[n] = x8
            print('   chunk %03d: %d recordings loaded from cache' % (k, len(names)))
            continue
        t0 = time.time()
        arrays, metas = {}, {}
        with ThreadPoolExecutor(max_workers=16) as ex:
            for n, (x8, x4, meta) in zip(names, ex.map(load_shared_audio, names)):
                arrays[n + '__8k'] = x8
                arrays[n + '__4k'] = x4
                metas[n] = meta
                AUDIO4[n] = x4
                AUDIO_META[n] = meta
                if need8:
                    AUDIO8[n] = x8
        atomic_savez(fp, names=np.array(names), meta=np.array(json.dumps(metas)), **arrays)
        del arrays
        print('   chunk %03d: %d recordings decoded and cached in %.0f s' % (k, len(names), time.time() - t0))

    for n in NAMES_ALL:
        assert len(AUDIO4[n]) == AUDIO_META[n]['L4'] and AUDIO_META[n]['L4'] >= 1, n
        assert np.all(np.isfinite(AUDIO4[n])), 'non-finite audio in ' + n
    _L4 = np.array([AUDIO_META[n]['L4'] for n in NAMES_ALL]) / TARGET_SR
    _nw = np.array([AUDIO_META[n]['n_win'] for n in NAMES_ALL])
    _drop = np.array([AUDIO_META[n]['dropped_tail_s'] for n in NAMES_ALL])
    print('analysed audio: %.1f min total | per recording median %.1f s (min %.1f, max %.1f)' % (
        _L4.sum() / 60, np.median(_L4), _L4.min(), _L4.max()))
    print('windows per recording: median %d, max %d, total %d | tails dropped (<2 s): %d recordings, %.1f s in all | padded short recordings: %d' % (
        np.median(_nw), _nw.max(), _nw.sum(), int((_drop > 0).sum()), _drop.sum(),
        sum(AUDIO_META[n]['padded_short'] for n in NAMES_ALL)))
    print('native sample rates:', dict(collections.Counter(AUDIO_META[n]['sr_raw'] for n in NAMES_ALL)))
    print('RAM free: %.1f GB' % (psutil.virtual_memory().available / 1e9))
''')

md(r'''
## Stage 3: generator

Windows from all recordings are stacked and pushed through the generator in batches (32 on an
A100 or L4, 16 on a T4; the batch halves automatically on a CUDA out-of-memory error). Each window
is z-scored, receives the scenario one-hot `[1, 0, 0]`, and its output is polarity-normalised
before the windows are concatenated and the padding is trimmed. The synthetic ECG is cached in
float16 and always round-tripped through float16 before detection, so a resumed session and a
fresh session see the same trace. The cache stores the SHA-256 of the checkpoint that generated it
and is regenerated when the checkpoint differs.
''')

code('s3_generator', r'''
def gen_batch(segs):
    x = torch.from_numpy(np.stack(segs)).unsqueeze(1).to(device)
    s = torch.zeros(len(segs), 3, device=device)
    s[:, 0] = 1.0
    try:
        with torch.no_grad():
            out = gen(x, s).squeeze(1).float().cpu().numpy()
    finally:
        del x, s
    if out.ndim == 1:
        out = out[None, :]
    return out

ECG = {}
OOM_EVENTS = 0
GEN_BATCH_USED = GAN_BATCH
with stage('S3 generator'):
    gen.eval()
    n_generated = 0
    for k, names in chunk_slices(NAMES_ALL):
        fp = CACHE_DIR / ('ecg_%03d.npz' % k)
        if ECG_CHUNK_OK[k]:
            got = read_cache(fp, lambda z, names=names: {n: z[n] for n in names} if ecg_chunk_matches(z, names) else None)
            if got is not None:
                ECG.update(got)
                print('   chunk %03d: synthetic ECG loaded from cache' % k)
                continue
            # readable in Stage 2, not now: fetch this chunk's 8 kHz audio from the audio cache and regenerate it
            with np.load(CACHE_DIR / ('audio_%03d.npz' % k), allow_pickle=False) as z:
                for n in names:
                    AUDIO8[n] = z[n + '__8k']
        assert all(n in AUDIO8 for n in names), '8 kHz audio missing; rerun Stage 2 with the generator cache incomplete'
        jobs = []
        for n in names:
            x8 = AUDIO8[n]
            for w in range(AUDIO_META[n]['n_win']):
                seg = x8[w * SIGNAL_LENGTH:(w + 1) * SIGNAL_LENGTH]
                real = len(seg)
                assert real > 0, (n, w)
                if real < SIGNAL_LENGTH:
                    seg = np.pad(seg, (0, SIGNAL_LENGTH - real))
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                jobs.append((n, w, real, seg.astype(np.float32)))
        per = {n: [] for n in names}
        b, i = GEN_BATCH_USED, 0
        t0 = time.time()
        bar = tqdm(total=len(jobs), desc='chunk %03d generator' % k, leave=False)
        while i < len(jobs):
            blk = jobs[i:i + b]
            try:
                out = gen_batch([j[3] for j in blk])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                OOM_EVENTS += 1
                if b == 1:
                    raise
                b = max(1, b // 2)
                GEN_BATCH_USED = b
                bar.write('CUDA out of memory: generator batch -> %d' % b)
                continue
            for (n, w, real, _), e in zip(blk, out):
                e = ensure_positive_polarity(e[:real])          # polarity decided on the real part of a padded trailing window
                per[n].append((w, e.astype(np.float16)))
            i += len(blk)
            bar.update(len(blk))
        bar.close()
        arrays = {}
        for n in names:
            pieces = [e for _, e in sorted(per[n], key=lambda t: t[0])]
            full = np.concatenate(pieces)
            assert len(full) == AUDIO_META[n]['L8'], (n, len(full), AUDIO_META[n]['L8'])
            assert np.all(np.isfinite(full.astype(np.float32))), 'non-finite generator output in ' + n
            ECG[n] = full
            arrays[n] = full
        atomic_savez(fp, names=np.array(names), ckpt_sha256=np.array(CKPT_SHA256), **arrays)   # the cache is tied to the checkpoint
        n_generated += len(jobs)
        print('   chunk %03d: %d windows generated in %.0f s (batch %d)' % (k, len(jobs), time.time() - t0, b))
        del jobs, per, arrays
        torch.cuda.empty_cache()
    _nw = np.array([AUDIO_META[n]['n_win'] for n in NAMES_ALL])
    print('windows per recording: median %d, mean %.2f, max %d | windows generated this session: %d | OOM events: %d' % (
        np.median(_nw), _nw.mean(), _nw.max(), n_generated, OOM_EVENTS))
    AUDIO8.clear()
    gc.collect()
    print('RAM free: %.1f GB' % (psutil.virtual_memory().available / 1e9))
''')

md(r'''
## Stage 4: boundary arms and rate-fidelity diagnostics

Every arm is a function that returns a list of cycle dictionaries with `s1`, `systole`, `s2`,
`diastole` as 4 kHz sample-index pairs, plus a per-recording diagnostics dictionary. No boundary
may lie outside the analysed signal; this is asserted for every cycle of every arm. Arms A and B
and arms C, D, E, E2 are built in the worker pool (the TSV parsing included); a worker exception is
recorded in `ARM_FAILURES` (persisted to `arm_failures.json`) and the recording is dropped with the
reason "worker error", never as "too few cycles". The arm caches carry a hash of every arm constant,
a code version and the generator checkpoint's SHA-256, so a change to `FUSION`, `JOIN_MS`,
`REALIGN_MS`, `MATCH_MS`, `N_MIN_CYCLES`, to the arm code or to the checkpoint rebuilds them
instead of reusing stale boundaries.

**Arm A, `syn_fixed`.** The generator's windows are z-scored (mean removed, divided by the SD) and
polarity-normalised one by one, so the concatenated trace has a step at every window join. The
detector's 5 to 30 Hz band-pass would turn that step into a spurious peak, so the SOS envelope
(4th-order Butterworth 5 to 30 Hz in second-order sections, `sosfiltfilt`, squared, 80 ms boxcar,
that is a moving average over 80 ms) is computed **per generator window** and concatenated
(`_env_by_window`); the prominence ladder (0.3 / 0.15 / 0.08 / 0.03 of the envelope SD, tried in
turn until at least three peaks appear) and the 300 ms minimum distance are then applied once,
globally, exactly as the verbatim `_detect_sos` applies them. `_detect_sos` itself stays in the
notebook for provenance and is no longer called. If the ladder yields fewer than three peaks the
recording falls to an absolute-envelope detector and that path is recorded (`envelope-fallback`); if
that also fails the path is `none` and the recording is later dropped from the common set. Peaks
within 100 ms of a window join are de-duplicated (the larger envelope value wins), every peak is
realigned to the raw maximum within 40 ms, two realigned peaks closer than 300 ms lose the smaller
one (`n_close_pairs_dropped`), and the fixed 15 / 35 / 10 / 40 %RR phases follow after
`phases_from` clamps the train to the signal.

**Arm B, `syn_fusion`.** Acoustic S1 evidence is the 20 to 150 Hz band of the 4 kHz PCG,
rectified, smoothed with a 20 ms boxcar and normalised by its 95th percentile. Two peak lists are
taken from that envelope: the thinned list (250 ms minimum distance, prominence 0.1), which drives
the insertions, and the rescue list (prominence 0.1, no spacing rule, height at least
`rescue_height` = 0.50 of the p95 level). Each synthetic R-peak searches [R - 40 ms, R + 120 ms]
for the strongest thinned peak; if none is there but a rescue-list maximum is, the candidate is
still supported and anchored at that maximum's 50 % onset (`n_rescued`). The rescue rule exists
because the 250 ms thinning suppresses S1 wherever S2 is louder: with perfect anchors it raises the
support fraction from 0.82 to 0.98 and lowers false fallbacks from 11.7 % to 1.0 %, while wrong
anchors stay unsupported (0.39). When at least half the candidates are supported, unsupported ones
are removed and unmatched thinned peaks (no synthetic candidate within 150 ms) are inserted when
they split a gap of at least 1.6 times the median supported RR into two plausible halves. Otherwise
`fusion_fallback` is set and the supported acoustic onsets are kept as they are: only the removals
and insertions are skipped (a design choice; reverting to the raw synthetic anchors would discard
validated timing). The S1 anchor is the acoustic onset, the last 50 % crossing before the peak,
which sits about 17 ms after the annotated S1 onset. S2 onset is the 50 % crossing of the strongest
envelope local maximum inside [anchor + 0.25 RR, anchor + 0.55 RR] (the 250 ms spacing is a
beat-spacing rule and is not applied inside the systolic search window, because S2 sits 150 to
330 ms after S1 in this cohort); it lands about 18 to 22 ms after the annotated S2 onset and is
mildly murmur-dependent (23 % of cycles more than 40 ms late in murmur-present recordings against
10 % in murmur-absent ones); without one, the fixed-fraction S2 is used. Phases: S1 runs from onset
for min(0.15 RR, 120 ms), systole to the S2 onset, S2 for min(0.10 RR, 100 ms), diastole to the
next anchor. Malformed cycles are dropped and counted. Every constant is fixed in advance; nothing
is tuned on labels.

**Arm C, `pcg_seg`.** The dataset's per-beat annotation via `tsv_cycles_to_samples`, restricted to
cycles that lie inside the analysed portion. The verbatim converter truncates each boundary with
`int(t * sr)` separately, so adjacent annotated phases can overlap or leave a gap of one sample;
`phase_overlaps()` counts these and arm C is checked with the ordering rule only. The fraction of
the analysed portion covered by annotated cycles is recorded as `annotated_coverage_fraction`.

**Arm D, `pcg_seg_fixed`.** The fixed fractions applied inside each annotated cycle's own extent
(S1 onset to diastole end); the next cycle's S1 onset is used as the end only when the two cycles
abut within two samples. Interior annotation gaps are counted (`n_annotation_gaps`) and never
spanned by a pseudo-cycle. This isolates the cost of the fixed phase model from the cost of anchor
timing.

**Arm E, `equal_time`.** 75 bpm pseudo-cycles in four equal quarters (`equal_time_cycles`,
verbatim). This is the mechanism study's control and the one the validation gate checks.

**Arm E2, `equal_time_matched`.** `get_nongated_cycles`, verbatim from the published ablation: the
analysed portion is cut into N equal pseudo-cycles, N being the number of annotated cycles of arm C
inside that portion, each in four equal quarters; seconds are converted to samples with
`int(t * sr)` as that pipeline did, so adjacent pseudo-cycles can differ by one sample at a join.
E2 inherits the annotated **count**, not the annotated rate: wherever annotation covers only part
of the portion its pseudo-cycles run slow, which the diagnostics show next to the coverage
fraction. Stage 4 asserts per recording that the E2 cycle count equals the arm C count.

**Diagnostics.** For every recording and arm, against arm C: cycles, median RR and implied rate,
RR-CV, RR-jump rate (over 50 %), fraction of cycles above 180 bpm, Hungarian one-to-one matching of
cycle starts to annotated S1 onsets at 150 ms (the assignment that minimises the total absolute
offset, each start matched at most once) giving insertion rate, deletion rate and median signed
offset, over-detection ratio, and the recording-level median RR for the across-recording
correlation. Sign convention: `median_offset_ms` is the arm's cycle start minus the annotated S1
onset (negative = the arm precedes S1); `arm_mean_syn_ac_offset_ms` is the acoustic S1 onset
minus the synthetic R-peak (positive = the acoustic onset follows R). Arm A adds the detector path
and the non-finite fraction; arm B adds the support fraction, rescues, insertions, removals, the
fallback flag and the mean absolute synthetic-to-acoustic offset.
''')

code('s4_arm_defs', r'''
JOIN_MS, REALIGN_MS, MATCH_MS = 100, 40, 150
FUSION = dict(band_hz=(20, 150), env_smooth_ms=20, peak_dist_ms=250, peak_prom=0.10,
              search_lo_ms=-40, search_hi_ms=120, match_ms=150, gap_factor=1.6, support_min=0.5,
              s2_lo=0.25, s2_hi=0.55, s1_frac=0.15, s1_cap_ms=120, s2_frac=0.10, s2_cap_ms=100,
              s1_onset_back_ms=150, s2_onset_back_ms=100, rescue_height=0.50)
AC_SOS = sig.butter(4, list(FUSION['band_hz']), btype='bandpass', fs=TARGET_SR, output='sos')
ARM_CODE_VERSION = 'R1-2026-09-30'          # bumped whenever an arm's construction changes, so earlier arm caches are rebuilt
# the generator checkpoint is part of the hash: boundaries built from another checkpoint's synthetic ECG are never reused
ARM_CFG_HASH = hashlib.sha1(json.dumps(dict(FUSION=FUSION, JOIN_MS=JOIN_MS, REALIGN_MS=REALIGN_MS, MATCH_MS=MATCH_MS,
                                            N_MIN_CYCLES=N_MIN_CYCLES, code=ARM_CODE_VERSION, checkpoint=CKPT_SHA256),
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]

def check_cycles(cyc, L, tag, strict=True):
    """Every boundary inside [0, L], every phase ordered (start <= end), phase starts non-decreasing.
    strict=True additionally requires the eight boundaries to be monotone; that holds for every arm
    built here. The dataset annotation (arm C) occasionally overlaps adjacent phases by a sample or
    two, so arm C and its jittered copies are checked with strict=False and phase_overlaps() counts them."""
    for c in cyc:
        seq = [v for k in PHASE_KEYS for v in c[k]]
        assert all(0 <= v <= L for v in seq), '%s: boundary outside the signal %s (L=%d)' % (tag, seq, L)
        assert all(c[k][0] <= c[k][1] for k in PHASE_KEYS), '%s: phase with end before start %s' % (tag, seq)
        starts = [c[k][0] for k in PHASE_KEYS]
        assert all(a <= b for a, b in zip(starts, starts[1:])), '%s: phase starts not ordered %s' % (tag, seq)
        if strict:
            assert all(a <= b for a, b in zip(seq, seq[1:])), '%s: boundaries not monotone %s' % (tag, seq)
    return cyc

def phase_overlaps(cyc):
    """(number of adjacent-phase overlaps, largest overlap in samples) in an annotated cycle list."""
    n, mx = 0, 0
    for c in cyc:
        for k1, k2 in zip(PHASE_KEYS, PHASE_KEYS[1:]):
            ov = c[k1][1] - c[k2][0]
            if ov > 0:
                n += 1
                mx = max(mx, ov)
    return n, mx

def envelope_fallback_peaks(ecg, fs, min_ms=300):
    w = int(0.05 * fs)
    sm = np.convolve(np.abs(ecg), np.ones(w) / w, mode='same')
    pk, _ = sig.find_peaks(sm, distance=int(min_ms / 1000 * fs), height=np.percentile(sm, 75))
    return pk

def _env_by_window(ecg, fs, n_win):
    """5-30 Hz SOS envelope per generator window, concatenated: the step between independently
    z-scored, independently polarity-flipped windows never enters the filter."""
    return np.concatenate([_sos_env(ecg[w * SIGNAL_LENGTH:(w + 1) * SIGNAL_LENGTH], fs) for w in range(n_win)])

def detect_arm_a(ecg32, fs, n_win):
    """Per-window SOS envelope, one global prominence ladder and minimum distance -> (peaks at 8 kHz, diagnostics).
    The verbatim _detect_sos is kept for provenance; its ladder and distance are reproduced here on the per-window envelope."""
    d = {}
    env = _env_by_window(ecg32, fs, n_win)
    assert len(env) == len(ecg32), (len(env), len(ecg32))
    d['nonfinite_frac'] = float(np.mean(~np.isfinite(env)))
    ref = float(np.std(env))
    md = int(300 / 1000 * fs)
    pk, path = np.array([], int), 'none'
    if np.isfinite(ref) and ref > 0:
        for pf in (0.3, 0.15, 0.08, 0.03):                     # the same ladder as _detect_sos
            cand, _ = sig.find_peaks(env, distance=md, prominence=ref * pf)
            if len(cand) >= 3:
                pk, path = cand, 'primary'
                break
    if len(pk) < 3:
        pk = envelope_fallback_peaks(ecg32, fs)
        path = 'envelope-fallback' if len(pk) >= 3 else 'none'
        if len(pk) < 3:
            pk = np.array([], int)
        env = np.abs(ecg32)
    d['detector_path'] = path
    pk = np.asarray(pk, int)
    tol = int(JOIN_MS / 1000 * fs)
    keep = np.ones(len(pk), bool)
    near = removed = 0
    for j in [w * SIGNAL_LENGTH for w in range(1, n_win)]:
        idx = np.flatnonzero(np.abs(pk - j) <= tol)
        near += len(idx)
        if len(idx) > 1:
            best = idx[int(np.argmax(env[pk[idx]]))]
            for q in idx:
                if q != best:
                    keep[q] = False
                    removed += 1
    pk = pk[keep]
    r = int(REALIGN_MS / 1000 * fs)
    out = []
    for p in pk:
        a, b = max(0, int(p) - r), min(len(ecg32), int(p) + r + 1)
        out.append(a + int(np.argmax(ecg32[a:b])))
    pk = np.array(sorted(set(out)), int)
    # realignment can pull two peaks inside the 300 ms minimum distance: drop the smaller (by envelope) of such a pair
    close_dropped = 0
    while len(pk) >= 2:
        j = np.flatnonzero(np.diff(pk) < md)
        if len(j) == 0:
            break
        j0 = int(j[0])
        pk = np.delete(pk, j0 if env[pk[j0]] < env[pk[j0 + 1]] else j0 + 1)
        close_dropped += 1
    d.update(n_peaks_near_join=int(near), n_join_dedup=int(removed), n_close_pairs_dropped=int(close_dropped), n_peaks=int(len(pk)))
    return pk, d

def arm_a_cycles(name):
    ecg = ECG[name].astype(np.float32)          # float16 cache -> float32 for detection, always
    L4 = AUDIO_META[name]['L4']
    pk8, d = detect_arm_a(ecg, GAN_SR, AUDIO_META[name]['n_win'])
    pk4 = (pk8 * TARGET_SR / GAN_SR).astype(int)
    cyc = phases_from(pk4, L4)
    return check_cycles(cyc, L4, 'syn_fixed ' + name), d, pk4

def acoustic_envelope(x4, sr=TARGET_SR):
    f = sig.sosfiltfilt(AC_SOS, x4.astype(np.float64))
    w = int(FUSION['env_smooth_ms'] / 1000 * sr)
    env = np.convolve(np.abs(f), np.ones(w) / w, mode='same')
    return env / (np.percentile(env, 95) + 1e-12)

def onset_50(env, peak, back):
    """Last sample before `peak` where env crosses 50 % of the peak height, within `back` samples.
    Returns (onset, found)."""
    h = 0.5 * env[peak]
    a = max(0, int(peak) - int(back))
    below = np.flatnonzero(env[a:peak] < h)
    if len(below) == 0:
        return int(a), False
    return int(a + below[-1] + 1), True

def arm_b_cycles(name, pk4):
    x4 = AUDIO4[name]
    L4 = len(x4)
    sr = TARGET_SR
    env = acoustic_envelope(x4)
    ac_pk, _ = sig.find_peaks(env, distance=int(FUSION['peak_dist_ms'] / 1000 * sr), prominence=FUSION['peak_prom'])
    ac_pk = np.asarray(ac_pk, int)
    raw_pk, _ = sig.find_peaks(env, prominence=FUSION['peak_prom'])
    strong = raw_pk[env[raw_pk] >= FUSION['rescue_height']]     # local maxima at >= 50 % of the p95 level, no spacing rule (rescue list)
    back1 = int(FUSION['s1_onset_back_ms'] / 1000 * sr)
    ac_on = [onset_50(env, p, back1) for p in ac_pk]          # (onset sample, crossing found) per thinned acoustic peak
    r4 = np.asarray(pk4, int)
    d = dict(n_syn=int(len(r4)), n_acoustic=int(len(ac_pk)), supported_frac=np.nan, mean_abs_syn_ac_offset_ms=np.nan,
             mean_syn_ac_offset_ms=np.nan, n_removed=0, n_inserted=0, n_rescued=0, fusion_fallback=False, n_onset_fallback=0,
             n_malformed=0, frac_s2_acoustic=np.nan)
    if len(r4) < 2:
        return [], d
    lo, hi = int(FUSION['search_lo_ms'] / 1000 * sr), int(FUSION['search_hi_ms'] / 1000 * sr)
    used = np.zeros(len(ac_pk), bool)
    supported = np.zeros(len(r4), bool)
    anchors, offs, onset_fb, rescued = [], [], 0, 0
    for i, r in enumerate(r4):
        cand = np.flatnonzero((ac_pk >= r + lo) & (ac_pk <= r + hi) & ~used)
        if len(cand):
            j = cand[int(np.argmax(env[ac_pk[cand]]))]
            used[j] = True
            on, found = ac_on[j]
        else:
            resc = strong[(strong >= r + lo) & (strong <= r + hi)]
            if len(resc) == 0:
                anchors.append(int(r))
                continue
            rescued += 1
            on, found = onset_50(env, int(resc[int(np.argmax(env[resc]))]), back1)
        supported[i] = True
        if found:
            anchors.append(int(on))
            offs.append((int(on) - int(r)) / sr * 1000.0)
        else:
            anchors.append(int(r))
            onset_fb += 1
    frac = float(supported.mean())
    d.update(supported_frac=frac, n_onset_fallback=int(onset_fb), n_rescued=int(rescued),
             mean_abs_syn_ac_offset_ms=float(np.mean(np.abs(offs))) if offs else np.nan,
             mean_syn_ac_offset_ms=float(np.mean(offs)) if offs else np.nan)
    if frac >= FUSION['support_min']:
        anc = np.array([a for a, s in zip(anchors, supported) if s], int)
        d['n_removed'] = int((~supported).sum())
        rr_sup = np.diff(anc)
        med = float(np.median(rr_sup)) if len(rr_sup) else np.nan
        ins = 0
        if np.isfinite(med) and med > 0:
            mtol = int(FUSION['match_ms'] / 1000 * sr)
            for j in range(len(ac_pk)):
                if used[j] or np.min(np.abs(r4 - ac_pk[j])) <= mtol:
                    continue
                o = int(ac_on[j][0]) if ac_on[j][1] else int(ac_pk[j])
                k = int(np.searchsorted(anc, o))
                if k == 0 or k == len(anc):
                    continue
                gap = anc[k] - anc[k - 1]
                if gap >= FUSION['gap_factor'] * med and (o - anc[k - 1]) >= 0.5 * med and (anc[k] - o) >= 0.5 * med:
                    anc = np.insert(anc, k, o)
                    ins += 1
        d['n_inserted'] = int(ins)
        anc = list(anc)
    else:
        anc = list(anchors)                  # fallback: the supported acoustic onsets stay; only removals and insertions are skipped
        d['fusion_fallback'] = True
    anc = sorted(set(int(a) for a in anc if 0 <= a <= L4))
    cyc, malformed, s2_ac = [], 0, 0
    s1_cap, s2_cap = int(FUSION['s1_cap_ms'] / 1000 * sr), int(FUSION['s2_cap_ms'] / 1000 * sr)
    back2 = int(FUSION['s2_onset_back_ms'] / 1000 * sr)
    for a, b in zip(anc, anc[1:]):
        rr = b - a
        if rr <= 0:
            malformed += 1
            continue
        s1e = a + min(int(FUSION['s1_frac'] * rr), s1_cap)
        w0, w1 = a + int(FUSION['s2_lo'] * rr), min(a + int(FUSION['s2_hi'] * rr), L4)
        s2on, found = a + int(0.50 * rr), False
        if w1 > w0 + 2:
            loc, _ = sig.find_peaks(env[w0:w1], prominence=FUSION['peak_prom'])
            if len(loc):
                p = w0 + int(loc[int(np.argmax(env[w0:w1][loc]))])
                cand_on, ok = onset_50(env, p, back2)
                if ok:
                    s2on, found = cand_on, True
        s2e = s2on + min(int(FUSION['s2_frac'] * rr), s2_cap)
        if not (a < s1e < s2on < s2e < b):
            malformed += 1
            continue
        s2_ac += int(found)
        cyc.append({'cycle': len(cyc), 's1': (a, s1e), 'systole': (s1e, s2on), 's2': (s2on, s2e), 'diastole': (s2e, b)})
    d.update(n_malformed=int(malformed), frac_s2_acoustic=(s2_ac / len(cyc)) if cyc else np.nan)
    return check_cycles(cyc, L4, 'syn_fusion ' + name), d

def arm_c_cycles(name):
    L4 = AUDIO_META[name]['L4']
    # tsv_cycles_to_samples (verbatim) truncates each boundary with int(t * sr) separately, so adjacent annotated phases
    # can overlap or leave a gap of one sample; phase_overlaps() counts them and arm C is checked with strict=False
    cyc_all = tsv_cycles_to_samples(REC[name][4], TARGET_SR)
    cyc = [c for c in cyc_all if all(0 <= c[k][0] <= L4 and 0 <= c[k][1] <= L4 for k in PHASE_KEYS)]
    n_ov, max_ov = phase_overlaps(cyc)
    covered = sum(int(c['diastole'][1]) - int(c['s1'][0]) for c in cyc)                          # annotated samples inside the portion
    span = (int(cyc[-1]['diastole'][1]) - int(cyc[0]['s1'][0])) if cyc else 0                     # first S1 onset to last diastole end
    return (check_cycles(cyc, L4, 'pcg_seg ' + name, strict=False),
            dict(n_annotated=len(cyc_all), n_beyond_portion=len(cyc_all) - len(cyc), n_phase_overlaps=n_ov, max_phase_overlap_samples=max_ov,
                 annotated_coverage_fraction=float(covered / L4) if L4 else np.nan, annotated_span_fraction=float(span / L4) if L4 else np.nan))

def arm_d_cycles(name, cyc_c):
    """Fixed fractions inside each annotated cycle's own extent; the next S1 onset is the end only when the cycles abut."""
    L4 = AUDIO_META[name]['L4']
    cyc, n_gap = [], 0
    for a, b in zip(cyc_c, list(cyc_c[1:]) + [None]):
        s = int(a['s1'][0])
        abut = b is not None and abs(int(b['s1'][0]) - int(a['diastole'][1])) <= 2
        e = int(b['s1'][0]) if abut else int(a['diastole'][1])
        n_gap += int(b is not None and not abut)
        seg = segment_phases_samples([s, min(e, L4)], L4)
        if seg:
            seg[0]['cycle'] = len(cyc)
            cyc.append(seg[0])
    return check_cycles(cyc, L4, 'pcg_seg_fixed ' + name), dict(n_anchors=len(cyc_c), n_annotation_gaps=int(n_gap))

def arm_e_cycles(name):
    L4 = AUDIO_META[name]['L4']
    return check_cycles(equal_time_cycles(L4, TARGET_SR), L4, 'equal_time ' + name), {}

def arm_e2_cycles(name, cyc_c):
    L4 = AUDIO_META[name]['L4']
    sec = get_nongated_cycles(L4, TARGET_SR, len(cyc_c))
    # seconds -> samples with int(t * sr), as the published pipeline did: adjacent pseudo-cycles can differ by one sample at a join
    cyc = [{'cycle': i, **{k: (int(c[k][0] * TARGET_SR), int(c[k][1] * TARGET_SR)) for k in PHASE_KEYS}} for i, c in enumerate(sec)]
    return check_cycles(cyc, L4, 'equal_time_matched ' + name), dict(n_matched_to_annotation=len(cyc_c))

def match_hungarian(a, b, tol):
    """Optimal one-to-one matching within tol samples -> (n_matched, signed deltas a - b)."""
    if len(a) == 0 or len(b) == 0:
        return 0, np.array([], float)
    D = np.abs(np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :])
    ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e9))
    ok = D[ri, ci] <= tol
    return int(ok.sum()), (np.asarray(a, float)[ri[ok]] - np.asarray(b, float)[ci[ok]])

def cycle_starts(cyc):
    return np.array(sorted(set(int(c['s1'][0]) for c in cyc)), int)

def rate_diag(cyc, ref_cyc, sr=TARGET_SR):
    st, ref = rhythm(cyc), rhythm(ref_cyc)
    a, b = cycle_starts(cyc), cycle_starts(ref_cyc)
    m, dl = match_hungarian(a, b, int(MATCH_MS / 1000 * sr))
    dl_ms = dl / sr * 1000.0
    return dict(n_cycles=int(len(cyc)), median_rr_ms=st['rr_ms'], bpm=st['bpm'], rr_cv=st['cv'], rr_jump=st['jump'],
                frac_fast=st['fast'], ref_n_cycles=int(len(ref_cyc)), ref_median_rr_ms=ref['rr_ms'], ref_bpm=ref['bpm'],
                n_matched=int(m), insertion_rate=(len(a) - m) / len(a) if len(a) else np.nan,
                deletion_rate=(len(b) - m) / len(b) if len(b) else np.nan,
                median_offset_ms=float(np.median(dl_ms)) if len(dl_ms) else np.nan,
                median_abs_offset_ms=float(np.median(np.abs(dl_ms))) if len(dl_ms) else np.nan,
                overdetection_ratio=len(a) / len(b) if len(b) else np.nan), dl_ms

def _ab_worker(name):
    try:
        cyc_a, d_a, pk4 = arm_a_cycles(name)
        cyc_b, d_b = arm_b_cycles(name, pk4)
        return name, (cyc_a, d_a, pk4, cyc_b, d_b), None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

def _cd_worker(name):
    try:
        cyc_c, d_c = arm_c_cycles(name)
        cyc_d, d_d = arm_d_cycles(name, cyc_c)
        cyc_e, d_e = arm_e_cycles(name)
        cyc_e2, d_e2 = arm_e2_cycles(name, cyc_c)
        return name, (cyc_c, d_c, cyc_d, d_d, cyc_e, d_e, cyc_e2, d_e2), None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

ARMS = ['syn_fixed', 'syn_fusion', 'pcg_seg', 'pcg_seg_fixed', 'equal_time', 'equal_time_matched']
ARM_LETTER = {'syn_fixed': 'A', 'syn_fusion': 'B', 'pcg_seg': 'C', 'pcg_seg_fixed': 'D', 'equal_time': 'E', 'equal_time_matched': 'E2'}
ARM_LABEL = {'syn_fixed': 'A synthetic-fixed', 'syn_fusion': 'B synthetic-fusion', 'pcg_seg': 'C PCG segmentation',
             'pcg_seg_fixed': 'D PCG-seg + fixed', 'equal_time': 'E equal-time 75 bpm', 'equal_time_matched': 'E2 equal-time matched'}
EQUAL_TIME_DEFINITIONS = {
    'equal_time': 'equal_time_cycles (verbatim, analysis source): fixed 75 bpm pseudo-cycles, four equal quarters; the mechanism study control used by the validation gate',
    'equal_time_matched': 'get_nongated_cycles (verbatim, ABLATION_GATING_VALIDATION.ipynb): the analysed portion cut into N equal pseudo-cycles, N = annotated cycles of arm C inside that portion, four equal quarters, seconds -> samples by int(t*sr); the published control, which inherits the annotated count and not the annotated rate'}
print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)
''')

code('s4_run', r'''
CYC = {arm: {} for arm in ARMS}
ARM_DIAG = {arm: {} for arm in ARMS}
PK_A = {}
ARM_FAILURES = []

def read_arm_cache(fp):
    """{name: payload} from an arm cache written under the current ARM_CFG_HASH; {} when missing, unreadable or stale."""
    if not fp.exists():
        return {}
    try:
        with open(fp, 'rb') as fh:
            obj = pickle.load(fh)
        if not (isinstance(obj, dict) and obj.get('cfg') == ARM_CFG_HASH and isinstance(obj.get('data'), dict)):
            raise ValueError('stale')
        return obj['data']
    except Exception as e:
        why = 'stale (arm configuration hash differs)' if isinstance(e, ValueError) and str(e) == 'stale' else 'unreadable (%s)' % type(e).__name__
        CACHE_REBUILDS.append((fp.name, why))
        print('   %s cache %s: rebuilding' % (why, fp.name))
        fp.unlink()
        return {}

_AB_ERROR = ([], {'detector_path': 'error', 'nonfinite_frac': np.nan, 'n_peaks_near_join': 0, 'n_join_dedup': 0, 'n_close_pairs_dropped': 0, 'n_peaks': 0},
             np.array([], int), [], {'fusion_fallback': True, 'supported_frac': np.nan, 'n_rescued': 0, 'n_inserted': 0, 'n_removed': 0, 'n_malformed': 0})
_CD_ERROR = ([], {'worker_error': True, 'n_annotated': 0, 'n_beyond_portion': 0, 'n_phase_overlaps': 0, 'max_phase_overlap_samples': 0,
                  'annotated_coverage_fraction': np.nan, 'annotated_span_fraction': np.nan},
             [], {'n_anchors': 0, 'n_annotation_gaps': 0}, [], {}, [], {'n_matched_to_annotation': 0})

with stage('S4 boundaries'):
    # ---- arms A and B (worker pool; the generator output is already on disk and in RAM) ----
    fp_ab = CACHE_DIR / 'cycles_arms_ab.pkl'
    ab = read_arm_cache(fp_ab)
    if ab and not all(n in ab for n in NAMES_ALL):
        print('   arms A/B cache incomplete (%d of %d): computing the missing recordings' % (sum(n in ab for n in NAMES_ALL), len(NAMES_ALL)))
    todo = [n for n in NAMES_ALL if n not in ab]
    if todo:
        t0 = time.time()
        with _pool(N_PROC) as ex:
            for name, payload, err in tqdm(ex.map(_ab_worker, todo, chunksize=8), total=len(todo), desc='arms A+B', leave=False):
                if err is not None:
                    ARM_FAILURES.append((name, 'A/B', err))
                    payload = _AB_ERROR
                ab[name] = payload
        atomic_pickle(fp_ab, {'cfg': ARM_CFG_HASH, 'data': ab})
        print('   arms A/B built for %d recordings in %.0f s' % (len(todo), time.time() - t0))
    else:
        print('   arms A/B loaded from cache')
    for n in NAMES_ALL:
        cyc_a, d_a, pk4, cyc_b, d_b = ab[n]
        CYC['syn_fixed'][n], ARM_DIAG['syn_fixed'][n], PK_A[n] = cyc_a, d_a, pk4
        CYC['syn_fusion'][n], ARM_DIAG['syn_fusion'][n] = cyc_b, d_b
    del ab

    # ---- arms C, D, E, E2 (the TSV parsing runs in the pool as well) ----
    fp_cd = CACHE_DIR / 'cycles_arms_cdee2.pkl'
    cd = read_arm_cache(fp_cd)
    todo = [n for n in NAMES_ALL if n not in cd]
    if todo:
        t0 = time.time()
        with _pool(N_PROC) as ex:
            for name, payload, err in tqdm(ex.map(_cd_worker, todo, chunksize=16), total=len(todo), desc='arms C/D/E/E2', leave=False):
                if err is not None:
                    ARM_FAILURES.append((name, 'C/D/E/E2', err))
                    payload = _CD_ERROR
                cd[name] = payload
        atomic_pickle(fp_cd, {'cfg': ARM_CFG_HASH, 'data': cd})
        print('   arms C/D/E/E2 built for %d recordings in %.0f s' % (len(todo), time.time() - t0))
    else:
        print('   arms C/D/E/E2 loaded from cache')
    for n in NAMES_ALL:
        cyc_c, d_c, cyc_d, d_d, cyc_e, d_e, cyc_e2, d_e2 = cd[n]
        CYC['pcg_seg'][n], ARM_DIAG['pcg_seg'][n] = cyc_c, d_c
        CYC['pcg_seg_fixed'][n], ARM_DIAG['pcg_seg_fixed'][n] = cyc_d, d_d
        CYC['equal_time'][n], ARM_DIAG['equal_time'][n] = cyc_e, d_e
        CYC['equal_time_matched'][n], ARM_DIAG['equal_time_matched'][n] = cyc_e2, dict(d_e2, annotated_coverage_fraction=d_c.get('annotated_coverage_fraction', np.nan))
    del cd
    with open(RUN_DIR / 'arm_failures.json', 'w') as fh:
        json.dump(ARM_FAILURES, fh, indent=1)

    # re-check every cached cycle against the signal length (the assertion the design demands) and the E2 count
    for arm in ARMS:
        for n in NAMES_ALL:
            check_cycles(CYC[arm][n], AUDIO_META[n]['L4'], arm + ' ' + n, strict=(arm != 'pcg_seg'))
    for n in NAMES_ALL:
        assert len(CYC['equal_time_matched'][n]) == len(CYC['pcg_seg'][n]), 'E2 must carry exactly the arm C cycle count: %s' % n
    _ov = [ARM_DIAG['pcg_seg'][n]['n_phase_overlaps'] for n in NAMES_ALL]
    print('   all boundaries of all arms lie inside their signals; E2 cycle count equals the arm C count in every recording; annotation phase overlaps (arm C): %d cycles in %d recordings, largest %d samples' % (
        int(np.sum(_ov)), int(np.sum(np.array(_ov) > 0)), max(ARM_DIAG['pcg_seg'][n]['max_phase_overlap_samples'] for n in NAMES_ALL)))

    # ---- the common kept set ----
    KEEP, DROPPED = [], []
    _err = {n for n, _, _ in ARM_FAILURES} | {n for n in NAMES_ALL if ARM_DIAG['syn_fixed'][n].get('detector_path') == 'error' or ARM_DIAG['pcg_seg'][n].get('worker_error')}
    for n in NAMES_ALL:
        bad = [arm for arm in ARMS if len(CYC[arm][n]) < N_MIN_CYCLES]
        if bad:
            DROPPED.append((n, [arm + (' (worker error)' if n in _err else '') for arm in bad]))
        else:
            KEEP.append(n)
    KEEP_ARR = np.array(KEEP)
    print('\nkept recordings (>= %d cycles in every arm): %d of %d | dropped: %d' % (N_MIN_CYCLES, len(KEEP), len(NAMES_ALL), len(DROPPED)))
    for pat, cnt in collections.Counter(tuple(b) for _, b in DROPPED).most_common():
        print('   dropped %4d  fewer than %d cycles in: %s' % (cnt, N_MIN_CYCLES, ', '.join(pat)))
    if ARM_FAILURES:
        print('   worker failures: %d (persisted to arm_failures.json), e.g. %s' % (len(ARM_FAILURES), ARM_FAILURES[:3]))
    assert len(KEEP) >= 20, 'too few recordings survive; inspect the drop reasons above'

    # ---- diagnostics against arm C ----
    OFFSETS = {arm: {} for arm in ARMS}
    DIAG_ROWS = []
    for n in NAMES_ALL:
        ref = CYC['pcg_seg'][n]
        for arm in ARMS:
            rd, dl = rate_diag(CYC[arm][n], ref)
            OFFSETS[arm][n] = dl
            row = dict(recording=n, patient_id=PID_OF[n], age_band=AGE_OF.get(PID_OF[n], 'Unknown'), murmur=LAB_OF[n]['murmur'],
                       arm=arm, kept=n in set(KEEP), analysed_s=AUDIO_META[n]['L4'] / TARGET_SR, n_windows=AUDIO_META[n]['n_win'])
            row.update(rd)
            for k_, v_ in ARM_DIAG[arm][n].items():
                row['arm_' + k_] = v_
            DIAG_ROWS.append(row)
    DIAG = pd.DataFrame(DIAG_ROWS)
    DIAG.to_csv(RUN_DIR / 'diagnostics.csv', index=False)
    atomic_pickle(CACHE_DIR / 'offsets_ms.pkl', OFFSETS)
    print('diagnostics.csv written: %d rows (%d recordings x %d arms)' % (len(DIAG), len(NAMES_ALL), len(ARMS)))

    # ---- summary table (medians over kept recordings) and rate correlation ----
    DK = DIAG[DIAG.kept]
    cols = ['n_cycles', 'median_rr_ms', 'bpm', 'rr_cv', 'rr_jump', 'frac_fast', 'insertion_rate', 'deletion_rate',
            'median_offset_ms', 'median_abs_offset_ms', 'overdetection_ratio']
    DIAG_SUMMARY, RR_PEARSON = {}, {}
    print('\nrate-fidelity diagnostics, medians over %d kept recordings (arm C is the reference, so its own row is the identity)' % len(KEEP))
    print('%-22s %6s %7s %6s %6s %6s %7s %6s %6s %8s %8s %6s %7s' % (
        'arm', 'cycles', 'RR ms', 'bpm', 'RRcv', 'jump', '>180bpm', 'ins', 'del', 'off ms', '|off| ms', 'x det', 'r(RR)'))
    print('-' * 118)
    for arm in ARMS:
        sub = DK[DK.arm == arm]
        med = {c: float(np.nanmedian(sub[c].astype(float))) for c in cols}
        ok = np.isfinite(sub['median_rr_ms'].astype(float)) & np.isfinite(sub['ref_median_rr_ms'].astype(float))
        r = float(np.corrcoef(sub.loc[ok, 'median_rr_ms'].astype(float), sub.loc[ok, 'ref_median_rr_ms'].astype(float))[0, 1]) if ok.sum() > 2 else np.nan
        v = sub['median_abs_offset_ms'].astype(float)
        v = v[np.isfinite(v)]
        pooled = np.concatenate([OFFSETS[arm][n] for n in KEEP if len(OFFSETS[arm][n])]) if any(len(OFFSETS[arm][n]) for n in KEEP) else np.array([])
        DIAG_SUMMARY[arm] = dict(med, n=int(len(sub)), rr_pearson_r=r, n_rate_pairs=int(ok.sum()),
                                 abs_offset_iqr_ms=[float(np.percentile(v, 25)), float(np.percentile(v, 75))] if len(v) >= 4 else [None, None],
                                 offset_sd_ms=float(np.std(pooled)) if len(pooled) >= 2 else None, n_matched_pairs=int(len(pooled)))
        RR_PEARSON[arm] = r
        print('%-22s %6.0f %7.0f %6.0f %6.3f %6.3f %7.3f %6.3f %6.3f %+8.1f %8.1f %6.2f %7.3f' % (
            arm, med['n_cycles'], med['median_rr_ms'], med['bpm'], med['rr_cv'], med['rr_jump'], med['frac_fast'],
            med['insertion_rate'], med['deletion_rate'], med['median_offset_ms'], med['median_abs_offset_ms'], med['overdetection_ratio'], r))
    pa = DK[DK.arm == 'syn_fixed']
    DIAG_SUMMARY['syn_fixed']['detector_paths'] = dict(collections.Counter(pa['arm_detector_path']))
    DIAG_SUMMARY['syn_fixed']['nonfinite_frac_mean'] = float(np.nanmean(pa['arm_nonfinite_frac'].astype(float)))
    DIAG_SUMMARY['syn_fixed']['peaks_near_join_total'] = int(pa['arm_n_peaks_near_join'].sum())
    DIAG_SUMMARY['syn_fixed']['join_dedup_total'] = int(pa['arm_n_join_dedup'].sum())
    DIAG_SUMMARY['syn_fixed']['close_pairs_dropped_total'] = int(pa['arm_n_close_pairs_dropped'].sum())
    pb = DK[DK.arm == 'syn_fusion']
    DIAG_SUMMARY['syn_fusion'].update(
        fusion_fallback_frac=float(pb['arm_fusion_fallback'].astype(bool).mean()),
        supported_frac_median=float(np.nanmedian(pb['arm_supported_frac'].astype(float))),
        n_inserted_total=int(pb['arm_n_inserted'].sum()), n_removed_total=int(pb['arm_n_removed'].sum()),
        n_rescued_total=int(pb['arm_n_rescued'].sum()), n_malformed_total=int(pb['arm_n_malformed'].sum()),
        mean_abs_syn_ac_offset_ms_median=float(np.nanmedian(pb['arm_mean_abs_syn_ac_offset_ms'].astype(float))),
        frac_s2_acoustic_median=float(np.nanmedian(pb['arm_frac_s2_acoustic'].astype(float))))
    pdd = DK[DK.arm == 'pcg_seg_fixed']
    DIAG_SUMMARY['pcg_seg_fixed'].update(n_annotation_gaps_total=int(pdd['arm_n_annotation_gaps'].sum()),
                                         frac_recordings_with_gaps=float((pdd['arm_n_annotation_gaps'] > 0).mean()))
    pc = DK[DK.arm == 'pcg_seg']
    _cov = pc['arm_annotated_coverage_fraction'].astype(float)
    DIAG_SUMMARY['equal_time_matched'].update(annotated_coverage_fraction_median=float(np.nanmedian(_cov)),
                                              n_recordings_coverage_below_0_8=int((_cov < 0.8).sum()),
                                              note='E2 inherits the annotated cycle count, not the annotated rate; its implied bpm falls with the coverage fraction')
    print('\narm A detector paths:', DIAG_SUMMARY['syn_fixed']['detector_paths'],
          '| mean non-finite fraction %.4f | peaks within 100 ms of a join %d (de-duplicated %d) | close pairs dropped after realignment %d' % (
              DIAG_SUMMARY['syn_fixed']['nonfinite_frac_mean'], DIAG_SUMMARY['syn_fixed']['peaks_near_join_total'],
              DIAG_SUMMARY['syn_fixed']['join_dedup_total'], DIAG_SUMMARY['syn_fixed']['close_pairs_dropped_total']))
    print('arm B: fallback in %.1f%% of recordings | median supported fraction %.2f | rescued %d, inserted %d, removed %d, malformed %d | median |R - acoustic S1 onset| %.0f ms | acoustic S2 found in %.0f%% of cycles (median per recording)' % (
        100 * DIAG_SUMMARY['syn_fusion']['fusion_fallback_frac'], DIAG_SUMMARY['syn_fusion']['supported_frac_median'],
        DIAG_SUMMARY['syn_fusion']['n_rescued_total'], DIAG_SUMMARY['syn_fusion']['n_inserted_total'], DIAG_SUMMARY['syn_fusion']['n_removed_total'],
        DIAG_SUMMARY['syn_fusion']['n_malformed_total'], DIAG_SUMMARY['syn_fusion']['mean_abs_syn_ac_offset_ms_median'], 100 * DIAG_SUMMARY['syn_fusion']['frac_s2_acoustic_median']))
    print('arm D: %d interior annotation gaps in %.0f%% of recordings, none spanned by a pseudo-cycle' % (
        DIAG_SUMMARY['pcg_seg_fixed']['n_annotation_gaps_total'], 100 * DIAG_SUMMARY['pcg_seg_fixed']['frac_recordings_with_gaps']))
    print('arm E2 inherits the annotated count, not the rate: median implied bpm %.0f vs annotation %.0f | annotated coverage of the analysed portion: median %.2f, %d of %d kept recordings below 0.8' % (
        DIAG_SUMMARY['equal_time_matched']['bpm'], DIAG_SUMMARY['pcg_seg']['bpm'], DIAG_SUMMARY['equal_time_matched']['annotated_coverage_fraction_median'],
        DIAG_SUMMARY['equal_time_matched']['n_recordings_coverage_below_0_8'], len(KEEP)))
    print('offset spread (matched pairs): arm A SD %s ms, per-recording median |offset| IQR %s | arm B SD %s ms, IQR %s' % (
        DIAG_SUMMARY['syn_fixed']['offset_sd_ms'] and '%.0f' % DIAG_SUMMARY['syn_fixed']['offset_sd_ms'], DIAG_SUMMARY['syn_fixed']['abs_offset_iqr_ms'],
        DIAG_SUMMARY['syn_fusion']['offset_sd_ms'] and '%.0f' % DIAG_SUMMARY['syn_fusion']['offset_sd_ms'], DIAG_SUMMARY['syn_fusion']['abs_offset_iqr_ms']))

    # ---- global offset search (verbatim best_offset) on a sample ----
    EXP_PK = {n: cycle_starts(CYC['pcg_seg'][n]) for n in NAMES_ALL}
    GLOBAL_OFFSET = {}
    for arm, pk_of in [('syn_fixed', lambda n: PK_A[n]), ('syn_fusion', lambda n: cycle_starts(CYC['syn_fusion'][n]))]:
        usable = [n for n in KEEP if len(pk_of(n)) >= 4 and len(EXP_PK[n]) >= 4]
        samp = usable[::max(1, len(usable) // 300)]
        if len(samp) >= 5:
            off_ms, f1_, curve = best_offset(pk_of, samp)
            GLOBAL_OFFSET[arm] = dict(offset_ms=int(off_ms), median_f1_at_75ms=float(f1_), n_sample=len(samp))
            print('%-12s offset maximising agreement with S1 onset: %+d ms (median F1 %.1f%% at 75 ms, %d recordings); negative = peak precedes S1' % (
                arm, off_ms, 100 * f1_, len(samp)))
''')

md(r'''
## Stage 5: features

The verbatim 464-feature extractor runs at 4 kHz on the shared audio for every arm, on the common
kept set only. Work is spread over the worker pool (forked processes with single-threaded BLAS,
pinned again with `threadpoolctl` inside each worker; serial where fork is unavailable); results
are written in chunks to a resumable part file (tied to the cycles it was built from, like the
final file) and finally to one `.npz` per arm, written
atomically (temporary file, then `os.replace`) with the feature names, recording names, patient
ids, labels and a content hash of the boundaries it was built from, so a cache built from other
cycles is never reused. An unreadable cache file is removed, rebuilt and listed in the manifest.
Feature names are fixed alphabetically from arm C on the first kept recording and every arm is
aligned to that order.
''')

code('s5_features', r'''
def _pool_init():
    warnings.simplefilter('ignore', RuntimeWarning)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

_JOB_ARMS = {}          # {arm_tag: {name: cycles}} visible to forked workers
_JOB_EXTRACTOR = None   # callable(pcg, cycles, sr) -> dict

def _feature_worker(name):
    try:
        pcg = AUDIO4[name]
        return name, {tag: _JOB_EXTRACTOR(pcg, cyc_of[name], TARGET_SR) for tag, cyc_of in _JOB_ARMS.items()}, None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

def cycles_hash(cyc_of, names):
    """Content hash of what a feature matrix was built from: cycle boundaries as plain ints (independent of numpy
    scalar types, so a fresh run and a resumed run agree), or the job value itself for the window ablation."""
    h = hashlib.sha1()
    for n in names:
        v = cyc_of[n]
        if isinstance(v, list):
            h.update(json.dumps([[int(x) for k in PHASE_KEYS for x in c[k]] for c in v]).encode())
        else:
            h.update(json.dumps(jsonable(v)).encode())
        h.update(b'|')
    return h.hexdigest()[:16]

FEATURE_CACHE_HASHES = {}      # {tag: cycles_hash} for every feature matrix used in this run (written to the manifest)

def build_feature_matrices(arm_cycles, names, feature_names, part_tag, extractor=None):
    """arm_cycles: {tag: {name: cycles}}. Returns {tag: X float32 (n x d)}. Cached per tag as
    cache/features_<tag>.npz (names, feature names and cycles hash must all match); partial progress in cache/feat_part_<part_tag>.npz."""
    global _JOB_ARMS, _JOB_EXTRACTOR
    out, need, chash = {}, [], {}
    for tag in arm_cycles:
        chash[tag] = cycles_hash(arm_cycles[tag], names)
        FEATURE_CACHE_HASHES[tag] = chash[tag]

        def _ok(z, tag=tag):
            if list(z['rec_names']) == list(names) and list(z['feature_names']) == list(feature_names) and str(z['cycles_hash']) == chash[tag]:
                return z['X'].astype(np.float32)
            return None

        X = read_cache(CACHE_DIR / ('features_%s.npz' % tag), _ok)
        if X is not None:
            out[tag] = X
            print('   %-24s loaded from cache %s' % (tag, X.shape))
        else:
            need.append(tag)
    if not need:
        return out
    _JOB_ARMS = {tag: arm_cycles[tag] for tag in need}
    _JOB_EXTRACTOR = extractor or extract_all_v4_features
    part = CACHE_DIR / ('feat_part_%s.npz' % part_tag)
    part_hash = json.dumps({tag: chash[tag] for tag in need}, sort_keys=True)     # the part file is tied to the cycles it was built from
    Xs = {tag: np.zeros((len(names), len(feature_names)), np.float32) for tag in need}
    have = np.zeros(len(names), bool)
    if part.exists():
        try:
            with np.load(part, allow_pickle=False) as z:
                if (set(z.files) == set(need) | {'have', 'rec_names', 'cycles_hashes'} and list(z['rec_names']) == list(names)
                        and str(z['cycles_hashes']) == part_hash):
                    Xs = {tag: z[tag].copy() for tag in need}
                    have = z['have'].copy()
                    print('   resuming %s: %d of %d recordings already done' % (part_tag, int(have.sum()), len(names)))
                else:
                    print('   part file of %s was built from other recordings or cycles: starting again' % part_tag)
        except Exception as e:
            print('   ignoring unreadable part file (%s)' % type(e).__name__)
    todo = np.flatnonzero(~have).tolist()
    pos = {n: i for i, n in enumerate(names)}
    failed = []
    t0 = time.time()
    if todo:
        with _pool(N_PROC, _pool_init) as ex:
            bar = tqdm(total=len(todo), desc='features ' + part_tag, leave=False)
            for s in range(0, len(todo), CHUNK):
                blk = todo[s:s + CHUNK]
                for name, feats, err in ex.map(_feature_worker, [names[i] for i in blk], chunksize=4):
                    i = pos[name]
                    if err is not None:
                        failed.append((name, err))
                    else:
                        for tag in need:
                            Xs[tag][i] = [float(feats[tag].get(k, 0.0)) for k in feature_names]
                        have[i] = True
                    bar.update(1)
                tmp = str(part) + '.tmp.npz'
                np.savez(tmp, have=have, rec_names=np.array(names), cycles_hashes=np.array(part_hash), **Xs)
                os.replace(tmp, str(part))
            bar.close()
    if failed:
        raise RuntimeError('feature extraction failed for %d recordings, e.g. %s' % (len(failed), failed[:3]))
    assert have.all(), 'feature rows missing'
    for tag in need:
        X = np.nan_to_num(Xs[tag], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        assert X.shape == (len(names), len(feature_names))
        atomic_savez_compressed(CACHE_DIR / ('features_%s.npz' % tag), X=X, feature_names=np.array(feature_names),
                                rec_names=np.array(names), patient_ids=np.array([PID_OF[n] for n in names]),
                                y_murmur=np.array([LAB_OF[n]['murmur'] for n in names]),
                                y_grading=np.array([LAB_OF[n]['grading'] for n in names]),
                                y_timing=np.array([LAB_OF[n]['timing'] for n in names]),
                                y_shape=np.array([LAB_OF[n]['shape'] for n in names]),
                                cycles_hash=np.array(chash[tag]))
        out[tag] = X
        print('   %-24s built %s (cycles hash %s)' % (tag, X.shape, chash[tag]))
    if part.exists():
        os.remove(part)
    print('   %d matrices built in %.1f min' % (len(need), (time.time() - t0) / 60))
    return out

with stage('S5 features'):
    _first = KEEP[0]
    FEATURE_NAMES = sorted(extract_all_v4_features(AUDIO4[_first], CYC['pcg_seg'][_first], TARGET_SR).keys())
    assert len(FEATURE_NAMES) == 464, 'expected 464 features, got %d' % len(FEATURE_NAMES)
    print('feature names fixed from arm C on %s: %d features' % (_first, len(FEATURE_NAMES)))
    FEATS = build_feature_matrices({arm: CYC[arm] for arm in ARMS}, KEEP, FEATURE_NAMES, 'arms')
    Y = {key: np.array([LAB_OF[n][key] for n in KEEP]) for key in ['murmur', 'grading', 'timing', 'shape']}
    G = np.array([PID_OF[n] for n in KEEP])
    AGE = np.array([AGE_OF.get(PID_OF[n], 'Unknown') for n in KEEP])
    for arm in ARMS:
        assert FEATS[arm].shape == (len(KEEP), 464), arm
        assert np.all(np.isfinite(FEATS[arm])), arm
    print('kept set: %d recordings, %d patients | age bands: %s' % (len(KEEP), len(np.unique(G)), dict(collections.Counter(AGE.tolist()))))
''')

md(r'''
## Stage 6: folds

`StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=42)` grouped by patient, drawn once per
task on the common kept set from the labels and patient ids only, and saved to `folds/<task>.npz`
with the recording names and patient ids so the CNN notebook uses the identical splits. Every arm
and every ablation reads these splits; nothing draws its own. In a reduced run (`DRY_RUN` or
`LIMIT_RECORDINGS`) a task whose labelled set has fewer than `N_SPLITS` patients or a class with
fewer than two members is skipped with a message and listed in `results.json` under
`sample.skipped_tasks`; every later stage iterates the remaining tasks. A full run never skips a
task.
''')

code('s6_folds', r'''
FOLDS = {}
TASK_KEY = dict(TASKS)        # every task's label key, kept even when a task is skipped below
SKIPPED_TASKS = {}

def draw_or_load_folds(task, names_all, y_all, groups_all, fname):
    """Patient-grouped stratified folds on one labelled set (y == -1 excluded), cached in folds/<fname>.
    Returns the fold record, or None when fewer than two folds are possible."""
    mask = y_all != -1
    ym, gm, nm = y_all[mask], groups_all[mask], names_all[mask]
    fp = FOLD_DIR / fname

    def _ok(z):
        if list(z['names']) == list(nm) and np.array_equal(z['y'], ym):
            k = int(z['n_splits'])
            return dict(mask=mask, y=ym, groups=gm, names=nm, splits=[(z['train_%02d' % i], z['test_%02d' % i]) for i in range(k)], n_splits=k)
        return None

    got = read_cache(fp, _ok)
    if got is not None:
        print('   %-10s folds loaded from %s' % (task, fp.name))
        return got
    n_groups = len(np.unique(gm))
    min_class = int(min(c for c in np.bincount(ym) if c > 0)) if len(ym) else 0
    k = min(N_SPLITS, n_groups, min_class)          # the source's rule; only a reduced run ever lowers it
    if k < 2:
        return None
    if k < N_SPLITS:
        print('   %-10s %d patients, smallest class %d: using %d folds (reduced runs only)' % (task, n_groups, min_class, k))
    splits = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(ym, ym, groups=gm))
    for tr, te in splits:
        assert not set(gm[tr]) & set(gm[te]), 'patient overlap between train and test'
    fold_id = np.full(len(ym), -1)
    for i, (_, te) in enumerate(splits):
        fold_id[te] = i
    assert (fold_id >= 0).all()
    arrays = {'train_%02d' % i: tr for i, (tr, _) in enumerate(splits)}
    arrays.update({'test_%02d' % i: te for i, (_, te) in enumerate(splits)})
    atomic_savez(fp, names=nm, patient_ids=gm, y=ym, fold_id=fold_id, n_splits=np.array(k),
                 keep_names=names_all, mask_over_keep=mask, seed=np.array(SEED), **arrays)
    print('   %-10s folds drawn and saved to %s' % (task, fp.name))
    return dict(mask=mask, y=ym, groups=gm, names=nm, splits=splits, n_splits=k)

with stage('S6 folds'):
    for t_ in list(TASKS):
        y_ = Y[TASKS[t_]]
        m_ = y_ != -1
        n_pat = int(len(np.unique(G[m_]))) if m_.any() else 0
        min_class = int(min(c for c in np.bincount(y_[m_]) if c > 0)) if m_.any() else 0
        if not VALIDATION_APPLICABLE and (n_pat < N_SPLITS or min_class < 2):
            SKIPPED_TASKS[t_] = 'reduced run: %d labelled patients (need %d) and smallest class %d (need 2)' % (n_pat, N_SPLITS, min_class)
            print('   %-10s SKIPPED (%s)' % (t_, SKIPPED_TASKS[t_]))
            del TASKS[t_]
            continue
        f_ = draw_or_load_folds(t_, KEEP_ARR, y_, G, t_ + '.npz')
        assert f_ is not None, 'not enough patients or classes for grouped CV on %s' % t_
        FOLDS[t_] = f_
        sizes = [len(te) for _, te in f_['splits']]
        print('      %d labelled, %d patients, classes %s, %d folds, test sizes %d-%d' % (
            len(f_['y']), len(np.unique(f_['groups'])), dict(collections.Counter(f_['y'].tolist())), f_['n_splits'], min(sizes), max(sizes)))
    assert 'detection' in TASKS, 'the detection task cannot be skipped'
    if SKIPPED_TASKS:
        print('   tasks in this run: %s | skipped: %s' % (list(TASKS), SKIPPED_TASKS))
''')

md(r'''
## Stage 7: classifiers and evaluation

Per arm and task: `StandardScaler` fitted on the training fold only. The MLP is the verbatim
`LightweightMLP` (464-128-64-C, batch-norm, ReLU, dropout 0.4 / 0.3) trained with AdamW
(lr 1e-3, weight decay 1e-4), mini-batches of 32 drawn from the fold held in GPU memory,
cross-entropy with label smoothing 0.1 and inverse-frequency class weights, gradient-norm clipping
at 1.0, up to 250 epochs, early stopping with patience 25 epochs on the macro-F1 of an inner
patient-disjoint `GroupShuffleSplit(test_size=0.15, random_state=42)` validation split, best state
restored, torch seeded per fold. Two details differ from the thesis V5 training loop and are stated
here so nobody reads them as V5: V5 balanced the classes with an inverse-frequency
`WeightedRandomSampler`, this notebook weights the loss instead; V5 stopped early on inner
accuracy, this notebook stops on inner macro-F1. The per-class counts of every inner validation
split are recorded in `fit_info`. The random forest is 300 trees, depth 15, balanced class weights.
Pooled out-of-fold labels and class probabilities are stored for both. A second recipe,
`fit_mlp_reference` (focal loss gamma 2 with class weights, full-batch steps, 300 epochs,
validation every 5 epochs, patience 5 checks, constant seed), exists only for the validation gate,
which must compare like with like against the mechanism study's numbers.

Metrics on the pooled out-of-fold predictions: accuracy, macro-F1, balanced accuracy, per-class
sensitivity and specificity, AUROC (binary, or one-vs-rest macro: each class scored against the
other two and the three areas averaged), AUPRC (one-vs-rest macro with label binarisation), the
confusion matrix and the majority-class baseline; per fold, accuracy and macro-F1. Uncertainty
comes from a patient-level bootstrap with 2000 resamples: patient draws are made once per task and
turned into a recording-weight matrix, so every confusion-matrix metric is computed for all
replicates in one float64 matrix product, and the same draws score both arms of every paired
difference. A resample whose truth holds a single class is masked (the verbatim oracle's rule) and
`n_boot_valid` records how many remain; AUROC and AUPRC are bootstrapped with a vectorised
weighted-rank computation that is asserted equal to sklearn before it is used. The vectorised
bootstrap is checked against the verbatim `boot_ci` and `boot_diff_ci` oracles on the detection
task and on one three-class task before it is trusted. Fold-wise paired t-tests are reported
alongside. The paired differences cover every arm against the two equal-time controls, B and C
against A, C against B, and two pairs that separate the sources of loss: C minus D (annotated
phase boundaries against fixed RR fractions on the same annotated cycle starts, the cost of the
fixed phase model) and A minus D (synthetic against annotated anchors under the same fixed
fractions, the cost of anchor timing). Every cached CV result stores the labels, a hash of the feature matrix, a hash of the
classifier configuration and the recipe, and is rebuilt when any of them differs.
''')

code('s7_classifier_defs', V_MLP + '\n\n' + r'''
MLP_CFG = dict(lr=1e-3, weight_decay=1e-4, batch=32, max_epochs=250, patience=25, label_smoothing=0.1,
               inner_val=0.15, clip_grad_norm=1.0)
RF_CFG = dict(n_estimators=300, max_depth=15, class_weight='balanced', random_state=SEED, n_jobs=-1)
REFERENCE_MLP_CFG = dict(model='ReferenceMLP, default init (Mechanism cell 30)', loss='focal gamma 2.0 with inverse-frequency class weights',
                         batch='full', max_epochs=300, val_every=5, patience_checks=5, seed='constant SEED', inner_val=0.15,
                         tf32='on during the gate fits, as in the reference run')
CFG_HASH = hashlib.sha1(json.dumps(dict(mlp=MLP_CFG, rf=RF_CFG, reference=REFERENCE_MLP_CFG, seed=SEED, n_splits=N_SPLITS),
                                   sort_keys=True, default=str).encode()).hexdigest()[:16]

def fit_mlp(Xtr, ytr, gtr, ncls, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    info = {'inner_split': 'GroupShuffleSplit', 'recipe': 'primary'}
    if len(np.unique(gtr)) >= 3:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=MLP_CFG['inner_val'], random_state=SEED).split(Xtr, ytr, groups=gtr))
    else:
        tr, va = np.arange(len(ytr)), np.arange(len(ytr))       # reduced runs only; recorded
        info['inner_split'] = 'degenerate: fewer than 3 patients in the training fold'
    Xt = torch.tensor(Xtr[tr], dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr[tr], dtype=torch.long, device=DEV)
    Xv = torch.tensor(Xtr[va], dtype=torch.float32, device=DEV)
    yv = ytr[va]
    cnt = np.bincount(ytr[tr], minlength=ncls).astype(float)
    w = torch.tensor(cnt.sum() / (ncls * np.maximum(cnt, 1.0)), dtype=torch.float32, device=DEV)
    model = LightweightMLP(Xtr.shape[1], ncls).to(DEV)
    opt = AdamW(model.parameters(), lr=MLP_CFG['lr'], weight_decay=MLP_CFG['weight_decay'])
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=MLP_CFG['label_smoothing'])
    g = torch.Generator().manual_seed(seed)
    n, bs = len(tr), MLP_CFG['batch']
    n_steps = max(1, n // bs) if n > bs else 1
    labs = list(range(ncls))
    best, state, bad, best_ep, ep = -1.0, None, 0, -1, -1
    for ep in range(MLP_CFG['max_epochs']):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for s in range(n_steps):
            idx = perm[s * bs:(s + 1) * bs] if n > bs else perm
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MLP_CFG['clip_grad_norm'])
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        f = f1_score(yv, pv, average='macro', labels=labs, zero_division=0)
        if f > best:
            best, bad, best_ep = f, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_CFG['patience']:
                break
    model.load_state_dict(state)
    model.eval()
    info.update(best_epoch=int(best_ep), epochs_run=int(ep + 1), best_val_macro_f1=float(best), n_inner_train=int(n), n_inner_val=int(len(va)),
                inner_val_class_counts=np.bincount(yv, minlength=ncls).tolist())
    return model, info

class FocalLoss(nn.Module):                      # Mechanism_Rhythm_Integrity cell 30, verbatim semantics
    def __init__(self, g=2.0, w=None):
        super().__init__()
        self.g, self.w = g, w

    def forward(self, lo, y):
        ce = F.cross_entropy(lo, y, weight=self.w, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.g * ce).mean()

class ReferenceMLP(nn.Module):       # Mechanism_Rhythm_Integrity cell 30 as written there: PyTorch default initialisation, no Kaiming step
    def __init__(self, d, c, hidden=(128, 64), drops=(0.4, 0.3)):
        super().__init__()
        L, prev = [], d
        for h, p in zip(hidden, drops):
            L += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        L.append(nn.Linear(prev, c))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)

def fit_mlp_reference(Xtr, ytr, gtr, ncls, seed=SEED):
    """The recipe that produced the gate's reference values: the reference study's own model class (default
    initialisation, not the V5 class with its Kaiming step), focal(2.0), full batch, 300 epochs, validation every
    5 epochs, patience 5 checks, constant seed."""
    torch.manual_seed(seed)
    labs = list(range(ncls))
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, groups=gtr))
    Xt = torch.tensor(Xtr[tr], dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr[tr], dtype=torch.long, device=DEV)
    Xv = torch.tensor(Xtr[va], dtype=torch.float32, device=DEV)
    cnt = np.bincount(ytr[tr], minlength=ncls).astype(float)
    w = torch.tensor(cnt.sum() / (ncls * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    m = ReferenceMLP(Xtr.shape[1], ncls).to(DEV)
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = FocalLoss(2.0, w)
    best, state, bad = -1.0, None, 0
    for ep in range(300):
        m.train()
        opt.zero_grad()
        crit(m(Xt), yt).backward()
        opt.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad():
                pv = m(Xv).argmax(1).cpu().numpy()
            f = f1_score(ytr[va], pv, average='macro', labels=labs, zero_division=0)
            if f > best:
                best, bad, state = f, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 5:
                    break
    if state:
        m.load_state_dict(state)
    return m.eval(), dict(inner_split='GroupShuffleSplit', recipe='reference-focal-fullbatch', best_val_macro_f1=float(best),
                          inner_val_class_counts=np.bincount(ytr[va], minlength=ncls).tolist())

@torch.no_grad()
def mlp_proba(model, X):
    return torch.softmax(model(torch.tensor(X, dtype=torch.float32, device=DEV)), 1).cpu().numpy()

def run_cv(X, task, tag, scaler='fold', do_mlp=True, do_rf=True, seed=SEED, recipe='primary', folds=None):
    """Patient-grouped CV on the saved folds (or on `folds`). Cached under cache/cv/<tag>__<task>.npz; the cache is reused
    only when the recording names, labels, feature-matrix hash, configuration hash, recipe and flags all match."""
    Fd = folds if folds is not None else FOLDS[task]
    Xm, ym, gm, nm = X[Fd['mask']], Fd['y'], Fd['groups'], Fd['names']
    ncls = NCLS[task]
    n = len(ym)
    fp = CV_DIR / ('%s__%s.npz' % (tag, task))
    x_hash = hashlib.sha1(np.ascontiguousarray(Xm, dtype=np.float32).tobytes()).hexdigest()[:16]

    def _ok(z):
        if (list(z['names']) == list(nm) and np.array_equal(z['y'], ym) and str(z['cfg_hash']) == CFG_HASH and str(z['x_hash']) == x_hash
                and str(z['recipe']) == recipe and bool(z['do_mlp']) == do_mlp and bool(z['do_rf']) == do_rf and str(z['scaler']) == scaler):
            res = {k: z[k] for k in z.files if k not in ('names', 'y', 'do_mlp', 'do_rf', 'scaler', 'fit_info', 'cfg_hash', 'x_hash', 'recipe')}
            res['fit_info'] = json.loads(str(z['fit_info']))
            res['names'], res['y'], res['groups'] = nm, ym, gm
            return res
        return None

    got = read_cache(fp, _ok)
    if got is not None:
        return got
    t0 = time.time()
    res = dict(fold_id=np.full(n, -1), y=ym, groups=gm, names=nm,
               pred_mlp=np.full(n, -1), proba_mlp=np.zeros((n, ncls), np.float32), fold_f1_mlp=[], fold_acc_mlp=[],
               pred_rf=np.full(n, -1), proba_rf=np.zeros((n, ncls), np.float32), fold_f1_rf=[], fold_acc_rf=[], fit_info=[])
    sc_full = StandardScaler().fit(Xm) if scaler == 'full' else None
    labs = list(range(ncls))
    for k, (tr, te) in enumerate(Fd['splits']):
        res['fold_id'][te] = k
        if scaler == 'fold':
            sc = StandardScaler().fit(Xm[tr])
            Xtr, Xte = sc.transform(Xm[tr]), sc.transform(Xm[te])
        else:
            Xtr, Xte = sc_full.transform(Xm[tr]), sc_full.transform(Xm[te])
        if do_mlp:
            if recipe == 'primary':
                mdl, info = fit_mlp(Xtr, ym[tr], gm[tr], ncls, seed + k)
            else:
                mdl, info = fit_mlp_reference(Xtr, ym[tr], gm[tr], ncls, seed)   # constant seed, as the reference run
            pr = mlp_proba(mdl, Xte)
            res['proba_mlp'][te] = pr
            res['pred_mlp'][te] = pr.argmax(1)
            res['fold_f1_mlp'].append(f1_score(ym[te], pr.argmax(1), average='macro', labels=labs, zero_division=0))
            res['fold_acc_mlp'].append(accuracy_score(ym[te], pr.argmax(1)))
            res['fit_info'].append(info)
            del mdl
        if do_rf:
            rf = RandomForestClassifier(**RF_CFG).fit(Xtr, ym[tr])
            pr = np.zeros((len(te), ncls), np.float32)
            pr[:, rf.classes_] = rf.predict_proba(Xte)
            res['proba_rf'][te] = pr
            res['pred_rf'][te] = pr.argmax(1)
            res['fold_f1_rf'].append(f1_score(ym[te], pr.argmax(1), average='macro', labels=labs, zero_division=0))
            res['fold_acc_rf'].append(accuracy_score(ym[te], pr.argmax(1)))
            del rf
    assert (res['fold_id'] >= 0).all()
    for k_ in ('fold_f1_mlp', 'fold_acc_mlp', 'fold_f1_rf', 'fold_acc_rf'):
        res[k_] = np.array(res[k_], float)
    res['seconds'] = np.array(time.time() - t0)
    save = {k: v for k, v in res.items() if k not in ('y', 'groups', 'names', 'fit_info')}
    atomic_savez(fp, names=nm, y=ym, do_mlp=np.array(do_mlp), do_rf=np.array(do_rf), scaler=np.array(scaler), recipe=np.array(recipe),
                 cfg_hash=np.array(CFG_HASH), x_hash=np.array(x_hash), fit_info=np.array(json.dumps(res['fit_info'])), **save)
    return res

# ------------------------------- metrics and bootstrap -------------------------------
def cm_metrics(M):
    """M: (..., C, C) confusion counts, rows = truth. Returns arrays over the leading axes."""
    M = np.asarray(M, np.float64)
    tp = np.diagonal(M, axis1=-2, axis2=-1)
    row, col, tot = M.sum(-1), M.sum(-2), M.sum((-1, -2))
    fp_, fn_ = col - tp, row - tp
    tn = tot[..., None] - row - col + tp
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp_ > 0, tp / np.maximum(tp + fp_, 1e-12), 0.0)
        rec = np.where(row > 0, tp / np.maximum(row, 1e-12), 0.0)
        spec = np.where(tn + fp_ > 0, tn / np.maximum(tn + fp_, 1e-12), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        acc = tp.sum(-1) / np.maximum(tot, 1e-12)
        sup = row > 0
        bacc = (rec * sup).sum(-1) / np.maximum(sup.sum(-1), 1)
    return dict(acc=acc, macro_f1=f1.mean(-1), bacc=bacc, recall=rec, specificity=spec)

def confusion(y, pred, ncls):
    return confusion_matrix(y, pred, labels=list(range(ncls))).astype(float)

def score_metrics(y, proba, ncls, w=None):
    """(AUROC, AUPRC, note). OvR macro for three classes; None when undefined."""
    P = proba.astype(np.float64)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    present = np.unique(y if w is None else y[w > 0])
    if len(present) < 2:
        return np.nan, np.nan, 'fewer than two classes present'
    if ncls == 2:
        return (roc_auc_score(y, P[:, 1], sample_weight=w), average_precision_score(y, P[:, 1], sample_weight=w), '')
    if len(present) < ncls:
        return np.nan, np.nan, 'a class is absent, OvR macro undefined'
    Yb = label_binarize(y, classes=list(range(ncls)))
    return (roc_auc_score(y, P, multi_class='ovr', average='macro', sample_weight=w),
            average_precision_score(Yb, P, average='macro', sample_weight=w), '')

def _boot_binary_scores(yb, s, W):
    """Weighted AUROC and average precision for every bootstrap replicate at once. yb: (n,) 0/1 truth, s: (n,) scores,
    W: (B, n) resample multiplicities. Same distinct-threshold set and tie handling as sklearn's roc_curve and
    precision_recall_curve; NaN where a replicate lacks positives or negatives."""
    order = np.argsort(-s, kind='mergesort')
    s_, y_ = s[order], yb[order].astype(np.float64)
    Wo = W[:, order].astype(np.float64)
    tp = np.cumsum(Wo * y_, 1)
    fp_ = np.cumsum(Wo * (1.0 - y_), 1)
    last = np.r_[np.flatnonzero(np.diff(s_) != 0), len(s_) - 1]          # last index of every distinct score
    tp_t, fp_t = tp[:, last], fp_[:, last]
    P, N = tp[:, -1:], fp_[:, -1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr, fpr = tp_t / P, fp_t / N
        prec = np.where(tp_t + fp_t > 0, tp_t / np.maximum(tp_t + fp_t, 1e-300), 0.0)
        z = np.zeros((len(W), 1))
        tpr0, fpr0 = np.concatenate([z, tpr[:, :-1]], 1), np.concatenate([z, fpr[:, :-1]], 1)
        auroc = ((fpr - fpr0) * (tpr + tpr0) / 2.0).sum(1)                   # trapezoid over the ROC = weighted rank statistic
        ap = ((tpr - tpr0) * prec).sum(1)                                     # sum of (R_k - R_k-1) P_k, as average_precision_score
    ok = (P[:, 0] > 0) & (N[:, 0] > 0)
    return np.where(ok, auroc, np.nan), np.where(ok, ap, np.nan)

def boot_score_metrics(y, proba, ncls, W):
    """(AUROC, AUPRC) per replicate: binary, or the one-vs-rest macro average over classes (sklearn's 'ovr', 'macro').
    NaN where a class has no weight in a replicate, where sklearn would refuse to score."""
    P = proba.astype(np.float64)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    if ncls == 2:
        return _boot_binary_scores((y == 1).astype(int), P[:, 1], W)
    per = [_boot_binary_scores((y == c).astype(int), P[:, c], W) for c in range(ncls)]
    return np.stack([p[0] for p in per], 1).mean(1), np.stack([p[1] for p in per], 1).mean(1)

_W_CACHE = {}
def patient_weights(groups, n_boot, seed=SEED):
    """Vectorised patient-level resampling: (n_boot x n) multiplicity of every recording.
    Draws exactly what the verbatim boot_ci draws: rng.choice(pats, size=len(pats), replace=True)."""
    key = (hashlib.sha1(np.asarray(groups).tobytes()).hexdigest(), n_boot, seed)
    if key in _W_CACHE:
        return _W_CACHE[key]
    rng = np.random.default_rng(seed)
    pats, inv = np.unique(groups, return_inverse=True)
    W = np.zeros((n_boot, len(groups)), np.int32)
    for b in range(n_boot):
        take = rng.choice(pats, size=len(pats), replace=True)
        W[b] = np.bincount(np.searchsorted(pats, take), minlength=len(pats))[inv]
    _W_CACHE[key] = W
    return W

def boot_confusions(y, pred, W, ncls):
    onehot = np.zeros((len(y), ncls * ncls), np.float64)
    onehot[np.arange(len(y)), y * ncls + pred] = 1.0
    return (W.astype(np.float64) @ onehot).reshape(len(W), ncls, ncls)

def ci95(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return [None, None]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def pct(ci):
    return [None if v is None else 100 * v for v in ci]

def ci_str(ci, fmt='%5.1f'):
    """'[lo, hi]' for printing, or an open marker when a bound is undefined (fewer than 10 valid bootstrap draws)."""
    if not ci or ci[0] is None or ci[1] is None:
        return '[  n/a  ]'
    return '[' + (fmt % ci[0]) + ', ' + (fmt % ci[1]) + ']'

def evaluate_predictions(y, pred, proba, groups, ncls, fold_f1=None, fold_acc=None, with_scores=True, n_boot=None):
    """Pooled metrics with patient-bootstrap CIs; percentages for rates, [0,1] for AUROC/AUPRC. Resamples whose truth
    holds one class are masked, as the verbatim oracle skips them; n_boot_valid counts the rest."""
    n_boot = n_boot or N_BOOT
    M0 = confusion(y, pred, ncls)
    m0 = cm_metrics(M0)
    assert abs(m0['macro_f1'] - f1_score(y, pred, average='macro', labels=list(range(ncls)), zero_division=0)) < 1e-9
    assert abs(m0['acc'] - accuracy_score(y, pred)) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # sklearn warns when y_pred holds a class absent from y_true
        assert abs(m0['bacc'] - balanced_accuracy_score(y, pred)) < 1e-9
    W = patient_weights(groups, n_boot)
    Mb = boot_confusions(y, pred, W, ncls)
    valid = (Mb.sum(2) > 0).sum(1) >= 2          # the verbatim oracle's rule
    mb = cm_metrics(Mb)
    for k_ in ('acc', 'macro_f1', 'bacc'):
        mb[k_] = np.where(valid, mb[k_], np.nan)
    mb['recall'] = np.where(valid[:, None], mb['recall'], np.nan)
    mb['specificity'] = np.where(valid[:, None], mb['specificity'], np.nan)
    out = dict(n=int(len(y)), n_patients=int(len(np.unique(groups))), n_boot=int(n_boot), n_boot_valid=int(valid.sum()),
               accuracy=100 * float(m0['acc']), accuracy_ci=pct(ci95(mb['acc'])),
               macro_f1=100 * float(m0['macro_f1']), macro_f1_ci=pct(ci95(mb['macro_f1'])),
               balanced_accuracy=100 * float(m0['bacc']), balanced_accuracy_ci=pct(ci95(mb['bacc'])),
               sensitivity=[100 * float(v) for v in m0['recall']], sensitivity_ci=[pct(ci95(mb['recall'][:, c])) for c in range(ncls)],
               specificity=[100 * float(v) for v in m0['specificity']], specificity_ci=[pct(ci95(mb['specificity'][:, c])) for c in range(ncls)],
               confusion_matrix=M0.astype(int).tolist(), boot_macro_f1=mb['macro_f1'], boot_acc=mb['acc'])
    if with_scores and proba is not None:
        au, ap, note = score_metrics(y, proba, ncls)
        out.update(auroc=float(au), auprc=float(ap), score_note=note)
        if note == '':
            bau, bap = boot_score_metrics(y, proba, ncls, W)
            out.update(auroc_ci=ci95(bau), auprc_ci=ci95(bap), n_boot_valid_scores=int(np.isfinite(bau).sum()))
        else:
            out.update(auroc_ci=[None, None], auprc_ci=[None, None], n_boot_valid_scores=0)
    if fold_f1 is not None and len(fold_f1):
        out.update(fold_macro_f1=[100 * float(v) for v in fold_f1], fold_macro_f1_mean=100 * float(np.mean(fold_f1)), fold_macro_f1_sd=100 * float(np.std(fold_f1)),
                   fold_accuracy=[100 * float(v) for v in fold_acc], fold_accuracy_mean=100 * float(np.mean(fold_acc)), fold_accuracy_sd=100 * float(np.std(fold_acc)))
    return out

def majority_baseline(y, ncls):
    cnt = np.bincount(y, minlength=ncls)
    top = int(np.argmax(cnt))
    pred = np.full(len(y), top)
    return dict(majority_class=top, accuracy=100 * float(accuracy_score(y, pred)),
                macro_f1=100 * float(f1_score(y, pred, average='macro', labels=list(range(ncls)), zero_division=0)),
                class_counts=cnt.tolist())

def paired_difference(ea, eb, fold_a=None, fold_b=None):
    """ea, eb: evaluate_predictions outputs sharing the same bootstrap draws. P(>0) and the CI use the finite draws only."""
    out = {}
    for m, key in [('macro_f1', 'boot_macro_f1'), ('accuracy', 'boot_acc')]:
        d = 100 * (ea[key] - eb[key])
        fin = np.isfinite(d)
        out[m] = dict(point=float(ea[m] - eb[m]), boot_mean=float(np.mean(d[fin])) if fin.any() else None, ci=ci95(d),
                      p_gt0=float(np.mean(d[fin] > 0)) if fin.any() else None, n_boot_valid=int(fin.sum()))
    if fold_a is not None and fold_b is not None and len(fold_a) == len(fold_b) and len(fold_a) >= 3:
        a, b = 100 * np.asarray(fold_a), 100 * np.asarray(fold_b)
        if np.allclose(a, b):
            out['fold_ttest'] = dict(t=None, p=None, n_folds=int(len(a)), mean_diff=0.0, note='identical fold scores')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t_, p_ = ttest_rel(a, b)
            out['fold_ttest'] = dict(t=float(t_), p=float(p_), n_folds=int(len(a)), mean_diff=float(np.mean(a - b)))
    return out

def strip_boot(e):
    return {k: v for k, v in e.items() if not k.startswith('boot_')}

print('classifier and evaluation machinery defined | MLP', MLP_CFG, '| RF', RF_CFG, '| configuration hash', CFG_HASH)
''')

code('s7_run', r'''
# C-D: annotated phase boundaries against fixed RR fractions on the same annotated cycle starts (the cost of the fixed phase
# model, R3-M4). A-D: synthetic anchors against annotated anchors under the same fixed fractions (the cost of anchor timing).
PAIRS = [('syn_fixed', 'equal_time'), ('syn_fusion', 'equal_time'), ('pcg_seg', 'equal_time'), ('pcg_seg_fixed', 'equal_time'),
         ('syn_fusion', 'syn_fixed'), ('pcg_seg', 'syn_fixed'), ('pcg_seg', 'syn_fusion'),
         ('pcg_seg', 'pcg_seg_fixed'), ('syn_fixed', 'pcg_seg_fixed'),
         ('syn_fixed', 'equal_time_matched'), ('syn_fusion', 'equal_time_matched'), ('pcg_seg', 'equal_time_matched'),
         ('equal_time_matched', 'equal_time')]

SCHEMA = {
    'run': 'flags, paths, seeds, sample sizes, tasks run and equal-time definitions of this run',
    'sample': 'stage-1 counts, kept set (names), drop reasons, tasks skipped in a reduced run, per-task n / patients / class counts',
    'diagnostics_summary': 'per arm: medians over kept recordings of the rate-fidelity diagnostics, Pearson r of median RR vs annotation, offset SD (signed, matched pairs) and IQR of per-recording median |offset|, detector paths (A), fusion statistics (B), annotation gaps (D), annotated coverage (E2). Sign convention: median_offset_ms = arm cycle start minus annotated S1 onset (negative = arm precedes S1); arm_mean_syn_ac_offset_ms = acoustic S1 onset minus synthetic R-peak (positive = acoustic onset follows R)',
    'global_offset_ms': 'offset maximising S1-onset agreement for arms A and B (verbatim best_offset); negative = peak precedes S1',
    'arms.<task>.<arm>': 'n, n_patients, majority baseline, and for mlp / rf: accuracy, macro_f1, balanced_accuracy, sensitivity and specificity per class (all in %), auroc and auprc (0-1), *_ci = 95% patient-bootstrap percentile interval over the valid resamples (n_boot_valid; [null, null] when fewer than 10), confusion_matrix (rows = truth), fold_* = per-fold scores, fit_info = early-stopping record per fold',
    'paired.<task>.<X-Y>.<clf>': 'pairs A-E, B-E, C-E, D-E, B-A, C-A, C-B, C-D (annotated phases against fixed fractions on the same cycle starts), A-D (synthetic against annotated anchors under the same fixed fractions), A-E2, B-E2, C-E2, E2-E; X minus Y on macro_f1 and accuracy: point estimate, bootstrap mean, 95% CI, P(diff>0) over the finite paired draws (n_boot_valid) from the same patient resamples; fold_ttest = scipy ttest_rel over the per-fold scores',
    'validation_gate': 'sample check (per-task n and patients against the reference run) and arm C / arm E macro-F1 under the reference recipe (ReferenceMLP, TF32 on) against the mechanism-study values; a cell is within when |delta| <= GATE_TOL or the reference value lies inside the 95% patient-bootstrap interval of the gate value (ci, within per cell); refset = the same on the reference-rule set when the kept set differs',
    'ablations': 'window_length (R1-M2), timing_budget (R1-M1), redundancy (R1-m3), scaling (R3-m3), subgroups (R3-M4)',
    'stage_seconds': 'wall-clock seconds per stage in this session',
    'units': 'rates in percent (0-100); auroc / auprc in [0,1]; offsets in ms; RR in ms',
}
RESULTS = {'schema': SCHEMA,
           'run': dict(run_dir=str(RUN_DIR), started=datetime.datetime.now().isoformat(timespec='seconds'), dry_run=DRY_RUN,
                       limit_recordings=LIMIT_RECORDINGS, n_boot=N_BOOT, seed=SEED, n_splits=N_SPLITS, gate_tol=GATE_TOL, n_min_cycles=N_MIN_CYCLES,
                       run_flags=RUN_FLAGS, dataset=str(AUDIO_PATH), n_pairs=N_PAIRS, checkpoint=str(GAN_CHECKPOINT), checkpoint_sha256=CKPT_SHA256,
                       gpu=GPU_NAME, pool=POOL_KIND, mlp_cfg=MLP_CFG, rf_cfg=RF_CFG, reference_mlp_cfg=REFERENCE_MLP_CFG, fusion_cfg=FUSION,
                       match_ms=MATCH_MS, join_ms=JOIN_MS, realign_ms=REALIGN_MS, cfg_hash=CFG_HASH, arm_cfg_hash=ARM_CFG_HASH,
                       equal_time_definitions=EQUAL_TIME_DEFINITIONS, arms=ARMS, arm_letters=ARM_LETTER, arm_labels=ARM_LABEL, tasks=list(TASKS)),
           'sample': dict(n_indexed=len(NAMES_ALL), n_kept=len(KEEP), n_dropped=len(DROPPED), kept_names=list(KEEP),
                          dropped={n: bad for n, bad in DROPPED}, excluded_stage1=EXCLUDED, skipped_tasks=SKIPPED_TASKS,
                          per_task={t: dict(n=int(len(FOLDS[t]['y'])), n_patients=int(len(np.unique(FOLDS[t]['groups']))),
                                            n_splits=FOLDS[t]['n_splits'], class_counts=np.bincount(FOLDS[t]['y'], minlength=NCLS[t]).tolist()) for t in TASKS}),
           'diagnostics_summary': DIAG_SUMMARY, 'global_offset_ms': GLOBAL_OFFSET,
           'arms': {}, 'paired': {}, 'validation_gate': None, 'ablations': {}}

def write_results():
    RESULTS['stage_seconds'] = dict(STAGE_SECONDS)
    tmp = str(RUN_DIR / 'results.json') + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(jsonable(RESULTS), fh, indent=1)
    os.replace(tmp, str(RUN_DIR / 'results.json'))

OOF = {}
def write_oof():
    """Pooled out-of-fold predictions of every arm and every ablation seen so far; rewritten after each stage that adds some."""
    tmp = str(RUN_DIR / 'oof_predictions.npz') + '.tmp.npz'
    np.savez_compressed(tmp, arms=np.array(ARMS), tasks=np.array(list(TASKS)), **OOF)
    os.replace(tmp, str(RUN_DIR / 'oof_predictions.npz'))

def add_oof(task, tag, r, clfs=('mlp', 'rf')):
    for clf in clfs:
        OOF['%s__%s__%s__pred' % (task, tag, clf)] = r['pred_' + clf]
        OOF['%s__%s__%s__proba' % (task, tag, clf)] = r['proba_' + clf]

CV, EVAL = {}, {}
with stage('S7 classifiers and evaluation'):
    for task in TASKS:
        CV[task], EVAL[task] = {}, {}
        RESULTS['arms'].setdefault(task, {})
        for arm in ARMS:
            CV[task][arm] = run_cv(FEATS[arm], task, arm)
            print('   %-10s %-20s MLP macro-F1 %5.1f acc %5.1f | RF macro-F1 %5.1f acc %5.1f | %.0f s' % (
                task, arm, 100 * f1_score(CV[task][arm]['y'], CV[task][arm]['pred_mlp'], average='macro', labels=list(range(NCLS[task])), zero_division=0),
                100 * accuracy_score(CV[task][arm]['y'], CV[task][arm]['pred_mlp']),
                100 * f1_score(CV[task][arm]['y'], CV[task][arm]['pred_rf'], average='macro', labels=list(range(NCLS[task])), zero_division=0),
                100 * accuracy_score(CV[task][arm]['y'], CV[task][arm]['pred_rf']), float(CV[task][arm]['seconds'])))

    # ---- bootstrap oracle check: vectorised == verbatim boot_ci / boot_diff_ci (same draws), detection and one 3-class task ----
    def _oracle_check(task):
        r_, e_ = CV[task]['pcg_seg'], CV[task]['equal_time']
        ncls = NCLS[task]
        v0, lo0, hi0 = boot_ci(r_['y'], r_['pred_mlp'], r_['groups'], metric='f1', n_boot=N_BOOT, seed=SEED)
        d0, dlo0, dhi0 = boot_diff_ci(r_['y'], r_['pred_mlp'], e_['pred_mlp'], r_['groups'], metric='f1', n_boot=N_BOOT, seed=SEED)
        ea = evaluate_predictions(r_['y'], r_['pred_mlp'], None, r_['groups'], ncls, with_scores=False)
        eb = evaluate_predictions(e_['y'], e_['pred_mlp'], None, e_['groups'], ncls, with_scores=False)
        lo1, hi1 = ea['macro_f1_ci']
        dlo1, dhi1 = paired_difference(ea, eb)['macro_f1']['ci']
        Mb_ = boot_confusions(r_['y'], r_['pred_mlp'], patient_weights(r_['groups'], N_BOOT), ncls)
        valid_ = (Mb_.sum(2) > 0).sum(1) >= 2
        # the verbatim oracle averages macro-F1 over the classes present in a resample's truth or prediction, the vectorised path
        # over every class; the two agree exactly when every valid resample shows all classes (always true for detection)
        all_present = bool(np.all(((Mb_.sum(2) > 0) | (Mb_.sum(1) > 0))[valid_].all(1)))
        print('   %-10s arm C: verbatim boot_ci [%.3f, %.3f] vs vectorised [%.3f, %.3f] | C - E: verbatim boot_diff_ci [%.3f, %.3f] vs vectorised [%.3f, %.3f] | %d of %d one-class resamples masked' % (
            task, lo0, hi0, lo1, hi1, dlo0, dhi0, dlo1, dhi1, int((~valid_).sum()), N_BOOT))
        if all_present:
            assert abs(lo0 - lo1) < 1e-6 and abs(hi0 - hi1) < 1e-6 and abs(dlo0 - dlo1) < 1e-6 and abs(dhi0 - dhi1) < 1e-6, \
                'vectorised bootstrap disagrees with the verbatim oracle on %s' % task
            print('   %-10s vectorised bootstrap reproduces the verbatim oracle exactly' % task)
        else:
            assert not VALIDATION_APPLICABLE, 'a full run must be comparable to the oracle on %s' % task
            print('   %-10s not compared: some valid resamples lack a class entirely, where the oracle averages over fewer classes (reduced runs only)' % task)

    _oracle_check('detection')
    _three = [t for t in TASKS if NCLS[t] == 3]
    if _three:
        _oracle_check(_three[0])

    # ---- vectorised AUROC / AUPRC bootstrap == sklearn with sample weights (point estimate and three replicates) ----
    for task in ['detection'] + _three[:1]:
        r_ = CV[task]['pcg_seg']
        ncls = NCLS[task]
        W_ = patient_weights(r_['groups'], N_BOOT)
        Wchk = np.vstack([np.ones((1, len(r_['y'])), np.int32), W_[:3]])
        au_, ap_ = boot_score_metrics(r_['y'], r_['proba_mlp'], ncls, Wchk)
        for b in range(len(Wchk)):
            sa, sp, note = score_metrics(r_['y'], r_['proba_mlp'], ncls, w=None if b == 0 else Wchk[b].astype(float))
            if note == '':
                assert abs(sa - au_[b]) < 1e-9 and abs(sp - ap_[b]) < 1e-9, ('vectorised AUROC / AUPRC disagree with sklearn', task, b, sa, au_[b], sp, ap_[b])
            else:
                assert not (np.isfinite(au_[b]) or np.isfinite(ap_[b])), ('vectorised score defined where sklearn refuses', task, b)
        print('   %-10s vectorised AUROC / AUPRC equal sklearn (weighted) on the point estimate and three replicates' % task)

    # ---- pooled metrics with CIs ----
    for task in TASKS:
        ncls = NCLS[task]
        for arm in ARMS:
            r = CV[task][arm]
            EVAL[task][arm] = {}
            for clf in ('mlp', 'rf'):
                EVAL[task][arm][clf] = evaluate_predictions(r['y'], r['pred_' + clf], r['proba_' + clf], r['groups'], ncls,
                                                            r['fold_f1_' + clf], r['fold_acc_' + clf])
            RESULTS['arms'][task][arm] = dict(n=int(len(r['y'])), n_patients=int(len(np.unique(r['groups']))),
                                              majority=majority_baseline(r['y'], ncls),
                                              mlp=strip_boot(EVAL[task][arm]['mlp']), rf=strip_boot(EVAL[task][arm]['rf']),
                                              fit_info=r['fit_info'], cv_seconds=float(r['seconds']))
        RESULTS['paired'][task] = {}
        for a, b in PAIRS:
            ra, rb = CV[task][a], CV[task][b]
            assert np.array_equal(ra['y'], rb['y']) and np.array_equal(ra['groups'], rb['groups'])
            RESULTS['paired'][task]['%s-%s' % (ARM_LETTER[a], ARM_LETTER[b])] = {
                'arms': [a, b],
                'mlp': paired_difference(EVAL[task][a]['mlp'], EVAL[task][b]['mlp'], ra['fold_f1_mlp'], rb['fold_f1_mlp']),
                'rf': paired_difference(EVAL[task][a]['rf'], EVAL[task][b]['rf'], ra['fold_f1_rf'], rb['fold_f1_rf'])}
    write_results()

    # ---- pooled out-of-fold predictions ----
    for task in TASKS:
        Fd = FOLDS[task]
        OOF[task + '__y'] = Fd['y']
        OOF[task + '__names'] = Fd['names']
        OOF[task + '__patient_ids'] = Fd['groups']
        OOF[task + '__fold_id'] = CV[task][ARMS[0]]['fold_id']
        for arm in ARMS:
            add_oof(task, arm, CV[task][arm])
    write_oof()
    print('written results.json and oof_predictions.npz')

    # ---- tables ----
    for clf in ('mlp', 'rf'):
        print('\n%s macro-F1 [95%% patient-bootstrap CI], pooled out-of-fold; [n/a] = fewer than 10 valid bootstrap draws' % clf.upper())
        print('%-24s' % 'arm' + ''.join('%22s' % t for t in TASKS))
        print('-' * (24 + 22 * len(TASKS)))
        for arm in ARMS:
            print('%-24s' % ARM_LABEL[arm] + ''.join('%22s' % ('%5.1f %s' % (EVAL[t][arm][clf]['macro_f1'], ci_str(EVAL[t][arm][clf]['macro_f1_ci']))) for t in TASKS))
        print('%-24s' % 'majority class' + ''.join('%22.1f' % RESULTS['arms'][t][ARMS[0]]['majority']['macro_f1'] for t in TASKS))
    print('\npaired differences in MLP macro-F1 (patient bootstrap 95% CI, P(diff>0) over finite draws; fold t-test p)')
    for task in TASKS:
        print('  ' + task)
        for key, v in RESULTS['paired'][task].items():
            m = v['mlp']['macro_f1']
            tt = v['mlp'].get('fold_ttest', {})
            print('     %-6s %+6.1f  %s  P(>0)=%s   fold t-test p=%s' % (
                key, m['point'], ci_str(m['ci'], '%+6.1f'), ('%.3f' % m['p_gt0']) if m['p_gt0'] is not None else 'n/a',
                ('%.3f' % tt['p']) if tt.get('p') is not None else 'n/a'))
''')

md(r'''
## Stage 8: validation gate

Before any new arm is trusted, arm C (PCG segmentation) and arm E (equal-time 75 bpm) must
reproduce the mechanism study's reference macro-F1 values on the same recordings. Two things are
checked. First the sample: per task, n and the number of patients must equal the reference run
(detection 3006 / 874, grading 612 / 178, timing 608 / 177, shape 605 / 176); when they differ the
recordings dropped per arm are printed. Second the numbers: the two gate arms are refitted with the
**reference recipe** (`fit_mlp_reference`: the reference study's own model class `ReferenceMLP`
with PyTorch default initialisation, not the V5 class with its Kaiming step; focal loss gamma 2 with
class weights, full-batch steps, 300 epochs, validation every 5 epochs, patience 5 checks, constant
seed, inner `GroupShuffleSplit` 0.15), because that is the recipe behind the reference values; TF32
is switched on for these fits only, as it was in the reference run, and restored afterwards. The
primary recipe's value is printed beside each cell for orientation.

**The rule.** A cell is within when its deviation from the reference is at most `GATE_TOL` points,
or when the reference value lies inside the 95 % patient-bootstrap interval of the gate value. The
second clause is what "a mismatch larger than bootstrap noise" means: the reference run itself
moves by several points from one seed to the next on the three-class tasks, so a fixed 3-point band
alone would fail a faithful reproduction about as often as not. Every cell's value, deviation,
interval and verdict are printed and stored (`ci`, `within`). The gate passes when the sample
matches and every cell is within; a sample mismatch or any cell outside sets
`VALIDATION_OK = False`, the figure cells are then skipped (unless `FORCE_FIGURES` is set) and the
manifest records the failure.

If the kept set differs because arms A or B dropped recordings (arms C, D, E and E2 keep exactly
the 3006 reference recordings), the two gate arms are additionally evaluated on the reference-rule
set (at least two whole-recording TSV cycles and at least 1 s of audio) with folds drawn on that
set and the same rule; when that set matches the reference sample and every cell is within, the
gate passes on that basis, says so in its headline, and records `decided_on = reference-rule set`.
Numbers from such a run must be reported with the kept-set n, not 3006. The shipped synthetic arm's
reference values are printed next to arm A for orientation only; arm A uses the corrected detector,
so it is not gated. Runs on a subset (`DRY_RUN` or `LIMIT_RECORDINGS`) cannot be gated: the cell
prints "not applicable" and runs no reference fits.
''')

code('s8_gate', r'''
REFERENCE = {'pcg_seg': {'detection': 79.2, 'grading': 58.3, 'timing': 49.0, 'shape': 46.6},
             'equal_time': {'detection': 69.6, 'grading': 51.0, 'timing': 39.5, 'shape': 39.0}}
# arm A is not gated (corrected detector); two rows are printed for orientation: the shipped detector and the mechanism study's SOS arm
REFERENCE_INFO = {'A / shipped': {'detection': 71.5, 'grading': 52.7, 'timing': 41.6, 'shape': 40.5},
                  'A / sos': {'detection': 69.3, 'grading': 50.2, 'timing': 44.1, 'shape': 40.3}}
REFERENCE_N = {'detection': (3006, 874), 'grading': (612, 178), 'timing': (608, 177), 'shape': (605, 176)}  # Mechanism cell 30 output
GATE_RULE = ('a cell is within when |delta| <= GATE_TOL or the reference value lies inside the 95% patient-bootstrap interval of the '
             'gate value; the gate passes when the sample matches the reference run and every cell is within')

def gate_cell(cv, task, arm, extra=None):
    """One gate cell: reference-recipe macro-F1 with its 95 % patient-bootstrap interval and the verdict of the rule."""
    e = evaluate_predictions(cv['y'], cv['pred_mlp'], None, cv['groups'], NCLS[task], with_scores=False)
    v, ci, ref = e['macro_f1'], e['macro_f1_ci'], REFERENCE[arm][task]
    by_tol = abs(v - ref) <= GATE_TOL
    by_ci = ci[0] is not None and ci[1] is not None and ci[0] <= ref <= ci[1]
    cell = dict(arm=arm, task=task, reference=ref, value_reference_recipe=v, delta=v - ref, ci=ci, within_tolerance=bool(by_tol),
                reference_inside_ci=bool(by_ci), within=bool(by_tol or by_ci), n=int(len(cv['y'])), n_boot_valid=e['n_boot_valid'])
    cell.update(extra or {})
    return cell

def gate_row(c, primary=None):
    verdict = 'within (tolerance)' if c['within_tolerance'] else ('within (reference inside the 95% interval)' if c['reference_inside_ci'] else 'OUTSIDE')
    return '%-12s %-10s %9.1f %10.1f %+7.1f  %-16s %s %s' % (c['arm'], c['task'], c['reference'], c['value_reference_recipe'], c['delta'], ci_str(c['ci']),
                                                         ('%8.1f ' % primary) if primary is not None else '', verdict)

@contextlib.contextmanager
def gate_tf32():
    """The reference run had TF32 on (Mechanism cell 30). The gate fits run with both flags on; the flags are restored afterwards."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield prev
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev

with stage('S8 validation gate'):
    gate = {'applicable': VALIDATION_APPLICABLE, 'tolerance': GATE_TOL, 'rule': GATE_RULE,
            'recipe': 'reference-focal-fullbatch (fit_mlp_reference: ReferenceMLP with PyTorch default initialisation, TF32 on during the fits)',
            'reference_n': 3006, 'sample': {}, 'cells': [], 'decided_on': None}
    # ---- sample check: per-task n and patients against the reference run ----
    print('sample check against the reference run (per task)')
    for task, (n_ref, p_ref) in REFERENCE_N.items():
        n_, p_ = (int(len(FOLDS[task]['y'])), int(len(np.unique(FOLDS[task]['groups'])))) if task in FOLDS else (0, 0)
        gate['sample'][task] = dict(n=n_, n_reference=n_ref, patients=p_, patients_reference=p_ref, same=(n_ == n_ref and p_ == p_ref), skipped=task not in FOLDS)
        print('   %-10s n %4d (reference %4d)  patients %3d (reference %3d)%s' % (
            task, n_, n_ref, p_, p_ref, '' if gate['sample'][task]['same'] else ('  <-- skipped in this run' if task not in FOLDS else '  <-- differs')))
    SAMPLE_OK = all(v['same'] for v in gate['sample'].values())
    gate['sample_ok'] = SAMPLE_OK

    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        print('\nVALIDATION GATE NOT APPLICABLE: this run uses %d recordings (DRY_RUN=%s, LIMIT_RECORDINGS=%d); the reference is the full 3006-recording set, so the reference-recipe fits are not run' % (
            len(KEEP), DRY_RUN, LIMIT_RECORDINGS))
    else:
        by_arm = dict(collections.Counter(a for _, bad in DROPPED for a in bad))
        if not SAMPLE_OK:
            print('!!! kept set differs from the reference set; recordings dropped per arm:', by_arm)
        REFSET_OK, worst_ref = False, None
        with gate_tf32() as _tf32_prev:
            # ---- the two gate arms refitted with the reference recipe ----
            GATE_CV = {t: {a: run_cv(FEATS[a], t, a + '__gate', do_rf=False, recipe='reference') for a in REFERENCE} for t in TASKS}
            print('\nrule: ' + GATE_RULE)
            print('%-12s %-10s %9s %10s %7s  %-16s %8s  %s' % ('arm', 'task', 'reference', 'ref.recipe', 'delta', '95% interval', 'primary', 'verdict'))
            print('-' * 100)
            for arm, refs in REFERENCE.items():
                for task in refs:
                    if task not in TASKS:
                        continue
                    c = gate_cell(GATE_CV[task][arm], task, arm, dict(value_primary_recipe=EVAL[task][arm]['mlp']['macro_f1']))
                    gate['cells'].append(c)
                    print(gate_row(c, c['value_primary_recipe']))
            for lab, refs in REFERENCE_INFO.items():
                for task, ref in refs.items():
                    if task in TASKS:
                        v_ = EVAL[task]['syn_fixed']['mlp']['macro_f1']
                        print('%-12s %-10s %9.1f %10s %+7.1f  %-16s %8.1f   (information only, primary recipe)' % (lab, task, ref, '', v_ - ref, '', v_))
            worst = max([abs(c['delta']) for c in gate['cells']] or [0.0])
            gate['worst_abs_delta'] = worst
            gate['n_cells_outside'] = int(sum(not c['within'] for c in gate['cells']))

            # ---- reference-rule set when the kept set differs: >= 2 whole-recording TSV cycles and >= 1 s of audio ----
            if not SAMPLE_OK:
                REF_RULE_SET = [n for n in NAMES_ALL if ARM_DIAG['pcg_seg'][n].get('n_annotated', 0) >= 2 and AUDIO_META[n]['n_raw'] / AUDIO_META[n]['sr_raw'] >= 1.0]
                ref_names = np.array(REF_RULE_SET)
                gref = np.array([PID_OF[n] for n in REF_RULE_SET])
                print('\nreference-rule set (>= 2 whole-recording TSV cycles, >= 1 s): %d recordings, %d patients; %d of them outside the kept set' % (
                    len(REF_RULE_SET), len(np.unique(gref)), len(set(REF_RULE_SET) - set(KEEP))))
                ref_feats = build_feature_matrices({a + '__refset': CYC[a] for a in REFERENCE}, REF_RULE_SET, FEATURE_NAMES, 'gate_refset')
                gate['refset'] = dict(rule='>= 2 whole-recording TSV cycles and >= 1 s of audio', n=len(REF_RULE_SET), n_patients=int(len(np.unique(gref))), sample={}, cells=[])
                print('%-12s %-10s %9s %10s %7s  %-16s  %s' % ('arm', 'task', 'reference', 'ref.set', 'delta', '95% interval', 'verdict'))
                for task in TASKS:
                    yref = np.array([LAB_OF[n][TASK_KEY[task]] for n in REF_RULE_SET])
                    fd = draw_or_load_folds(task, ref_names, yref, gref, task + '__refset.npz')
                    assert fd is not None, 'reference-rule set folds could not be drawn for %s' % task
                    n_, p_ = int(len(fd['y'])), int(len(np.unique(fd['groups'])))
                    gate['refset']['sample'][task] = dict(n=n_, patients=p_, same=((n_, p_) == REFERENCE_N[task]))
                    for arm in REFERENCE:
                        r = run_cv(ref_feats[arm + '__refset'], task, arm + '__gate_refset', do_rf=False, recipe='reference', folds=fd)
                        c = gate_cell(r, task, arm, dict(patients=p_))
                        gate['refset']['cells'].append(c)
                        print(gate_row(c))
                worst_ref = max([abs(c['delta']) for c in gate['refset']['cells']] or [0.0])
                gate['refset']['sample_ok'] = all(v['same'] for v in gate['refset']['sample'].values()) and set(gate['refset']['sample']) == set(REFERENCE_N)
                gate['refset']['worst_abs_delta'] = worst_ref
                gate['refset']['n_cells_outside'] = int(sum(not c['within'] for c in gate['refset']['cells']))
                REFSET_OK = gate['refset']['sample_ok'] and gate['refset']['n_cells_outside'] == 0 and len(gate['refset']['cells']) > 0
                print('reference-rule set: sample %s the reference, %d of %d cells outside, worst deviation %.1f points' % (
                    'matches' if gate['refset']['sample_ok'] else 'does not match', gate['refset']['n_cells_outside'], len(gate['refset']['cells']), worst_ref))
        gate['tf32'] = dict(on_during_gate_fits=True, restored=dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32)),
                            before=dict(matmul=bool(_tf32_prev[0]), cudnn=bool(_tf32_prev[1])))
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == _tf32_prev, 'TF32 flags were not restored after the gate fits'

        KEPT_OK = SAMPLE_OK and gate['n_cells_outside'] == 0 and len(gate['cells']) > 0
        if KEPT_OK:
            VALIDATION_OK = True
            gate['decided_on'] = 'kept set'
            print('\nVALIDATION GATE PASSED: the sample equals the reference run and every cell is within (worst deviation %.1f points; tolerance %.1f, or the reference inside the 95%% interval)' % (worst, GATE_TOL))
        elif not SAMPLE_OK and REFSET_OK:
            VALIDATION_OK = True
            gate['decided_on'] = 'reference-rule set'
            print('\nVALIDATION GATE PASSED ON THE REFERENCE-RULE SET, NOT ON THE KEPT SET.\n'
                  '   The kept set has %d recordings (the reference run has 3006); recordings dropped per arm: %s.\n'
                  '   On the reference sample (%d recordings, %d patients) both gate arms reproduce the reference values: worst deviation %.1f points, every cell within.\n'
                  '   Report every number of this run with the kept-set n (%d), not with 3006.\n'
                  '   Kept-set worst deviation: %.1f points (%d of %d cells outside); an excess over the reference-sample deviation is a sample effect of the\n'
                  '   dropped recordings, not a difference in the recipe.' % (
                      len(KEEP), by_arm, gate['refset']['n'], gate['refset']['n_patients'], worst_ref, len(KEEP), worst, gate['n_cells_outside'], len(gate['cells'])))
        elif not SAMPLE_OK:
            VALIDATION_OK = False
            print('\n' + '!' * 78 + '\n!!! VALIDATION GATE FAILED ON THE SAMPLE: per-task n or patient counts differ from the reference run and the\n!!! reference-rule set does not repair it (its sample %s the reference; %s of its cells outside, worst deviation %s).\n!!! Inspect the drop reasons before trusting any new arm.\n' % (
                'matches' if gate.get('refset', {}).get('sample_ok') else 'does not match', gate.get('refset', {}).get('n_cells_outside', 'n/a'),
                ('%.1f' % worst_ref) if worst_ref is not None else 'n/a') + '!' * 78)
        else:
            VALIDATION_OK = False
            print('\n' + '!' * 78 + '\n!!! VALIDATION GATE FAILED: %d of %d cells are outside both the %.1f-point tolerance and their 95%% interval (worst deviation %.1f).\n!!! The reference arms do not reproduce. The figures are not drawn. Inspect the diagnostics,\n!!! the kept set and the classifier configuration before trusting any new arm.\n' % (
                gate['n_cells_outside'], len(gate['cells']), GATE_TOL, worst) + '!' * 78)
    gate['ok'] = VALIDATION_OK
    RESULTS['validation_gate'] = gate
    write_results()
''')

md(r'''
## Stage 9: ablations

* **Window length (R1-M2).** Detection with the boundary-free features only
  (`extract_recording_level_features` + `extract_soft_murmur_features`, 206 values) computed on the
  first 4, 8 and 16 s and on the full analysed recording, each excerpt max-abs renormalised after
  truncation as the loader normalises a whole recording; same folds, MLP and RF.
* **Timing budget (R1-M1).** Arm C boundaries jittered rigidly per cycle with sigma in
  {0, 10, 20, 40, 80, 160} ms (`jitter_cycles`, verbatim; one generator per sigma seeded
  `SEED + sigma`), same features and folds, MLP macro-F1 with CIs. A cycle whose shifted boundary
  would leave the signal is dropped by the verbatim function; the count per sigma is recorded.
  Arms A and B are placed on the curve by the sigma that would give their macro-F1
  (`equivalent_sigma`, verbatim), which is the valid placement; their measured offset SD and
  median |offset| are reported beside it.
* **Redundancy (R1-m3).** Pearson correlation matrix of the arm C features, the number of pairs with
  |r| > 0.9, the PCA dimension for 95 % variance, and macro-F1 after dropping one feature from each
  such pair (greedy, in alphabetical feature order). The selection is label-free and fitted once on
  all kept recordings; no label enters it, so it is not a per-fold step. The statistics set of the
  supplement's feature-redundancy section (pairs above 0.8 / 0.9 / 0.95 / 0.99 with the features
  involved, the clusters and the independent groups, coincident pairs at |r| >= 0.9999, constant
  features, median and mean |r|, PCA components for 90 / 95 / 99 % of the variance) is written for
  every arm to `results.json` under `ablations.redundancy.per_arm`, so the two can be compared.
* **Scaling variant (R3-m3).** Arm C with the scaler fitted on the full matrix before the split, to
  quantify the optimistic bias the original Discussion disclosed.
* **Subgroups (R3-M4).** Arms A, B and C by heart-rate tertile, once from the arm C median RR and
  once from the arm A (`syn_fixed`) median RR as the design brief requires, and by CirCor age band;
  macro-F1 and accuracy with CIs for every stratum with at least 10 patients, and the arm A / B
  boundary offset per stratum.

The pooled out-of-fold predictions of every ablation (window, jitter, reduced, full-matrix scaler)
are added to `oof_predictions.npz`.
''')

code('s9_window', r'''
RESULTS['ablations'] = RESULTS.get('ablations') or {}
WINDOW_S = [4, 8, 16, None]

def _recording_level_extractor(pcg, n_samples, sr):
    x = pcg if n_samples is None else pcg[:n_samples]
    x = x / (np.max(np.abs(x)) + 1e-10)          # renormalised after truncation, as the loader normalises a whole recording
    f = {}
    f.update(extract_recording_level_features(x, sr))
    f.update(extract_soft_murmur_features(x, sr))
    return f

with stage('S9a window length'):
    WIN_NAMES = sorted(_recording_level_extractor(AUDIO4[KEEP[0]], None, TARGET_SR).keys())
    print('boundary-free feature count: %d' % len(WIN_NAMES))
    win_jobs = {}
    for w in WINDOW_S:
        tag = 'window_%s' % ('full' if w is None else '%02ds' % w)
        win_jobs[tag] = {n: (None if w is None else int(w * TARGET_SR)) for n in KEEP}
    WIN_FEATS = build_feature_matrices(win_jobs, KEEP, WIN_NAMES, 'window', extractor=_recording_level_extractor)
    WINDOW = {}
    for w in WINDOW_S:
        tag = 'window_%s' % ('full' if w is None else '%02ds' % w)
        n_short = int(sum(AUDIO_META[n]['L4'] < (w or 0) * TARGET_SR for n in KEEP))
        r = run_cv(WIN_FEATS[tag], 'detection', tag)
        add_oof('detection', tag, r)
        WINDOW[tag] = dict(window_s=w, n_recordings_shorter_than_window=n_short,
                           mlp=strip_boot(evaluate_predictions(r['y'], r['pred_mlp'], r['proba_mlp'], r['groups'], 2, r['fold_f1_mlp'], r['fold_acc_mlp'])),
                           rf=strip_boot(evaluate_predictions(r['y'], r['pred_rf'], r['proba_rf'], r['groups'], 2, r['fold_f1_rf'], r['fold_acc_rf'])))
        print('   %-12s MLP macro-F1 %5.1f %s | RF %5.1f %s | %d recordings shorter than the window' % (
            tag, WINDOW[tag]['mlp']['macro_f1'], ci_str(WINDOW[tag]['mlp']['macro_f1_ci']), WINDOW[tag]['rf']['macro_f1'], ci_str(WINDOW[tag]['rf']['macro_f1_ci']), n_short))
    RESULTS['ablations']['window_length'] = dict(feature_count=len(WIN_NAMES), task='detection', results=WINDOW, renormalised_after_truncation=True,
                                                 reference_arm_c_mlp=EVAL['detection']['pcg_seg']['mlp']['macro_f1'],
                                                 reference_arm_e_mlp=EVAL['detection']['equal_time']['mlp']['macro_f1'])
    write_results()
    write_oof()
''')

code('s9_timing', r'''
SIGMAS_MS = [0, 10, 20, 40, 80, 160]
with stage('S9b timing budget'):
    jit_jobs, jit_short, jit_dropped = {}, {}, {}
    n_cyc_c = int(sum(len(CYC['pcg_seg'][n]) for n in KEEP))
    for s in SIGMAS_MS[1:]:
        rng = np.random.default_rng(SEED + s)
        tag = 'jitter_%03d' % s
        jit_jobs[tag] = {n: jitter_cycles(CYC['pcg_seg'][n], s, TARGET_SR, rng, AUDIO_META[n]['L4']) for n in KEEP}
        for n in KEEP:
            check_cycles(jit_jobs[tag][n], AUDIO_META[n]['L4'], tag + ' ' + n, strict=False)
        jit_short[tag] = int(sum(len(c) < 2 for c in jit_jobs[tag].values()))
        jit_dropped[tag] = n_cyc_c - int(sum(len(c) for c in jit_jobs[tag].values()))
    print('   cycles dropped by the jitter (a shifted boundary outside the signal), of %d arm C cycles: %s' % (
        n_cyc_c, ', '.join('sigma %d: %d' % (s, jit_dropped['jitter_%03d' % s]) for s in SIGMAS_MS[1:])))
    JIT_FEATS = build_feature_matrices(jit_jobs, KEEP, FEATURE_NAMES, 'jitter')
    TIMING = {t: {} for t in TASKS}
    for task in TASKS:
        for s in SIGMAS_MS:
            if s == 0:
                e = EVAL[task]['pcg_seg']['mlp']
                TIMING[task][s] = dict(macro_f1=e['macro_f1'], macro_f1_ci=e['macro_f1_ci'], accuracy=e['accuracy'], accuracy_ci=e['accuracy_ci'],
                                       n_dropped_cycles=0, source='arm C')
                continue
            tag = 'jitter_%03d' % s
            r = run_cv(JIT_FEATS[tag], task, tag, do_rf=False)
            add_oof(task, tag, r, clfs=('mlp',))
            e = evaluate_predictions(r['y'], r['pred_mlp'], None, r['groups'], NCLS[task], r['fold_f1_mlp'], r['fold_acc_mlp'], with_scores=False)
            TIMING[task][s] = dict(macro_f1=e['macro_f1'], macro_f1_ci=e['macro_f1_ci'], accuracy=e['accuracy'], accuracy_ci=e['accuracy_ci'],
                                   recordings_with_under_2_cycles=jit_short[tag], n_dropped_cycles=jit_dropped[tag], source=tag)
        f1s = [TIMING[task][s]['macro_f1'] for s in SIGMAS_MS]
        TIMING[task]['equivalent_sigma_ms'] = {arm: equivalent_sigma(SIGMAS_MS, f1s, EVAL[task][arm]['mlp']['macro_f1']) for arm in ('syn_fixed', 'syn_fusion')}
        print('   %-10s ' % task + ' '.join('s=%3d: %5.1f' % (s, TIMING[task][s]['macro_f1']) for s in SIGMAS_MS) +
              ' | A %.1f -> sigma %s | B %.1f -> sigma %s' % (
                  EVAL[task]['syn_fixed']['mlp']['macro_f1'], TIMING[task]['equivalent_sigma_ms']['syn_fixed'],
                  EVAL[task]['syn_fusion']['mlp']['macro_f1'], TIMING[task]['equivalent_sigma_ms']['syn_fusion']))
    ARM_OFFSET_SUMMARY = {arm: dict(median_abs_offset_ms=DIAG_SUMMARY[arm]['median_abs_offset_ms'], median_offset_ms=DIAG_SUMMARY[arm]['median_offset_ms'],
                                    offset_sd_ms=DIAG_SUMMARY[arm]['offset_sd_ms'], abs_offset_iqr_ms=DIAG_SUMMARY[arm]['abs_offset_iqr_ms'],
                                    insertion_rate=DIAG_SUMMARY[arm]['insertion_rate'], deletion_rate=DIAG_SUMMARY[arm]['deletion_rate'])
                          for arm in ('syn_fixed', 'syn_fusion')}
    print('   arm A: median |offset| %.0f ms, offset SD %s ms, insertion %.2f, deletion %.2f | arm B: median |offset| %.0f ms, offset SD %s ms, insertion %.2f, deletion %.2f' % (
        ARM_OFFSET_SUMMARY['syn_fixed']['median_abs_offset_ms'], ARM_OFFSET_SUMMARY['syn_fixed']['offset_sd_ms'] and '%.0f' % ARM_OFFSET_SUMMARY['syn_fixed']['offset_sd_ms'],
        ARM_OFFSET_SUMMARY['syn_fixed']['insertion_rate'], ARM_OFFSET_SUMMARY['syn_fixed']['deletion_rate'],
        ARM_OFFSET_SUMMARY['syn_fusion']['median_abs_offset_ms'], ARM_OFFSET_SUMMARY['syn_fusion']['offset_sd_ms'] and '%.0f' % ARM_OFFSET_SUMMARY['syn_fusion']['offset_sd_ms'],
        ARM_OFFSET_SUMMARY['syn_fusion']['insertion_rate'], ARM_OFFSET_SUMMARY['syn_fusion']['deletion_rate']))
    RESULTS['ablations']['timing_budget'] = dict(sigmas_ms=SIGMAS_MS, curves=TIMING, arm_offsets=ARM_OFFSET_SUMMARY, n_cycles_arm_c=n_cyc_c,
                                                 dropped_cycles_per_sigma=jit_dropped, classifier='mlp',
                                                 note='the equivalent sigma (curve value equal to the arm macro-F1) is the valid placement of an arm on the curve; the median |offset| is not a sigma')
    write_results()
    write_oof()
''')

code('s9_redundancy_scaling', r'''
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def redundancy_stats(X):
    """Label-free redundancy statistics of one feature matrix over all kept recordings, with the definitions of the
    supplement's feature-redundancy section: pairs above a threshold among all d(d-1)/2 pairs, the features involved in
    at least one such pair, clusters = connected components of the pair graph, groups = d - features + clusters,
    coincident pairs (|r| >= 0.9999), constant features, median and mean |r|, and the number of principal components of
    the standardised matrix that carry 90 / 95 / 99 % of the variance. Returns (R, constant mask, statistics)."""
    Xc = np.asarray(X, np.float64)
    d = Xc.shape[1]
    const = Xc.std(0) == 0
    Xs = StandardScaler().fit_transform(Xc)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(Xs.T)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    absr = np.abs(R[np.triu_indices(d, 1)])
    by_thr = {}
    for thr in (0.8, 0.9, 0.95, 0.99):
        adj = np.abs(R) > thr
        np.fill_diagonal(adj, False)
        involved = adj.any(1)
        _, lab = connected_components(csr_matrix(adj), directed=False)
        sizes = np.bincount(lab[involved]) if involved.any() else np.array([], int)
        sizes = sizes[sizes >= 2]
        by_thr['%.2f' % thr] = dict(pairs=int((absr > thr).sum()), features=int(involved.sum()), clusters=int(len(sizes)),
                                     largest=int(sizes.max()) if len(sizes) else 0, groups=int(d - involved.sum() + len(sizes)))
    cum = np.cumsum(PCA(svd_solver='full').fit(Xs[:, ~const]).explained_variance_ratio_)
    pca = {p: int(min(len(cum), np.searchsorted(cum, p / 100.0 - 1e-12) + 1)) for p in (90, 95, 99)}
    stats = dict(n_recordings=int(Xc.shape[0]), n_features=int(d), n_pairs_total=int(len(absr)), n_constant=int(const.sum()),
                 constant_features=[FEATURE_NAMES[j] for j in np.flatnonzero(const)] if d == len(FEATURE_NAMES) else [],
                 by_threshold=by_thr, n_pairs_abs_r_ge_0_9999=int((absr >= 0.9999).sum()),
                 median_abs_r=float(np.median(absr)), mean_abs_r=float(np.mean(absr)),
                 pca_components_90pct=pca[90], pca_components_95pct=pca[95], pca_components_99pct=pca[99])
    return R, const, stats

with stage('S9c redundancy'):
    RED_ARM = {}
    print('   %-20s %5s %28s %11s %16s' % ('arm', 'const', '|r|>0.9: pairs/feat/clusters', 'coincident', 'PCA 90/95/99 %'))
    for arm in ARMS:
        R_, const_, RED_ARM[arm] = redundancy_stats(FEATS[arm])
        if arm == 'pcg_seg':
            R, const = R_, const_
        s_, t9 = RED_ARM[arm], RED_ARM[arm]['by_threshold']['0.90']
        print('   %-20s %5d %28s %11d %16s' % (arm, s_['n_constant'], '%d / %d / %d' % (t9['pairs'], t9['features'], t9['clusters']), s_['n_pairs_abs_r_ge_0_9999'],
                                             '%d / %d / %d' % (s_['pca_components_90pct'], s_['pca_components_95pct'], s_['pca_components_99pct'])))
    rs = RED_ARM['pcg_seg']
    d = R.shape[0]
    n_pairs = rs['by_threshold']['0.90']['pairs']
    kept_idx, dropped_idx = [], []
    for j in range(d):
        if kept_idx and np.any(np.abs(R[j, kept_idx]) > 0.9):
            dropped_idx.append(j)
        else:
            kept_idx.append(j)
    atomic_np_save(CACHE_DIR / 'redundancy_corr_pcg_seg.npy', R.astype(np.float32))
    print('   arm C: features %d | constant %d | pairs with |r|>0.9: %d | PCA components for 95%% variance: %d | dropped %d -> %d features (label-free selection on all kept recordings)' % (
        d, rs['n_constant'], n_pairs, rs['pca_components_95pct'], len(dropped_idx), len(kept_idx)))
    RED = {}
    for task in TASKS:
        r = run_cv(FEATS['pcg_seg'][:, kept_idx], task, 'pcg_seg_reduced', do_rf=False)
        add_oof(task, 'pcg_seg_reduced', r, clfs=('mlp',))
        e = evaluate_predictions(r['y'], r['pred_mlp'], None, r['groups'], NCLS[task], r['fold_f1_mlp'], r['fold_acc_mlp'], with_scores=False)
        diff = paired_difference(e, EVAL[task]['pcg_seg']['mlp'], r['fold_f1_mlp'], CV[task]['pcg_seg']['fold_f1_mlp'])
        RED[task] = dict(reduced=strip_boot(e), full_macro_f1=EVAL[task]['pcg_seg']['mlp']['macro_f1'], reduced_minus_full=diff)
        print('   %-10s full %5.1f | reduced %5.1f %s | diff %+.1f %s' % (
            task, RED[task]['full_macro_f1'], e['macro_f1'], ci_str(e['macro_f1_ci']), diff['macro_f1']['point'], ci_str(diff['macro_f1']['ci'], '%+.1f')))
    RESULTS['ablations']['redundancy'] = dict(
        arm='pcg_seg', n_features=d, n_constant=rs['n_constant'], n_pairs_abs_r_over_0_9=n_pairs,
        n_features_in_pairs_over_0_9=rs['by_threshold']['0.90']['features'], n_clusters_over_0_9=rs['by_threshold']['0.90']['clusters'],
        n_pairs_abs_r_ge_0_9999=rs['n_pairs_abs_r_ge_0_9999'],
        pca_components_90pct=rs['pca_components_90pct'], pca_components_95pct=rs['pca_components_95pct'], pca_components_99pct=rs['pca_components_99pct'],
        n_dropped=len(dropped_idx), n_kept=len(kept_idx),
        selection='label-free: pairs with |r| > 0.9 in the Pearson matrix of the arm C features over all kept recordings, no labels used; one feature of each pair dropped greedily in alphabetical order; the same reduced set is used in every fold',
        definitions='pairs: |r| above the threshold among all d(d-1)/2 feature pairs; features: in at least one such pair; clusters: connected components of the pair graph; groups: d - features + clusters; coincident: |r| >= 0.9999; PCA on the standardised matrix (constant features excluded), smallest number of components reaching the variance level',
        dropped_features=[FEATURE_NAMES[j] for j in dropped_idx], per_task=RED, per_arm=RED_ARM, classifier='mlp')
    write_results()
    write_oof()

with stage('S9d scaling variant'):
    SCAL = {}
    for task in TASKS:
        r = run_cv(FEATS['pcg_seg'], task, 'pcg_seg_fullscaler', scaler='full', do_rf=False)
        add_oof(task, 'pcg_seg_fullscaler', r, clfs=('mlp',))
        e = evaluate_predictions(r['y'], r['pred_mlp'], None, r['groups'], NCLS[task], r['fold_f1_mlp'], r['fold_acc_mlp'], with_scores=False)
        diff = paired_difference(e, EVAL[task]['pcg_seg']['mlp'], r['fold_f1_mlp'], CV[task]['pcg_seg']['fold_f1_mlp'])
        SCAL[task] = dict(full_matrix_scaler=strip_boot(e), in_fold_macro_f1=EVAL[task]['pcg_seg']['mlp']['macro_f1'], full_minus_in_fold=diff)
        print('   %-10s in-fold %5.1f | full-matrix scaler %5.1f %s | bias %+.1f %s' % (
            task, SCAL[task]['in_fold_macro_f1'], e['macro_f1'], ci_str(e['macro_f1_ci']), diff['macro_f1']['point'], ci_str(diff['macro_f1']['ci'], '%+.1f')))
    RESULTS['ablations']['scaling'] = dict(arm='pcg_seg', classifier='mlp', per_task=SCAL)
    write_results()
    write_oof()
''')

code('s9_subgroups', r'''
SUB_ARMS = ['syn_fixed', 'syn_fusion', 'pcg_seg']
AGE_ORDER = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult', 'Unknown']
MIN_STRATUM_PATIENTS = 10
with stage('S9e subgroups'):
    _dc = DIAG[DIAG.arm == 'pcg_seg'].set_index('recording')
    _offA = DIAG[DIAG.arm == 'syn_fixed'].set_index('recording')
    _offB = DIAG[DIAG.arm == 'syn_fusion'].set_index('recording')
    HR_KEEP_C = np.array([float(_dc.loc[n, 'bpm']) for n in KEEP])
    HR_KEEP_A = np.array([float(_offA.loc[n, 'bpm']) for n in KEEP])

    def _tertiles(hr):
        fin = np.isfinite(hr)
        q1, q2 = (np.nanpercentile(hr[fin], [33.3, 66.7]) if fin.sum() >= 3 else (np.nan, np.nan))
        band = np.where(~fin, 'unknown', np.where(hr <= q1, 'low', np.where(hr <= q2, 'mid', 'high')))
        return band, dict(cut_bpm=[float(q1), float(q2)], n_without_rate=int((~fin).sum()))

    SUBG = {}
    for task in TASKS:
        Fd = FOLDS[task]
        age = AGE[Fd['mask']]
        names_m = Fd['names']
        band_c, ex_c = _tertiles(HR_KEEP_C[Fd['mask']])
        band_a, ex_a = _tertiles(HR_KEEP_A[Fd['mask']])
        strata = {'hr_tertile': (band_c, ['low', 'mid', 'high'], dict(ex_c, source='arm C median RR (bpm), tertiles within the task set')),
                  'hr_tertile_arm_a': (band_a, ['low', 'mid', 'high'], dict(ex_a, source='arm A (syn_fixed) median RR (bpm), tertiles within the task set')),
                  'age_band': (age, [a for a in AGE_ORDER if a in set(age.tolist())], {})}
        SUBG[task] = {}
        for sname, (band, levels, extra) in strata.items():
            SUBG[task][sname] = dict(extra, levels={})
            for lvl in levels:
                sel = band == lvl
                entry = dict(n=int(sel.sum()), n_patients=int(len(np.unique(Fd['groups'][sel]))) if sel.any() else 0,
                             median_abs_offset_ms={'syn_fixed': float(np.nanmedian(_offA.loc[names_m[sel], 'median_abs_offset_ms'].astype(float))) if sel.any() else None,
                                                   'syn_fusion': float(np.nanmedian(_offB.loc[names_m[sel], 'median_abs_offset_ms'].astype(float))) if sel.any() else None},
                             insertion_rate={'syn_fixed': float(np.nanmedian(_offA.loc[names_m[sel], 'insertion_rate'].astype(float))) if sel.any() else None,
                                             'syn_fusion': float(np.nanmedian(_offB.loc[names_m[sel], 'insertion_rate'].astype(float))) if sel.any() else None},
                             class_counts=np.bincount(Fd['y'][sel], minlength=NCLS[task]).tolist() if sel.any() else [])
                if entry['n_patients'] >= MIN_STRATUM_PATIENTS:
                    for arm in SUB_ARMS:
                        r = CV[task][arm]
                        e = evaluate_predictions(r['y'][sel], r['pred_mlp'][sel], None, r['groups'][sel], NCLS[task], with_scores=False)
                        entry[arm] = dict(macro_f1=e['macro_f1'], macro_f1_ci=e['macro_f1_ci'], accuracy=e['accuracy'], accuracy_ci=e['accuracy_ci'], n_boot_valid=e['n_boot_valid'])
                else:
                    entry['note'] = 'fewer than %d patients: no subgroup estimate' % MIN_STRATUM_PATIENTS
                SUBG[task][sname]['levels'][lvl] = entry
        print('   %-10s HR tertile cuts: arm C %.0f / %.0f bpm, arm A %.0f / %.0f bpm | ' % (task, ex_c['cut_bpm'][0], ex_c['cut_bpm'][1], ex_a['cut_bpm'][0], ex_a['cut_bpm'][1]) +
              ' | '.join('%s n=%d' % (l, SUBG[task]['hr_tertile']['levels'][l]['n']) for l in ['low', 'mid', 'high']) + ' | age: ' + ', '.join(
              '%s n=%d' % (l, v['n']) for l, v in SUBG[task]['age_band']['levels'].items()))
        for sname in ('hr_tertile', 'hr_tertile_arm_a'):
            for lvl in ['low', 'mid', 'high']:
                en = SUBG[task][sname]['levels'][lvl]
                if 'pcg_seg' in en:
                    print('       %-16s %-5s A %5.1f  B %5.1f  C %5.1f  | |offset| A %s ms, B %s ms' % (
                        sname, lvl, en['syn_fixed']['macro_f1'], en['syn_fusion']['macro_f1'], en['pcg_seg']['macro_f1'],
                        '%.0f' % en['median_abs_offset_ms']['syn_fixed'] if en['median_abs_offset_ms']['syn_fixed'] is not None else 'n/a',
                        '%.0f' % en['median_abs_offset_ms']['syn_fusion'] if en['median_abs_offset_ms']['syn_fusion'] is not None else 'n/a'))
    RESULTS['ablations']['subgroups'] = dict(arms=SUB_ARMS, classifier='mlp', stratifiers=['hr_tertile', 'hr_tertile_arm_a', 'age_band'],
                                             heart_rate_sources={'hr_tertile': 'arm C median RR (bpm), tertiles within each task set',
                                                                 'hr_tertile_arm_a': 'arm A (syn_fixed) median RR (bpm), tertiles within each task set'},
                                             min_patients_per_stratum=MIN_STRATUM_PATIENTS, per_task=SUBG)
    write_results()
''')

md(r'''
## Stage 10: figures

European Heart Journal conventions: vector PDF plus 600 dpi PNG, 88 mm or 180 mm wide, sans-serif
with nothing below 8 pt (ticks 8, labels 9, titles 10; a helper asserts this before every save),
one palette (blue #184f95, amber #eda100, greys; arm B is drawn in the dark amber with a hatch in
bars and histograms and a dash-dot line, so it never reads as arm A), panel letters, error bars =
95 % patient-bootstrap CI (an open marker or a paler bar means the CI is undefined, fewer than 10
valid bootstrap draws), percentage axes on a shared range within each figure, and n printed with
every caption. A layout check runs before every save: every drawn text must lie inside the canvas
and no two drawn texts may overlap, otherwise the cell stops with a message naming them, so a
clipped or collided figure is never written silently. The figure cells read only the written outputs (`results.json`,
`cache/offsets_ms.pkl`, `cache/redundancy_corr_pcg_seg.npy`), so Stage 10 can be re-run on a
finished folder: run the flag, environment, override and Stage 0 cells with `RESUME_DIR` set, then
the three figure cells. The figures are saved to `figures/` in the run folder and are not shown
inline. If the validation gate failed the figure cells print the failure and skip the drawing
(unless `FORCE_FIGURES` is set); the provenance and manifest cells below still run, so the manifest
records the failure. A figure that fails its own checks is reported, the other figures are still
drawn, and the error is raised again at the very end of the notebook, after the manifest is written.
''')

code('s10_fig_style', r'''
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM = 1 / 25.4
W_SINGLE, W_DOUBLE = 88 * MM, 180 * MM
BLUE, AMBER = '#184f95', '#eda100'
BLUE_LIGHT, AMBER_DARK = '#7fa3d1', '#b37700'
GREY_D, GREY_M, GREY_L = '#404040', '#8c8c8c', '#cfcfcf'
ARM_COLOR = {'syn_fixed': AMBER, 'syn_fusion': AMBER_DARK, 'pcg_seg': BLUE, 'pcg_seg_fixed': BLUE_LIGHT,
             'equal_time': GREY_M, 'equal_time_matched': GREY_D}
ARM_HATCH = {'syn_fusion': '////'}                                              # arm B carries a hatch in bars and histograms
ARM_LS = {'syn_fixed': '--', 'syn_fusion': (0, (4, 1.5, 1, 1.5)), 'equal_time': '-.'}   # arm B: dash-dot, never arm A's dashes
ARM_MARKER = {'syn_fixed': 'o', 'syn_fusion': 's', 'pcg_seg': 'D', 'pcg_seg_fixed': '^', 'equal_time': 'v', 'equal_time_matched': 'h'}
TT = {'detection': 'Detection', 'grading': 'Grading', 'timing': 'Timing', 'shape': 'Shape'}
CLASS_NAMES = {'detection': ['absent', 'present'], 'grading': ['I', 'II', 'III+'], 'timing': ['holo', 'early', 'mid'], 'shape': ['plat.', 'decr.', 'diam.']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 10,
                     'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2, 'hatch.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 110, 'savefig.pad_inches': 0.02})
FIG_CAPTIONS = {}

# ---- everything the figures need is read from the written outputs, so this stage can be re-run on a finished folder ----
with open(RUN_DIR / 'results.json') as fh:
    RES = json.load(fh)
FIG_ARMS = list(RES['run']['arms'])
FIG_TASKS = list(RES['run']['tasks'])
FIG_ARM_LETTER = dict(RES['run']['arm_letters'])
FIG_ARM_LABEL = dict(RES['run']['arm_labels'])
FIG_MATCH_MS = float(RES['run']['match_ms'])
FIG_N = {t: int(RES['arms'][t][FIG_ARMS[0]]['n']) for t in FIG_TASKS}
FIG_KEEP = list(RES['sample']['kept_names'])
FIG_DIAG = RES['diagnostics_summary']
FIG_VALIDATION_OK = RES['validation_gate']['ok'] if RES.get('validation_gate') else None
with open(CACHE_DIR / 'offsets_ms.pkl', 'rb') as fh:
    FIG_OFFSETS = pickle.load(fh)

def n_caption():
    return ', '.join('%s %d' % (TT[t], FIG_N[t]) for t in FIG_TASKS)

def panel_letter(ax, k, x=-0.14, y=1.04):
    ax.text(x, y, 'ABCDEFGHIJKL'[k], transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')

def assert_min_font(fig, minimum=8.0):
    small = [(t.get_text()[:30], float(t.get_fontsize())) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < minimum - 1e-6]
    assert not small, 'text below %g pt: %s' % (minimum, small[:5])

def assert_layout(fig, name, tol_pt=0.5):
    """Stop when a drawn text leaves the canvas or two drawn texts overlap (titles, axis labels, tick labels, panel
    letters, legend entries, annotations), so a clipped or collided figure is never written silently."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tol = tol_pt * fig.dpi / 72.0
    all_ticks, drawn_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for lab in (tick.label1, tick.label2):
                    all_ticks.add(id(lab))
                    if ax.get_visible() and ax.axison and lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                        drawn_ticks.add(id(lab))
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_ticks and id(t) not in drawn_ticks:
            continue                                   # tick label outside the view or on a switched-off axis: not drawn
        ax = getattr(t, 'axes', None)
        if ax is not None and not ax.get_visible():
            continue
        items.append((t.get_text().replace('\n', ' ')[:30], t.get_window_extent(r)))
    fb = fig.bbox
    outside = [s for s, b in items if b.x0 < fb.x0 - tol or b.y0 < fb.y0 - tol or b.x1 > fb.x1 + tol or b.y1 > fb.y1 + tol]
    assert not outside, '%s: text outside the canvas: %s' % (name, outside[:6])
    clash = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
                clash.append((items[i][0], items[j][0]))
    assert not clash, '%s: overlapping text: %s' % (name, clash[:6])
    return len(items)

def save_fig(fig, name, caption):
    assert_min_font(fig, 8.0)
    n_text = assert_layout(fig, name)
    fig.savefig(FIG_DIR / (name + '.pdf'))
    fig.savefig(FIG_DIR / (name + '.png'), dpi=600)
    plt.close(fig)
    FIG_CAPTIONS[name] = caption
    print('saved %s.pdf / .png (layout check passed on %d text items)\n   %s' % (name, n_text, caption))

def shared_range(values, pad=3.0, lo_floor=0.0, hi_ceil=100.0):
    v = np.array([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return lo_floor, hi_ceil
    return max(lo_floor, math.floor(v.min() - pad)), min(hi_ceil, math.ceil(v.max() + pad))

def ci_defined(ci):
    return bool(ci) and ci[0] is not None and ci[1] is not None

def err_from_ci(point, ci):
    """Asymmetric error-bar lengths; zero where a bound is undefined or (rarely) on the wrong side of the point."""
    lo, hi = (ci or [None, None])
    return [[max(0.0, point - lo) if lo is not None else 0.0], [max(0.0, hi - point) if hi is not None else 0.0]]

def marker_style(color, ci):
    """Filled marker with a white edge when the CI is defined; open marker (white face, coloured edge) when it is not."""
    if ci_defined(ci):
        return dict(markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
    return dict(markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.1)

def grid(n, ncols, panel_h, w=None, **kw):
    """Axes grid for n panels of the given height; unused axes are hidden. Returns (fig, list of n axes)."""
    rows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(w or W_DOUBLE, panel_h * rows), constrained_layout=True, squeeze=False, **kw)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]

FIGURE_ERRORS = {}

def run_figure(fn):
    """Draw one figure. A layout or data problem in it is printed, recorded for the manifest and raised again at the very
    end of the notebook (after the manifest is written); the other figures are still drawn."""
    try:
        fn()
    except Exception as e:
        plt.close('all')
        FIGURE_ERRORS[fn.__name__] = '%s: %s' % (type(e).__name__, e)
        print('!' * 78 + '\n!!! FIGURE NOT WRITTEN (%s): %s\n' % (fn.__name__, FIGURE_ERRORS[fn.__name__]) + '!' * 78)

# a failed gate does not stop the notebook here: the figure cells are skipped and the provenance and manifest cells still run
DRAW_FIGURES = not (FIG_VALIDATION_OK is False and not FORCE_FIGURES)
if not DRAW_FIGURES:
    print('!' * 78 + '\n!!! THE VALIDATION GATE FAILED: the figures are not drawn (set FORCE_FIGURES = True only for diagnostic use).\n'
          '!!! The provenance and manifest cells below still run, so run_manifest.json records the failure.\n' + '!' * 78)
if FIG_VALIDATION_OK is None:
    print('NOTE: figures are drawn from a reduced, non-gated run (DRY_RUN or LIMIT_RECORDINGS).')
print('figure style set: %.2f in single, %.2f in double column; minimum font 8 pt and the layout check are asserted on save | inputs read from results.json: %d tasks %s, %d arms, %d kept recordings' % (
    W_SINGLE, W_DOUBLE, len(FIG_TASKS), FIG_TASKS, len(FIG_ARMS), len(FIG_KEEP)))
''')

code('s10_figures_main', r'''
def fig_F1():
    """Forest plot: macro-F1 per arm and task (MLP)."""
    A = RES['arms']
    vals = [A[t][a]['mlp']['macro_f1'] for t in FIG_TASKS for a in FIG_ARMS] + [c for t in FIG_TASKS for a in FIG_ARMS for c in A[t][a]['mlp']['macro_f1_ci'] if c is not None]
    xlo, xhi = shared_range(vals, pad=4)
    fig, axes = grid(len(FIG_TASKS), len(FIG_TASKS), 2.7, sharex=True)
    ypos = np.arange(len(FIG_ARMS))[::-1]
    for k, (ax, t) in enumerate(zip(axes, FIG_TASKS)):
        for yp, a in zip(ypos, FIG_ARMS):
            e = A[t][a]['mlp']
            ax.errorbar(e['macro_f1'], yp, xerr=err_from_ci(e['macro_f1'], e['macro_f1_ci']), fmt=ARM_MARKER[a], color=ARM_COLOR[a],
                        ms=4.5, capsize=2, elinewidth=0.9, **marker_style(ARM_COLOR[a], e['macro_f1_ci']))
        ax.axvline(A[t][FIG_ARMS[0]]['majority']['macro_f1'], color=GREY_M, ls=':', lw=0.9, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([FIG_ARM_LABEL[a] for a in FIG_ARMS] if k == 0 else [])
        ax.set_xlim(xlo, xhi)
        ax.set_title('%s\nn = %d' % (TT[t], FIG_N[t]))
        ax.set_xlabel('macro-F1 (%)')
        ax.grid(axis='x', alpha=0.25, lw=0.4)
        panel_letter(ax, k)
    fig.legend(handles=[Line2D([], [], color=GREY_M, ls=':', lw=0.9, label='majority-class macro-F1')], loc='outside lower center', frameon=False)
    save_fig(fig, 'F1_forest_macro_f1', 'Macro-F1 per arm and task, MLP, pooled out-of-fold, 95%% patient-bootstrap CI (open marker = CI undefined, fewer than 10 valid draws); dotted line = majority-class macro-F1; E2 = equal-time control with the annotated cycle count, not the annotated rate; n = %s' % n_caption())

def fig_F2():
    """Paired differences against the equal-time control (E)."""
    keys = ['A-E', 'B-E', 'C-E', 'D-E', 'E2-E']
    PD = RES['paired']
    dvals = [PD[t][k]['mlp']['macro_f1']['point'] for t in FIG_TASKS for k in keys] + \
            [c for t in FIG_TASKS for k in keys for c in PD[t][k]['mlp']['macro_f1']['ci'] if c is not None]
    ylo, yhi = shared_range(dvals, pad=2, lo_floor=-100, hi_ceil=100)
    fig, axes = grid(len(FIG_TASKS), len(FIG_TASKS), 2.5, sharey=True)
    for k, (ax, t) in enumerate(zip(axes, FIG_TASKS)):
        for i, key in enumerate(keys):
            a = PD[t][key]['arms'][0]
            m = PD[t][key]['mlp']['macro_f1']
            ax.errorbar(i, m['point'], yerr=err_from_ci(m['point'], m['ci']), fmt=ARM_MARKER[a], color=ARM_COLOR[a], ms=5, capsize=2, elinewidth=0.9,
                        **marker_style(ARM_COLOR[a], m['ci']))
        ax.axhline(0, color=GREY_D, lw=0.7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([key.split('-')[0] for key in keys])
        ax.set_ylim(ylo, yhi)
        ax.set_title('%s\nn = %d' % (TT[t], FIG_N[t]))
        ax.set_xlabel('arm minus E')
        if k == 0:
            ax.set_ylabel('macro-F1 difference (points)')
        ax.grid(axis='y', alpha=0.25, lw=0.4)
        panel_letter(ax, k)
    others = sorted(set(PD[FIG_TASKS[0]]) - set(keys))
    save_fig(fig, 'F2_paired_vs_equal_time', 'Paired patient-bootstrap differences in MLP macro-F1 against the 75 bpm equal-time control (E): the pairs %s, 95%% CI (open marker = CI undefined). The other pairs (%s; C-D = annotated phases against fixed fractions on the same cycle starts, A-D = synthetic against annotated anchors under the same fixed fractions) are in results.json under paired; n = %s' % (
        ', '.join(keys), ', '.join(others), n_caption()))

def fig_F3():
    """Timing budget: arm C under boundary jitter, with arms A and B placed on the curve."""
    A = RES['arms']
    TBd = RES['ablations']['timing_budget']
    TB, SIG = TBd['curves'], [int(s) for s in TBd['sigmas_ms']]
    yv = [TB[t][str(s)]['macro_f1'] for t in FIG_TASKS for s in SIG] + [c for t in FIG_TASKS for s in SIG for c in TB[t][str(s)]['macro_f1_ci'] if c is not None] + \
         [A[t][a]['mlp']['macro_f1'] for t in FIG_TASKS for a in ('syn_fixed', 'syn_fusion', 'equal_time')]
    ylo, yhi = shared_range(yv, pad=3)
    ncol = 2 if len(FIG_TASKS) > 1 else 1
    fig, axes = grid(len(FIG_TASKS), ncol, 2.3, sharex=True, sharey=True)
    iqr = FIG_DIAG['syn_fixed'].get('abs_offset_iqr_ms') or [None, None]
    SD_LS = {'syn_fixed': ':', 'syn_fusion': (0, (1, 3))}      # arm B: loosely dotted, never arm A's dots
    for k, (ax, t) in enumerate(zip(axes, FIG_TASKS)):
        xs = np.array(SIG, float)
        ys = np.array([TB[t][str(s)]['macro_f1'] for s in SIG])
        lo = np.array([TB[t][str(s)]['macro_f1_ci'][0] if TB[t][str(s)]['macro_f1_ci'][0] is not None else np.nan for s in SIG])
        hi = np.array([TB[t][str(s)]['macro_f1_ci'][1] if TB[t][str(s)]['macro_f1_ci'][1] is not None else np.nan for s in SIG])
        ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.12, lw=0)
        ax.plot(xs, ys, '-D', color=BLUE, ms=4, markerfacecolor='white', markeredgewidth=1.1)
        if iqr[0] is not None:
            ax.axvspan(iqr[0], iqr[1], color=AMBER, alpha=0.15, lw=0)
        for a in ('syn_fixed', 'syn_fusion'):
            lvl = A[t][a]['mlp']['macro_f1']
            ax.axhline(lvl, color=ARM_COLOR[a], ls=ARM_LS[a], lw=1.0)
            sd = FIG_DIAG[a].get('offset_sd_ms')
            if sd is not None:
                ax.axvline(sd, color=ARM_COLOR[a], ls=SD_LS[a], lw=1.0)
            se = TB[t].get('equivalent_sigma_ms', {}).get(a)
            if se is not None and np.isfinite(se) and se <= max(SIG):
                ax.plot([se], [lvl], marker=ARM_MARKER[a], color=ARM_COLOR[a], ms=5.5, markerfacecolor='white', markeredgewidth=1.2, ls='none', zorder=5)
        ax.axhline(A[t]['equal_time']['mlp']['macro_f1'], color=GREY_M, ls=ARM_LS['equal_time'], lw=0.9)
        ax.set_xticks(SIG)
        ax.set_xticklabels(['' if s == 10 else str(s) for s in SIG])       # the 10 ms tick stays, its label would touch '20'
        ax.set_ylim(ylo, yhi)
        ax.set_title('%s (n=%d)' % (TT[t], FIG_N[t]))
        if k % ncol == 0:
            ax.set_ylabel('macro-F1 (%)')
        if k >= len(FIG_TASKS) - ncol:
            ax.set_xlabel('boundary jitter sigma (ms)')
        ax.grid(alpha=0.25, lw=0.4)
        panel_letter(ax, k)
    H = [Line2D([], [], color=BLUE, marker='D', markerfacecolor='white', markeredgewidth=1.1, label='C + jitter (95% CI band)'),
         Line2D([], [], color=AMBER, ls=ARM_LS['syn_fixed'], label='A: macro-F1'),
         Line2D([], [], color=AMBER, ls=SD_LS['syn_fixed'], lw=1.0, label='A: SD of signed offset'),
         Patch(facecolor=AMBER, alpha=0.15, label='A: IQR of median |offset|'),
         Line2D([], [], color=AMBER, marker='o', markerfacecolor='white', markeredgewidth=1.2, ls='none', label='A: equivalent sigma'),
         Line2D([], [], color=AMBER_DARK, ls=ARM_LS['syn_fusion'], label='B: macro-F1'),
         Line2D([], [], color=AMBER_DARK, ls=SD_LS['syn_fusion'], lw=1.0, label='B: SD of signed offset'),
         Line2D([], [], color=AMBER_DARK, marker='s', markerfacecolor='white', markeredgewidth=1.2, ls='none', label='B: equivalent sigma'),
         Line2D([], [], color=GREY_M, ls=ARM_LS['equal_time'], label='E: equal-time 75 bpm')]
    fig.legend(handles=H, loc='outside lower center', ncol=3, frameon=False)
    save_fig(fig, 'F3_timing_budget', 'MLP macro-F1 as arm C boundaries are jittered (sigma 0, 10, 20, 40, 80, 160 ms; the 10 ms tick is unlabelled). Horizontal lines: arm A and B levels; open markers on the curve: the sigma at which the curve equals each level (equivalent sigma, the valid placement); dotted verticals: SD of the signed matched offset of each arm (the median |offset| is not a sigma); shaded band: IQR of arm A\'s per-recording median |offset|; n = %s' % n_caption())

def fig_F4():
    """Boundary-offset histograms. Arms C and D start every cycle at the annotated S1 onset and have no panel."""
    harms = [a for a in ['syn_fixed', 'syn_fusion', 'equal_time', 'equal_time_matched'] if a in FIG_OFFSETS]
    pooled = {a: np.concatenate([np.asarray(FIG_OFFSETS[a][n], float) for n in FIG_KEEP if len(FIG_OFFSETS[a][n])]) if any(len(FIG_OFFSETS[a][n]) for n in FIG_KEEP) else np.array([]) for a in harms}
    edges = np.linspace(-FIG_MATCH_MS, FIG_MATCH_MS, 31)
    ncol_h = 3
    rows_h = int(math.ceil((len(harms) + 1) / ncol_h))
    fig, axg = plt.subplots(rows_h, ncol_h, figsize=(W_DOUBLE, 2.1 * rows_h), constrained_layout=True, sharey=True, squeeze=False)
    flat = axg.ravel().tolist()
    ymax = 0
    for k, (ax, a) in enumerate(zip(flat, harms)):
        v = pooled[a]
        if len(v):
            cnt, _, _ = ax.hist(v, bins=edges, weights=np.full(len(v), 100.0 / len(v)), color=ARM_COLOR[a], edgecolor='white', lw=0.3, hatch=ARM_HATCH.get(a))
            ymax = max(ymax, cnt.max())
            ax.axvline(np.median(v), color=GREY_D, lw=0.9, ls='--')
        ax.axvline(0, color=GREY_D, lw=0.6)
        ax.set_title('%s\nn = %d matched starts' % (FIG_ARM_LABEL[a], len(v)))
        panel_letter(ax, k, x=-0.10)
        if k % ncol_h == 0:
            ax.set_ylabel('matched cycle starts (%)')
        ax.set_xlim(-FIG_MATCH_MS, FIG_MATCH_MS)
    for ax in flat[:len(harms)]:
        ax.set_ylim(0, max(1.0, math.ceil(ymax * 1.1)))
    info = flat[len(harms)]
    info.axis('off')
    info.legend(handles=[Line2D([], [], color=GREY_D, lw=0.6, label='zero offset (annotated S1 onset)'),
                         Line2D([], [], color=GREY_D, lw=0.9, ls='--', label='median offset')], loc='upper left', frameon=False)
    info.text(0.02, 0.55, 'Arms C and D start every cycle at\nthe annotated S1 onset (offset 0\nby construction) and have no panel.',
              transform=info.transAxes, va='top', ha='left', fontsize=8, color=GREY_D)
    for ax in flat[len(harms) + 1:]:
        ax.set_visible(False)
    fig.supxlabel('offset of the cycle start to the annotated S1 onset (ms)', fontsize=9)
    save_fig(fig, 'F4_offset_histograms', 'Signed offset of each arm\'s cycle start to the annotated S1 onset for Hungarian-matched pairs within %d ms (negative = arm precedes S1); solid line = zero offset, dashed = median; n = matched pairs over %d recordings. Arms C and D start every cycle at the annotated S1 onset (offset 0 by construction) and have no panel. Arm B anchors sit at the 50%% envelope crossing, about 17 ms after the annotated S1 onset (its S2 onset 18 to 22 ms late and mildly murmur-dependent: 23%% vs 10%% of cycles more than 40 ms late). E2 inherits the annotated cycle count, not the rate, so its starts drift wherever annotation coverage is partial.' % (int(FIG_MATCH_MS), len(FIG_KEEP)))

def fig_F5():
    """Subgroups: heart-rate tertiles (from arm C and from arm A) and CirCor age band."""
    SGd = RES['ablations']['subgroups']
    SG, SUBA = SGd['per_task'], list(SGd['arms'])
    MINP = int(SGd.get('min_patients_per_stratum', 10))
    strat = [s for s in ('hr_tertile', 'hr_tertile_arm_a', 'age_band') if all(s in SG[t] for t in FIG_TASKS)]
    STRAT_LABEL = {'hr_tertile': 'tertile of heart rate from PCG-segmentation boundaries (arm C)',
                   'hr_tertile_arm_a': 'tertile of heart rate from synthetic R-peaks (arm A)', 'age_band': 'CirCor age band'}
    ABBR = {'Neonate': 'Neo', 'Infant': 'Inf', 'Child': 'Chi', 'Adolescent': 'Ado', 'Young Adult': 'YA', 'Unknown': 'Unk'}

    def shown(t, s):                                 # strata with an estimate for every arm (at least MINP patients)
        return [l for l, v in SG[t][s]['levels'].items() if all(a in v for a in SUBA)]

    sv = [SG[t][s]['levels'][l][a]['macro_f1'] for t in FIG_TASKS for s in strat for l in shown(t, s) for a in SUBA] + \
         [c for t in FIG_TASKS for s in strat for l in shown(t, s) for a in SUBA for c in SG[t][s]['levels'][l][a]['macro_f1_ci'] if c is not None]
    ylo, yhi = shared_range(sv, pad=3)
    ytop = yhi + 2.0 if yhi >= 100 else yhi              # head-room so an error bar that reaches 100 ends in a visible cap
    fig = plt.figure(figsize=(W_DOUBLE, 2.45 * len(strat) + 0.35), constrained_layout=True)
    subfigs = np.atleast_1d(fig.subfigures(len(strat) + 1, 1, height_ratios=[2.45] * len(strat) + [0.35]))
    wbar = 0.26
    for r_, sname in enumerate(strat):
        axs = subfigs[r_].subplots(1, len(FIG_TASKS), sharey=True, squeeze=False)[0]
        for c_, t in enumerate(FIG_TASKS):
            ax = axs[c_]
            lev = SG[t][sname]['levels']
            levels = shown(t, sname)
            dropped = [l for l in lev if l not in levels]
            xs = np.arange(len(levels))
            for j, a in enumerate(SUBA):
                if not levels:
                    continue
                pts = np.array([lev[l][a]['macro_f1'] for l in levels], float)
                cis = [lev[l][a]['macro_f1_ci'] for l in levels]
                err = np.array([[max(0.0, p - ci[0]) if ci_defined(ci) else 0.0 for p, ci in zip(pts, cis)],
                                [max(0.0, ci[1] - p) if ci_defined(ci) else 0.0 for p, ci in zip(pts, cis)]])
                bars = ax.bar(xs + (j - 1) * wbar, pts, width=wbar, color=ARM_COLOR[a], yerr=err, capsize=1.5,
                              error_kw=dict(lw=0.6, ecolor=GREY_D), edgecolor='white', lw=0.4, hatch=ARM_HATCH.get(a))
                for bar_, ci in zip(bars, cis):
                    if not ci_defined(ci):
                        bar_.set_alpha(0.45)
            fits = len(levels) * max([len(l) for l in levels] or [0]) <= 21               # about 21 characters of 8 pt text span one panel
            names = levels if fits else [ABBR.get(l, l) for l in levels]                  # full names where they fit, else the abbreviations of the caption
            labs = []
            for l, nm in zip(levels, names):
                lacks_class = any(int(c) == 0 for c in lev[l].get('class_counts', []))    # fixed-label macro-F1 has a ceiling there
                labs.append('%s%s\n%d' % (nm, '*' if lacks_class else '', lev[l]['n']))
            ax.set_xticks(xs)
            ax.set_xticklabels(labs)
            ax.set_xlim(-0.6, max(len(levels), 1) - 0.4)
            ax.set_ylim(ylo, ytop)
            ax.set_yticks([v for v in range(0, 101, 20) if ylo <= v <= ytop])           # the same ticks in every row
            if dropped:                                  # strata under the reporting threshold are not drawn; they are named under the panel
                lines, cur = ['under %d patients:' % MINP], ''
                for it in ['%s %d' % (l, lev[l]['n']) for l in dropped]:
                    cand = (cur + ', ' + it) if cur else it
                    if cur and len(cand) > 22:
                        lines.append(cur + ',')
                        cur = it
                    else:
                        cur = cand
                lines.append(cur)
                ax.set_xlabel('\n'.join(lines), fontsize=8, color=GREY_M)
            if r_ == 0:
                ax.set_title(TT[t])
            if c_ == 0:
                ax.set_ylabel('macro-F1 (%)')
            ax.grid(axis='y', alpha=0.25, lw=0.4)
            panel_letter(ax, r_ * len(FIG_TASKS) + c_)
        subfigs[r_].supxlabel(STRAT_LABEL[sname], fontsize=9)
    subfigs[-1].legend(handles=[Patch(facecolor=ARM_COLOR[a], hatch=ARM_HATCH.get(a), edgecolor='white', label=FIG_ARM_LABEL[a]) for a in SUBA],
                       loc='center', ncol=3, frameon=False)
    save_fig(fig, 'F5_subgroups', 'MLP macro-F1 of arms A, B and C by tertile of heart rate (from the PCG-segmentation boundaries of arm C and, as a second stratifier, from the synthetic R-peaks of arm A) and by CirCor age band, 95%% CI; the number under each level is its n; strata with fewer than %d patients have no estimate, are not drawn and are listed under the panel with their n; * = the stratum lacks a class, so its fixed-label macro-F1 has a ceiling below 100; Neo, Inf, Chi, Ado, YA, Unk = Neonate, Infant, Child, Adolescent, Young Adult, age not recorded; paler bars = CI undefined; total n = %s' % (MINP, n_caption()))

def fig_F6():
    """Window length: detection with the boundary-free features on the first 4, 8, 16 s and the full recording."""
    WLd = RES['ablations']['window_length']
    WL = WLd['results']
    tags = list(WL.keys())
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.5), constrained_layout=True)
    # the boundary-free features belong to no arm: the classifiers are drawn in black, the arm colours are kept for the reference lines
    for clf, col, mk, ls_, dx in [('mlp', 'black', 'D', '-', -0.05), ('rf', 'black', 'o', '--', 0.05)]:
        pts = [WL[t][clf]['macro_f1'] for t in tags]
        cis = [WL[t][clf]['macro_f1_ci'] for t in tags]
        err = np.array([[max(0.0, p - ci[0]) if ci_defined(ci) else 0.0 for p, ci in zip(pts, cis)], [max(0.0, ci[1] - p) if ci_defined(ci) else 0.0 for p, ci in zip(pts, cis)]])
        ax.errorbar(np.arange(len(tags)) + dx, pts, yerr=err, fmt=mk, ls=ls_, color=col, ms=4, capsize=2, elinewidth=0.9, label=clf.upper() + ', %d boundary-free features' % WLd['feature_count'])
        for i, (p, ci) in enumerate(zip(pts, cis)):
            if not ci_defined(ci):
                ax.plot([i + dx], [p], marker=mk, color=col, markerfacecolor='white', markeredgewidth=1.1, ms=4, ls='none', zorder=5)
    ax.axhline(WLd['reference_arm_c_mlp'], color=ARM_COLOR['pcg_seg'], ls=(0, (6, 2)), lw=1.0, label='arm C, all 464 features (MLP)')
    ax.axhline(WLd['reference_arm_e_mlp'], color=ARM_COLOR['equal_time'], ls=ARM_LS['equal_time'], lw=1.0, label='arm E, all 464 features (MLP)')
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(['%d s' % WL[t]['window_s'] if WL[t]['window_s'] is not None else 'full' for t in tags])
    ax.set_xlabel('analysed window (first N seconds)')
    ax.set_ylabel('detection macro-F1 (%)')
    wv = [WL[t][c]['macro_f1'] for t in tags for c in ('mlp', 'rf')] + [x for t in tags for c in ('mlp', 'rf') for x in WL[t][c]['macro_f1_ci'] if x is not None] + \
         [WLd['reference_arm_c_mlp'], WLd['reference_arm_e_mlp']]
    ax.set_ylim(*shared_range(wv, pad=3))
    fig.legend(loc='outside lower center', ncol=1, frameon=False)
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    save_fig(fig, 'F6_window_length', 'Murmur detection with the %d boundary-free features on the first 4, 8, 16 s and the full recording (audio renormalised after truncation), MLP and RF, 95%% CI (open marker = CI undefined); blue dashed and grey dash-dot lines = arms C and E with all 464 features (MLP); n = %d' % (WLd['feature_count'], FIG_N['detection']))

if DRAW_FIGURES:
    with stage('S10 figures'):
        for _fig in (fig_F1, fig_F2, fig_F3, fig_F4, fig_F5, fig_F6):
            run_figure(_fig)
else:
    print('S10 figures skipped: the validation gate failed and FORCE_FIGURES is False. The provenance and manifest cells still run.')
''')

code('s10_figures_supplement', r'''
from matplotlib.colors import LinearSegmentedColormap
cmap_blue = LinearSegmentedColormap.from_list('blue', ['#ffffff', BLUE])

def fig_F7():
    """Confusion matrices for arms A, B, C: arm as column header, task as row label, letter and macro-F1 above each panel."""
    A = RES['arms']
    carms = ['syn_fixed', 'syn_fusion', 'pcg_seg']
    fig, axg = plt.subplots(len(FIG_TASKS), len(carms), figsize=(W_DOUBLE, 2.2 * len(FIG_TASKS)), constrained_layout=True, squeeze=False)
    for r_, t in enumerate(FIG_TASKS):
        for c_, a in enumerate(carms):
            ax = axg[r_, c_]
            cm = np.array(A[t][a]['mlp']['confusion_matrix'], float)
            rowsum = np.maximum(cm.sum(1, keepdims=True), 1)
            pct_ = 100 * cm / rowsum
            ax.imshow(pct_, cmap=cmap_blue, vmin=0, vmax=100)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    # white text only where it out-contrasts the dark grey on the blue ramp
                    ax.text(j, i, '%.0f%%\n(%d)' % (pct_[i, j], cm[i, j]), ha='center', va='center', fontsize=8, color='white' if pct_[i, j] > 62 else GREY_D)
            names = CLASS_NAMES[t]
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names)
            ax.set_yticklabels(names if c_ == 0 else [])
            ax.set_title('ABCDEFGHIJKL'[r_ * len(carms) + c_], loc='left', fontweight='bold')
            ax.set_title('macro-F1 %.1f' % A[t][a]['mlp']['macro_f1'], loc='right', fontsize=9)
            if r_ == 0:
                ax.annotate(FIG_ARM_LABEL[a], xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            if r_ == len(FIG_TASKS) - 1:
                ax.set_xlabel('predicted class')
            if c_ == 0:
                ax.set_ylabel('%s\ntrue class' % TT[t])
            for s in ax.spines.values():
                s.set_visible(False)
    save_fig(fig, 'F7_confusion_matrices', 'Pooled out-of-fold confusion matrices (MLP) for arms A, B and C (columns) and each task (rows), row-normalised percentages with counts; shape classes: plat. = plateau, decr. = decrescendo, diam. = diamond; n = %s' % n_caption())

def fig_F8():
    """Redundancy heatmap of the arm C features."""
    R = np.load(CACHE_DIR / 'redundancy_corr_pcg_seg.npy')
    rd = RES['ablations']['redundancy']
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.9), constrained_layout=True)
    im = ax.imshow(np.abs(R), cmap=cmap_blue, vmin=0, vmax=1, interpolation='nearest')
    ticks = list(range(0, R.shape[0], 100))                  # a few index ticks so the axes carry a scale
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('feature index (alphabetical)')
    ax.set_ylabel('feature index (alphabetical)')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('|Pearson r|')
    save_fig(fig, 'F8_redundancy_heatmap', 'Absolute Pearson correlation between the %d arm C features over %d recordings (label-free): %d pairs exceed |r| = 0.9, involving %d features in %d clusters; %d pairs coincide (|r| >= 0.9999); %d features are constant; %d, %d and %d principal components carry 90%%, 95%% and 99%% of the variance' % (
        rd['n_features'], len(FIG_KEEP), rd['n_pairs_abs_r_over_0_9'], rd['n_features_in_pairs_over_0_9'], rd['n_clusters_over_0_9'], rd['n_pairs_abs_r_ge_0_9999'],
        rd['n_constant'], rd['pca_components_90pct'], rd['pca_components_95pct'], rd['pca_components_99pct']))

if DRAW_FIGURES:
    with stage('S10 figures (supplement)'):
        for _fig in (fig_F7, fig_F8):
            run_figure(_fig)
    print('figures written to', FIG_DIR, '| not shown inline: open the PDF or PNG files in the run folder')
else:
    print('S10 supplementary figures skipped: the validation gate failed and FORCE_FIGURES is False.')
''')

md(r'''
## Provenance of the verbatim blocks

Recorded at build time by `build_gating_campaign.py`: the source file, line range or function name,
and the SHA-256 of the exact text that was embedded in this notebook. The run manifest copies it,
together with `NOTEBOOK_SOURCE_SHA256`, the SHA-256 over every other cell's source at build time,
which Stage 11 compares with the live notebook when Colab exposes it.
''')

code('provenance', 'VERBATIM_PROVENANCE = ' + PROV_JSON + '\n'
     "NOTEBOOK_SOURCE_SHA256 = '__NOTEBOOK_SOURCE_SHA256__'   # SHA-256 over every other cell's source, filled in by the builder\n"
     "print('%d verbatim blocks embedded | notebook source sha256 %s' % (len(VERBATIM_PROVENANCE), NOTEBOOK_SOURCE_SHA256[:16]))\n"
     "for _p in VERBATIM_PROVENANCE:\n"
     "    print('   %-40s %-28s %s' % (_p['source'], _p.get('lines') or _p.get('function'), _p['sha256'][:16]))")

md(r'''
## Stage 11: run manifest

Versions, GPU, seeds, run flags, dataset path and pair count, checkpoint SHA-256, the worker-pool
kind, per-stage seconds for this session and cumulative over every session of the run folder, a
git-style blob hash of every cache file (taken at write time; only files written by an earlier
notebook version are hashed here), the configuration, arm and feature-cache hashes, the cache
rebuilds and worker failures, the validation gate result, the verbatim-block provenance, the
build-time hash of the notebook's cell sources (and the live notebook's hash when Colab exposes
it), and which equal-time definition every results entry and figure used.
''')

code('s11_manifest', r'''
def notebook_source_hash(cells):
    """SHA-256 over the sources of every cell except the provenance cell: the rule the builder applies at build time."""
    h = hashlib.sha256()
    for c in cells:
        src = c.get('source', '') if isinstance(c, dict) else c
        src = ''.join(src) if isinstance(src, list) else src
        if src.startswith('VERBATIM_PROVENANCE = '):
            continue
        h.update(src.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()

with stage('S11 manifest'):
    _t11 = time.time()
    live_hash = None
    try:
        from google.colab import _message
        _nb = _message.blocking_request('get_ipynb', request='', timeout_sec=30)
        live_hash = notebook_source_hash(_nb['ipynb']['cells'])
    except Exception as e:
        print('   live notebook hash unavailable outside Colab (%s); the build-time hash is recorded' % type(e).__name__)
    cache_files = sorted(list(CACHE_DIR.glob('*.npz')) + list(CACHE_DIR.glob('*.pkl')) + list(CACHE_DIR.glob('*.npy')) +
                         list(CV_DIR.glob('*.npz')) + list(FOLD_DIR.glob('*.npz')))
    n_now = 0
    for f in cache_files:
        rel = str(f.relative_to(RUN_DIR))
        if rel not in CACHE_HASHES or CACHE_HASHES[rel].get('bytes') != os.path.getsize(f):
            CACHE_HASHES[rel] = dict(git_blob_sha1=git_blob_sha1(f), bytes=os.path.getsize(f))   # only files not hashed at write time
            n_now += 1
    cache_hashes = {str(f.relative_to(RUN_DIR)): CACHE_HASHES[str(f.relative_to(RUN_DIR))] for f in cache_files}
    eq_usage = {}
    for t in TASKS:
        eq_usage['arms.%s.equal_time' % t] = 'equal_time'
        eq_usage['arms.%s.equal_time_matched' % t] = 'equal_time_matched'
        for key in RESULTS['paired'][t]:
            a_, b_ = RESULTS['paired'][t][key]['arms']
            used = [x for x in (a_, b_) if x.startswith('equal_time')]
            if used:
                eq_usage['paired.%s.%s' % (t, key)] = ' and '.join(used)
    eq_usage['validation_gate.equal_time'] = 'equal_time'
    eq_usage['ablations.window_length.reference_arm_e_mlp'] = 'equal_time'
    eq_usage['figures.F1_forest_macro_f1'] = 'equal_time and equal_time_matched'
    eq_usage['figures.F2_paired_vs_equal_time'] = 'equal_time'
    eq_usage['figures.F3_timing_budget.reference_line'] = 'equal_time'
    eq_usage['figures.F4_offset_histograms'] = 'equal_time and equal_time_matched'
    eq_usage['figures.F5_subgroups'] = 'none (arms A, B, C only)'
    eq_usage['figures.F6_window_length.reference_line'] = 'equal_time'
    eq_usage['figures.F7_confusion_matrices'] = 'none (arms A, B, C only)'
    eq_usage['figures.F8_redundancy_heatmap'] = 'none (arm C features)'
    _cum = stage_seconds_cumulative()
    _cum['S11 manifest'] = _cum.get('S11 manifest', 0.0) + (time.time() - _t11)
    MANIFEST = dict(
        created=datetime.datetime.now().isoformat(timespec='seconds'), run_dir=str(RUN_DIR), resumed_from=RESUMED_FROM,
        session_id=SESSION_ID, sessions=STAGE_SESSIONS_PREV + [SESSION_ID], dry_run=DRY_RUN,
        limit_recordings=LIMIT_RECORDINGS, n_boot=N_BOOT, seed=SEED, n_splits=N_SPLITS, gate_tol=GATE_TOL, n_min_cycles=N_MIN_CYCLES, run_flags=RUN_FLAGS,
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                      sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, pandas=pd.__version__, pywt=pywt.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        gpu=GPU_NAME, cpu_count=os.cpu_count(), feature_workers=N_PROC, pool=dict(kind=POOL_KIND, uses=len(POOL_USES)),
        determinism=dict(cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                         matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        dataset=dict(training_data=str(AUDIO_PATH), metadata=str(METADATA_PATH), paired_wav_tsv=N_PAIRS,
                     n_indexed=len(NAMES_ALL), n_kept=len(KEEP), n_dropped=len(DROPPED), n_excluded_stage1=len(EXCLUDED),
                     tasks=list(TASKS), skipped_tasks=SKIPPED_TASKS),
        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),
        generator=dict(batch_start=GAN_BATCH, batch_final=GEN_BATCH_USED, oom_events=OOM_EVENTS, signal_length=SIGNAL_LENGTH, gan_sr=GAN_SR, ecg_cache_dtype='float16'),
        classifier=dict(mlp=MLP_CFG, rf=RF_CFG, reference=REFERENCE_MLP_CFG), fusion=FUSION,
        arm_constants=dict(join_ms=JOIN_MS, realign_ms=REALIGN_MS, match_ms=MATCH_MS, arm_code_version=ARM_CODE_VERSION),
        hashes=dict(cfg_hash=CFG_HASH, arm_cfg_hash=ARM_CFG_HASH, feature_cycles_hashes=FEATURE_CACHE_HASHES,
                    notebook_source_sha256_build=NOTEBOOK_SOURCE_SHA256, notebook_source_sha256_live=live_hash,
                    notebook_matches_build=(live_hash == NOTEBOOK_SOURCE_SHA256) if live_hash else None),
        arm_failures=ARM_FAILURES, cache_rebuilds=CACHE_REBUILDS,
        equal_time_definitions=EQUAL_TIME_DEFINITIONS, equal_time_usage_by_results_entry=eq_usage,
        stage_seconds_this_session=STAGE_SECONDS, stage_seconds_cumulative=_cum,
        cache_files=cache_hashes, cache_files_hashed_at_write_time=len(cache_files) - n_now,
        validation_gate=RESULTS['validation_gate'], verbatim_provenance=VERBATIM_PROVENANCE, figures=FIG_CAPTIONS,
        figures_drawn=DRAW_FIGURES, figure_errors=FIGURE_ERRORS,
        outputs=dict(results='results.json', oof='oof_predictions.npz', diagnostics='diagnostics.csv', index='index.json',
                     arm_failures='arm_failures.json', run_flags='run_flags.json', stage_seconds='stage_seconds.json', cache_hashes='cache_hashes.json',
                     folds=[str(p.name) for p in sorted(FOLD_DIR.glob('*.npz'))]))
    with open(RUN_DIR / 'run_manifest.json', 'w') as fh:
        json.dump(jsonable(MANIFEST), fh, indent=1)
    RESULTS['run']['finished'] = MANIFEST['created']
    write_results()
    print('run_manifest.json written | %d cache files (%d hashed now, the rest at write time) | elapsed this session %.1f min | cumulative over %d session(s) %.1f min' % (
        len(cache_hashes), n_now, sum(STAGE_SECONDS.values()) / 60, len(STAGE_SESSIONS_PREV) + 1, sum(_cum.values()) / 60))
    print('\nstage seconds (this session):')
    for k_, v_ in STAGE_SECONDS.items():
        print('   %-36s %8.1f' % (k_, v_))
    print('\nrun folder:', RUN_DIR)
    print('validation gate:', RESULTS['validation_gate']['ok'], '| decided on:', RESULTS['validation_gate'].get('decided_on'), '| figures drawn:', DRAW_FIGURES,
          '| pool:', POOL_KIND, '| cache rebuilds:', len(CACHE_REBUILDS), '| worker failures:', len(ARM_FAILURES), '| skipped tasks:', list(SKIPPED_TASKS))

# a figure that failed its checks was reported in Stage 10; it is raised here, after the manifest is on disk
if FIGURE_ERRORS:
    raise RuntimeError('figures not written (the manifest records this too): %s' % FIGURE_ERRORS)
''')

# ----------------------------------------------------------------------------------------------
# Assemble, check, write
# ----------------------------------------------------------------------------------------------

def main():
    nb = new_notebook()
    nb.metadata = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
        'colab': {'name': 'R1_Gating_Campaign.ipynb', 'provenance': [], 'machine_shape': 'hm', 'gpuType': 'A100'},
        'built_by': 'build_gating_campaign.py',
        'verbatim_provenance': PROVENANCE,
    }
    n_code = n_md = 0
    tmpdir = tempfile.mkdtemp(prefix='gating_cells_')
    for kind, name, text in CELLS:
        assert '—' not in text, 'em dash found in cell %s' % (name or 'markdown')
        assert isinstance(text, str) and text.strip(), 'empty cell %s' % name
        if kind == 'markdown':
            nb.cells.append(new_markdown_cell(text))
            n_md += 1
        else:
            compile(text, name, 'exec')                       # syntax check in-process
            path = os.path.join(tmpdir, '%02d_%s.py' % (len(nb.cells), name))
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text + '\n')
            py_compile.compile(path, doraise=True)            # and via py_compile, as requested
            cell = new_code_cell(text)
            cell.metadata['name'] = name
            nb.cells.append(cell)
            n_code += 1
    # notebook source hash: every cell except the provenance cell (Stage 11 applies the same rule to the live notebook)
    h = hashlib.sha256()
    for c in nb.cells:
        if c.source.startswith('VERBATIM_PROVENANCE = '):
            continue
        h.update(c.source.encode('utf-8'))
        h.update(b'\x00')
    nb_hash = h.hexdigest()
    n_prov = 0
    for c in nb.cells:
        if c.source.startswith('VERBATIM_PROVENANCE = '):
            assert '__NOTEBOOK_SOURCE_SHA256__' in c.source
            c.source = c.source.replace('__NOTEBOOK_SOURCE_SHA256__', nb_hash)
            n_prov += 1
    assert n_prov == 1, 'exactly one provenance cell expected'
    nb.metadata['notebook_source_sha256'] = nb_hash
    nbformat.validate(nb)
    nbformat.write(nb, str(OUT_NB))
    print('notebook source sha256 (all cells but provenance): %s' % nb_hash)
    print('wrote %s' % OUT_NB)
    print('cells: %d code + %d markdown = %d' % (n_code, n_md, len(nb.cells)))
    print('verbatim blocks: %d' % len(PROVENANCE))
    for p in PROVENANCE:
        print('   %-40s %-28s %s' % (p['source'], p.get('lines') or p.get('function'), p['sha256'][:16]))
    print('cell sources also py_compiled in %s' % tmpdir)


if __name__ == '__main__':
    main()
