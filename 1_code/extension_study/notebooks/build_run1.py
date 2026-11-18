#!/usr/bin/env python
"""
build_run1.py: assembles Extension_Run1.ipynb (08_EXTENSION_DESIGN.md, Run 1) from the R1 gating-campaign builder.

The R1 builder (Revision-R1/notebooks/build_gating_campaign.py) is imported unchanged; its cells are checked to
reproduce the R1 notebook byte for byte (notebook source hash), then patched by name. Every patch asserts that its
anchor text occurs exactly once, so a change in the R1 builder stops this script instead of producing a silent
mismatch.

Run 1 changes (and nothing else):
  1. run folder JBHI-Revision/Extension_Study/runs/run1/<timestamp>; run flags record the run kind and seeds;
  2. groups = children (Patient ID union Additional ID, 816); the fold assignment of every task is embedded (built
     locally by extstudy/splits.py, SHA-256 in runs/folds_child/folds_manifest.json) and checked against the notebook's
     own names, labels, and child IDs before use (full run); reduced runs draw child-grouped folds as R1 did;
  3. the kept set is computed from the six R1 arms and must equal the 3,000 R1 recordings (full run);
  4. new arms: Springer S_wpk and S_pub (Panel E's GPL-3 port, embedded, run on the notebook's own analysed audio,
     label-free; fewer than 2 cycles falls back to equal-time windows and is counted; more than 5% = failed arm) and
     the matched-coverage copies S_wpk_annot and E_annot (selection to the annotated stretches only, like E2);
  5. the MLP's probabilities are averaged over five seed bases (42, 1042, 2042, 3042, 4042) plus the fold index;
     per-seed probabilities are saved;
  6. the R1 validation gate (reference values from patient-grouped folds and a single seed) is replaced by a
     label-blind integrity gate: pinned kept set, embedded folds, zero child overlap, R1 feature matrices reproduced
     when the R1 run folder is reachable, and a repeat fit that returns identical probabilities;
  7. the R1 reviewer ablations (Stage 9) and R1 figures (Stage 10) are not run; the composite endpoint, shared
     child bootstrap, and hypothesis tests are computed locally from oof_predictions.npz (extstudy/stats.py);
  8. a DRY_RUN or LIMIT_RECORDINGS run stops before Stage 7: one logged scoring per arm, in the full run only.

Usage:  python build_run1.py   (writes Extension_Run1.ipynb beside this file)
"""
import base64, hashlib, importlib.util, json, os, sys, zlib
from pathlib import Path

import numpy as np
import nbformat

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
R1_BUILDER = Path('E:/NUST MS Thesis/Final Draft/Submitted File/Revision-R1/notebooks/build_gating_campaign.py')
R1_NOTEBOOK = R1_BUILDER.parent / 'R1_Gating_Campaign.ipynb'
OUT_NB = HERE / 'Extension_Run1.ipynb'
FOLD_DIR = STUDY / 'runs' / 'folds_child'
SPRINGER_DIR = STUDY / 'extstudy' / 'springer'
R1_RUN_ON_DRIVE = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ load the R1 builder and check its base
spec = importlib.util.spec_from_file_location('r1_builder', str(R1_BUILDER))
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)                         # populates r1.CELLS; reads the analysis sources it slices from
CELLS = r1.CELLS


def source_hash(cells):
    h = hashlib.sha256()
    for kind, name, text in cells:
        if text.startswith('VERBATIM_PROVENANCE = '):
            continue
        h.update(text.encode('utf-8')); h.update(b'\x00')
    return h.hexdigest()


R1_META = json.load(open(R1_NOTEBOOK, encoding='utf-8'))['metadata']
BASE_HASH = source_hash(CELLS)
assert BASE_HASH == R1_META['notebook_source_sha256'], 'the R1 builder no longer reproduces R1_Gating_Campaign.ipynb'
print('R1 base reproduced: notebook source sha256 %s' % BASE_HASH)


def idx(name):
    hits = [i for i, c in enumerate(CELLS) if c[1] == name]
    assert len(hits) == 1, 'cell %s: %d hits' % (name, len(hits))
    return hits[0]


def patch(name, old, new):
    i = idx(name)
    kind, nm, text = CELLS[i]
    assert text.count(old) == 1, 'cell %s: anchor found %d times: %r' % (name, text.count(old), old[:80])
    CELLS[i] = (kind, nm, text.replace(old, new))


def insert_after(name, cells):
    i = idx(name)
    CELLS[i + 1:i + 1] = cells


# ------------------------------------------------------------------ embedded data: folds, pinned kept set, Springer
man = json.load(open(FOLD_DIR / 'folds_manifest.json'))
folds = {}
for t in ('detection', 'grading', 'timing', 'shape'):
    z = np.load(FOLD_DIR / (t + '.npz'), allow_pickle=False)
    folds[t] = dict(names=z['names'].tolist(), child=z['patient_ids'].tolist(), pid=z['patient_ids_original'].tolist(),
                    y=z['y'].tolist(), fold_id=z['fold_id'].tolist(), sha256=man[t]['sha256'], children=man[t]['children'])
PINNED_KEEP = np.load(FOLD_DIR / 'detection.npz')['keep_names'].tolist()
assert all(np.load(FOLD_DIR / (t + '.npz'))['names'].dtype == np.dtype('<U11') for t in folds)
assert len(PINNED_KEEP) == 3000
FOLD_B64 = base64.b64encode(zlib.compress(json.dumps(dict(folds=folds, pinned_keep=PINNED_KEEP)).encode(), 9)).decode()
FOLD_B64_SHA = sha256_bytes(FOLD_B64.encode())

SPRINGER_SRC = (SPRINGER_DIR / 'springer_py.py').read_text(encoding='utf-8')
SPRINGER_B64 = base64.b64encode(SPRINGER_SRC.encode('utf-8')).decode()
PARAMS = {f: base64.b64encode((SPRINGER_DIR / 'params' / f).read_bytes()).decode()
          for f in ('Springer_B_matrix.mat', 'Springer_pi_vector.mat', 'Springer_total_obs_distribution.mat')}
PARAM_SHA = {f: sha256_bytes(base64.b64decode(v)) for f, v in PARAMS.items()}
SPRINGER_SHA = sha256_bytes(SPRINGER_SRC.encode('utf-8'))

# ------------------------------------------------------------------ 1. header and flags
CELLS[0] = ('markdown', None, r'''# Extension Run 1: child-grouped folds, five-seed ensembles, and the Springer arms

Run 1 of the EHJ-DH extension (`08_EXTENSION_DESIGN.md`). Built by `extension_study/notebooks/build_run1.py` from the
R1 gating-campaign notebook, whose cells are reproduced byte for byte and then patched; the patch list is in that
script's docstring. Everything the R1 notebook held fixed is still fixed: the shared analysed audio, the 464-feature
extractor, the classifier configuration, and the cycle format.

**What changes.** Folds are grouped by child (Patient ID union Additional ID: 816 children; the R1 folds split 54 of
58 children enrolled twice), and their assignment is embedded and verified. The MLP's probabilities are averaged over
five seed bases. Two Springer arms run on the same audio, label-free: **S_wpk** (published parameters, heart-rate
search 40 to 220 bpm at the highest local maximum, systole floor 0.10 s; the primary automatic comparator) and
**S_pub** (published heart-rate rule). Two matched-coverage copies keep only cycles inside the annotated stretches:
**S_wpk_annot** and **E_annot**. The R1 validation gate is replaced by a label-blind integrity gate; the R1
ablations and figures are not rerun.

**How to run.** Runtime > Change runtime type > A100 GPU with High-RAM; set `DRY_RUN` in the next cell; Runtime > Run
all. The cell under "Environment: libraries, Drive, GPU" mounts Drive, installs the one package Colab lacks
(PyWavelets), and imports everything else (numpy, pandas, scipy, scikit-learn, PyTorch, tqdm, psutil, matplotlib, all
preinstalled on Colab). Nothing else needs installing. A dry run stops before Stage 7 by design.

**What is computed elsewhere.** The composite macro-F1 (mean of grading, timing, and shape), the shared child-level
bootstrap, and the hypothesis tests run locally on `oof_predictions.npz` (`extstudy/stats.py`), and every scoring
is logged in the append-only ledger.

| arm | key | boundaries |
|---|---|---|
| A | `syn_fixed` | R-peaks from the reconstructed ECG, fixed 15/35/10/40 %RR phases |
| B | `syn_fusion` | synthetic R-peaks refined by the acoustic S1/S2 envelope |
| C | `pcg_seg` | the dataset's annotation (reference only) |
| D | `pcg_seg_fixed` | annotated S1 onsets, fixed-fraction phases |
| E | `equal_time` | 75 bpm pseudo-cycles (control) |
| E2 | `equal_time_matched` | E with the annotated cycle count |
| S | `springer_wpk` | Springer LR-HSMM, 40 to 220 bpm highest local maximum, systole floor 0.10 s |
| S_pub | `springer_pub` | Springer with the published heart-rate rule |
| S_annot | `springer_wpk_annot` | S cycles inside the annotated stretches |
| E_annot | `equal_time_annot` | E cycles inside the annotated stretches |
''')

PREREG = STUDY / 'prereg' / 'PREREG_EXT.md'
PREREG_SHA = sha256_bytes(PREREG.read_bytes())
assert open(STUDY / 'prereg' / 'PREREG_EXT.sha256').read().split()[0] == PREREG_SHA, 'PREREG_EXT.md changed after it was frozen'
patch('flags', "SEED = 42\n", """SEED = 42
PREREG_SHA256 = '%s'   # extension_study/prereg/PREREG_EXT.md, frozen 16 Oct 2026 before any Run 1 classification
ENSEMBLE_SEEDS = (42, 1042, 2042, 3042, 4042)   # MLP probabilities averaged over these seed bases plus the fold index (08 design 2.2)""" % PREREG_SHA + """
R1_RUN_DIR = '%s'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip
S_FALLBACK_FAIL = 0.05    # an arm with more than 5%% of recordings on the equal-time fallback is reported as failed
""" % R1_RUN_ON_DRIVE)

STAGE_SRC = r'''
    # ---- Extension Run 1: stage the dataset on the Colab disk. Drive's mount dropped ("Transport endpoint is not
    # connected") when the shared-audio stage read thousands of WAV files from it in parallel; the copy runs one file at a
    # time with retries and a forced remount, sizes are checked, and every later read uses the local copy ----
    import shutil as _shutil, errno as _errno
    def _drive_retry(fn, *a, tries=6):
        for _k in range(tries):
            try:
                return fn(*a)
            except OSError as _e:
                if _k == tries - 1:
                    raise
                if _e.errno in (_errno.ENOTCONN, _errno.ECONNABORTED, _errno.EIO) and IN_COLAB:
                    try:
                        from google.colab import drive as _gd
                        _gd.mount('/content/drive', force_remount=True)
                    except Exception as _me:
                        print('   remount attempt failed:', type(_me).__name__)
                time.sleep(3 * (_k + 1))
    AUDIO_PATH_DRIVE = AUDIO_PATH
    if str(AUDIO_PATH).startswith('/content/drive'):
        _local = Path('/content/circor_local/training_data')
        _local.mkdir(parents=True, exist_ok=True)
        _src = _drive_retry(lambda: sorted([p for p in AUDIO_PATH.iterdir() if p.suffix in ('.wav', '.tsv')], key=lambda p: p.name))
        _t0, _n_new = time.time(), 0
        for _p in tqdm(_src, desc='staging CirCor on the Colab disk', leave=False):
            _q = _local / _p.name
            _size = _drive_retry(lambda: _p.stat().st_size)
            if _q.exists() and _q.stat().st_size == _size:
                continue
            _drive_retry(_shutil.copyfile, _p, _q)
            assert _q.stat().st_size == _size, 'size mismatch after copy: %s' % _p.name
            _n_new += 1
        _drive_retry(_shutil.copyfile, METADATA_PATH, _local.parent / 'training_data.csv')
        assert paired_count(_local) == N_PAIRS, 'local copy has %d pairs, Drive has %d' % (paired_count(_local), N_PAIRS)
        AUDIO_PATH, BASE_PATH = _local, _local.parent
        METADATA_PATH = BASE_PATH / 'training_data.csv'
        print('dataset staged on the Colab disk: %d files (%d copied now) in %.0f s -> %s' % (len(_src), _n_new, time.time() - _t0, AUDIO_PATH))
'''
patch('resolve', "    METADATA_PATH = BASE_PATH / 'training_data.csv'\n", "    METADATA_PATH = BASE_PATH / 'training_data.csv'\n" + STAGE_SRC.strip('\n') + '\n')
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/R1_Gating_Campaign'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))")
patch('resolve', "                     n_min_cycles=N_MIN_CYCLES, dry_run_subset=DRY_RUN_SUBSET if DRY_RUN else None)",
      "                     n_min_cycles=N_MIN_CYCLES, dry_run_subset=DRY_RUN_SUBSET if DRY_RUN else None,\n"
      "                     run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)")

_ienv = idx('env') - 1
assert CELLS[_ienv][0] == 'markdown' and CELLS[_ienv][2].startswith('## Environment\n')
CELLS[_ienv] = ('markdown', None, CELLS[_ienv][2].replace(
    '## Environment\n', '## Environment: libraries, Drive, GPU\n\nThis cell mounts Google Drive, installs PyWavelets (the only '
    'package Colab lacks), imports every library the notebook needs, prints the GPU, CPU count and RAM, and fixes the '
    'determinism settings.\n', 1))

# ------------------------------------------------------------------ 2. embedded folds, child map, Springer
EXT_CELL = r'''
# ---------------- Extension Run 1: embedded folds, child map, Springer (label-free) ----------------
import base64 as _b64, zlib as _zlib, types as _types, io as _io
import scipy.io
FOLD_B64 = '%(fold_b64)s'
assert hashlib.sha256(FOLD_B64.encode()).hexdigest() == '%(fold_sha)s', 'embedded fold data corrupted'
EMBED = json.loads(_zlib.decompress(_b64.b64decode(FOLD_B64)).decode())
EMBED_FOLDS, PINNED_KEEP = EMBED['folds'], EMBED['pinned_keep']

def child_map(meta):
    """Patient ID -> child ID: connected components of Patient ID and Additional ID, labelled by the smallest Patient ID."""
    parent = {int(p): int(p) for p in meta['Patient ID']}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p, a in zip(meta['Patient ID'], meta['Additional ID']):
        if pd.notna(a) and int(a) in parent:
            ra, rb = find(int(p)), find(int(a))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {p: find(p) for p in parent}

CHILD_OF = child_map(metadata_df)
print('children: %%d patients in %%d children (%%d enrolled twice)' %% (len(CHILD_OF), len(set(CHILD_OF.values())), len(CHILD_OF) - len(set(CHILD_OF.values()))))

def fold_record_hash(names, child, y, fold_id):
    """extstudy.splits.fold_hash: SHA-256 over names (fixed width 11, as in the fold files), child IDs, labels, and fold
    IDs (int64)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.array(names, dtype='<U11')).tobytes())
    for a in (child, y, fold_id):
        h.update(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes())
    return h.hexdigest()

for _t, _f in EMBED_FOLDS.items():
    assert fold_record_hash(_f['names'], _f['child'], _f['y'], _f['fold_id']) == _f['sha256'], 'embedded %%s folds do not match their hash' %% _t
print('embedded child-grouped folds verified against their SHA-256:', {t: f['sha256'][:12] for t, f in EMBED_FOLDS.items()})

# Springer: Panel E's GPL-3 port of the PhysioNet 2016 sample2016 code, executed in its own namespace
SPRINGER_SRC = _b64.b64decode('%(springer_b64)s').decode('utf-8')
assert hashlib.sha256(SPRINGER_SRC.encode('utf-8')).hexdigest() == '%(springer_sha)s'
# written to a real file and imported (numba's on-disk cache needs a source file); the Viterbi forward pass then runs
# as plain Python, the code Panel E checked and the local tests ran (about 0.3 s per recording and variant), not
# numba's compiled copy, whose floating-point results can differ in the last bits and flip a tied path
import importlib.util as _ilu
_sp_path = (Path('/content') if IN_COLAB else RUN_DIR) / 'springer_py_run1.py'
os.environ.setdefault('NUMBA_CACHE_DIR', str(_sp_path.parent / 'numba_cache'))
_sp_path.write_text(SPRINGER_SRC, encoding='utf-8')
_spec = _ilu.spec_from_file_location('springer_py_run1', str(_sp_path))
springer = _ilu.module_from_spec(_spec)
sys.modules['springer_py_run1'] = springer
_spec.loader.exec_module(springer)
if hasattr(springer._viterbi_forward, 'py_func'):
    springer._viterbi_forward = springer._viterbi_forward.py_func
SPRINGER_VITERBI = 'plain Python' if not hasattr(springer._viterbi_forward, 'py_func') else 'compiled'
assert SPRINGER_VITERBI == 'plain Python'
_PARAM_B64 = %(params)s
_PARAM_SHA = %(param_sha)s
for _k, _v in _PARAM_B64.items():
    assert hashlib.sha256(_b64.b64decode(_v)).hexdigest() == _PARAM_SHA[_k], 'Springer parameter file %%s corrupted' %% _k
_mats = {k: scipy.io.loadmat(_io.BytesIO(_b64.b64decode(v)), squeeze_me=True) for k, v in _PARAM_B64.items()}
SPRINGER_PARAMS = ([np.asarray(b, float) for b in _mats['Springer_B_matrix.mat']['Springer_B_matrix']],
                   np.asarray([float(p) for p in _mats['Springer_pi_vector.mat']['Springer_pi_vector']]),
                   (np.asarray(_mats['Springer_total_obs_distribution.mat']['Springer_total_obs_distribution'][0], float),
                    np.asarray(_mats['Springer_total_obs_distribution.mat']['Springer_total_obs_distribution'][1], float)))
S_VARIANTS = {'springer_wpk': dict(hr_min=40, hr_max=220, sys_min_s=0.10, hr_method='peak'),
              'springer_pub': dict(hr_min=30, hr_max=120, sys_min_s=0.20, hr_method='argmax')}
S_CFG_HASH = hashlib.sha1(json.dumps(dict(src='%(springer_sha)s', params=_PARAM_SHA, variants=S_VARIANTS, n_min=N_MIN_CYCLES, viterbi=SPRINGER_VITERBI),
                                     sort_keys=True).encode()).hexdigest()[:16]

def states_to_cycles(states_1k, L4, sr_states=1000, sr_out=TARGET_SR):
    """Complete S1 -> systole -> S2 -> diastole runs of the state sequence as R1 cycles at 4 kHz."""
    st = np.asarray(states_1k)
    ch = np.flatnonzero(np.diff(st)) + 1
    starts, ends = np.concatenate([[0], ch]), np.concatenate([ch, [len(st)]])
    runs = [(int(st[s]), int(s), int(e)) for s, e in zip(starts, ends)]
    k = sr_out / sr_states
    cyc, i = [], 0
    while i + 3 < len(runs):
        if [r[0] for r in runs[i:i + 4]] == [1, 2, 3, 4]:
            b = [min(L4, int(round(x * k))) for r in runs[i:i + 4] for x in (r[1], r[2])]
            c = dict(cycle=len(cyc), s1=(b[0], b[1]), systole=(b[2], b[3]), s2=(b[4], b[5]), diastole=(b[6], b[7]))
            if c['diastole'][1] > c['s1'][0]:
                cyc.append(c)
            i += 4
        else:
            i += 1
    return cyc

def annotated_spans(cyc_c):
    """Annotated stretches of arm C: runs of abutting cycles, (first S1 onset, last diastole end) in samples."""
    spans = []
    for c in cyc_c:
        a, b = int(c['s1'][0]), int(c['diastole'][1])
        if spans and a - spans[-1][1] <= 2:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans

def restrict_to(cyc, spans):
    """Cycles whose S1 onset lies inside an annotated stretch (selection only; boundaries untouched)."""
    out = [c for c in cyc if any(a <= c['s1'][0] < b for a, b in spans)]
    return [dict(c, cycle=i) for i, c in enumerate(out)]

def _s_worker(name):
    """Both Springer variants on the shared analysed audio; fewer than N_MIN_CYCLES cycles -> equal-time fallback."""
    try:
        L4 = AUDIO_META[name]['L4']
        x4 = np.asarray(AUDIO4[name][:L4], float)
        out = {}
        for arm, kw in S_VARIANTS.items():
            states, info = springer.run_springer(x4, TARGET_SR, SPRINGER_PARAMS, return_info=True, **kw)
            cyc = states_to_cycles(states, L4)
            d = dict(hr_bpm=float(info['hr']), systolic_s=float(info['systolic_time']), n_cycles_springer=len(cyc),
                     unassigned_frac=float(np.mean(np.asarray(states) == 0)), fallback=False)
            if len(cyc) < N_MIN_CYCLES:
                cyc, d['fallback'] = equal_time_cycles(L4, TARGET_SR), True
            out[arm] = (cyc, d)
        return name, out, None
    except Exception as e:
        return name, None, '%%s: %%s' %% (type(e).__name__, e)
print('Springer arms ready: %%s | configuration hash %%s | source sha256 %%s | Viterbi %%s' %% (list(S_VARIANTS), S_CFG_HASH, '%(springer_sha)s'[:16], SPRINGER_VITERBI))
''' % dict(fold_b64=FOLD_B64, fold_sha=FOLD_B64_SHA, springer_b64=SPRINGER_B64, springer_sha=SPRINGER_SHA,
           params=json.dumps(PARAMS), param_sha=json.dumps(PARAM_SHA))

patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS
ARM_LETTER.update({'springer_wpk': 'S', 'springer_pub': 'S_pub', 'springer_wpk_annot': 'S_annot', 'equal_time_annot': 'E_annot'})
ARM_LABEL.update({'springer_wpk': 'S Springer (40-220 bpm, peak)', 'springer_pub': 'S_pub Springer (published HR rule)',
                  'springer_wpk_annot': 'S_annot Springer, annotated stretches', 'equal_time_annot': 'E_annot equal-time, annotated stretches'})
print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)""")
insert_after('s4_arm_defs', [('code', 'ext_run1_defs', EXT_CELL.strip('\n'))])

# ------------------------------------------------------------------ 3/4. Springer arms in Stage 4; kept set from R1 arms, pinned
patch('s4_run', """    with open(RUN_DIR / 'arm_failures.json', 'w') as fh:
        json.dump(ARM_FAILURES, fh, indent=1)""", """    # ---- Springer arms S_wpk and S_pub (label-free) and the matched-coverage copies ----
    fp_s = CACHE_DIR / 'cycles_arms_springer.pkl'
    s_cache = {}
    if fp_s.exists():
        try:
            with open(fp_s, 'rb') as fh:
                _o = pickle.load(fh)
            s_cache = _o['data'] if _o.get('cfg') == S_CFG_HASH else {}
        except Exception:
            s_cache = {}
    todo = [n for n in NAMES_ALL if n not in s_cache]
    if todo:
        t0 = time.time()
        with _pool(N_PROC) as ex:
            for name, payload, err in tqdm(ex.map(_s_worker, todo, chunksize=8), total=len(todo), desc='arms S', leave=False):
                if err is not None:
                    ARM_FAILURES.append((name, 'S', err))
                    L4_ = AUDIO_META[name]['L4']
                    payload = {a: (equal_time_cycles(L4_, TARGET_SR), dict(fallback=True, worker_error=err)) for a in S_VARIANTS}
                s_cache[name] = payload
        atomic_pickle(fp_s, {'cfg': S_CFG_HASH, 'data': s_cache})
        print('   arms S built for %d recordings in %.0f s' % (len(todo), time.time() - t0))
    else:
        print('   arms S loaded from cache')
    for n in NAMES_ALL:
        for a in S_ARMS:
            CYC[a][n], ARM_DIAG[a][n] = s_cache[n][a]
        spans = annotated_spans(CYC['pcg_seg'][n])
        for a, src in (('springer_wpk_annot', 'springer_wpk'), ('equal_time_annot', 'equal_time')):
            c_ = restrict_to(CYC[src][n], spans)
            fb = len(c_) < N_MIN_CYCLES
            CYC[a][n] = CYC[src][n] if fb else c_
            ARM_DIAG[a][n] = dict(fallback=bool(fb), n_annotated_spans=len(spans), n_cycles_selected=len(c_))
    del s_cache
    S_FALLBACK = {a: float(np.mean([bool(ARM_DIAG[a][n].get('fallback')) for n in NAMES_ALL])) for a in S_ARMS + COV_ARMS}
    print('   fallback share (equal-time windows used):', {a: round(100 * v, 2) for a, v in S_FALLBACK.items()}, '%; arms above',
          100 * S_FALLBACK_FAIL, '% are reported as failed:', [a for a, v in S_FALLBACK.items() if v > S_FALLBACK_FAIL])
    with open(RUN_DIR / 'arm_failures.json', 'w') as fh:
        json.dump(ARM_FAILURES, fh, indent=1)""")
patch('s4_run', "        bad = [arm for arm in ARMS if len(CYC[arm][n]) < N_MIN_CYCLES]",
      "        bad = [arm for arm in R1_ARMS if len(CYC[arm][n]) < N_MIN_CYCLES]")
patch('s4_run', "    assert len(KEEP) >= 20, 'too few recordings survive; inspect the drop reasons above'",
      """    assert len(KEEP) >= 20, 'too few recordings survive; inspect the drop reasons above'
    KEEP_PINNED_OK = None
    if VALIDATION_APPLICABLE:
        KEEP_PINNED_OK = list(KEEP) == list(PINNED_KEEP)
        assert KEEP_PINNED_OK, 'the kept set differs from the 3,000 R1 recordings: %d kept, %d not in R1, %d R1 missing' % (
            len(KEEP), len(set(KEEP) - set(PINNED_KEEP)), len(set(PINNED_KEEP) - set(KEEP)))
        print('   kept set equals the 3,000 R1 recordings (pinned)')""")
patch('s4_run', "    pa = DK[DK.arm == 'syn_fixed']", """    for a in S_ARMS + COV_ARMS:
        DIAG_SUMMARY[a]['fallback_frac'] = S_FALLBACK[a]
        DIAG_SUMMARY[a]['failed_arm'] = bool(S_FALLBACK[a] > S_FALLBACK_FAIL)
    for a in S_ARMS:
        _hr = np.array([ARM_DIAG[a][n].get('hr_bpm', np.nan) for n in KEEP], float)
        DIAG_SUMMARY[a]['springer_hr_bpm_median'] = float(np.nanmedian(_hr))
    pa = DK[DK.arm == 'syn_fixed']""")

# ------------------------------------------------------------------ 2. child groups and R1 feature reproduction
patch('s5_features', "    G = np.array([PID_OF[n] for n in KEEP])",
      "    G = np.array([CHILD_OF[PID_OF[n]] for n in KEEP])                 # groups = children (Patient ID union Additional ID)")
patch('s5_features', "    print('kept set: %d recordings, %d patients | age bands: %s' % (len(KEEP), len(np.unique(G)), dict(collections.Counter(AGE.tolist()))))",
      """    print('kept set: %d recordings, %d children | age bands: %s' % (len(KEEP), len(np.unique(G)), dict(collections.Counter(AGE.tolist()))))
    FEATURE_REPRO = None
    if VALIDATION_APPLICABLE and R1_RUN_DIR and Path(R1_RUN_DIR, 'cache').is_dir():
        FEATURE_REPRO = {}
        for arm in R1_ARMS:
            fp_ = Path(R1_RUN_DIR) / 'cache' / ('features_%s.npz' % arm)
            if fp_.exists():
                z_ = np.load(fp_, allow_pickle=False)
                same_names = list(z_['rec_names']) == list(KEEP)
                FEATURE_REPRO[arm] = dict(same_names=bool(same_names),
                                          max_abs_diff=float(np.max(np.abs(z_['X'] - FEATS[arm]))) if same_names else None)
        print('   R1 feature matrices reproduced:', FEATURE_REPRO)""")

# ------------------------------------------------------------------ 2. folds from the embedded child-grouped assignment
patch('s6_folds', "        f_ = draw_or_load_folds(t_, KEEP_ARR, y_, G, t_ + '.npz')",
      """        if VALIDATION_APPLICABLE:
            f_ = embedded_folds(t_, KEEP_ARR, y_, G, t_ + '.npz')
        else:
            f_ = draw_or_load_folds(t_, KEEP_ARR, y_, G, t_ + '.npz')""")
patch('s6_folds', "with stage('S6 folds'):", '''FOLD_CHECK = {}

def embedded_folds(task, names_all, y_all, groups_all, fname):
    """The embedded child-grouped assignment, checked against this notebook's names, labels and child IDs, saved
    in the R1 fold-file layout (so the CNN notebook reads the identical splits)."""
    e = EMBED_FOLDS[task]
    mask = y_all != -1
    ym, gm, nm = y_all[mask], groups_all[mask], names_all[mask]
    checks = dict(names=list(nm) == list(e['names']), labels=bool(np.array_equal(ym, np.asarray(e['y']))),
                  children=bool(np.array_equal(gm, np.asarray(e['child']))))
    fold_id = np.asarray(e['fold_id'], np.int64)
    k = int(fold_id.max()) + 1
    splits = [(np.flatnonzero(fold_id != i), np.flatnonzero(fold_id == i)) for i in range(k)]
    checks['child_overlap_zero'] = all(not set(gm[tr]) & set(gm[te]) for tr, te in splits)
    checks['hash'] = fold_record_hash(nm, gm, ym, fold_id) == e['sha256']
    FOLD_CHECK[task] = checks
    assert all(checks.values()), 'embedded folds for %s fail their checks: %s' % (task, checks)
    arrays = {'train_%02d' % i: tr for i, (tr, _) in enumerate(splits)}
    arrays.update({'test_%02d' % i: te for i, (_, te) in enumerate(splits)})
    atomic_savez(FOLD_DIR / fname, names=nm, patient_ids=gm, patient_ids_original=np.array([PID_OF[n] for n in nm]), y=ym,
                 fold_id=fold_id, n_splits=np.array(k), keep_names=names_all, mask_over_keep=mask, seed=np.array(SEED),
                 grouping=np.array('child = Patient ID union Additional ID'), sha256=np.array(e['sha256']), **arrays)
    print('   %-10s embedded child-grouped folds verified (%d recordings, %d children, sha256 %s)' % (task, len(nm), len(np.unique(gm)), e['sha256'][:12]))
    return dict(mask=mask, y=ym, groups=gm, names=nm, splits=splits, n_splits=k)

with stage('S6 folds'):''')

# ------------------------------------------------------------------ 5. five-seed MLP ensembles
patch('s7_classifier_defs', """            if recipe == 'primary':
                mdl, info = fit_mlp(Xtr, ym[tr], gm[tr], ncls, seed + k)
            else:
                mdl, info = fit_mlp_reference(Xtr, ym[tr], gm[tr], ncls, seed)   # constant seed, as the reference run
            pr = mlp_proba(mdl, Xte)""", """            if recipe == 'primary':                 # Run 1: probabilities averaged over the five seed bases plus the fold index
                prs = []
                for sb in ENSEMBLE_SEEDS:
                    mdl, info = fit_mlp(Xtr, ym[tr], gm[tr], ncls, sb + k)
                    prs.append(mlp_proba(mdl, Xte))
                    res['fit_info'].append(dict(info, seed=int(sb + k)))
                    del mdl
                res['proba_mlp_seeds'][te] = np.stack(prs, -1)
                pr = np.mean(prs, 0).astype(np.float32)
                mdl = None
            else:
                mdl, info = fit_mlp_reference(Xtr, ym[tr], gm[tr], ncls, seed)   # constant seed, as the reference run
                pr = mlp_proba(mdl, Xte)""")
patch('s7_classifier_defs', """            res['fit_info'].append(info)
            del mdl""", """            if recipe != 'primary':
                res['fit_info'].append(info)
            del mdl""")
patch('s7_classifier_defs', """               pred_rf=np.full(n, -1), proba_rf=np.zeros((n, ncls), np.float32), fold_f1_rf=[], fold_acc_rf=[], fit_info=[])""",
      """               pred_rf=np.full(n, -1), proba_rf=np.zeros((n, ncls), np.float32), fold_f1_rf=[], fold_acc_rf=[], fit_info=[],
               proba_mlp_seeds=np.zeros((n, ncls, len(ENSEMBLE_SEEDS)), np.float32))""")
patch('s7_classifier_defs', """    fp = CV_DIR / ('%s__%s.npz' % (tag, task))""",
      """    fp = CV_DIR / ('%s__%s__ens%d.npz' % (tag, task, len(ENSEMBLE_SEEDS) if recipe == 'primary' else 1))""")

# ------------------------------------------------------------------ 5b. the fold-by-seed fits run in parallel GPU workers
# The MLP is small and trained with batch 32 (the frozen recipe), so one fit leaves the A100 idle; throughput comes from
# running many fits at once. The worker module holds the exact source of LightweightMLP, fit_mlp, and mlp_proba,
# lifted from the cell at build time; every job is the same function with the same seed, so the probabilities equal
# those of a one-at-a-time run (checked in the integrity gate).
import ast as _ast
_s7 = CELLS[idx('s7_classifier_defs')][2]
_lines = _s7.split('\n')
_defs = {n.name: '\n'.join(_lines[min([d.lineno for d in n.decorator_list] + [n.lineno]) - 1:n.end_lineno]) for n in _ast.parse(_s7).body
         if isinstance(n, (_ast.FunctionDef, _ast.ClassDef)) and n.name in ('LightweightMLP', 'fit_mlp', 'mlp_proba')}
assert set(_defs) == {'LightweightMLP', 'fit_mlp', 'mlp_proba'}
WORKER_SRC = '''"""Extension Run 1 MLP worker: LightweightMLP, fit_mlp and mlp_proba copied verbatim from the notebook cell."""
import os, random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

MLP_CFG, SEED, DEV = None, None, None


def init(cfg, seed):
    """Same determinism settings as the notebook's environment cell."""
    global MLP_CFG, SEED, DEV
    MLP_CFG, SEED = dict(cfg), seed
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' + _defs['LightweightMLP'] + '\n\n\n' + _defs['fit_mlp'] + '\n\n\n' + _defs['mlp_proba'] + '''


def fit_job(Xtr, ytr, gtr, Xte, ncls, seed):
    """One fit, as run_cv performs it in-process: fit_mlp, then mlp_proba on the test rows."""
    model, info = fit_mlp(Xtr, ytr, gtr, ncls, seed)
    pr = mlp_proba(model, Xte)
    return pr, info
'''
compile(WORKER_SRC, 'run1_mlp_worker.py', 'exec')
WORKER_B64 = base64.b64encode(WORKER_SRC.encode('utf-8')).decode()
WORKER_SHA = sha256_bytes(WORKER_SRC.encode('utf-8'))

assert '@torch.no_grad()' in _defs['mlp_proba'], 'decorator missing from the worker copy'
patch('s7_classifier_defs', "@torch.no_grad()\ndef mlp_proba(model, X):", '''# ---- Extension Run 1: parallel GPU workers for the fold-by-seed MLP fits ----
GPU_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))
MLP_POOL, _mlpw = None, None
if (DEV.type == 'cuda' or os.environ.get('RUN1_POOL_ON_CPU')) and GPU_WORKERS > 1:
    import base64 as _b64w, importlib as _importlib
    _wsrc = _b64w.b64decode('%s').decode('utf-8')
    assert hashlib.sha256(_wsrc.encode('utf-8')).hexdigest() == '%s', 'worker source corrupted'
    _wdir = Path('/content') if IN_COLAB else RUN_DIR
    (_wdir / 'run1_mlp_worker.py').write_text(_wsrc, encoding='utf-8')
    if str(_wdir) not in sys.path:
        sys.path.insert(0, str(_wdir))
    _mlpw = _importlib.import_module('run1_mlp_worker')
    MLP_POOL = ProcessPoolExecutor(max_workers=GPU_WORKERS, mp_context=mp.get_context('spawn'),
                                   initializer=_mlpw.init, initargs=(MLP_CFG, SEED))
    print('MLP fits run in %%d parallel %%s worker processes (spawn); worker source sha256 %%s' %% (GPU_WORKERS, DEV.type.upper(), '%s'[:16]))
else:
    print('MLP fits run one at a time in this process (no GPU worker pool)')

@torch.no_grad()
def mlp_proba(model, X):''' % (WORKER_B64, WORKER_SHA, WORKER_SHA))
patch('s7_classifier_defs', """    labs = list(range(ncls))
    for k, (tr, te) in enumerate(Fd['splits']):""", """    labs = list(range(ncls))
    futs = {}
    if do_mlp and recipe == 'primary' and MLP_POOL is not None:     # submit every fold-by-seed fit; RF runs below meanwhile
        for k, (tr, te) in enumerate(Fd['splits']):
            if scaler == 'fold':
                sc_ = StandardScaler().fit(Xm[tr])
                Xtr_, Xte_ = sc_.transform(Xm[tr]), sc_.transform(Xm[te])
            else:
                Xtr_, Xte_ = sc_full.transform(Xm[tr]), sc_full.transform(Xm[te])
            for sb in ENSEMBLE_SEEDS:
                futs[(k, sb)] = MLP_POOL.submit(_mlpw.fit_job, Xtr_, ym[tr], gm[tr], Xte_, ncls, sb + k)
    for k, (tr, te) in enumerate(Fd['splits']):""")
patch('s7_classifier_defs', """                for sb in ENSEMBLE_SEEDS:
                    mdl, info = fit_mlp(Xtr, ym[tr], gm[tr], ncls, sb + k)
                    prs.append(mlp_proba(mdl, Xte))
                    res['fit_info'].append(dict(info, seed=int(sb + k)))
                    del mdl""", """                for sb in ENSEMBLE_SEEDS:
                    if futs:
                        p_, info = futs[(k, sb)].result()
                    else:
                        mdl, info = fit_mlp(Xtr, ym[tr], gm[tr], ncls, sb + k)
                        p_ = mlp_proba(mdl, Xte)
                        del mdl
                    prs.append(p_)
                    res['fit_info'].append(dict(info, seed=int(sb + k)))""")

# ------------------------------------------------------------------ pairs that include the new arms
patch('s7_run', """         ('equal_time_matched', 'equal_time')]""", """         ('equal_time_matched', 'equal_time'),
         ('springer_wpk', 'equal_time'), ('springer_pub', 'equal_time'), ('pcg_seg', 'springer_wpk'), ('springer_wpk', 'springer_pub'),
         ('springer_wpk', 'syn_fixed'), ('springer_wpk_annot', 'equal_time_annot'), ('pcg_seg', 'equal_time_annot')]""")
patch('s7_run', "CV, EVAL = {}, {}\nwith stage('S7 classifiers and evaluation'):",
      "CV, EVAL = {}, {}\n"
      "if DRY_RUN or LIMIT_RECORDINGS:   # Run 1 rule: one logged scoring per arm; a smoke test stops before any classification\n"
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')\n"
      "with stage('S7 classifiers and evaluation'):")
patch('s7_run', "    for task in TASKS:\n        CV[task], EVAL[task] = {}, {}",
      "    RESULTS['run'].update(ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', springer_cfg_hash=S_CFG_HASH, springer_variants=S_VARIANTS,\n"
      "                          dataset_on_drive=str(AUDIO_PATH_DRIVE), dataset_read_from=str(AUDIO_PATH), prereg_sha256=PREREG_SHA256,\n"
      "                          fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL, fold_sha256={t: EMBED_FOLDS[t]['sha256'] for t in EMBED_FOLDS})\n"
      "    for task in TASKS:\n        CV[task], EVAL[task] = {}, {}")
patch('s7_run', "def add_oof(task, tag, r, clfs=('mlp', 'rf')):\n    for clf in clfs:",
      "def add_oof(task, tag, r, clfs=('mlp', 'rf')):\n"
      "    if 'proba_mlp_seeds' in r:\n"
      "        OOF['%s__%s__mlp__proba_seeds' % (task, tag)] = r['proba_mlp_seeds']\n"
      "    OOF['%s__%s__names' % (task, tag)] = np.asarray(r['names'])\n"
      "    OOF['%s__%s__groups' % (task, tag)] = np.asarray(r['groups'])\n"
      "    OOF['%s__%s__y' % (task, tag)] = np.asarray(r['y'])\n"
      "    for clf in clfs:")

# ------------------------------------------------------------------ 6. integrity gate replaces the R1 validation gate
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, r'''## Stage 8: integrity gate (Run 1)

The R1 gate compared arms C and E with reference values from patient-grouped folds and one seed; Run 1 changes both
on purpose, so that gate no longer applies. Run 1 checks only things that do not depend on any classification result:
the kept set equals the 3,000 R1 recordings; every task's folds equal the embedded child-grouped assignment (names,
labels, child IDs, hash) with no child on both sides of a split; the R1 feature matrices are reproduced (when the R1
run folder is reachable); and a repeat of one five-seed fit returns identical probabilities. The gate passes when every
applicable check passes; a failure is recorded and the run is not used.''')
CELLS[i8] = ('code', 's8_gate', r'''
with stage('S8 integrity gate'):
    gate = dict(kind='integrity (Run 1)', applicable=bool(VALIDATION_APPLICABLE))
    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        gate.update(ok=None, decided_on='not applicable (reduced run)')
        print('integrity gate: not applicable in a reduced run')
    else:
        gate['kept_set_pinned'] = bool(KEEP_PINNED_OK)
        gate['folds'] = FOLD_CHECK
        gate['folds_ok'] = all(all(v.values()) for v in FOLD_CHECK.values()) and set(FOLD_CHECK) == set(TASKS)
        gate['feature_reproduction'] = FEATURE_REPRO
        gate['features_ok'] = None if not FEATURE_REPRO else all(v['same_names'] and v['max_abs_diff'] is not None and v['max_abs_diff'] < 1e-4
                                                                for v in FEATURE_REPRO.values())
        # repeat fit: fold 0 of detection, arm E, five seeds, twice
        Fd_ = FOLDS['detection']; tr_, te_ = Fd_['splits'][0]
        Xm_ = FEATS['equal_time'][Fd_['mask']]
        sc_ = StandardScaler().fit(Xm_[tr_])
        reps = []
        for _ in range(2):
            prs_ = []
            for sb in ENSEMBLE_SEEDS:
                m_, _i = fit_mlp(sc_.transform(Xm_[tr_]), Fd_['y'][tr_], Fd_['groups'][tr_], 2, sb)
                prs_.append(mlp_proba(m_, sc_.transform(Xm_[te_])))
            reps.append(np.mean(prs_, 0))
        gate['repeat_fit_max_abs_diff'] = float(np.max(np.abs(reps[0] - reps[1])))
        gate['repeat_fit_ok'] = gate['repeat_fit_max_abs_diff'] < 1e-6
        checks = [gate['kept_set_pinned'], gate['folds_ok'], gate['repeat_fit_ok']] + ([gate['features_ok']] if gate['features_ok'] is not None else [])
        if MLP_POOL is not None:     # the parallel workers return what an in-process fit returns
            Xtr0_, Xte0_ = sc_.transform(Xm_[tr_]), sc_.transform(Xm_[te_])
            p_pool_, _ = MLP_POOL.submit(_mlpw.fit_job, Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], Xte0_, 2, ENSEMBLE_SEEDS[0]).result()
            m0_, _ = fit_mlp(Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], 2, ENSEMBLE_SEEDS[0])
            gate['pool_vs_inprocess_max_abs_diff'] = float(np.max(np.abs(p_pool_ - mlp_proba(m0_, Xte0_))))
            gate['pool_ok'] = gate['pool_vs_inprocess_max_abs_diff'] < 1e-6
            checks.append(gate['pool_ok'])
        VALIDATION_OK = bool(all(checks))
        gate.update(ok=VALIDATION_OK, decided_on='integrity checks')
        print('integrity gate:', 'PASSED' if VALIDATION_OK else 'FAILED', {k: v for k, v in gate.items() if k not in ('folds', 'feature_reproduction')})
    RESULTS['validation_gate'] = gate
    write_results()
    if MLP_POOL is not None:
        MLP_POOL.shutdown()
''')

# ------------------------------------------------------------------ 7. Stage 9 and 10 cells are not run in Run 1
i9 = idx('s8_gate') + 1
iprov = next(i for i, c in enumerate(CELLS) if c[0] == 'code' and c[2].startswith('VERBATIM_PROVENANCE = '))
iprov_md = iprov - 1
assert CELLS[iprov_md][0] == 'markdown'
removed = [c[1] or 'markdown' for c in CELLS[i9:iprov_md]]
CELLS[i9:iprov_md] = [
    ('markdown', None, '## Stages 9 and 10: not run\n\nThe R1 reviewer ablations and R1 figures are not part of Run 1; Run 1 figures are drawn locally from `results.json` and `oof_predictions.npz`.'),
    ('code', 's9_s10_not_run', "FIG_CAPTIONS, DRAW_FIGURES, FIGURE_ERRORS = {}, False, {}\nprint('Stages 9 and 10 are not run in Extension Run 1')")]
print('removed R1 cells:', removed)

# ------------------------------------------------------------------ write
r1.OUT_NB = OUT_NB
r1.CELLS = CELLS
r1.main()
nb = nbformat.read(str(OUT_NB), as_version=4)
nb.metadata['colab']['name'] = 'Extension_Run1.ipynb'
nb.metadata['built_by'] = 'extension_study/notebooks/build_run1.py (patches over build_gating_campaign.py)'
nb.metadata['r1_base_notebook_source_sha256'] = BASE_HASH
nb.metadata['embedded'] = dict(folds_sha256={t: f['sha256'] for t, f in folds.items()}, springer_src_sha256=SPRINGER_SHA,
                               springer_params_sha256=PARAM_SHA, pinned_keep_n=len(PINNED_KEEP))
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Run 1 notebook written: %s (%d cells)' % (OUT_NB, len(nb.cells)))
