#!/usr/bin/env python
"""
build_run2.py: assembles Extension_Run2.ipynb (prereg section 7, Run 2) from the locked Extension_Run1.ipynb.

The Run 1 notebook is read cell by cell (its file hash must equal prereg/RUN1_LOCK.json) and patched by cell name;
every patch asserts that its anchor occurs exactly once. What Run 2 changes, and nothing else:
  1. run folder Extension_Study/runs/run2/<timestamp>; run flags record the run kind and the T/W checkpoint stamp;
  2. no generator: the GAN cells (v_generator, v_ckpt_load, s3_generator) are left out, the checkpoint search and the
     8 kHz audio are skipped; arms A and B belong to Run 1 and are not rebuilt;
  3. the frozen T and W models (checkpoint folder 20261018_0908 on Drive, every file and weight hash checked against
     the embedded freeze.json) run label-free on the shared analysed audio (extstudy/infer.py: 30 s windows, seed
     averages, shared decoder); cycles: W through arm B's construction (arm_b_cycles, verbatim), T through its own S1
     and S2 heads (extstudy/cycles.py: t_cycles); fewer than 2 cycles -> equal-time fallback, counted; more than 5%
     = failed arm; matched-coverage copies T_annot and W_annot restrict cycles to the annotated stretches as in Run 1;
  4. the kept set is the 3,000 Run 1 recordings (embedded); arms C and E are rebuilt only to reproduce Run 1;
  5. classified arms: E (reproduction control: its features and out-of-fold predictions must equal Run 1's, embedded),
     T, W, T_annot, W_annot; the five-seed MLP, random forest, folds and bootstrap are Run 1's cells unchanged;
  6. integrity gate: kept set, embedded folds, Run 1 cycle hashes of C, E and E_annot, E features and E predictions
     reproduced, repeat fit, pool check. The bootstrap self-checks use T instead of C (C is not rescored);
  7. a DRY_RUN or LIMIT_RECORDINGS run stops before Stage 7 (one logged scoring per arm, full run only).
H2 to H4 are computed locally (notebooks/merge_run2.py, then extstudy/stats.py with t_frozen=True) from this run's
out-of-fold predictions merged with Run 1's.

Usage:  python build_run2.py   (writes Extension_Run2.ipynb beside this file)
"""
import base64, hashlib, json, os, py_compile, tempfile, zlib
from pathlib import Path

import numpy as np
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PHD = STUDY.parent
RUN1_NB = HERE / 'Extension_Run1.ipynb'
OUT_NB = HERE / 'Extension_Run2.ipynb'
TW_STAMP = '20261018_0908'
TW_LOCAL = STUDY / 'runs' / 'run2_freeze' / TW_STAMP
RUN1_DIR = PHD / 'Results' / 'A100 Results' / 'extracted' / 'runs' / 'run1' / '20261016_1516'
RUN1_ZIP = PHD / 'Results' / 'A100 Results' / 'runs-20261016T171455Z-1-001.zip'
MODULES = ['__init__.py', 'frontend.py', 'model.py', 'robust_beats.py', 'decoder.py', 'infer.py', 'cycles.py']

sha = lambda b: hashlib.sha256(b).hexdigest()
lock = json.load(open(STUDY / 'prereg' / 'RUN1_LOCK.json'))
assert sha(RUN1_NB.read_bytes()) == lock['run1_notebook_file_sha256'], 'Extension_Run1.ipynb differs from its lock'
nb1 = json.load(open(RUN1_NB, encoding='utf-8'))
CELLS = [(c['cell_type'], c.get('metadata', {}).get('name'), ''.join(c['source']) if isinstance(c['source'], list) else c['source'])
         for c in nb1['cells']]


def idx(name):
    hits = [i for i, c in enumerate(CELLS) if c[1] == name]
    assert len(hits) == 1, 'cell %s: %d hits' % (name, len(hits))
    return hits[0]


def patch(name, old, new):
    i = idx(name)
    kind, nm, text = CELLS[i]
    assert text.count(old) == 1, 'cell %s: anchor found %d times: %r' % (name, text.count(old), old[:90])
    CELLS[i] = (kind, nm, text.replace(old, new))


def replace_cell(name, text):
    i = idx(name)
    CELLS[i] = (CELLS[i][0], name, text.strip('\n'))


def drop(name, note):
    """Remove a code cell and turn the markdown cell right before it (if any) into a one-line note."""
    i = idx(name)
    if i > 0 and CELLS[i - 1][0] == 'markdown':
        CELLS[i - 1] = ('markdown', None, note)
        del CELLS[i]
    else:
        CELLS[i] = ('markdown', None, note)


def insert_after(name, cells):
    i = idx(name)
    CELLS[i + 1:i + 1] = cells


# ------------------------------------------------------------------ embedded data
MODS = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
for m, b in MODS.items():
    assert 'circor' not in b.decode('utf-8').lower(), 'CirCor reference in %s' % m
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in MODS.items()}
MOD_SHA = {m: sha(b) for m, b in MODS.items()}
FREEZE_BYTES = (TW_LOCAL / 'freeze.json').read_bytes()
FREEZE_SHA = sha(FREEZE_BYTES)
FREEZE_REC = json.loads(FREEZE_BYTES)
assert FREEZE_REC['smoke'] is False and FREEZE_REC['label_check']['status'] == 'PASSED'

man1 = json.load(open(RUN1_DIR / 'run_manifest.json'))
RUN1_CYCLES_HASH = {a: man1['hashes']['feature_cycles_hashes'][a] for a in ('pcg_seg', 'equal_time', 'equal_time_annot')}
import zipfile
with zipfile.ZipFile(RUN1_ZIP) as zf, zf.open('runs/run1/20261016_1516/cache/features_equal_time.npz') as fh:
    import io
    zE = np.load(io.BytesIO(fh.read()), allow_pickle=False)
    RUN1_E_FEATURES_SHA = sha(np.ascontiguousarray(zE['X'].astype(np.float32)).tobytes())
    assert str(zE['cycles_hash']) == RUN1_CYCLES_HASH['equal_time']
o1 = np.load(RUN1_DIR / 'oof_predictions.npz', allow_pickle=False)
RUN1_E = {}
for t in ('detection', 'grading', 'timing', 'shape'):
    p = '%s__equal_time__' % t
    RUN1_E[t] = dict(names=o1[p + 'names'].astype(str).tolist(), y=o1[p + 'y'].astype(int).tolist(),
                     mlp_pred=o1[p + 'mlp__pred'].astype(int).tolist(), mlp_proba=o1[p + 'mlp__proba'].astype(float).round(8).tolist(),
                     rf_pred=o1[p + 'rf__pred'].astype(int).tolist())
RUN1_E_B64 = base64.b64encode(zlib.compress(json.dumps(RUN1_E).encode(), 9)).decode()
RUN1_E_SHA = sha(RUN1_E_B64.encode())
OOF1_SHA = sha((RUN1_DIR / 'oof_predictions.npz').read_bytes())
PREREG_SHA = sha((STUDY / 'prereg' / 'PREREG_EXT.md').read_bytes())
assert open(STUDY / 'prereg' / 'PREREG_EXT.sha256').read().split()[0] == PREREG_SHA

# ------------------------------------------------------------------ 0. header
assert CELLS[0][0] == 'markdown'
CELLS[0] = ('markdown', None, r'''# Extension Run 2: the frozen ECG-supervised timing network T and the waveform network W on CirCor

Run 2 of the EHJ-DH extension (`prereg/PREREG_EXT.md`, section 7). Built by `extension_study/notebooks/build_run2.py`
from the locked Run 1 notebook, patched cell by cell; the patch list is in that script's docstring. The shared
analysed audio, the 464-feature extractor, the five-seed MLP, the random forest, the child-grouped folds and the
bootstrap are Run 1's cells, unchanged.

**What runs.** T and W were trained on paired adult PCG-ECG data only and frozen on 18 October 2026 (checkpoint folder
`Extension_Study/checkpoints/20261018_0908`; every file and weight hash is checked against the embedded freeze
record). They run label-free on each recording; T's cycles come from its S1 and S2 heads, W's synthetic ECG goes
through arm B's construction. Arms C and E are rebuilt only to show that this run reproduces Run 1 exactly: E's
features and out-of-fold predictions must equal Run 1's.

**How to run.** Runtime > Change runtime type > **A100 GPU** with High-RAM (Run 1's GPU class: arm E's predictions are
compared with Run 1's bit for bit); leave `DRY_RUN = False`; Runtime > Run all. Expected time about 1.5 to 2 hours.

**Afterwards.** Download the run folder `Extension_Study/runs/run2/<timestamp>` (results.json, oof_predictions.npz,
diagnostics.csv, run_manifest.json). H2 to H4 are computed locally from this run's predictions merged with Run 1's.

| arm | key | boundaries |
|---|---|---|
| E | `equal_time` | 75 bpm pseudo-cycles (reproduction control) |
| **T** | `timing_net` | frozen T: S1 events from the shared decoder, S2 from its S2 head, 50% acoustic onsets |
| W | `waveform_net` | frozen W's synthetic ECG -> R-peaks -> arm B's acoustic construction |
| T_annot | `timing_net_annot` | T cycles inside the annotated stretches |
| W_annot | `waveform_net_annot` | W cycles inside the annotated stretches |
''')

# ------------------------------------------------------------------ 1. flags
patch('flags', "R1_RUN_DIR = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip",
      "R1_RUN_DIR = None          # Run 2 checks reproduction against Run 1's embedded hashes instead\n"
      "TW_STAMP = '%s'   # frozen T and W (Extension_Study/checkpoints/<stamp>)\n"
      "TW_CHECKPOINT_DIR = None   # None = the Drive folder above; a local path for a local run\n"
      "REQUIRE_GPU = 'A100'       # full run only: Run 1's GPU class, so arm E reproduces bit for bit" % TW_STAMP)
patch('flags', "PREREG_SHA256 = '755b47da43b250671391aae8672b182278cb4561314343581024ab1f581d8974'   # extension_study/prereg/PREREG_EXT.md, frozen 16 Oct 2026 before any Run 1 classification",
      "PREREG_SHA256 = '%s'   # extension_study/prereg/PREREG_EXT.md, frozen 16 Oct 2026" % PREREG_SHA)

# ------------------------------------------------------------------ 2. environment: GPU class for the full run
patch('env', "GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32",
      "GAN_BATCH = None    # Run 2 runs no generator\n"
      "if (not DRY_RUN) and LIMIT_RECORDINGS == 0 and REQUIRE_GPU:\n"
      "    assert REQUIRE_GPU in GPU_NAME, 'Run 2 must run on an %s (Run 1 class) so arm E reproduces Run 1; this runtime has %s' % (REQUIRE_GPU, GPU_NAME)")
patch('env', "print('generator batch', GAN_BATCH, '| feature workers', N_PROC)", "print('feature workers', N_PROC)")

# ------------------------------------------------------------------ 3. resolve: no generator; run2 folder; T/W checkpoint
i = idx('resolve')
txt = CELLS[i][2]
a = txt.index('    # ---- generator checkpoint ----')
b = txt.index('    # ---- output folder and run flags ----')
CELLS[i] = (CELLS[i][0], 'resolve', txt[:a] + '''    # ---- Run 2: no generator; the frozen T and W checkpoint folder ----
    GAN_CHECKPOINT, CKPT_SHA256, CKPT_HASHES = None, None, {}
    TW_SRC = Path(TW_CHECKPOINT_DIR) if TW_CHECKPOINT_DIR else Path(DRIVE_ROOT + '/JBHI-Revision/Extension_Study/checkpoints/' + TW_STAMP)
    assert (TW_SRC / 'freeze.json').exists(), 'frozen T/W checkpoint folder not found: %s' % TW_SRC

''' + txt[b:])
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run2'))")
patch('resolve', "run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)",
      "run='extension_run2', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256, tw_stamp=TW_STAMP)")
patch('resolve', "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),",
      "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('TW_SRC', TW_SRC),")
patch('resolve', "    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))",
      "    print('paired wav+tsv files: %d | frozen T/W folder: %s' % (N_PAIRS, TW_SRC))")

patch('resolve', """    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
""", "")
# ------------------------------------------------------------------ 4. generator cells out; 8 kHz audio never needed
drop('v_generator', '*Run 2: the generator architecture cell of Run 1 is not used (no synthetic ECG from the GAN).*')
drop('v_ckpt_load', '*Run 2: the generator checkpoint is not loaded.*')
drop('s3_generator', '## Stage 3: not run\n\nRun 2 builds no GAN arms; arms A and B belong to Run 1.')
patch('s2_audio', "    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True",
      "    ECG_CHUNK_OK[k] = True       # Run 2: no generator, so the 8 kHz audio is never loaded")

# ------------------------------------------------------------------ 5. arm lists
patch('s4_arm_defs', """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS""", """R1_ARMS = list(ARMS)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
TW_ARMS = ['timing_net', 'waveform_net']
TW_COV_ARMS = ['timing_net_annot', 'waveform_net_annot']
ARMS = ['equal_time'] + TW_ARMS + TW_COV_ARMS                              # classified in Run 2 (E: reproduction control)
CYC_ARMS = ['pcg_seg', 'equal_time', 'equal_time_annot'] + TW_ARMS + TW_COV_ARMS   # cycles built in Run 2""")
patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """ARM_LETTER.update({'springer_wpk': 'S', 'springer_pub': 'S_pub', 'springer_wpk_annot': 'S_annot', 'equal_time_annot': 'E_annot',
                   'timing_net': 'T', 'waveform_net': 'W', 'timing_net_annot': 'T_annot', 'waveform_net_annot': 'W_annot'})
ARM_LABEL.update({'equal_time_annot': 'E_annot equal-time, annotated stretches', 'timing_net': 'T timing network (frozen)',
                  'waveform_net': 'W waveform network (frozen)', 'timing_net_annot': 'T_annot, annotated stretches',
                  'waveform_net_annot': 'W_annot, annotated stretches'})
print('arm definitions ready: classified', ARMS, '| cycles', CYC_ARMS, '| arm configuration hash', ARM_CFG_HASH)""")

# ------------------------------------------------------------------ 6. T/W definitions (embedded code, frozen models)
EXT2 = r'''
# ---------------- Extension Run 2: frozen T and W, label-free inference and cycle export ----------------
import base64 as _b64r, zlib as _zr, shutil as _sh2
_MOD_B64 = %(mod_b64)s
_MOD_SHA = %(mod_sha)s
_code_root = (Path('/content') if IN_COLAB else RUN_DIR) / 'run2_code'
(_code_root / 'extstudy').mkdir(parents=True, exist_ok=True)
for _m, _b in _MOD_B64.items():
    _src = _b64r.b64decode(_b)
    assert hashlib.sha256(_src).hexdigest() == _MOD_SHA[_m], 'embedded module %%s corrupted' %% _m
    (_code_root / 'extstudy' / _m).write_bytes(_src)
for _k in [k for k in sys.modules if k == 'extstudy' or k.startswith('extstudy.')]:
    del sys.modules[_k]
sys.path.insert(0, str(_code_root))
from extstudy import infer as TWI, cycles as TWC
assert TWC.FUSION == FUSION and TWC.PHASE_KEYS == tuple(PHASE_KEYS) and TWC.TARGET_SR == TARGET_SR, 'cycle constants differ from arm B'

FREEZE_SHA = '%(freeze_sha)s'
_tw_local = (Path('/content') if IN_COLAB else RUN_DIR) / 'tw_checkpoints' / TW_STAMP
_tw_local.mkdir(parents=True, exist_ok=True)
for _f in sorted(p.name for p in TW_SRC.iterdir() if p.suffix in ('.json', '.pt')):
    _drive_retry(_sh2.copyfile, TW_SRC / _f, _tw_local / _f) if IN_COLAB else _sh2.copyfile(TW_SRC / _f, _tw_local / _f)
assert hashlib.sha256((_tw_local / 'freeze.json').read_bytes()).hexdigest() == FREEZE_SHA, 'freeze.json differs from the frozen record'
FREEZE_REC = json.load(open(_tw_local / 'freeze.json'))
T_MODELS, _ = TWI.load_models(str(_tw_local), 'T', DEV)          # file and weight hashes checked inside
W_MODELS, _ = TWI.load_models(str(_tw_local), 'W', DEV)
TW_WEIGHTS_OK = True
TW_CFG = dict(freeze_sha256=FREEZE_SHA, modules=_MOD_SHA, win_frames=TWI.WIN_FRAMES, overlap=TWI.OVERLAP, t_export=TWC.T_EXPORT,
              n_min=N_MIN_CYCLES, fusion=FUSION)
TW_CFG_HASH = hashlib.sha1(json.dumps(TW_CFG, sort_keys=True, default=str).encode()).hexdigest()[:16]
print('frozen T and W loaded and verified: %%s | T/W configuration hash %%s' %% ([n for n, _ in T_MODELS + W_MODELS], TW_CFG_HASH))

RUN1_CYCLES_HASH = %(run1_cycles)s
RUN1_E_FEATURES_SHA = '%(run1_e_feat)s'
_E1 = '%(run1_e_b64)s'
assert hashlib.sha256(_E1.encode()).hexdigest() == '%(run1_e_sha)s', 'embedded Run 1 arm E predictions corrupted'
RUN1_E = json.loads(_zr.decompress(_b64r.b64decode(_E1)).decode())
RUN1_OOF_SHA256 = '%(oof1_sha)s'

TW_RAW = {}          # {name: dict(bt, bw, p_s2, ...)} for the forked cycle workers

def _tw_worker(name):
    """T cycles from its heads; W's R-peaks through arm B (verbatim arm_b_cycles); equal-time fallback below 2 cycles."""
    try:
        L4 = AUDIO_META[name]['L4']
        x4 = np.asarray(AUDIO4[name][:L4], np.float64)
        raw = TW_RAW[name]
        cyc_t, d_t = TWC.t_cycles(x4, raw['bt'], raw['p_s2'], TWI.FRAME_T0, TWI.FRAME_S)
        pk4 = np.round(raw['bw'] * TARGET_SR).astype(int)
        pk4 = pk4[(pk4 >= 0) & (pk4 < L4)]
        cyc_w, d_w = arm_b_cycles(name, pk4)
        out = {}
        for arm, cyc, d in (('timing_net', cyc_t, d_t), ('waveform_net', cyc_w, d_w)):
            d = dict(d, fallback=False, n_cycles_net=len(cyc))
            if len(cyc) < N_MIN_CYCLES:
                cyc, d['fallback'] = equal_time_cycles(L4, TARGET_SR), True
            out[arm] = (cyc, d)
        return name, out, None
    except Exception as e:
        return name, None, '%%s: %%s' %% (type(e).__name__, e)
''' % dict(mod_b64=json.dumps(MOD_B64), mod_sha=json.dumps(MOD_SHA), freeze_sha=FREEZE_SHA, run1_cycles=json.dumps(RUN1_CYCLES_HASH),
           run1_e_feat=RUN1_E_FEATURES_SHA, run1_e_b64=RUN1_E_B64, run1_e_sha=RUN1_E_SHA, oof1_sha=OOF1_SHA)
insert_after('ext_run1_defs', [('code', 'ext_run2_defs', EXT2.strip('\n'))])

# ------------------------------------------------------------------ 7. Stage 4 for Run 2
replace_cell('s4_run', r'''
CYC = {arm: {} for arm in CYC_ARMS}
ARM_DIAG = {arm: {} for arm in CYC_ARMS}
ARM_FAILURES = []
_CD_ERROR = ([], {'worker_error': True, 'n_annotated': 0, 'n_beyond_portion': 0, 'n_phase_overlaps': 0, 'max_phase_overlap_samples': 0,
                  'annotated_coverage_fraction': np.nan, 'annotated_span_fraction': np.nan},
             [], {'n_anchors': 0, 'n_annotation_gaps': 0}, [], {}, [], {'n_matched_to_annotation': 0})

with stage('S4 boundaries'):
    # ---- arms C and E (Run 1's worker; rebuilt only to reproduce Run 1 and to give the annotated stretches) ----
    cd = {}
    t0 = time.time()
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_cd_worker, NAMES_ALL, chunksize=16), total=len(NAMES_ALL), desc='arms C/E', leave=False):
            if err is not None:
                ARM_FAILURES.append((name, 'C/E', err))
                payload = _CD_ERROR
            cd[name] = payload
    for n in NAMES_ALL:
        cyc_c, d_c, _cd_d, _dd_d, cyc_e, d_e, _ce2, _de2 = cd[n]
        CYC['pcg_seg'][n], ARM_DIAG['pcg_seg'][n] = cyc_c, d_c
        CYC['equal_time'][n], ARM_DIAG['equal_time'][n] = cyc_e, d_e
    del cd
    print('   arms C and E built for %d recordings in %.0f s' % (len(NAMES_ALL), time.time() - t0))

    # ---- the kept set: Run 1's 3,000 recordings (full run) ----
    if VALIDATION_APPLICABLE:
        _missing = [n for n in PINNED_KEEP if n not in set(NAMES_ALL)]
        assert not _missing, 'Run 1 recordings missing from this index: %s' % _missing[:5]
        KEEP = list(PINNED_KEEP)
    else:
        KEEP = [n for n in NAMES_ALL if len(CYC['pcg_seg'][n]) >= N_MIN_CYCLES and len(CYC['equal_time'][n]) >= N_MIN_CYCLES]
    _kset = set(KEEP)
    DROPPED = [(n, ['not in the Run 1 kept set']) for n in NAMES_ALL if n not in _kset]
    KEEP_ARR = np.array(KEEP)
    KEEP_PINNED_OK = (list(KEEP) == list(PINNED_KEEP)) if VALIDATION_APPLICABLE else None
    print('   kept set: %d recordings%s' % (len(KEEP), ' (the Run 1 kept set, pinned)' if VALIDATION_APPLICABLE else ''))

    # ---- frozen T and W on the GPU: seed-averaged traces, beats from the shared decoder ----
    fp_tw = CACHE_DIR / 'tw_traces.pkl'
    if fp_tw.exists():
        try:
            with open(fp_tw, 'rb') as fh:
                _o = pickle.load(fh)
            if _o.get('cfg') == TW_CFG_HASH:
                TW_RAW.update(_o['data'])
        except Exception:
            pass
    todo = [n for n in KEEP if n not in TW_RAW]
    t0 = time.time()
    for n in tqdm(todo, desc='T and W inference', leave=False):
        x4 = np.asarray(AUDIO4[n][:AUDIO_META[n]['L4']], np.float64)
        spec = TWI.spectrogram(x4, TARGET_SR)
        tt = TWI.forward(T_MODELS, spec, 'T', DEV)
        tw = TWI.forward(W_MODELS, spec, 'W', DEV)
        bt, rec_t = TWI.beats(tt, 'T')
        bw, rec_w = TWI.beats(tw, 'W')
        TW_RAW[n] = dict(bt=bt, bw=bw, p_s2=tt[1].astype(np.float32), rec_t=rec_t, rec_w=rec_w,
                         trace_sha_t=TWI.trace_hash(tt), trace_sha_w=TWI.trace_hash(tw))
    if todo:
        atomic_pickle(fp_tw, {'cfg': TW_CFG_HASH, 'data': TW_RAW})
        print('   T and W run on %d recordings in %.0f s' % (len(todo), time.time() - t0))
    else:
        print('   T and W traces loaded from cache')
    TW_TRACES_SHA = hashlib.sha256(''.join(TW_RAW[n]['trace_sha_t'] + TW_RAW[n]['trace_sha_w'] for n in KEEP).encode()).hexdigest()

    # ---- T and W cycles (worker pool) and the matched-coverage copies ----
    t0 = time.time()
    tw_out = {}
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_tw_worker, KEEP, chunksize=8), total=len(KEEP), desc='arms T/W', leave=False):
            if err is not None:
                ARM_FAILURES.append((name, 'T/W', err))
                L4_ = AUDIO_META[name]['L4']
                payload = {a: (equal_time_cycles(L4_, TARGET_SR), dict(fallback=True, worker_error=err)) for a in TW_ARMS}
            tw_out[name] = payload
    for n in KEEP:
        for a in TW_ARMS:
            CYC[a][n], ARM_DIAG[a][n] = tw_out[n][a]
        spans = annotated_spans(CYC['pcg_seg'][n])
        for a, src in (('timing_net_annot', 'timing_net'), ('waveform_net_annot', 'waveform_net'), ('equal_time_annot', 'equal_time')):
            c_ = restrict_to(CYC[src][n], spans)
            fb = len(c_) < N_MIN_CYCLES
            CYC[a][n] = CYC[src][n] if fb else c_
            ARM_DIAG[a][n] = dict(fallback=bool(fb), n_annotated_spans=len(spans), n_cycles_selected=len(c_))
    del tw_out
    print('   arms T and W built in %.0f s' % (time.time() - t0))
    TW_FALLBACK = {a: float(np.mean([bool(ARM_DIAG[a][n].get('fallback')) for n in KEEP])) for a in TW_ARMS + TW_COV_ARMS}
    TW_FAILED = [a for a, v in TW_FALLBACK.items() if v > S_FALLBACK_FAIL]
    print('   fallback share (equal-time windows used):', {a: round(100 * v, 2) for a, v in TW_FALLBACK.items()}, '%; arms above',
          100 * S_FALLBACK_FAIL, '% are reported as failed:', TW_FAILED)
    with open(RUN_DIR / 'arm_failures.json', 'w') as fh:
        json.dump(ARM_FAILURES, fh, indent=1)
    for arm in CYC_ARMS:
        for n in KEEP:
            check_cycles(CYC[arm][n], AUDIO_META[n]['L4'], arm + ' ' + n, strict=(arm != 'pcg_seg'))
    print('   every boundary of every arm lies inside its signal')

    # ---- diagnostics against arm C (descriptive; Gate B of the design) ----
    DIAG_ARMS = ['pcg_seg', 'equal_time'] + TW_ARMS + TW_COV_ARMS
    OFFSETS = {arm: {} for arm in DIAG_ARMS}
    DIAG_ROWS = []
    for n in KEEP:
        ref = CYC['pcg_seg'][n]
        for arm in DIAG_ARMS:
            rd, dl = rate_diag(CYC[arm][n], ref)
            OFFSETS[arm][n] = dl
            row = dict(recording=n, patient_id=PID_OF[n], age_band=AGE_OF.get(PID_OF[n], 'Unknown'), murmur=LAB_OF[n]['murmur'],
                       arm=arm, kept=True, analysed_s=AUDIO_META[n]['L4'] / TARGET_SR, n_windows=AUDIO_META[n]['n_win'])
            row.update(rd)
            for k_, v_ in ARM_DIAG[arm][n].items():
                row['arm_' + k_] = v_
            if arm in TW_ARMS:
                rec = TW_RAW[n]['rec_t' if arm == 'timing_net' else 'rec_w']
                row.update({'dec_' + k_: v_ for k_, v_ in rec.items()})
            DIAG_ROWS.append(row)
    DIAG = pd.DataFrame(DIAG_ROWS)
    DIAG.to_csv(RUN_DIR / 'diagnostics.csv', index=False)
    DK = DIAG
    cols = ['n_cycles', 'median_rr_ms', 'bpm', 'rr_cv', 'rr_jump', 'frac_fast', 'insertion_rate', 'deletion_rate',
            'median_offset_ms', 'median_abs_offset_ms', 'overdetection_ratio']
    DIAG_SUMMARY = {}
    print('\nrate-fidelity diagnostics against the annotation, medians over %d kept recordings' % len(KEEP))
    print('%-22s %6s %7s %6s %6s %6s %6s %8s %8s %6s' % ('arm', 'cycles', 'RR ms', 'bpm', 'ins', 'del', 'RRcv', 'off ms', '|off| ms', 'x det'))
    for arm in DIAG_ARMS:
        sub = DK[DK.arm == arm]
        med = {c: float(np.nanmedian(sub[c].astype(float))) for c in cols}
        pooled = np.concatenate([OFFSETS[arm][n] for n in KEEP if len(OFFSETS[arm][n])]) if any(len(OFFSETS[arm][n]) for n in KEEP) else np.array([])
        DIAG_SUMMARY[arm] = dict(med, n=int(len(sub)), offset_sd_ms=float(np.std(pooled)) if len(pooled) >= 2 else None, n_matched_pairs=int(len(pooled)))
        if arm in TW_FALLBACK:
            DIAG_SUMMARY[arm].update(fallback_frac=TW_FALLBACK[arm], failed_arm=bool(TW_FALLBACK[arm] > S_FALLBACK_FAIL))
        print('%-22s %6.0f %7.0f %6.0f %6.3f %6.3f %6.3f %+8.1f %8.1f %6.2f' % (
            arm, med['n_cycles'], med['median_rr_ms'], med['bpm'], med['insertion_rate'], med['deletion_rate'], med['rr_cv'],
            med['median_offset_ms'], med['median_abs_offset_ms'], med['overdetection_ratio']))
    _pt = DK[DK.arm == 'timing_net']
    DIAG_SUMMARY['timing_net'].update(onset_fallback_total=int(_pt['arm_n_onset_fallback'].fillna(0).sum()),
                                      malformed_total=int(_pt['arm_n_malformed'].fillna(0).sum()),
                                      frac_s2_acoustic_median=float(np.nanmedian(_pt['arm_frac_s2_acoustic'].astype(float))),
                                      s2_head_at_choice_median=float(np.nanmedian(_pt['arm_s2_head_at_choice_median'].astype(float))))
    _pw = DK[DK.arm == 'waveform_net']
    DIAG_SUMMARY['waveform_net'].update(fusion_fallback_frac=float(_pw['arm_fusion_fallback'].astype(bool).mean()),
                                        supported_frac_median=float(np.nanmedian(_pw['arm_supported_frac'].astype(float))),
                                        malformed_total=int(_pw['arm_n_malformed'].fillna(0).sum()))
    GLOBAL_OFFSET = {}
    print('arm T: S2 from an acoustic onset in %.0f%% of cycles (median per recording), median S2 probability at the chosen point %.2f | '
          'arm W: arm B fallback in %.1f%% of recordings, median supported fraction %.2f' % (
              100 * DIAG_SUMMARY['timing_net']['frac_s2_acoustic_median'], DIAG_SUMMARY['timing_net']['s2_head_at_choice_median'],
              100 * DIAG_SUMMARY['waveform_net']['fusion_fallback_frac'], DIAG_SUMMARY['waveform_net']['supported_frac_median']))
''')

# ------------------------------------------------------------------ 8. features: E reproduction recorded
patch('s5_features', "        print('   R1 feature matrices reproduced:', FEATURE_REPRO)",
      "        print('   R1 feature matrices reproduced:', FEATURE_REPRO)\n"
      "    E_FEATURES_SHA = hashlib.sha256(np.ascontiguousarray(FEATS['equal_time'].astype(np.float32)).tobytes()).hexdigest()\n"
      "    print('   arm E features against Run 1: %s (sha256 %s)' % (('reproduced' if E_FEATURES_SHA == RUN1_E_FEATURES_SHA else 'NOT reproduced') if VALIDATION_APPLICABLE else 'not applicable (reduced run)', E_FEATURES_SHA[:16]))")

# ------------------------------------------------------------------ 9. classification: pairs, self-checks on T, Run 2 messages
i = idx('s7_run')
txt = CELLS[i][2]
a = txt.index('PAIRS = [')
b = txt.index("('pcg_seg', 'equal_time_annot')]") + len("('pcg_seg', 'equal_time_annot')]")
CELLS[i] = (CELLS[i][0], 's7_run', txt[:a] + "PAIRS = [('timing_net', 'equal_time'), ('waveform_net', 'equal_time'), ('timing_net', 'waveform_net')]" + txt[b:])
patch('s7_run', "        r_, e_ = CV[task]['pcg_seg'], CV[task]['equal_time']", "        r_, e_ = CV[task]['timing_net'], CV[task]['equal_time']     # Run 2: T in place of C (C is not rescored)")
patch('s7_run', "        r_ = CV[task]['pcg_seg']", "        r_ = CV[task]['timing_net']")
patch('s7_run', "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')",
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 2 classifies only in the full, pre-registered run.')")
patch('s7_run', "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL,",
      "fallback_share=TW_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL, failed_arms=TW_FAILED, tw_stamp=TW_STAMP,\n"
      "                          tw_freeze_sha256=FREEZE_SHA, tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=TW_TRACES_SHA, run1_oof_sha256=RUN1_OOF_SHA256,")

# ------------------------------------------------------------------ 10. integrity gate for Run 2
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, r'''## Stage 8: integrity gate (Run 2)

Label-blind checks: the kept set equals the 3,000 Run 1 recordings; every task's folds equal the embedded child-grouped
assignment with no child on both sides; arms C, E and E_annot have Run 1's cycle hashes; arm E's feature matrix and
its out-of-fold predictions equal Run 1's (MLP predictions identical, probabilities within 1e-5, random forest
identical); the frozen T and W weights match the freeze record; a repeat five-seed fit returns identical
probabilities; the parallel workers return what an in-process fit returns. A failed gate means the run is not used.''')
CELLS[i8] = ('code', 's8_gate', r'''
with stage('S8 integrity gate'):
    gate = dict(kind='integrity (Run 2)', applicable=bool(VALIDATION_APPLICABLE))
    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        gate.update(ok=None, decided_on='not applicable (reduced run)')
        print('integrity gate: not applicable in a reduced run')
    else:
        gate['kept_set_pinned'] = bool(KEEP_PINNED_OK)
        gate['folds'] = FOLD_CHECK
        gate['folds_ok'] = all(all(v.values()) for v in FOLD_CHECK.values()) and set(FOLD_CHECK) == set(TASKS)
        gate['run1_cycles'] = {a: dict(run2=cycles_hash(CYC[a], KEEP), run1=RUN1_CYCLES_HASH[a]) for a in RUN1_CYCLES_HASH}
        gate['run1_cycles_ok'] = all(v['run2'] == v['run1'] for v in gate['run1_cycles'].values())
        gate['run1_e_features_ok'] = bool(E_FEATURES_SHA == RUN1_E_FEATURES_SHA)
        e_chk = {}
        for t in TASKS:
            r = CV[t]['equal_time']; ref = RUN1_E[t]
            same_rows = list(map(str, r['names'])) == ref['names'] and np.array_equal(np.asarray(r['y']), np.asarray(ref['y']))
            e_chk[t] = dict(same_rows=bool(same_rows),
                            mlp_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_mlp']), np.asarray(ref['mlp_pred']))),
                            mlp_proba_max_abs_diff=float(np.max(np.abs(np.asarray(r['proba_mlp'], float) - np.asarray(ref['mlp_proba'], float)))) if same_rows else None,
                            rf_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_rf']), np.asarray(ref['rf_pred']))))
        gate['run1_e_predictions'] = e_chk
        gate['run1_e_predictions_ok'] = all(v['mlp_pred_identical'] and v['rf_pred_identical'] and v['mlp_proba_max_abs_diff'] is not None
                                            and v['mlp_proba_max_abs_diff'] < 1e-5 for v in e_chk.values())
        gate['tw_weights_ok'] = bool(TW_WEIGHTS_OK)
        gate['tw_fallback'] = TW_FALLBACK
        gate['tw_failed_arms_reported'] = TW_FAILED
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
        checks = [gate['kept_set_pinned'], gate['folds_ok'], gate['run1_cycles_ok'], gate['run1_e_features_ok'],
                  gate['run1_e_predictions_ok'], gate['tw_weights_ok'], gate['repeat_fit_ok']]
        if MLP_POOL is not None:
            Xtr0_, Xte0_ = sc_.transform(Xm_[tr_]), sc_.transform(Xm_[te_])
            p_pool_, _ = MLP_POOL.submit(_mlpw.fit_job, Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], Xte0_, 2, ENSEMBLE_SEEDS[0]).result()
            m0_, _ = fit_mlp(Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], 2, ENSEMBLE_SEEDS[0])
            gate['pool_vs_inprocess_max_abs_diff'] = float(np.max(np.abs(p_pool_ - mlp_proba(m0_, Xte0_))))
            gate['pool_ok'] = gate['pool_vs_inprocess_max_abs_diff'] < 1e-6
            checks.append(gate['pool_ok'])
        VALIDATION_OK = bool(all(checks))
        gate.update(ok=VALIDATION_OK, decided_on='integrity checks')
        print('integrity gate:', 'PASSED' if VALIDATION_OK else 'FAILED',
              {k: v for k, v in gate.items() if k not in ('folds', 'run1_e_predictions', 'run1_cycles')})
        print('   Run 1 cycle hashes:', gate['run1_cycles'])
        print('   arm E against Run 1:', gate['run1_e_predictions'])
    RESULTS['validation_gate'] = gate
    write_results()
    if MLP_POOL is not None:
        MLP_POOL.shutdown()
''')

# ------------------------------------------------------------------ 11. manifest: T/W checkpoint in place of the generator
patch('s11_manifest', "        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),",
      "        checkpoint=dict(tw_checkpoint_dir=str(TW_SRC), freeze_sha256=FREEZE_SHA, models={k: v['weights_sha256'] for k, v in FREEZE_REC['models'].items()},\n"
      "                        tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=TW_TRACES_SHA, embedded_modules=_MOD_SHA),")
patch('s11_manifest', "        generator=dict(batch_start=GAN_BATCH, batch_final=GEN_BATCH_USED, oom_events=OOM_EVENTS, signal_length=SIGNAL_LENGTH, gan_sr=GAN_SR, ecg_cache_dtype='float16'),",
      "        generator=None,")

# ------------------------------------------------------------------ 12. markdown that described Run 1 only
for k, (kind, nm, text) in enumerate(CELLS):
    if kind == 'markdown' and 'Run 1' in text and nm is None and k > 0 and 'Run 2' not in text:
        CELLS[k] = (kind, nm, text.replace('(Run 1)', '(Run 1 cells, used unchanged in Run 2)'))

# ------------------------------------------------------------------ notebook source hash, then write
iprov = next(i for i, c in enumerate(CELLS) if c[0] == 'code' and c[2].startswith('VERBATIM_PROVENANCE = '))
h = hashlib.sha256()
for kind, name, text in CELLS:
    if text.startswith('VERBATIM_PROVENANCE = '):
        continue
    h.update(text.encode('utf-8')); h.update(b'\x00')
SRC_SHA = h.hexdigest()
ptxt = CELLS[iprov][2]
old = ptxt[ptxt.index("NOTEBOOK_SOURCE_SHA256 = '"):]
old = old[:old.index('\n')]
CELLS[iprov] = (CELLS[iprov][0], CELLS[iprov][1], ptxt.replace(old, "NOTEBOOK_SOURCE_SHA256 = '%s'   # SHA-256 over every other cell's source, filled in by build_run2.py" % SRC_SHA))

cells = []
for kind, name, text in CELLS:
    assert '—' not in text, 'em dash in cell %s' % name
    cells.append(new_markdown_cell(text) if kind == 'markdown' else new_code_cell(text, metadata={'name': name} if name else {}))
nb = new_notebook(cells=cells)
nb.metadata = dict(nb1['metadata'])
nb.metadata['colab'] = dict(nb1['metadata'].get('colab', {}), name='Extension_Run2.ipynb', machine_shape='hm', gpuType='A100')
nb.metadata['built_by'] = 'extension_study/notebooks/build_run2.py (patches over the locked Extension_Run1.ipynb)'
nb.metadata['notebook_source_sha256'] = SRC_SHA
nb.metadata['run1_notebook_file_sha256'] = lock['run1_notebook_file_sha256']
nb.metadata['embedded'] = dict(nb1['metadata'].get('embedded', {}), tw_freeze_sha256=FREEZE_SHA, modules_sha256=MOD_SHA,
                               run1_cycles_hashes=RUN1_CYCLES_HASH, run1_e_features_sha256=RUN1_E_FEATURES_SHA, run1_oof_sha256=OOF1_SHA)
tmp = Path(tempfile.mkdtemp())
for k, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % k)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Run 2 notebook written: %s (%d cells) | source sha256 %s' % (OUT_NB, len(nb.cells), SRC_SHA[:16]))
