#!/usr/bin/env python
"""
build_exp1.py: assembles Experiment1_InBandNoise.ipynb (Experiment 1; plan in
'Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md') from the locked Extension_Run1.ipynb, patched by cell name,
in the same way as build_run2.py and build_exp0c.py.

What changes from Run 1, and nothing else:
  1. run folder Extension_Study/runs/exp1/<timestamp>; no generator; the frozen T and W checkpoint folder of Run 2
     (every file and weight hash checked against the embedded freeze record);
  2. recordings: the Run 1 kept set restricted to those labelled for grading, timing or shape (610); tasks: those three;
  3. Stage 4: arms C and E (Run 1 worker); the primary Springer setting, T and W on the clean audio as reproduction
     controls; then the interference conditions of the plan (extstudy/noise.py: gauss, burst, esc50 at 10 and 0 dB):
     Springer, T and W segment the corrupted audio (segmentation-only arms); the both-stage arms share those cycles
     (and arm C's and E's) and read their features from the corrupted audio;
  4. Stage 5: features from the clean audio for the controls and segmentation-only arms, from the corrupted audio
     for the both-stage arms (the Run 1 workers read the global AUDIO4, which is bound per condition);
  5. folds (embedded, child-grouped), five-seed MLP and random forest: Run 1's cells unchanged;
  6. integrity gate: kept subset, folds, the five controls reproduce Runs 1 and 2, arm B's construction equals its
     verbatim copy, repeat fit, pool check.
H1 and the estimation contrasts are computed locally (notebooks/exp1_analysis.py).

Usage:  python build_exp1.py   (writes Experiment1_InBandNoise.ipynb beside this file)
"""
import base64, hashlib, json, py_compile, tempfile, zlib
from pathlib import Path

import numpy as np
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PHD = STUDY.parent
RUN1_NB = HERE / 'Extension_Run1.ipynb'
OUT_NB = HERE / 'Experiment1_InBandNoise.ipynb'
PLAN = Path('E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md')
TW_STAMP = '20261018_0908'
TW_LOCAL = STUDY / 'runs' / 'run2_freeze' / TW_STAMP
RUN1_DIR = PHD / 'Results' / 'A100 Results' / 'extracted' / 'runs' / 'run1' / '20261016_1516'
RUN2_DIR = STUDY / 'runs' / 'run2' / '20261018_1358'
SETS_JSON = STUDY / 'runs' / 'exp0c_prep' / 'exp0c_sets.json'
MODULES = ['__init__.py', 'frontend.py', 'model.py', 'robust_beats.py', 'decoder.py', 'infer.py', 'cycles.py', 'corrupt.py', 'noise.py']
EXP1_TASKS = ('grading', 'timing', 'shape')
CONTROLS = {'pcg_seg': ('run1', 'pcg_seg'), 'equal_time': ('run1', 'equal_time'), 'springer_wpk': ('run1', 'springer_wpk'),
            'timing_net': ('run2', 'timing_net'), 'waveform_net': ('run2', 'waveform_net')}
ESC50_URL = 'https://github.com/karolpiczak/ESC-50/archive/master.zip'

sha = lambda b: hashlib.sha256(b).hexdigest()
lock = json.load(open(STUDY / 'prereg' / 'RUN1_LOCK.json'))
assert sha(RUN1_NB.read_bytes()) == lock['run1_notebook_file_sha256'], 'Extension_Run1.ipynb differs from its lock'
PLAN_SHA = sha(PLAN.read_bytes())
assert open(PLAN.with_suffix('.sha256')).read().split()[0] == PLAN_SHA, 'EXP1_PREREG.md changed after it was hashed'
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
    i = idx(name)
    if i > 0 and CELLS[i - 1][0] == 'markdown':
        CELLS[i - 1] = ('markdown', None, note)
        del CELLS[i]
    else:
        CELLS[i] = ('markdown', None, note)


def insert_after(name, cells):
    i = idx(name)
    CELLS[i + 1:i + 1] = cells


# ------------------------------------------------------------------ embedded: code, freeze record, names, reference predictions
MODS = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
for m, b in MODS.items():
    assert 'circor' not in b.decode('utf-8').lower(), 'CirCor reference in %s' % m
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in MODS.items()}
MOD_SHA = {m: sha(b) for m, b in MODS.items()}
nb2 = json.load(open(HERE / 'Extension_Run2.ipynb', encoding='utf-8'))
for m in ('frontend.py', 'robust_beats.py', 'decoder.py', 'cycles.py'):             # the boundary-defining modules must be Run 2's
    assert MOD_SHA[m] == nb2['metadata']['embedded']['modules_sha256'][m], '%s differs from the Run 2 version' % m
RUN2_MODULES_DIFF = {m: dict(exp1=MOD_SHA[m], run2=nb2['metadata']['embedded']['modules_sha256'][m]) for m in ('model.py', 'infer.py')
                     if MOD_SHA[m] != nb2['metadata']['embedded']['modules_sha256'][m]}   # kind 'A' additions (Experiment 2); T/W unchanged, the gate reproduces Run 2
nb0b = json.load(open(HERE / 'Experiment0b_FailureConcentration.ipynb', encoding='utf-8'))
assert MOD_SHA['corrupt.py'] == nb0b['metadata']['embedded']['modules_sha256']['corrupt.py'], 'corrupt.py differs from the Experiment 0b version'
FREEZE_BYTES = (TW_LOCAL / 'freeze.json').read_bytes()
FREEZE_SHA = sha(FREEZE_BYTES)
FREEZE_REC = json.loads(FREEZE_BYTES)
assert FREEZE_REC['smoke'] is False and FREEZE_REC['label_check']['status'] == 'PASSED'
assert nb2['metadata']['embedded']['tw_freeze_sha256'] == FREEZE_SHA, 'freeze.json differs from the Run 2 record'
SETS = json.load(open(SETS_JSON))
NAMES = SETS['names']
assert len(NAMES) == 610
NAMES_B64 = base64.b64encode(zlib.compress(json.dumps(NAMES).encode(), 9)).decode()

o1 = np.load(RUN1_DIR / 'oof_predictions.npz', allow_pickle=False)
o2 = np.load(RUN2_DIR / 'oof_predictions.npz', allow_pickle=False)
REF = {}
for tag, (run, src) in CONTROLS.items():
    z = o1 if run == 'run1' else o2
    REF[tag] = {}
    for t in EXP1_TASKS:
        p = '%s__%s__' % (t, src)
        assert np.array_equal(z[p + 'names'], o1['%s__pcg_seg__names' % t])
        REF[tag][t] = dict(names=z[p + 'names'].astype(str).tolist(), y=z[p + 'y'].astype(int).tolist(),
                           mlp_pred=z[p + 'mlp__pred'].astype(int).tolist(), mlp_proba=z[p + 'mlp__proba'].astype(float).round(8).tolist(),
                           rf_pred=z[p + 'rf__pred'].astype(int).tolist())
REF_B64 = base64.b64encode(zlib.compress(json.dumps(REF).encode(), 9)).decode()
REF_SHA = sha(REF_B64.encode())
OOF1_SHA = sha((RUN1_DIR / 'oof_predictions.npz').read_bytes())
OOF2_SHA = sha((RUN2_DIR / 'oof_predictions.npz').read_bytes())

# ------------------------------------------------------------------ 0. header
CELLS[0] = ('markdown', None, r'''# Experiment 1: in-band interference at test time

How much of the gain of cardiac-phase gating over equal-time windows survives when interference confined to the murmur
band (20 to 500 Hz) and not locked to the cardiac cycle is added to the recordings? Plan: `Next Study - Beat-Accurate
Synthetic ECG/EXP1_PREREG.md` (hashed before any corrupted audio, boundary set, feature or score existed). Built by
`extension_study/notebooks/build_exp1.py` from the locked Run 1 notebook, patched cell by cell.

**What runs.** The 610 recordings labelled for grading, timing or shape. Five reproduction controls on the clean audio
(arms C, E, Springer, T and W; their predictions must equal Runs 1 and 2). Six conditions (gauss, burst, esc50 at
10 and 0 dB in-band SNR); in each, Springer, T and W segment the corrupted audio and the features are read from the
clean audio (`*_seg_*` arms), and arms E, C, Springer, T and W are scored with their boundaries and features read from
the corrupted audio (`*_both_*` arms). Everything goes through Run 1's feature extractor and five-seed MLP on the
child-grouped folds.

**How to run.** Runtime > Change runtime type > **A100 GPU** with High-RAM; leave `DRY_RUN = False`; Runtime > Run all.
The notebook fetches the ESC-50 archive from GitHub (about 600 MB); if that fails, it runs the gauss and burst
conditions and reports esc50 as not run. Expected time about 3 to 4 hours; if the session drops, set `RESUME_DIR` to
the run folder and run all again (traces, features and fits are cached).

**Afterwards.** Download `Extension_Study/runs/exp1/<timestamp>` without its `cache` folder (results.json,
oof_predictions.npz, exp1_boundaries.json, exp1_noise_manifest.json, run_manifest.json, stage_seconds.json and the
executed notebook); the analysis runs locally (`notebooks/exp1_analysis.py`).
''')

# ------------------------------------------------------------------ 1. flags, manual override, environment
patch('flags', "R1_RUN_DIR = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip",
      "R1_RUN_DIR = None\n"
      "TW_STAMP = '%s'   # frozen T and W (Extension_Study/checkpoints/<stamp>)\n"
      "TW_CHECKPOINT_DIR = None   # None = the Drive folder above; a local path for a local run\n"
      "REQUIRE_GPU = 'A100'       # full run only: the GPU class of Runs 1 and 2, so the controls reproduce bit for bit\n"
      "EXP1_PLAN_SHA256 = '%s'   # Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md\n"
      "EXP1_TASKS = ('grading', 'timing', 'shape')\n"
      "EXP1_KINDS = ('gauss', 'burst', 'esc50')         # interference kinds (extstudy/noise.py), in the plan's order\n"
      "EXP1_LEVELS = (10, 0)                            # in-band SNR in dB\n"
      "ESC50_URL = '%s'\n"
      "ESC50_CLASSES = ('crying_baby', 'breathing', 'coughing')" % (TW_STAMP, PLAN_SHA, ESC50_URL))
patch('manual_override', "MANUAL_OUTPUT_ROOT = None",
      "MANUAL_OUTPUT_ROOT = None\nMANUAL_ESC50_DIR = None\n# e.g. '/content/drive/MyDrive/ESC-50-master' (a folder with meta/esc50.csv and audio/); None = fetch the GitHub archive")
patch('env', "GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32",
      "GAN_BATCH = None\n"
      "if (not DRY_RUN) and LIMIT_RECORDINGS == 0 and REQUIRE_GPU:\n"
      "    assert REQUIRE_GPU in GPU_NAME, 'Experiment 1 must run on an %s (the class of Runs 1 and 2) so the controls reproduce; this runtime has %s' % (REQUIRE_GPU, GPU_NAME)")
patch('env', "print('generator batch', GAN_BATCH, '| feature workers', N_PROC)", "print('feature workers', N_PROC)")

# ------------------------------------------------------------------ 2. resolve: no generator; the T/W checkpoint folder; exp1 run folder
patch('resolve', """    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
""", "")
i = idx('resolve')
txt = CELLS[i][2]
a, b = txt.index('    # ---- generator checkpoint ----'), txt.index('    # ---- output folder and run flags ----')
CELLS[i] = (CELLS[i][0], 'resolve', txt[:a] + '''    # ---- Experiment 1: no generator; the frozen T and W checkpoint folder of Run 2 ----
    GAN_CHECKPOINT, CKPT_SHA256, CKPT_HASHES = None, None, {}
    TW_SRC = Path(TW_CHECKPOINT_DIR) if TW_CHECKPOINT_DIR else Path(DRIVE_ROOT + '/JBHI-Revision/Extension_Study/checkpoints/' + TW_STAMP)
    assert (TW_SRC / 'freeze.json').exists(), 'frozen T/W checkpoint folder not found: %s' % TW_SRC

''' + txt[b:])
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/exp1'))")
patch('resolve', "run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)",
      "run='exp1_inband_noise', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', exp1_plan_sha256=EXP1_PLAN_SHA256, tw_stamp=TW_STAMP)")
patch('resolve', "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),",
      "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('TW_SRC', TW_SRC),")
patch('resolve', "    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))",
      "    print('paired wav+tsv files: %d | frozen T/W folder: %s' % (N_PAIRS, TW_SRC))")
drop('v_generator', '*Experiment 1: the generator architecture cell is not used.*')
drop('v_ckpt_load', '*Experiment 1: no generator checkpoint is loaded.*')
drop('s3_generator', '## Stage 3: not run\n\nExperiment 1 builds no synthetic ECG from the GAN.')
patch('s2_audio', "    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True",
      "    ECG_CHUNK_OK[k] = True       # no generator: the 8 kHz audio is never loaded")

# ------------------------------------------------------------------ 3. arm lists
patch('s4_arm_defs', """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS""", """R1_ARMS = list(ARMS)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
CONTROL_ARMS = ['pcg_seg', 'equal_time', 'springer_wpk', 'timing_net', 'waveform_net']       # clean reproduction controls
EXP1_CONDS = []                                        # (condition, kind, SNR dB, condition seed); EXP1_PREREG.md
for _ki, _kind in enumerate(EXP1_KINDS):
    for _li, _snr in enumerate(EXP1_LEVELS):
        EXP1_CONDS.append(('%s%d' % (_kind, _snr), _kind, _snr, 300000 + 1000 * _ki + (_li + 1)))
SEG_TAGS = {c[0]: ['%s_seg_%s' % (a, c[0]) for a in ('S', 'T', 'W')] for c in EXP1_CONDS}
BOTH_TAGS = {c[0]: ['%s_both_%s' % (a, c[0]) for a in ('E', 'C', 'S', 'T', 'W')] for c in EXP1_CONDS}
BASE_OF = {'E': 'equal_time', 'C': 'pcg_seg', 'S': 'springer_wpk', 'T': 'timing_net', 'W': 'waveform_net'}
def _exp1_arms(conds):
    return list(CONTROL_ARMS) + [t for c in conds for t in SEG_TAGS[c[0]] + BOTH_TAGS[c[0]]]
ARMS = _exp1_arms(EXP1_CONDS)
assert len(ARMS) == 5 + 8 * len(EXP1_CONDS) and len(set(ARMS)) == len(ARMS)""")
patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """ARM_LETTER.update({'springer_wpk': 'S', 'timing_net': 'T', 'waveform_net': 'W'})
ARM_LABEL.update({'springer_wpk': 'S Springer (clean control)', 'timing_net': 'T timing network (clean control)', 'waveform_net': 'W waveform network (clean control)'})
for _c in EXP1_CONDS:
    for _t in SEG_TAGS[_c[0]] + BOTH_TAGS[_c[0]]:
        ARM_LETTER[_t] = _t
        ARM_LABEL[_t] = 'Experiment 1 ' + _t
print('Experiment 1: %d conditions %s, %d arms | arm configuration hash %s' % (len(EXP1_CONDS), [c[0] for c in EXP1_CONDS], len(ARMS), ARM_CFG_HASH))""")

# ------------------------------------------------------------------ 4. Experiment 1 definitions (embedded code, frozen models, interference)
EXP1_DEFS = r'''
# ---------------- Experiment 1: embedded code, frozen T and W, reference predictions, interference ----------------
import base64 as _b64x, zlib as _zx, shutil as _shx, urllib.request as _url, zipfile as _zf
_MOD_B64 = @@MOD_B64@@
_MOD_SHA = @@MOD_SHA@@
_code_root = (Path('/content') if IN_COLAB else RUN_DIR) / 'exp1_code'
(_code_root / 'extstudy').mkdir(parents=True, exist_ok=True)
for _m, _b in _MOD_B64.items():
    _src = _b64x.b64decode(_b)
    assert hashlib.sha256(_src).hexdigest() == _MOD_SHA[_m], 'embedded module %s corrupted' % _m
    (_code_root / 'extstudy' / _m).write_bytes(_src)
for _k in [k for k in sys.modules if k == 'extstudy' or k.startswith('extstudy.')]:
    del sys.modules[_k]
sys.path.insert(0, str(_code_root))
from extstudy import infer as TWI, cycles as TWC, corrupt as KOR, noise as NZ
assert TWC.FUSION == FUSION and TWC.PHASE_KEYS == tuple(PHASE_KEYS) and TWC.TARGET_SR == TARGET_SR, 'cycle constants differ from arm B'
assert set(EXP1_KINDS) <= set(NZ.KINDS) and tuple(NZ.ESC50_CLASSES) == tuple(ESC50_CLASSES), 'noise.py constants differ from the flags'

FREEZE_SHA = '@@FREEZE_SHA@@'
_tw_local = (Path('/content') if IN_COLAB else RUN_DIR) / 'tw_checkpoints' / TW_STAMP
_tw_local.mkdir(parents=True, exist_ok=True)
for _f in sorted(p.name for p in TW_SRC.iterdir() if p.suffix in ('.json', '.pt')):
    _drive_retry(_shx.copyfile, TW_SRC / _f, _tw_local / _f) if IN_COLAB else _shx.copyfile(TW_SRC / _f, _tw_local / _f)
assert hashlib.sha256((_tw_local / 'freeze.json').read_bytes()).hexdigest() == FREEZE_SHA, 'freeze.json differs from the frozen record'
FREEZE_REC = json.load(open(_tw_local / 'freeze.json'))
T_MODELS, _ = TWI.load_models(str(_tw_local), 'T', DEV)          # file and weight hashes checked inside
W_MODELS, _ = TWI.load_models(str(_tw_local), 'W', DEV)
TW_WEIGHTS_OK = True
TW_CFG = dict(freeze_sha256=FREEZE_SHA, modules=_MOD_SHA, win_frames=TWI.WIN_FRAMES, overlap=TWI.OVERLAP, t_export=TWC.T_EXPORT,
              n_min=N_MIN_CYCLES, fusion=FUSION)
TW_CFG_HASH = hashlib.sha1(json.dumps(TW_CFG, sort_keys=True, default=str).encode()).hexdigest()[:16]
print('frozen T and W loaded and verified: %s | T/W configuration hash %s' % ([n for n, _ in T_MODELS + W_MODELS], TW_CFG_HASH))

_R = '@@REF_B64@@'
assert hashlib.sha256(_R.encode()).hexdigest() == '@@REF_SHA@@', 'embedded reference predictions corrupted'
REF = json.loads(_zx.decompress(_b64x.b64decode(_R)).decode())      # Run 1: pcg_seg, equal_time, springer_wpk; Run 2: timing_net, waveform_net
del _R
EXP1_NAMES = json.loads(_zx.decompress(_b64x.b64decode('@@NAMES_B64@@')).decode())
assert len(EXP1_NAMES) == 610
RUN1_OOF_SHA256, RUN2_OOF_SHA256 = '@@OOF1_SHA@@', '@@OOF2_SHA@@'

# ---- ESC-50 clip bank (fetched once; a failure drops the esc50 conditions, which are then reported as not run) ----
ESC50_OK, ESC50_BANK, ESC50_NOTE = False, None, ''
try:
    if MANUAL_ESC50_DIR:
        _esc_root = Path(MANUAL_ESC50_DIR)
    else:
        _esc_zip = (Path('/content') if IN_COLAB else RUN_DIR) / 'esc50_master.zip'
        _esc_dir = _esc_zip.parent / 'esc50'
        if not (_esc_dir / 'ESC-50-master' / 'meta' / 'esc50.csv').exists():
            print('fetching the ESC-50 archive (about 600 MB) from %s ...' % ESC50_URL)
            _url.urlretrieve(ESC50_URL, str(_esc_zip))
            with _zf.ZipFile(_esc_zip) as _z:
                _z.extractall(str(_esc_dir))
        _esc_root = _esc_dir / 'ESC-50-master'
    ESC50_BANK = NZ.load_esc50_bank(str(_esc_root), TARGET_SR, ESC50_CLASSES)
    ESC50_OK = True
    print('ESC-50 bank: %d clips of %s | bank sha256 %s' % (len(ESC50_BANK['clips']), ESC50_CLASSES, ESC50_BANK['bank_sha256'][:16]))
except Exception as _e:
    ESC50_NOTE = '%s: %s' % (type(_e).__name__, _e)
    print('ESC-50 unavailable (%s): the esc50 conditions will be reported as not run' % ESC50_NOTE)
if not ESC50_OK:
    EXP1_CONDS = [c for c in EXP1_CONDS if c[1] != 'esc50']
    ARMS = _exp1_arms(EXP1_CONDS)
COND_OF = {c[0]: c for c in EXP1_CONDS}

# ---- audio binding: Run 1's workers read the global AUDIO4; a condition's corrupted audio is bound to that name ----
AUDIO4_CLEAN = None
def _bind_audio(d):
    """Bind the corrupted audio of a condition (or the clean audio, d=None) to the global name the workers read."""
    global AUDIO4, AUDIO4_CLEAN
    if AUDIO4_CLEAN is None:
        AUDIO4_CLEAN = AUDIO4
    AUDIO4 = AUDIO4_CLEAN if d is None else d

def make_noisy(cond, names):
    """The corrupted audio of one condition for the given recordings (deterministic per recording) and the mixing record."""
    c, kind, snr, seed = COND_OF[cond]
    src = AUDIO4_CLEAN if AUDIO4_CLEAN is not None else AUDIO4
    out, info = {}, {}
    for n in names:
        y, inf = NZ.mix(src[n][:AUDIO_META[n]['L4']], kind, snr, NZ.rng_for(seed, n), TARGET_SR, bank=ESC50_BANK)
        out[n], info[n] = y, inf
    return out, info

EXP1_RAW = {}          # {name: dict(bt, bw, p_s2, rec_t, rec_w, ...)} of the audio being segmented (visible to forked workers)

def _infer_tw(audio, names, cache_name):
    """Seed-averaged T and W traces and decoded beats on `audio`; cached per condition under the T/W configuration hash."""
    fp = CACHE_DIR / cache_name
    raw = {}
    if fp.exists():
        try:
            with open(fp, 'rb') as fh:
                _o = pickle.load(fh)
            if _o.get('cfg') == TW_CFG_HASH:
                raw = _o['data']
        except Exception:
            pass
    todo = [n for n in names if n not in raw]
    for n in tqdm(todo, desc='T and W inference ' + cache_name, leave=False):
        x4 = np.asarray(audio[n][:AUDIO_META[n]['L4']], np.float64)
        spec = TWI.spectrogram(x4, TARGET_SR)
        tt = TWI.forward(T_MODELS, spec, 'T', DEV)
        tw = TWI.forward(W_MODELS, spec, 'W', DEV)
        bt, rec_t = TWI.beats(tt, 'T')
        bw, rec_w = TWI.beats(tw, 'W')
        raw[n] = dict(bt=bt, bw=bw, p_s2=tt[1].astype(np.float32), rec_t=rec_t, rec_w=rec_w,
                      trace_sha_t=TWI.trace_hash(tt), trace_sha_w=TWI.trace_hash(tw))
    if todo:
        atomic_pickle(fp, {'cfg': TW_CFG_HASH, 'data': raw})
    return raw

def _tw_cycles_worker(name):
    """T cycles from its heads; W's R-peaks through arm B's construction (cycles.b_cycles, the verbatim copy); both on
    the audio bound to AUDIO4; equal-time fallback below N_MIN_CYCLES."""
    try:
        L4 = AUDIO_META[name]['L4']
        x4 = np.asarray(AUDIO4[name][:L4], np.float64)
        raw = EXP1_RAW[name]
        cyc_t, d_t = TWC.t_cycles(x4, raw['bt'], raw['p_s2'], TWI.FRAME_T0, TWI.FRAME_S)
        pk4 = np.round(raw['bw'] * TARGET_SR).astype(int)
        pk4 = pk4[(pk4 >= 0) & (pk4 < L4)]
        cyc_w, d_w = TWC.b_cycles(x4, pk4)
        out = {}
        for arm, cyc, d in (('T', cyc_t, d_t), ('W', cyc_w, d_w)):
            d = dict(d, fallback=False, n_cycles_net=len(cyc))
            if len(cyc) < N_MIN_CYCLES:
                cyc, d['fallback'] = equal_time_cycles(L4, TARGET_SR), True
            out[arm] = (cyc, d)
        return name, out, None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

def _springer_worker(name):
    """The primary Springer setting on the audio bound to AUDIO4; equal-time fallback below N_MIN_CYCLES."""
    try:
        L4 = AUDIO_META[name]['L4']
        x4 = np.asarray(AUDIO4[name][:L4], float)
        states, info = springer.run_springer(x4, TARGET_SR, SPRINGER_PARAMS, return_info=True, **S_VARIANTS['springer_wpk'])
        cyc = states_to_cycles(states, L4)
        d = dict(hr_bpm=float(info['hr']), systolic_s=float(info['systolic_time']), n_cycles_springer=len(cyc),
                 unassigned_frac=float(np.mean(np.asarray(states) == 0)), fallback=False)
        if len(cyc) < N_MIN_CYCLES:
            cyc, d['fallback'] = equal_time_cycles(L4, TARGET_SR), True
        return name, (cyc, d), None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

def _arm_b_check_worker(name):
    """On the clean audio, cycles.b_cycles must equal the notebook's arm_b_cycles (W's construction in Run 2)."""
    try:
        L4 = AUDIO_META[name]['L4']
        raw = EXP1_RAW[name]
        pk4 = np.round(raw['bw'] * TARGET_SR).astype(int)
        pk4 = pk4[(pk4 >= 0) & (pk4 < L4)]
        a = arm_b_cycles(name, pk4)[0]
        b = TWC.b_cycles(np.asarray(AUDIO4[name][:L4], np.float64), pk4)[0]
        return name, bool(a == b), None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)

def _bstats(cyc, name):
    """Boundary quality of one arm's cycles against arm C: cycle-start F1 and S2 onset offset (extstudy.corrupt)."""
    ref = CYC['pcg_seg'][name]
    bs = KOR.boundary_stats(cyc, ref, TARGET_SR)
    off, npair = KOR.s2_onset_offset_ms(cyc, ref, TARGET_SR)
    return dict(bs, s2_offset_ms=off, s2_pairs=npair)
print('Experiment 1 code ready: %d conditions %s | %d arms | ESC-50 %s | noise.py sha256 %s' % (
    len(EXP1_CONDS), [c[0] for c in EXP1_CONDS], len(ARMS), 'ready' if ESC50_OK else 'NOT RUN', _MOD_SHA['noise.py'][:16]))
'''
for k, v in dict(MOD_B64=json.dumps(MOD_B64), MOD_SHA=json.dumps(MOD_SHA), FREEZE_SHA=FREEZE_SHA, REF_B64=REF_B64, REF_SHA=REF_SHA,
                 NAMES_B64=NAMES_B64, OOF1_SHA=OOF1_SHA, OOF2_SHA=OOF2_SHA).items():
    EXP1_DEFS = EXP1_DEFS.replace('@@%s@@' % k, v)
assert '@@' not in EXP1_DEFS
insert_after('ext_run1_defs', [('code', 'exp1_defs', EXP1_DEFS.strip('\n'))])

# ------------------------------------------------------------------ 5. Stage 4 for Experiment 1
replace_cell('s4_run', r'''
CYC = {arm: {} for arm in ARMS}
ARM_DIAG = {arm: {} for arm in ARMS}
ARM_FAILURES = []
EXP1_INFO, EXP1_AUDIO_SHA, EXP1_COND_SHA = {}, {}, {}
with stage('S4 boundaries'):
    for _t in [t for t in list(TASKS) if t not in EXP1_TASKS]:
        del TASKS[_t]                                                          # Experiment 1: grading, timing and shape only
    _lab = lambda n: any(LAB_OF[n][TASKS[t]] != -1 for t in TASKS)
    _in1 = set(EXP1_NAMES)
    if VALIDATION_APPLICABLE:
        _missing = [n for n in PINNED_KEEP if n not in set(NAMES_ALL)]
        assert not _missing, 'Run 1 recordings missing from this index: %s' % _missing[:5]
        KEEP = [n for n in PINNED_KEEP if _lab(n)]
        assert sorted(KEEP) == sorted(EXP1_NAMES), 'the labelled kept set differs from the 610 recordings of the plan'
    else:
        KEEP = [n for n in NAMES_ALL if _lab(n) and n in _in1]                # reduced run: labelled recordings of the plan's set only
    _bind_audio(None)
    t0 = time.time()
    _cd = {}
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_cd_worker, KEEP, chunksize=16), total=len(KEEP), desc='arms C/E', leave=False):
            assert err is None, (name, err)
            _cd[name] = payload
    for n in KEEP:
        CYC['pcg_seg'][n], ARM_DIAG['pcg_seg'][n] = _cd[n][0], _cd[n][1]
        CYC['equal_time'][n], ARM_DIAG['equal_time'][n] = _cd[n][4], _cd[n][5]
    del _cd
    KEEP = [n for n in KEEP if len(CYC['pcg_seg'][n]) >= N_MIN_CYCLES] if not VALIDATION_APPLICABLE else KEEP
    KEEP_ARR = np.array(KEEP)
    DROPPED = []
    print('   recordings labelled for %s: %d | arms C and E built in %.0f s' % (list(TASKS), len(KEEP), time.time() - t0))

    def _segment(audio, label, cache_name):
        """Springer, T and W cycles on `audio` (None = clean), which the workers read as AUDIO4; returns ({S,T,W: {name: (cyc, d)}}, traces sha)."""
        global EXP1_RAW
        _bind_audio(audio)
        try:
            out = {'S': {}, 'T': {}, 'W': {}}
            with _pool(N_PROC) as ex:
                for name, payload, err in tqdm(ex.map(_springer_worker, KEEP, chunksize=8), total=len(KEEP), desc='Springer ' + label, leave=False):
                    if err is not None:
                        ARM_FAILURES.append((name, 'S ' + label, err))
                        payload = (equal_time_cycles(AUDIO_META[name]['L4'], TARGET_SR), dict(fallback=True, worker_error=err))
                    out['S'][name] = payload
            EXP1_RAW = _infer_tw(AUDIO4, KEEP, cache_name)
            with _pool(N_PROC) as ex:
                for name, payload, err in tqdm(ex.map(_tw_cycles_worker, KEEP, chunksize=8), total=len(KEEP), desc='T and W ' + label, leave=False):
                    if err is not None:
                        ARM_FAILURES.append((name, 'T/W ' + label, err))
                        L4_ = AUDIO_META[name]['L4']
                        payload = {a: (equal_time_cycles(L4_, TARGET_SR), dict(fallback=True, worker_error=err)) for a in ('T', 'W')}
                    out['T'][name], out['W'][name] = payload['T'], payload['W']
            traces_sha = hashlib.sha256(''.join(EXP1_RAW[n]['trace_sha_t'] + EXP1_RAW[n]['trace_sha_w'] for n in KEEP).encode()).hexdigest()
        finally:
            _bind_audio(None)
        return out, traces_sha

    # ---- clean reproduction controls: Springer, T and W ----
    t0 = time.time()
    _clean, TW_TRACES_SHA = _segment(None, 'clean', 'tw_traces_clean.pkl')
    for a, tag in (('S', 'springer_wpk'), ('T', 'timing_net'), ('W', 'waveform_net')):
        for n in KEEP:
            CYC[tag][n], ARM_DIAG[tag][n] = _clean[a][n]
    B_CHECK_OK = True
    with _pool(N_PROC) as ex:
        for name, ok, err in ex.map(_arm_b_check_worker, KEEP, chunksize=16):
            if err is not None or not ok:
                B_CHECK_OK = False
                ARM_FAILURES.append((name, 'arm B construction check', err or 'mismatch'))
    print('   clean controls built in %.0f s | cycles.b_cycles equals arm_b_cycles on every kept recording: %s' % (time.time() - t0, B_CHECK_OK))

    # ---- the interference conditions ----
    for cond, kind, snr, seed in EXP1_CONDS:
        t0 = time.time()
        _noisy, EXP1_INFO[cond] = make_noisy(cond, KEEP)
        EXP1_AUDIO_SHA[cond] = NZ.audio_hash(_noisy, KEEP)
        _seg, EXP1_COND_SHA[cond] = _segment(_noisy, cond, 'tw_traces_%s.pkl' % cond)
        for a in ('S', 'T', 'W'):
            for n in KEEP:
                CYC['%s_seg_%s' % (a, cond)][n], ARM_DIAG['%s_seg_%s' % (a, cond)][n] = _seg[a][n]
        for a in ('E', 'C', 'S', 'T', 'W'):
            src = BASE_OF[a] if a in ('E', 'C') else '%s_seg_%s' % (a, cond)
            for n in KEEP:
                CYC['%s_both_%s' % (a, cond)][n] = CYC[src][n]
                ARM_DIAG['%s_both_%s' % (a, cond)][n] = dict(ARM_DIAG[src][n], same_cycles_as=src)
        del _noisy, _seg
        _ach = [v['achieved_snr_db'] for v in EXP1_INFO[cond].values() if v['achieved_snr_db'] is not None]
        print('   %-8s segmented in %.0f s | achieved SNR median %.2f dB (n %d) | audio sha256 %s' % (
            cond, time.time() - t0, float(np.median(_ach)) if _ach else float('nan'), len(_ach), EXP1_AUDIO_SHA[cond][:12]))
    for arm in ARMS:
        for n in KEEP:
            check_cycles(CYC[arm][n], AUDIO_META[n]['L4'], arm + ' ' + n, strict=not (arm == 'pcg_seg' or arm.startswith('C_both')))

    # ---- boundary quality against arm C, per arm (descriptive) ----
    EXP1_BOUNDARY = {}
    for tag in [t for t in ARMS if t != 'pcg_seg']:
        rows = [dict(_bstats(CYC[tag][n], n), fallback=bool(ARM_DIAG[tag][n].get('fallback'))) for n in KEEP]
        off = np.array([r['s2_offset_ms'] for r in rows], float)
        has = np.array([r['s2_pairs'] > 0 for r in rows])
        EXP1_BOUNDARY[tag] = dict(
            fallback_frac=float(np.mean([r['fallback'] for r in rows])), f1_median=float(np.nanmedian([r['f1'] for r in rows])),
            f1_pooled=float(2 * sum(r['matched'] for r in rows) / max(1, sum(r['n'] + r['n_ref'] for r in rows))),
            s2_off_gt60_share=float(np.mean(np.abs(off[has]) > 60)) if has.any() else None, recordings_with_s2_pairs=int(has.sum()),
            offset_sd_ms_median=float(np.nanmedian([r['offset_sd_ms'] for r in rows])))
    S_FALLBACK = {tag: EXP1_BOUNDARY[tag]['fallback_frac'] for tag in EXP1_BOUNDARY}
    EXP1_SNR = {}
    for cond in EXP1_INFO:
        _ach = np.array([v['achieved_snr_db'] for v in EXP1_INFO[cond].values() if v['achieved_snr_db'] is not None], float)
        EXP1_SNR[cond] = dict(n=int(len(_ach)), median=float(np.median(_ach)) if len(_ach) else None,
                              p5=float(np.percentile(_ach, 5)) if len(_ach) else None, p95=float(np.percentile(_ach, 95)) if len(_ach) else None)
    with open(RUN_DIR / 'exp1_boundaries.json', 'w') as fh:
        json.dump(dict(plan_sha256=EXP1_PLAN_SHA256, n_recordings=len(KEEP), conditions_run=[c[0] for c in EXP1_CONDS], esc50_ok=ESC50_OK,
                       esc50_note=ESC50_NOTE, condition_seeds={c[0]: c[3] for c in EXP1_CONDS}, diagnostics=EXP1_BOUNDARY,
                       achieved_snr=EXP1_SNR, audio_sha256=EXP1_AUDIO_SHA, tw_traces_sha256=dict(clean=TW_TRACES_SHA, **EXP1_COND_SHA),
                       arm_b_construction_equal=B_CHECK_OK, worker_failures=ARM_FAILURES), fh, indent=1)
    with open(RUN_DIR / 'exp1_noise_manifest.json', 'w') as fh:
        json.dump(dict(plan_sha256=EXP1_PLAN_SHA256, noise_module_sha256=_MOD_SHA['noise.py'],
                       esc50=dict(ok=ESC50_OK, note=ESC50_NOTE, url=ESC50_URL, classes=list(ESC50_CLASSES),
                                  files=ESC50_BANK['files'] if ESC50_OK else None, bank_sha256=ESC50_BANK['bank_sha256'] if ESC50_OK else None),
                       per_recording={c: {n: {k: v for k, v in EXP1_INFO[c][n].items() if k != 'clips'} for n in KEEP} for c in EXP1_INFO},
                       clips_used={c: {n: EXP1_INFO[c][n].get('clips') for n in KEEP} for c in EXP1_INFO if COND_OF[c][1] == 'esc50'}), fh)
    print('\n%-16s %8s %9s %10s %9s' % ('arm', 'fallback', 'F1 med', 'S2>60ms', 'start SD'))
    for tag in [t for t in ARMS if t != 'pcg_seg']:
        v = EXP1_BOUNDARY[tag]
        print('%-16s %7.2f%% %9.3f %10s %8.1f' % (tag, 100 * v['fallback_frac'], v['f1_median'],
                                                   ('%.3f' % v['s2_off_gt60_share']) if v['s2_off_gt60_share'] is not None else 'n/a', v['offset_sd_ms_median']))
    DIAG_SUMMARY, GLOBAL_OFFSET, KEEP_PINNED_OK = {}, {}, (VALIDATION_APPLICABLE or None)
''')

# ------------------------------------------------------------------ 6. Stage 5: features from the clean or the corrupted audio
patch('s5_features', "    FEATS = build_feature_matrices({arm: CYC[arm] for arm in ARMS}, KEEP, FEATURE_NAMES, 'arms')",
      """    _clean_tags = list(CONTROL_ARMS) + [t for c in EXP1_CONDS for t in SEG_TAGS[c[0]]]
    _bind_audio(None)
    FEATS = build_feature_matrices({arm: CYC[arm] for arm in _clean_tags}, KEEP, FEATURE_NAMES, 'clean')
    for cond, kind, snr, seed in EXP1_CONDS:                          # both-stage arms: features read from the corrupted audio
        _noisy, _ = make_noisy(cond, KEEP)
        assert NZ.audio_hash(_noisy, KEEP) == EXP1_AUDIO_SHA[cond], 'the corrupted audio of %s did not reproduce' % cond
        _bind_audio(_noisy)
        try:
            FEATS.update(build_feature_matrices({arm: CYC[arm] for arm in BOTH_TAGS[cond]}, KEEP, FEATURE_NAMES, 'both_' + cond))
        finally:
            _bind_audio(None)
        del _noisy""")

# ------------------------------------------------------------------ 7. folds: detection is not part of Experiment 1
patch('s6_folds', "    assert 'detection' in TASKS, 'the detection task cannot be skipped'",
      "    assert (not VALIDATION_APPLICABLE) or set(TASKS) == set(EXP1_TASKS), 'Experiment 1 runs grading, timing and shape'")

# ------------------------------------------------------------------ 8. classification
i = idx('s7_run')
txt = CELLS[i][2]
a = txt.index('PAIRS = [')
b = txt.index("('pcg_seg', 'equal_time_annot')]") + len("('pcg_seg', 'equal_time_annot')]")
CELLS[i] = (CELLS[i][0], 's7_run', txt[:a] + 'PAIRS = []          # Experiment 1: contrasts are computed locally from oof_predictions.npz' + txt[b:])
patch('s7_run', "    _oracle_check('detection')", "    _oracle_check(list(TASKS)[0])")
patch('s7_run', "    for task in ['detection'] + _three[:1]:", "    for task in _three[:1]:")
patch('s7_run', "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')",
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Experiment 1 classifies only in the full run.')")
patch('s7_run', "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL,",
      "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL, exp1_plan_sha256=EXP1_PLAN_SHA256,\n"
      "                          exp1_conditions=[c[0] for c in EXP1_CONDS], exp1_boundaries=EXP1_BOUNDARY, exp1_achieved_snr=EXP1_SNR, esc50_ok=ESC50_OK,\n"
      "                          tw_stamp=TW_STAMP, tw_freeze_sha256=FREEZE_SHA, tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=TW_TRACES_SHA,\n"
      "                          run1_oof_sha256=RUN1_OOF_SHA256, run2_oof_sha256=RUN2_OOF_SHA256,")

# ------------------------------------------------------------------ 9. integrity gate
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, '## Stage 8: integrity gate (Experiment 1)\n\nThe recordings are the Run 1 kept set restricted to the three '
                 'tasks; the folds equal the embedded child-grouped assignment; the five clean controls reproduce their runs (arms C, E '
                 "and Springer against Run 1, T and W against Run 2: MLP predictions identical, probabilities within 1e-5, random forest "
                 "identical); arm B's construction equals its verbatim copy; the frozen weights match the freeze record; a repeat fit and "
                 'the worker pool return identical probabilities. A failed gate means the run is not used.')
CELLS[i8] = ('code', 's8_gate', r'''
with stage('S8 integrity gate'):
    gate = dict(kind='integrity (Experiment 1)', applicable=bool(VALIDATION_APPLICABLE))
    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        gate.update(ok=None, decided_on='not applicable (reduced run)')
        print('integrity gate: not applicable in a reduced run')
    else:
        gate['folds'] = FOLD_CHECK
        gate['folds_ok'] = all(all(v.values()) for v in FOLD_CHECK.values()) and set(FOLD_CHECK) == set(TASKS)
        rep = {}
        for tag in CONTROL_ARMS:
            rep[tag] = {}
            for t in TASKS:
                r = CV[t][tag]; ref = REF[tag][t]
                same_rows = list(map(str, r['names'])) == ref['names'] and np.array_equal(np.asarray(r['y']), np.asarray(ref['y']))
                rep[tag][t] = dict(same_rows=bool(same_rows),
                                   mlp_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_mlp']), np.asarray(ref['mlp_pred']))),
                                   mlp_proba_max_abs_diff=float(np.max(np.abs(np.asarray(r['proba_mlp'], float) - np.asarray(ref['mlp_proba'], float)))) if same_rows else None,
                                   rf_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_rf']), np.asarray(ref['rf_pred']))))
        gate['reproduction'] = rep
        gate['reproduction_ok'] = {tag: all(v['mlp_pred_identical'] and v['rf_pred_identical'] and v['mlp_proba_max_abs_diff'] is not None
                                            and v['mlp_proba_max_abs_diff'] < 1e-5 for v in rep[tag].values()) for tag in rep}
        gate['arm_b_construction_ok'] = bool(B_CHECK_OK)
        gate['tw_weights_ok'] = bool(TW_WEIGHTS_OK)
        gate['esc50_ok'] = bool(ESC50_OK)
        t0_ = list(TASKS)[0]
        Fd_ = FOLDS[t0_]; tr_, te_ = Fd_['splits'][0]
        Xm_ = FEATS['pcg_seg'][Fd_['mask']]
        sc_ = StandardScaler().fit(Xm_[tr_])
        reps = []
        for _ in range(2):
            prs_ = []
            for sb in ENSEMBLE_SEEDS:
                m_, _i = fit_mlp(sc_.transform(Xm_[tr_]), Fd_['y'][tr_], Fd_['groups'][tr_], NCLS[t0_], sb)
                prs_.append(mlp_proba(m_, sc_.transform(Xm_[te_])))
            reps.append(np.mean(prs_, 0))
        gate['repeat_fit_max_abs_diff'] = float(np.max(np.abs(reps[0] - reps[1])))
        gate['repeat_fit_ok'] = gate['repeat_fit_max_abs_diff'] < 1e-6
        checks = [gate['folds_ok'], gate['repeat_fit_ok'], gate['arm_b_construction_ok'], gate['tw_weights_ok']] + list(gate['reproduction_ok'].values())
        if MLP_POOL is not None:
            Xtr0_, Xte0_ = sc_.transform(Xm_[tr_]), sc_.transform(Xm_[te_])
            p_pool_, _ = MLP_POOL.submit(_mlpw.fit_job, Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], Xte0_, NCLS[t0_], ENSEMBLE_SEEDS[0]).result()
            m0_, _ = fit_mlp(Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], NCLS[t0_], ENSEMBLE_SEEDS[0])
            gate['pool_vs_inprocess_max_abs_diff'] = float(np.max(np.abs(p_pool_ - mlp_proba(m0_, Xte0_))))
            gate['pool_ok'] = gate['pool_vs_inprocess_max_abs_diff'] < 1e-6
            checks.append(gate['pool_ok'])
        VALIDATION_OK = bool(all(checks))
        gate.update(ok=VALIDATION_OK, decided_on='integrity checks')
        print('integrity gate:', 'PASSED' if VALIDATION_OK else 'FAILED', {k: v for k, v in gate.items() if k not in ('folds', 'reproduction')})
        for tag in rep:
            print('   %s against its run:' % tag, rep[tag])
    RESULTS['validation_gate'] = gate
    write_results()
    if MLP_POOL is not None:
        MLP_POOL.shutdown()
''')

# ------------------------------------------------------------------ 10. manifest
patch('s11_manifest', "        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),",
      "        checkpoint=dict(tw_checkpoint_dir=str(TW_SRC), freeze_sha256=FREEZE_SHA, models={k: v['weights_sha256'] for k, v in FREEZE_REC['models'].items()},\n"
      "                        tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=dict(clean=TW_TRACES_SHA, **EXP1_COND_SHA), embedded_modules=_MOD_SHA),\n"
      "        exp1=dict(plan_sha256=EXP1_PLAN_SHA256, conditions=[c[0] for c in EXP1_CONDS], condition_seeds={c[0]: c[3] for c in EXP1_CONDS},\n"
      "                  audio_sha256=EXP1_AUDIO_SHA, esc50_ok=ESC50_OK, esc50_bank_sha256=ESC50_BANK['bank_sha256'] if ESC50_OK else None, n_arms=len(ARMS)),")
patch('s11_manifest', "        generator=dict(batch_start=GAN_BATCH, batch_final=GEN_BATCH_USED, oom_events=OOM_EVENTS, signal_length=SIGNAL_LENGTH, gan_sr=GAN_SR, ecg_cache_dtype='float16'),",
      "        generator=None,")

# ------------------------------------------------------------------ 11. markdown that described Run 1 only
for k, (kind, nm, text) in enumerate(CELLS):
    if kind == 'markdown' and 'Run 1' in text and nm is None and k > 0 and 'Experiment 1' not in text:
        CELLS[k] = (kind, nm, text.replace('(Run 1)', '(Run 1 cells, used unchanged in Experiment 1)'))

# ------------------------------------------------------------------ write
iprov = next(i for i, c in enumerate(CELLS) if c[0] == 'code' and c[2].startswith('VERBATIM_PROVENANCE = '))
h = hashlib.sha256()
for kind, name, text in CELLS:
    if not text.startswith('VERBATIM_PROVENANCE = '):
        h.update(text.encode('utf-8')); h.update(b'\x00')
SRC_SHA = h.hexdigest()
ptxt = CELLS[iprov][2]
old = ptxt[ptxt.index("NOTEBOOK_SOURCE_SHA256 = '"):]
old = old[:old.index('\n')]
CELLS[iprov] = (CELLS[iprov][0], CELLS[iprov][1], ptxt.replace(old, "NOTEBOOK_SOURCE_SHA256 = '%s'   # filled in by build_exp1.py" % SRC_SHA))
cells = []
for kind, name, text in CELLS:
    assert '\u2014' not in text, 'em dash in cell %s' % name
    cells.append(new_markdown_cell(text) if kind == 'markdown' else new_code_cell(text, metadata={'name': name} if name else {}))
nb = new_notebook(cells=cells)
nb.metadata = dict(nb1['metadata'])
nb.metadata['colab'] = dict(nb1['metadata'].get('colab', {}), name='Experiment1_InBandNoise.ipynb', machine_shape='hm', gpuType='A100')
nb.metadata['built_by'] = 'extension_study/notebooks/build_exp1.py (patches over the locked Extension_Run1.ipynb)'
nb.metadata['notebook_source_sha256'] = SRC_SHA
nb.metadata['run1_notebook_file_sha256'] = lock['run1_notebook_file_sha256']
nb.metadata['embedded'] = dict(nb1['metadata'].get('embedded', {}), exp1_plan_sha256=PLAN_SHA, modules_sha256=MOD_SHA, tw_freeze_sha256=FREEZE_SHA,
                               run1_oof_sha256=OOF1_SHA, run2_oof_sha256=OOF2_SHA, reference_predictions_sha256=REF_SHA)
tmp = Path(tempfile.mkdtemp())
for k, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % k)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Experiment 1 notebook written: %s (%d cells) | source sha256 %s' % (OUT_NB, len(nb.cells), SRC_SHA[:16]))
