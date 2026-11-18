#!/usr/bin/env python
"""
build_run3.py: assembles Extension_Run3.ipynb (Experiment 2: the acoustic-label control arm T_ac on CirCor) from the
locked Extension_Run1.ipynb, patched by cell name, with the patch set of build_run2.py.

What changes from Run 2: the frozen checkpoint folder is the control arm's (kind A, three seeds), the classified arms
are E (reproduction control: features and out-of-fold predictions must equal Run 1's), T_ac and T_ac_annot, T_ac's
cycles come from its S1 and S2 heads through cycles.t_cycles exactly as T's did, and the run folder is
Extension_Study/runs/run3/<timestamp>. Everything else is Run 2's patch over Run 1's cells.
H1 and the estimation contrasts are computed locally (notebooks/exp2_analysis.py).

Usage:  python build_run3.py <checkpoint stamp, e.g. 20261103_1200_ac> [freeze folder] [output notebook]
        (by default the folder runs/run3_freeze/<stamp>/ must hold the downloaded freeze.json; the two optional
        arguments serve a dry build against a mock freeze record, written outside the notebooks folder)
"""
import base64, hashlib, json, py_compile, sys, tempfile, zlib
from pathlib import Path

import numpy as np
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PHD = STUDY.parent
RUN1_NB = HERE / 'Extension_Run1.ipynb'
OUT_NB = HERE / 'Extension_Run3.ipynb'
AC_STAMP = sys.argv[1] if len(sys.argv) > 1 else None
assert AC_STAMP, 'usage: python build_run3.py <checkpoint stamp> [freeze folder] [output notebook]'
AC_LOCAL = Path(sys.argv[2]) if len(sys.argv) > 2 else STUDY / 'runs' / 'run3_freeze' / AC_STAMP
if len(sys.argv) > 3:
    OUT_NB = Path(sys.argv[3])              # a dry build (mock freeze record) writes outside the notebooks folder
PLAN = Path('E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md')
RUN1_DIR = PHD / 'Results' / 'A100 Results' / 'extracted' / 'runs' / 'run1' / '20261016_1516'
RUN1_ZIP = PHD / 'Results' / 'A100 Results' / 'runs-20261016T171455Z-1-001.zip'
MODULES = ['__init__.py', 'frontend.py', 'model.py', 'robust_beats.py', 'decoder.py', 'infer.py', 'cycles.py']

sha = lambda b: hashlib.sha256(b).hexdigest()
lock = json.load(open(STUDY / 'prereg' / 'RUN1_LOCK.json'))
assert sha(RUN1_NB.read_bytes()) == lock['run1_notebook_file_sha256'], 'Extension_Run1.ipynb differs from its lock'
PLAN_SHA = sha(PLAN.read_bytes())
assert open(PLAN.with_suffix('.sha256')).read().split()[0] == PLAN_SHA, 'EXP2_PREREG.md changed after it was hashed'
nb1 = json.load(open(RUN1_NB, encoding='utf-8'))
nb2 = json.load(open(HERE / 'Extension_Run2.ipynb', encoding='utf-8'))
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


# ------------------------------------------------------------------ embedded data
MODS = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
for m, b in MODS.items():
    assert 'circor' not in b.decode('utf-8').lower(), 'CirCor reference in %s' % m
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in MODS.items()}
MOD_SHA = {m: sha(b) for m, b in MODS.items()}
for m in ('frontend.py', 'robust_beats.py', 'decoder.py', 'cycles.py'):             # the boundary-defining modules must be Run 2's
    assert MOD_SHA[m] == nb2['metadata']['embedded']['modules_sha256'][m], '%s differs from the Run 2 version' % m
FREEZE_BYTES = (AC_LOCAL / 'freeze.json').read_bytes()
FREEZE_SHA = sha(FREEZE_BYTES)
FREEZE_REC = json.loads(FREEZE_BYTES)
assert FREEZE_REC['smoke'] is False and all(k.startswith('A_') for k in FREEZE_REC['models']) and len(FREEZE_REC['models']) == 3
assert FREEZE_REC['exp2_plan_sha256'] == PLAN_SHA, 'the control arm was trained under a different plan'
assert FREEZE_REC['modules_sha256']['model.py'] == MOD_SHA['model.py'], 'model.py differs from the one the control arm trained with'

man1 = json.load(open(RUN1_DIR / 'run_manifest.json'))
RUN1_CYCLES_HASH = {a: man1['hashes']['feature_cycles_hashes'][a] for a in ('pcg_seg', 'equal_time', 'equal_time_annot')}
import io, zipfile
with zipfile.ZipFile(RUN1_ZIP) as zf, zf.open('runs/run1/20261016_1516/cache/features_equal_time.npz') as fh:
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
CELLS[0] = ('markdown', None, r'''# Extension Run 3: the acoustic-label control arm T_ac on CirCor (Experiment 2)

Scores the frozen control network A (the T architecture trained with S1 and S2 labels derived from the phonocardiogram
alone, no electrocardiogram; plan `Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md`) on CirCor exactly as Run 2
scored T. Built by `extension_study/notebooks/build_run3.py` from the locked Run 1 notebook with Run 2's patch set; the
shared analysed audio, the 464-feature extractor, the five-seed MLP, the random forest, the child-grouped folds and the
bootstrap are Run 1's cells, unchanged.

**What runs.** The frozen A models (checkpoint folder `Extension_Study/checkpoints/<stamp>`; every file and weight hash
checked against the embedded freeze record) run label-free on each recording; T_ac's cycles come from its S1 and S2
heads through the construction of T. Arms C and E are rebuilt only to show that this run reproduces Run 1 exactly.

**How to run.** Runtime > Change runtime type > **A100 GPU** with High-RAM; leave `DRY_RUN = False`; Runtime > Run all.
Expected time about 1.5 hours.

**Afterwards.** Download `Extension_Study/runs/run3/<timestamp>` (results.json, oof_predictions.npz, diagnostics.csv,
run_manifest.json). H1 and the contrasts are computed locally from this run's predictions merged with Runs 1 and 2.

| arm | key | boundaries |
|---|---|---|
| E | `equal_time` | 75 bpm pseudo-cycles (reproduction control) |
| **T_ac** | `timing_net_ac` | frozen A: S1 events from the shared decoder, S2 from its S2 head, 50% acoustic onsets |
| T_ac_annot | `timing_net_ac_annot` | T_ac cycles inside the annotated stretches |
'''.replace('<stamp>', AC_STAMP))

# ------------------------------------------------------------------ 1. flags
patch('flags', "R1_RUN_DIR = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip",
      "R1_RUN_DIR = None          # Run 3 checks reproduction against Run 1's embedded hashes instead\n"
      "TW_STAMP = '%s'   # frozen control arm A (Extension_Study/checkpoints/<stamp>)\n"
      "TW_CHECKPOINT_DIR = None   # None = the Drive folder above; a local path for a local run\n"
      "REQUIRE_GPU = 'A100'       # full run only: Run 1's GPU class, so arm E reproduces bit for bit\n"
      "EXP2_PLAN_SHA256 = '%s'   # Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md" % (AC_STAMP, PLAN_SHA))
patch('flags', "PREREG_SHA256 = '755b47da43b250671391aae8672b182278cb4561314343581024ab1f581d8974'   # extension_study/prereg/PREREG_EXT.md, frozen 16 Oct 2026 before any Run 1 classification",
      "PREREG_SHA256 = '%s'   # extension_study/prereg/PREREG_EXT.md, frozen 16 Oct 2026" % PREREG_SHA)

# ------------------------------------------------------------------ 2. environment
patch('env', "GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32",
      "GAN_BATCH = None    # Run 3 runs no generator\n"
      "if (not DRY_RUN) and LIMIT_RECORDINGS == 0 and REQUIRE_GPU:\n"
      "    assert REQUIRE_GPU in GPU_NAME, 'Run 3 must run on an %s (Run 1 class) so arm E reproduces Run 1; this runtime has %s' % (REQUIRE_GPU, GPU_NAME)")
patch('env', "print('generator batch', GAN_BATCH, '| feature workers', N_PROC)", "print('feature workers', N_PROC)")

# ------------------------------------------------------------------ 3. resolve
i = idx('resolve')
txt = CELLS[i][2]
a = txt.index('    # ---- generator checkpoint ----')
b = txt.index('    # ---- output folder and run flags ----')
CELLS[i] = (CELLS[i][0], 'resolve', txt[:a] + '''    # ---- Run 3: no generator; the frozen control arm's checkpoint folder ----
    GAN_CHECKPOINT, CKPT_SHA256, CKPT_HASHES = None, None, {}
    TW_SRC = Path(TW_CHECKPOINT_DIR) if TW_CHECKPOINT_DIR else Path(DRIVE_ROOT + '/JBHI-Revision/Extension_Study/checkpoints/' + TW_STAMP)
    assert (TW_SRC / 'freeze.json').exists(), 'frozen checkpoint folder not found: %s' % TW_SRC

''' + txt[b:])
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run3'))")
patch('resolve', "run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)",
      "run='extension_run3_acoustic_control', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256, tw_stamp=TW_STAMP, exp2_plan_sha256=EXP2_PLAN_SHA256)")
patch('resolve', "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),",
      "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('TW_SRC', TW_SRC),")
patch('resolve', "    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))",
      "    print('paired wav+tsv files: %d | frozen control-arm folder: %s' % (N_PAIRS, TW_SRC))")
patch('resolve', """    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
""", "")
drop('v_generator', '*Run 3: the generator architecture cell of Run 1 is not used.*')
drop('v_ckpt_load', '*Run 3: the generator checkpoint is not loaded.*')
drop('s3_generator', '## Stage 3: not run\n\nRun 3 builds no GAN arms.')
patch('s2_audio', "    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True",
      "    ECG_CHUNK_OK[k] = True       # Run 3: no generator, so the 8 kHz audio is never loaded")

# ------------------------------------------------------------------ 4. arm lists
patch('s4_arm_defs', """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS""", """R1_ARMS = list(ARMS)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
TW_ARMS = ['timing_net_ac']
TW_COV_ARMS = ['timing_net_ac_annot']
ARMS = ['equal_time'] + TW_ARMS + TW_COV_ARMS                              # classified in Run 3 (E: reproduction control)
CYC_ARMS = ['pcg_seg', 'equal_time', 'equal_time_annot'] + TW_ARMS + TW_COV_ARMS""")
patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """ARM_LETTER.update({'equal_time_annot': 'E_annot', 'timing_net_ac': 'T_ac', 'timing_net_ac_annot': 'T_ac_annot'})
ARM_LABEL.update({'equal_time_annot': 'E_annot equal-time, annotated stretches', 'timing_net_ac': 'T_ac acoustic-label control (frozen)',
                  'timing_net_ac_annot': 'T_ac_annot, annotated stretches'})
print('arm definitions ready: classified', ARMS, '| cycles', CYC_ARMS, '| arm configuration hash', ARM_CFG_HASH)""")

# ------------------------------------------------------------------ 5. control-arm definitions
EXT3 = r'''
# ---------------- Extension Run 3: the frozen control arm A, label-free inference and cycle export ----------------
import base64 as _b64r, zlib as _zr, shutil as _sh2
_MOD_B64 = @@MOD_B64@@
_MOD_SHA = @@MOD_SHA@@
_code_root = (Path('/content') if IN_COLAB else RUN_DIR) / 'run3_code'
(_code_root / 'extstudy').mkdir(parents=True, exist_ok=True)
for _m, _b in _MOD_B64.items():
    _src = _b64r.b64decode(_b)
    assert hashlib.sha256(_src).hexdigest() == _MOD_SHA[_m], 'embedded module %s corrupted' % _m
    (_code_root / 'extstudy' / _m).write_bytes(_src)
for _k in [k for k in sys.modules if k == 'extstudy' or k.startswith('extstudy.')]:
    del sys.modules[_k]
sys.path.insert(0, str(_code_root))
from extstudy import infer as TWI, cycles as TWC
assert TWC.FUSION == FUSION and TWC.PHASE_KEYS == tuple(PHASE_KEYS) and TWC.TARGET_SR == TARGET_SR, 'cycle constants differ from arm B'

FREEZE_SHA = '@@FREEZE_SHA@@'
_tw_local = (Path('/content') if IN_COLAB else RUN_DIR) / 'ac_checkpoints' / TW_STAMP
_tw_local.mkdir(parents=True, exist_ok=True)
for _f in sorted(p.name for p in TW_SRC.iterdir() if p.suffix in ('.json', '.pt')):
    _drive_retry(_sh2.copyfile, TW_SRC / _f, _tw_local / _f) if IN_COLAB else _sh2.copyfile(TW_SRC / _f, _tw_local / _f)
assert hashlib.sha256((_tw_local / 'freeze.json').read_bytes()).hexdigest() == FREEZE_SHA, 'freeze.json differs from the frozen record'
FREEZE_REC = json.load(open(_tw_local / 'freeze.json'))
A_MODELS, _ = TWI.load_models(str(_tw_local), 'A', DEV)          # file and weight hashes checked inside
TW_WEIGHTS_OK = True
TW_CFG = dict(freeze_sha256=FREEZE_SHA, modules=_MOD_SHA, win_frames=TWI.WIN_FRAMES, overlap=TWI.OVERLAP, t_export=TWC.T_EXPORT,
              n_min=N_MIN_CYCLES, fusion=FUSION, kind='A')
TW_CFG_HASH = hashlib.sha1(json.dumps(TW_CFG, sort_keys=True, default=str).encode()).hexdigest()[:16]
print('frozen control arm A loaded and verified: %s | configuration hash %s' % ([n for n, _ in A_MODELS], TW_CFG_HASH))

RUN1_CYCLES_HASH = @@RUN1_CYCLES@@
RUN1_E_FEATURES_SHA = '@@RUN1_E_FEAT@@'
_E1 = '@@RUN1_E_B64@@'
assert hashlib.sha256(_E1.encode()).hexdigest() == '@@RUN1_E_SHA@@', 'embedded Run 1 arm E predictions corrupted'
RUN1_E = json.loads(_zr.decompress(_b64r.b64decode(_E1)).decode())
RUN1_OOF_SHA256 = '@@OOF1_SHA@@'

TW_RAW = {}          # {name: dict(bt, p_s2, rec)} for the forked cycle workers

def _tw_worker(name):
    """T_ac cycles from its S1 and S2 heads (cycles.t_cycles, the construction of T); equal-time fallback below 2 cycles."""
    try:
        L4 = AUDIO_META[name]['L4']
        x4 = np.asarray(AUDIO4[name][:L4], np.float64)
        raw = TW_RAW[name]
        cyc_t, d_t = TWC.t_cycles(x4, raw['bt'], raw['p_s2'], TWI.FRAME_T0, TWI.FRAME_S)
        d = dict(d_t, fallback=False, n_cycles_net=len(cyc_t))
        if len(cyc_t) < N_MIN_CYCLES:
            cyc_t, d['fallback'] = equal_time_cycles(L4, TARGET_SR), True
        return name, {'timing_net_ac': (cyc_t, d)}, None
    except Exception as e:
        return name, None, '%s: %s' % (type(e).__name__, e)
'''
for k, v in dict(MOD_B64=json.dumps(MOD_B64), MOD_SHA=json.dumps(MOD_SHA), FREEZE_SHA=FREEZE_SHA, RUN1_CYCLES=json.dumps(RUN1_CYCLES_HASH),
                 RUN1_E_FEAT=RUN1_E_FEATURES_SHA, RUN1_E_B64=RUN1_E_B64, RUN1_E_SHA=RUN1_E_SHA, OOF1_SHA=OOF1_SHA).items():
    EXT3 = EXT3.replace('@@%s@@' % k, v)
assert '@@' not in EXT3
insert_after('ext_run1_defs', [('code', 'ext_run3_defs', EXT3.strip('\n'))])

# ------------------------------------------------------------------ 6. Stage 4: Run 2's cell with A in place of T and W
nb2_cells = {c.get('metadata', {}).get('name'): (''.join(c['source']) if isinstance(c['source'], list) else c['source']) for c in nb2['cells']}
s4 = nb2_cells['s4_run']
rep = [
    ("    # ---- frozen T and W on the GPU: seed-averaged traces, beats from the shared decoder ----", "    # ---- the frozen control arm A on the GPU: seed-averaged traces, beats from the shared decoder ----"),
    ("    for n in tqdm(todo, desc='T and W inference', leave=False):", "    for n in tqdm(todo, desc='control arm inference', leave=False):"),
    ("        tt = TWI.forward(T_MODELS, spec, 'T', DEV)\n        tw = TWI.forward(W_MODELS, spec, 'W', DEV)\n        bt, rec_t = TWI.beats(tt, 'T')\n        bw, rec_w = TWI.beats(tw, 'W')\n"
     "        TW_RAW[n] = dict(bt=bt, bw=bw, p_s2=tt[1].astype(np.float32), rec_t=rec_t, rec_w=rec_w,\n                         trace_sha_t=TWI.trace_hash(tt), trace_sha_w=TWI.trace_hash(tw))",
     "        tt = TWI.forward(A_MODELS, spec, 'A', DEV)\n        bt, rec_t = TWI.beats(tt, 'A')\n"
     "        TW_RAW[n] = dict(bt=bt, p_s2=tt[1].astype(np.float32), rec_t=rec_t, trace_sha_t=TWI.trace_hash(tt))"),
    ("        print('   T and W run on %d recordings in %.0f s' % (len(todo), time.time() - t0))", "        print('   control arm run on %d recordings in %.0f s' % (len(todo), time.time() - t0))"),
    ("        print('   T and W traces loaded from cache')", "        print('   control-arm traces loaded from cache')"),
    ("    TW_TRACES_SHA = hashlib.sha256(''.join(TW_RAW[n]['trace_sha_t'] + TW_RAW[n]['trace_sha_w'] for n in KEEP).encode()).hexdigest()",
     "    TW_TRACES_SHA = hashlib.sha256(''.join(TW_RAW[n]['trace_sha_t'] for n in KEEP).encode()).hexdigest()"),
    ("    # ---- T and W cycles (worker pool) and the matched-coverage copies ----", "    # ---- T_ac cycles (worker pool) and the matched-coverage copy ----"),
    ("desc='arms T/W', leave=False):", "desc='arm T_ac', leave=False):"),
    ("                ARM_FAILURES.append((name, 'T/W', err))", "                ARM_FAILURES.append((name, 'T_ac', err))"),
    ("        for a, src in (('timing_net_annot', 'timing_net'), ('waveform_net_annot', 'waveform_net'), ('equal_time_annot', 'equal_time')):",
     "        for a, src in (('timing_net_ac_annot', 'timing_net_ac'), ('equal_time_annot', 'equal_time')):"),
    ("    print('   arms T and W built in %.0f s' % (time.time() - t0))", "    print('   arm T_ac built in %.0f s' % (time.time() - t0))"),
    ("            if arm in TW_ARMS:\n                rec = TW_RAW[n]['rec_t' if arm == 'timing_net' else 'rec_w']",
     "            if arm in TW_ARMS:\n                rec = TW_RAW[n]['rec_t']"),
    ("    _pt = DK[DK.arm == 'timing_net']\n    DIAG_SUMMARY['timing_net'].update(", "    _pt = DK[DK.arm == 'timing_net_ac']\n    DIAG_SUMMARY['timing_net_ac'].update("),
]
for old, new in rep:
    assert s4.count(old) == 1, 'Run 2 s4_run anchor not found once: %r' % old[:80]
    s4 = s4.replace(old, new)
# the W summary block and the final print concern W only: drop them
a = s4.index("    _pw = DK[DK.arm == 'waveform_net']")
b = s4.index("    GLOBAL_OFFSET = {}")
s4 = s4[:a] + s4[b:]
a = s4.index("    print('arm T: S2 from an acoustic onset")
s4 = s4[:a] + ("    print('arm T_ac: S2 from an acoustic onset in %.0f%% of cycles (median per recording), median S2 probability at the chosen point %.2f' % (\n"
               "              100 * DIAG_SUMMARY['timing_net_ac']['frac_s2_acoustic_median'], DIAG_SUMMARY['timing_net_ac']['s2_head_at_choice_median']))\n")
replace_cell('s4_run', s4)

# ------------------------------------------------------------------ 7. features, classification, gate, manifest: Run 2's patches with the arm renamed
patch('s5_features', "        print('   R1 feature matrices reproduced:', FEATURE_REPRO)",
      "        print('   R1 feature matrices reproduced:', FEATURE_REPRO)\n"
      "    E_FEATURES_SHA = hashlib.sha256(np.ascontiguousarray(FEATS['equal_time'].astype(np.float32)).tobytes()).hexdigest()\n"
      "    print('   arm E features against Run 1: %s (sha256 %s)' % (('reproduced' if E_FEATURES_SHA == RUN1_E_FEATURES_SHA else 'NOT reproduced') if VALIDATION_APPLICABLE else 'not applicable (reduced run)', E_FEATURES_SHA[:16]))")
i = idx('s7_run')
txt = CELLS[i][2]
a = txt.index('PAIRS = [')
b = txt.index("('pcg_seg', 'equal_time_annot')]") + len("('pcg_seg', 'equal_time_annot')]")
CELLS[i] = (CELLS[i][0], 's7_run', txt[:a] + "PAIRS = [('timing_net_ac', 'equal_time')]" + txt[b:])
patch('s7_run', "        r_, e_ = CV[task]['pcg_seg'], CV[task]['equal_time']", "        r_, e_ = CV[task]['timing_net_ac'], CV[task]['equal_time']     # Run 3: T_ac in place of C (C is not rescored)")
patch('s7_run', "        r_ = CV[task]['pcg_seg']", "        r_ = CV[task]['timing_net_ac']")
patch('s7_run', "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')",
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 3 classifies only in the full, pre-registered run.')")
patch('s7_run', "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL,",
      "fallback_share=TW_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL, failed_arms=TW_FAILED, tw_stamp=TW_STAMP, exp2_plan_sha256=EXP2_PLAN_SHA256,\n"
      "                          tw_freeze_sha256=FREEZE_SHA, tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=TW_TRACES_SHA, run1_oof_sha256=RUN1_OOF_SHA256,")
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, nb2_cells.get('md32', '## Stage 8: integrity gate (Run 3)') if False else
                 '## Stage 8: integrity gate (Run 3)\n\nLabel-blind checks, as Run 2: the kept set equals the 3,000 Run 1 recordings; the folds equal '
                 'the embedded child-grouped assignment; arms C, E and E_annot have Run 1\'s cycle hashes; arm E\'s feature matrix and its '
                 'out-of-fold predictions equal Run 1\'s; the frozen control-arm weights match the freeze record; a repeat five-seed fit '
                 'returns identical probabilities; the parallel workers return what an in-process fit returns. A failed gate means the run is not used.')
CELLS[i8] = ('code', 's8_gate', nb2_cells['s8_gate'].replace("kind='integrity (Run 2)'", "kind='integrity (Run 3)'"))
patch('s11_manifest', "        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),",
      "        checkpoint=dict(ac_checkpoint_dir=str(TW_SRC), freeze_sha256=FREEZE_SHA, models={k: v['weights_sha256'] for k, v in FREEZE_REC['models'].items()},\n"
      "                        tw_cfg_hash=TW_CFG_HASH, tw_traces_sha256=TW_TRACES_SHA, embedded_modules=_MOD_SHA, exp2_plan_sha256=EXP2_PLAN_SHA256),")
patch('s11_manifest', "        generator=dict(batch_start=GAN_BATCH, batch_final=GEN_BATCH_USED, oom_events=OOM_EVENTS, signal_length=SIGNAL_LENGTH, gan_sr=GAN_SR, ecg_cache_dtype='float16'),",
      "        generator=None,")
for k, (kind, nm, text) in enumerate(CELLS):
    if kind == 'markdown' and 'Run 1' in text and nm is None and k > 0 and 'Run 3' not in text:
        CELLS[k] = (kind, nm, text.replace('(Run 1)', '(Run 1 cells, used unchanged in Run 3)'))

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
CELLS[iprov] = (CELLS[iprov][0], CELLS[iprov][1], ptxt.replace(old, "NOTEBOOK_SOURCE_SHA256 = '%s'   # filled in by build_run3.py" % SRC_SHA))
cells = []
for kind, name, text in CELLS:
    assert '\u2014' not in text, 'em dash in cell %s' % name
    cells.append(new_markdown_cell(text) if kind == 'markdown' else new_code_cell(text, metadata={'name': name} if name else {}))
nb = new_notebook(cells=cells)
nb.metadata = dict(nb1['metadata'])
nb.metadata['colab'] = dict(nb1['metadata'].get('colab', {}), name='Extension_Run3.ipynb', machine_shape='hm', gpuType='A100')
nb.metadata['built_by'] = 'extension_study/notebooks/build_run3.py (patches over the locked Extension_Run1.ipynb)'
nb.metadata['notebook_source_sha256'] = SRC_SHA
nb.metadata['run1_notebook_file_sha256'] = lock['run1_notebook_file_sha256']
nb.metadata['embedded'] = dict(nb1['metadata'].get('embedded', {}), ac_freeze_sha256=FREEZE_SHA, modules_sha256=MOD_SHA, exp2_plan_sha256=PLAN_SHA,
                               run1_cycles_hashes=RUN1_CYCLES_HASH, run1_e_features_sha256=RUN1_E_FEATURES_SHA, run1_oof_sha256=OOF1_SHA)
tmp = Path(tempfile.mkdtemp())
for k, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % k)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Run 3 notebook written: %s (%d cells) | source sha256 %s' % (OUT_NB, len(nb.cells), SRC_SHA[:16]))
