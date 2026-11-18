#!/usr/bin/env python
"""
build_exp0c.py: assembles Experiment0c_FailurePlacement.ipynb (next study, Experiment 0c; plan in
'Next Study - Beat-Accurate Synthetic ECG/EXP0C_PREREG.md') from the locked Extension_Run1.ipynb, patched by cell name,
in the same way as build_exp0b.py.

What changes from Run 1, and nothing else:
  1. run folder Extension_Study/runs/exp0c/<timestamp>; no generator (GAN cells out, no 8 kHz audio);
  2. recordings: the Run 1 kept set restricted to those labelled for grading, timing or shape; tasks: those three;
  3. boundary sets: arm C (reproduction control), Springer everywhere and W everywhere (reproduction controls against
     Run 1 and Run 2, from embedded cycles), and 30 sets: Experiment 0b's stylised failures placed on Springer's
     failure set or on as many random recordings, and Springer's and W's own cycles inside or outside their failure
     sets (extstudy/placement.py over extstudy/corrupt.py as Experiment 0b embedded it);
  4. features, folds (embedded, child-grouped), five-seed MLP and random forest: Run 1's cells unchanged;
  5. integrity gate: kept subset, folds, arm C, Springer and W reproduced, repeat fit, pool check.
Retention, the primary Delta and the secondary contrasts are computed locally (notebooks/exp0c_analysis.py).

Usage:  python build_exp0c.py   (writes Experiment0c_FailurePlacement.ipynb beside this file)
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
OUT_NB = HERE / 'Experiment0c_FailurePlacement.ipynb'
PLAN = Path('E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG/EXP0C_PREREG.md')
RUN1_DIR = PHD / 'Results' / 'A100 Results' / 'extracted' / 'runs' / 'run1' / '20261016_1516'
RUN2_DIR = STUDY / 'runs' / 'run2' / '20261018_1358'
SETS_JSON = STUDY / 'runs' / 'exp0c_prep' / 'exp0c_sets.json'
MODULES = ['__init__.py', 'corrupt.py', 'placement.py']
EXP0C_TASKS = ('grading', 'timing', 'shape')

sha = lambda b: hashlib.sha256(b).hexdigest()
lock = json.load(open(STUDY / 'prereg' / 'RUN1_LOCK.json'))
assert sha(RUN1_NB.read_bytes()) == lock['run1_notebook_file_sha256'], 'Extension_Run1.ipynb differs from its lock'
PLAN_SHA = sha(PLAN.read_bytes())
assert open(PLAN.with_suffix('.sha256')).read().split()[0] == PLAN_SHA, 'EXP0C_PREREG.md changed after it was hashed'
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


# ------------------------------------------------------------------ embedded: code, failure sets and cycles, reference predictions
MODS = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in MODS.items()}
MOD_SHA = {m: sha(b) for m, b in MODS.items()}
nb0b = json.load(open(HERE / 'Experiment0b_FailureConcentration.ipynb', encoding='utf-8'))
assert MOD_SHA['corrupt.py'] == nb0b['metadata']['embedded']['modules_sha256']['corrupt.py'], 'corrupt.py differs from the Experiment 0b version'
SETS = json.load(open(SETS_JSON))
F_S, F_W, NAMES = SETS['sets']['S']['recordings'], SETS['sets']['W']['recordings'], SETS['names']
assert len(NAMES) == 610 and len(F_S) == 163 and len(F_W) == 69
for k in ('S', 'W'):
    assert sha(SETS['embedded'][k]['b64'].encode()) == SETS['embedded'][k]['sha256'], 'embedded %s cycles corrupted' % k
assert all(v['recomputed'] == v['recorded'] for v in SETS['verify'].values())
SETS_EMB = dict(names=NAMES, F_S=F_S, F_W=F_W, c_hash=SETS['cycles_hash_610_C'],
                cyc={k: dict(b64=SETS['embedded'][k]['b64'], sha256=SETS['embedded'][k]['sha256'], hash=SETS['embedded'][k]['cycles_hash_610'])
                     for k in ('S', 'W')})
SETS_B64 = base64.b64encode(zlib.compress(json.dumps(SETS_EMB).encode(), 9)).decode()
SETS_SHA = sha(SETS_B64.encode())

o1 = np.load(RUN1_DIR / 'oof_predictions.npz', allow_pickle=False)
o2 = np.load(RUN2_DIR / 'oof_predictions.npz', allow_pickle=False)
REF = {}
for tag, z, src in (('pcg_seg', o1, 'pcg_seg'), ('s_all', o1, 'springer_wpk'), ('w_all', o2, 'waveform_net')):
    REF[tag] = {}
    for t in EXP0C_TASKS:
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
CELLS[0] = ('markdown', None, r'''# Experiment 0c: failure placed where Springer fails

Does a stylised segmentation failure cost more of the gain of accurate gating when it falls on the recordings where
Springer fails than when it falls on as many recordings drawn at random? Plan: `Next Study - Beat-Accurate Synthetic
ECG/EXP0C_PREREG.md` (hashed before any Experiment 0c feature or score existed). Built by
`extension_study/notebooks/build_exp0c.py` from the locked Run 1 notebook, patched cell by cell.

**What runs.** Arm C's annotated cycles (reproduction control); Springer everywhere and W everywhere (reproduction
controls against Run 1 and Run 2); and 30 sets:
- every cycle's S2 displaced by 100 ms in Springer's 163 failed recordings (five seeds) or in 163 random recordings
  (ten seeds);
- equal-time windows in Springer's failed recordings (one set) or in 163 random recordings (ten seeds);
- Springer's own cycles inside or outside its failure set, and W's inside or outside its own (69 recordings).

Each set goes through Run 1's feature extractor and five-seed MLP on the child-grouped folds, for the recordings
labelled for grading, timing or shape.

**How to run.** Runtime > Change runtime type > **A100 GPU** with High-RAM; leave `DRY_RUN = False`; Runtime > Run all.
Expected time about 1.5 to 2 hours; if the session drops, set `RESUME_DIR` to the run folder and run all again
(features and fits are cached).

**Afterwards.** Download `Extension_Study/runs/exp0c/<timestamp>` without its `cache` folder (results.json,
oof_predictions.npz, exp0c_boundaries.json, exp0c_s2_offsets.json, run_manifest.json, stage_seconds.json and the
executed notebook); the analysis runs locally.
''')

# ------------------------------------------------------------------ 1. flags and environment
patch('flags', "R1_RUN_DIR = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip",
      "R1_RUN_DIR = None\n"
      "EXP0C_PLAN_SHA256 = '%s'   # Next Study - Beat-Accurate Synthetic ECG/EXP0C_PREREG.md\n"
      "REQUIRE_GPU = 'A100'       # full run only: the GPU class of Runs 1 and 2, so arms C, Springer and W reproduce bit for bit\n"
      "EXP0C_TASKS = ('grading', 'timing', 'shape')\n"
      "EXP0C_SEEDS_FS = (1, 2, 3, 4, 5)                 # S2 failure on Springer's failure set: the directions vary\n"
      "EXP0C_SEEDS_RN = tuple(range(1, 11))             # random recording sets" % PLAN_SHA)
patch('env', "GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32",
      "GAN_BATCH = None\n"
      "if (not DRY_RUN) and LIMIT_RECORDINGS == 0 and REQUIRE_GPU:\n"
      "    assert REQUIRE_GPU in GPU_NAME, 'Experiment 0c must run on an %s (Run 1 class) so the controls reproduce; this runtime has %s' % (REQUIRE_GPU, GPU_NAME)")
patch('env', "print('generator batch', GAN_BATCH, '| feature workers', N_PROC)", "print('feature workers', N_PROC)")

# ------------------------------------------------------------------ 2. resolve: no generator; exp0c folder
patch('resolve', """    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
""", "")
i = idx('resolve')
txt = CELLS[i][2]
a, b = txt.index('    # ---- generator checkpoint ----'), txt.index('    # ---- output folder and run flags ----')
CELLS[i] = (CELLS[i][0], 'resolve', txt[:a] + '    GAN_CHECKPOINT, CKPT_SHA256, CKPT_HASHES = None, None, {}     # Experiment 0c runs no generator\n\n' + txt[b:])
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/exp0c'))")
patch('resolve', "run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)",
      "run='exp0c_failure_placement', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', exp0c_plan_sha256=EXP0C_PLAN_SHA256)")
patch('resolve', "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),",
      "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH),")
patch('resolve', "    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))",
      "    print('paired wav+tsv files: %d' % N_PAIRS)")
drop('v_generator', '*Experiment 0c: the generator architecture cell is not used.*')
drop('v_ckpt_load', '*Experiment 0c: no generator checkpoint is loaded.*')
drop('s3_generator', '## Stage 3: not run\n\nExperiment 0c builds no synthetic ECG.')
patch('s2_audio', "    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True",
      "    ECG_CHUNK_OK[k] = True       # no generator: the 8 kHz audio is never loaded")

# ------------------------------------------------------------------ 3. arm lists
patch('s4_arm_defs', """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS""", """R1_ARMS = list(ARMS)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
EXP0C_CONDS = []                                     # (tag, type, condition seed, selection, other arm); EXP0C_PREREG.md
for _s in EXP0C_SEEDS_FS:
    EXP0C_CONDS.append(('s2fs_s%d' % _s, 's2', 80000 + _s, 'F_S', None))
for _s in EXP0C_SEEDS_RN:
    EXP0C_CONDS.append(('s2rn_s%02d' % _s, 's2', 90000 + _s, 'random', None))
EXP0C_CONDS.append(('etfs', 'fail', 0, 'F_S', None))
for _s in EXP0C_SEEDS_RN:
    EXP0C_CONDS.append(('etrn_s%02d' % _s, 'fail', 110000 + _s, 'random', None))
EXP0C_CONDS += [('s_inside', 'inside', 0, 'F_S', 'S'), ('s_outside', 'outside', 0, 'F_S', 'S'),
                ('w_inside', 'inside', 0, 'F_W', 'W'), ('w_outside', 'outside', 0, 'F_W', 'W')]
EXP0C_CONTROLS = [('s_all', 'all', 0, 'none', 'S'), ('w_all', 'all', 0, 'none', 'W')]
COND_TAGS = [c[0] for c in EXP0C_CONDS]
CONTROL_TAGS = [c[0] for c in EXP0C_CONTROLS]
ALL_CONDS = EXP0C_CONDS + EXP0C_CONTROLS
assert len(COND_TAGS) == 30 and len(set(COND_TAGS + CONTROL_TAGS)) == 32
ARMS = ['pcg_seg'] + COND_TAGS + CONTROL_TAGS                            # arms C, Springer and W everywhere are controls""")
patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """ARM_LETTER.update({t: t for t in COND_TAGS + CONTROL_TAGS})
ARM_LABEL.update({t: 'Experiment 0c ' + t for t in COND_TAGS + CONTROL_TAGS})
print('Experiment 0c: arm C, %d sets and controls %s | arm configuration hash %s' % (len(COND_TAGS), CONTROL_TAGS, ARM_CFG_HASH))""")

EXP0C_DEFS = r'''
# ---------------- Experiment 0c: code, failure sets, Springer and W cycles, reference predictions (embedded) ----------------
import base64 as _b64x, zlib as _zx
_MOD_B64 = %(mod_b64)s
_MOD_SHA = %(mod_sha)s
_code_root = (Path('/content') if IN_COLAB else RUN_DIR) / 'exp0c_code'
(_code_root / 'extstudy').mkdir(parents=True, exist_ok=True)
for _m, _b in _MOD_B64.items():
    _src = _b64x.b64decode(_b)
    assert hashlib.sha256(_src).hexdigest() == _MOD_SHA[_m], 'embedded module %%s corrupted' %% _m
    (_code_root / 'extstudy' / _m).write_bytes(_src)
for _k in [k for k in sys.modules if k == 'extstudy' or k.startswith('extstudy.')]:
    del sys.modules[_k]
sys.path.insert(0, str(_code_root))
from extstudy import corrupt as KOR, placement as PL
_R = '%(ref_b64)s'
assert hashlib.sha256(_R.encode()).hexdigest() == '%(ref_sha)s', 'embedded reference predictions corrupted'
REF = json.loads(_zx.decompress(_b64x.b64decode(_R)).decode())          # {pcg_seg: Run 1 C, s_all: Run 1 Springer, w_all: Run 2 W}
_S = '%(sets_b64)s'
assert hashlib.sha256(_S.encode()).hexdigest() == '%(sets_sha)s', 'embedded failure sets corrupted'
_SETS = json.loads(_zx.decompress(_b64x.b64decode(_S)).decode())
EXP0C_NAMES, F_S, F_W, EXP0C_C_HASH = _SETS['names'], _SETS['F_S'], _SETS['F_W'], _SETS['c_hash']
OTHER = {}
for _k, _e in _SETS['cyc'].items():
    assert hashlib.sha256(_e['b64'].encode()).hexdigest() == _e['sha256'], 'embedded %%s cycles corrupted' %% _k
    OTHER[_k] = PL.unpack(_e['b64'])
    assert PL.cycles_hash(OTHER[_k], EXP0C_NAMES) == _e['hash'], 'embedded %%s cycles do not match their hash' %% _k
del _SETS, _S, _R
RUN1_OOF_SHA256, RUN2_OOF_SHA256 = '%(oof1_sha)s', '%(oof2_sha)s'
EXP0C_SEL = {}          # {tag: recordings selected}; filled in the main process before the worker pool starts

def _exp0c_worker(name):
    """Every Experiment 0c boundary set of one recording, with its boundary quality (before any fallback) and its S2
    offset (on the cycles that go to the features) against arm C."""
    try:
        L4 = AUDIO_META[name]['L4']
        ref = CYC['pcg_seg'][name]
        out = {}
        for tag, typ, seed, sel, arm in ALL_CONDS:
            cyc, info = PL.exp0c_cycles(ref, name, typ, seed, EXP0C_SEL[tag], OTHER[arm][name] if arm else None,
                                        equal_time_cycles, L4, TARGET_SR)
            bs = KOR.boundary_stats(cyc, ref, TARGET_SR)
            fb = len(cyc) < N_MIN_CYCLES
            if fb:
                cyc = equal_time_cycles(L4, TARGET_SR)
            off, npair = KOR.s2_onset_offset_ms(cyc, ref, TARGET_SR)
            out[tag] = (cyc, dict(bs, fallback=bool(fb), selected=name in EXP0C_SEL[tag], s2_offset_ms=off, s2_pairs=npair,
                                  **{'d_' + k: v for k, v in info.items()}))
        return name, out, None
    except Exception as e:
        return name, None, '%%s: %%s' %% (type(e).__name__, e)
print('Experiment 0c code ready: %%d sets + %%d controls | F_S %%d, F_W %%d | placement.py sha256 %%s' %% (
    len(EXP0C_CONDS), len(EXP0C_CONTROLS), len(F_S), len(F_W), _MOD_SHA['placement.py'][:16]))
''' % dict(mod_b64=json.dumps(MOD_B64), mod_sha=json.dumps(MOD_SHA), ref_b64=REF_B64, ref_sha=REF_SHA, sets_b64=SETS_B64,
           sets_sha=SETS_SHA, oof1_sha=OOF1_SHA, oof2_sha=OOF2_SHA)
insert_after('ext_run1_defs', [('code', 'exp0c_defs', EXP0C_DEFS.strip('\n'))])

# ------------------------------------------------------------------ 4. Stage 4 for Experiment 0c
replace_cell('s4_run', r'''
CYC = {arm: {} for arm in ARMS}
ARM_DIAG = {arm: {} for arm in ARMS}
ARM_FAILURES = []
with stage('S4 boundaries'):
    for _t in [t for t in list(TASKS) if t not in EXP0C_TASKS]:
        del TASKS[_t]                                                          # Experiment 0c: grading, timing and shape only
    _lab = lambda n: any(LAB_OF[n][TASKS[t]] != -1 for t in TASKS)
    _in0c = set(EXP0C_NAMES)
    if VALIDATION_APPLICABLE:
        _missing = [n for n in PINNED_KEEP if n not in set(NAMES_ALL)]
        assert not _missing, 'Run 1 recordings missing from this index: %s' % _missing[:5]
        KEEP = [n for n in PINNED_KEEP if _lab(n)]
        assert sorted(KEEP) == sorted(EXP0C_NAMES), 'the labelled kept set differs from the 610 recordings of the plan'
    else:
        KEEP = [n for n in NAMES_ALL if _lab(n) and n in _in0c]                # reduced run: recordings with embedded cycles only
    t0 = time.time()
    _cd = {}
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_cd_worker, KEEP, chunksize=16), total=len(KEEP), desc='arm C', leave=False):
            assert err is None, (name, err)
            _cd[name] = payload
    for n in KEEP:
        CYC['pcg_seg'][n], ARM_DIAG['pcg_seg'][n] = _cd[n][0], _cd[n][1]
    del _cd
    KEEP = [n for n in KEEP if len(CYC['pcg_seg'][n]) >= N_MIN_CYCLES] if not VALIDATION_APPLICABLE else KEEP
    KEEP_ARR = np.array(KEEP)
    DROPPED = []
    _sorted = sorted(KEEP)
    print('   recordings labelled for %s: %d | arm C built in %.0f s' % (list(TASKS), len(KEEP), time.time() - t0))
    if VALIDATION_APPLICABLE:
        assert PL.cycles_hash(CYC['pcg_seg'], _sorted) == EXP0C_C_HASH, 'arm C cycles differ from those the failure sets were fixed on'
    for _k, _F in (('S', F_S), ('W', F_W)):                                     # the failure sets recompute from the rebuilt arm C
        assert PL.failure_set(OTHER[_k], CYC['pcg_seg'], _sorted) == sorted(set(_F) & set(KEEP)), 'failure set %s does not recompute' % _k
    _named = {'F_S': set(F_S) & set(KEEP), 'F_W': set(F_W) & set(KEEP), 'none': set()}
    EXP0C_SEL.clear()
    for tag, typ, seed, sel, arm in ALL_CONDS:
        EXP0C_SEL[tag] = PL.select_count(KEEP, len(_named['F_S']), seed) if sel == 'random' else _named[sel]
    print('   failure sets recomputed: F_S %d, F_W %d; random sets of %d' % (len(_named['F_S']), len(_named['F_W']), len(_named['F_S'])))
    t0 = time.time()
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_exp0c_worker, KEEP, chunksize=4), total=len(KEEP), desc='Experiment 0c sets', leave=False):
            assert err is None, (name, err)
            for tag, (cyc, d) in payload.items():
                CYC[tag][name], ARM_DIAG[tag][name] = cyc, d
    for arm in ARMS:
        for n in KEEP:
            check_cycles(CYC[arm][n], AUDIO_META[n]['L4'], arm + ' ' + n, strict=False)
    EXP0C_BOUNDARY, _offsets = {}, {}
    for tag, typ, seed, sel, arm in ALL_CONDS:
        rows = [ARM_DIAG[tag][n] for n in KEEP]
        assert sum(r['selected'] for r in rows) == len(EXP0C_SEL[tag]), tag
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        off = np.array([r['s2_offset_ms'] for r in rows], float)
        has = np.array([r['s2_pairs'] > 0 for r in rows])
        EXP0C_BOUNDARY[tag] = dict(type=typ, seed=seed, selection=sel, arm=arm, n_selected=len(EXP0C_SEL[tag]),
                                   cycles=int(sum(r['d_n'] for r in rows)), cycles_displaced=int(sum(r['d_moved'] for r in rows)),
                                   cycles_clipped=int(sum(r['d_clipped'] for r in rows)),
                                   s2_off_gt60_share=float(np.mean(np.abs(off[has]) > 60)) if has.any() else None,
                                   recordings_with_s2_pairs=int(has.sum()), f1_median=med('f1'), offset_sd_ms_median=med('offset_sd_ms'),
                                   f1_pooled=float(2 * sum(r['matched'] for r in rows) / sum(r['n'] + r['n_ref'] for r in rows)),
                                   fallback_frac=float(np.mean([r['fallback'] for r in rows])))
        _offsets[tag] = dict(offset_ms=[None if not np.isfinite(v) else round(float(v), 4) for v in off],
                             pairs=[int(r['s2_pairs']) for r in rows], selected=[bool(r['selected']) for r in rows])
    with open(RUN_DIR / 'exp0c_boundaries.json', 'w') as fh:
        json.dump(dict(plan_sha256=EXP0C_PLAN_SHA256, n_recordings=len(KEEP), f_s=sorted(_named['F_S']), f_w=sorted(_named['F_W']),
                       conditions=EXP0C_BOUNDARY), fh, indent=1)
    with open(RUN_DIR / 'exp0c_s2_offsets.json', 'w') as fh:
        json.dump(dict(plan_sha256=EXP0C_PLAN_SHA256, names=list(KEEP), conditions=_offsets), fh)
    del _offsets
    print('   %d sets and %d controls built and checked in %.0f s' % (len(COND_TAGS), len(CONTROL_TAGS), time.time() - t0))
    for _g, _pre in (('S2 failure on F_S', 's2fs'), ('S2 failure at random', 's2rn'), ('total failure on F_S', 'etfs'),
                     ('total failure at random', 'etrn'), ('Springer inside', 's_inside'), ('Springer outside', 's_outside'),
                     ('W inside', 'w_inside'), ('W outside', 'w_outside'), ('Springer everywhere', 's_all'), ('W everywhere', 'w_all')):
        v = [EXP0C_BOUNDARY[t] for t in EXP0C_BOUNDARY if t.startswith(_pre)]
        print('   %-24s sets %2d | S2 >60 ms off in %.3f of recordings | boundary F1 median %.3f | fallback %.2f%%' % (
            _g, len(v), np.mean([x['s2_off_gt60_share'] for x in v if x['s2_off_gt60_share'] is not None] or [np.nan]),
            np.mean([x['f1_median'] for x in v]), 100 * np.mean([x['fallback_frac'] for x in v])))
    DIAG_SUMMARY, GLOBAL_OFFSET, KEEP_PINNED_OK = {}, {}, (VALIDATION_APPLICABLE or None)
''')

# ------------------------------------------------------------------ 5. folds: detection is not part of Experiment 0c
patch('s6_folds', "    assert 'detection' in TASKS, 'the detection task cannot be skipped'",
      "    assert (not VALIDATION_APPLICABLE) or set(TASKS) == set(EXP0C_TASKS), 'Experiment 0c runs grading, timing and shape'")

# ------------------------------------------------------------------ 6. classification
i = idx('s7_run')
txt = CELLS[i][2]
a = txt.index('PAIRS = [')
b = txt.index("('pcg_seg', 'equal_time_annot')]") + len("('pcg_seg', 'equal_time_annot')]")
CELLS[i] = (CELLS[i][0], 's7_run', txt[:a] + 'PAIRS = []          # Experiment 0c: comparisons are computed locally from oof_predictions.npz' + txt[b:])
patch('s7_run', "        r_, e_ = CV[task]['pcg_seg'], CV[task]['equal_time']", "        r_, e_ = CV[task]['pcg_seg'], CV[task][COND_TAGS[0]]     # Experiment 0c: no arm E here")
patch('s7_run', "    _oracle_check('detection')", "    _oracle_check(list(TASKS)[0])")
patch('s7_run', "    for task in ['detection'] + _three[:1]:", "    for task in _three[:1]:")
patch('s7_run', "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')",
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Experiment 0c classifies only in the full run.')")
patch('s7_run', "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL,",
      "exp0c_boundaries=EXP0C_BOUNDARY, exp0c_plan_sha256=EXP0C_PLAN_SHA256, run1_oof_sha256=RUN1_OOF_SHA256, run2_oof_sha256=RUN2_OOF_SHA256,")
patch('s7_run', "springer_cfg_hash=S_CFG_HASH, springer_variants=S_VARIANTS,", "")

# ------------------------------------------------------------------ 7. integrity gate
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, '## Stage 8: integrity gate (Experiment 0c)\n\nThe recordings are the Run 1 kept set restricted to the three '
                 'tasks; the folds equal the embedded child-grouped assignment; arm C reproduces Run 1, Springer everywhere reproduces '
                 "Run 1's Springer and W everywhere reproduces Run 2's W (MLP predictions identical, probabilities within 1e-5, random "
                 'forest identical); a repeat fit and the worker pool return identical probabilities.')
CELLS[i8] = ('code', 's8_gate', r'''
with stage('S8 integrity gate'):
    gate = dict(kind='integrity (Experiment 0c)', applicable=bool(VALIDATION_APPLICABLE))
    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        gate.update(ok=None, decided_on='not applicable (reduced run)')
        print('integrity gate: not applicable in a reduced run')
    else:
        gate['folds'] = FOLD_CHECK
        gate['folds_ok'] = all(all(v.values()) for v in FOLD_CHECK.values()) and set(FOLD_CHECK) == set(TASKS)
        rep = {}
        for tag in ('pcg_seg', 's_all', 'w_all'):
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
        checks = [gate['folds_ok'], gate['repeat_fit_ok']] + list(gate['reproduction_ok'].values())
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

# ------------------------------------------------------------------ 8. manifest
patch('s11_manifest', "        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),",
      "        checkpoint=None, exp0c=dict(plan_sha256=EXP0C_PLAN_SHA256, modules_sha256=_MOD_SHA, n_conditions=len(COND_TAGS), controls=CONTROL_TAGS),")
patch('s11_manifest', "        generator=dict(batch_start=GAN_BATCH, batch_final=GEN_BATCH_USED, oom_events=OOM_EVENTS, signal_length=SIGNAL_LENGTH, gan_sr=GAN_SR, ecg_cache_dtype='float16'),",
      "        generator=None,")

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
CELLS[iprov] = (CELLS[iprov][0], CELLS[iprov][1], ptxt.replace(old, "NOTEBOOK_SOURCE_SHA256 = '%s'   # filled in by build_exp0c.py" % SRC_SHA))
cells = []
for kind, name, text in CELLS:
    assert '\u2014' not in text, 'em dash in cell %s' % name
    cells.append(new_markdown_cell(text) if kind == 'markdown' else new_code_cell(text, metadata={'name': name} if name else {}))
nb = new_notebook(cells=cells)
nb.metadata = dict(nb1['metadata'])
nb.metadata['colab'] = dict(nb1['metadata'].get('colab', {}), name='Experiment0c_FailurePlacement.ipynb', machine_shape='hm', gpuType='A100')
nb.metadata['built_by'] = 'extension_study/notebooks/build_exp0c.py (patches over the locked Extension_Run1.ipynb)'
nb.metadata['notebook_source_sha256'] = SRC_SHA
nb.metadata['embedded'] = dict(exp0c_plan_sha256=PLAN_SHA, modules_sha256=MOD_SHA, run1_oof_sha256=OOF1_SHA, run2_oof_sha256=OOF2_SHA,
                               sets_sha256=SETS_SHA, reference_predictions_sha256=REF_SHA)
tmp = Path(tempfile.mkdtemp())
for k, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % k)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Experiment 0c notebook written: %s (%d cells) | source sha256 %s' % (OUT_NB, len(nb.cells), SRC_SHA[:16]))
