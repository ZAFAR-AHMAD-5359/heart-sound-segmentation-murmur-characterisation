#!/usr/bin/env python
"""
build_exp0.py: assembles Experiment0_BeatErrorBudget.ipynb (next study, Experiment 0; plan in
'Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md') from the locked Extension_Run1.ipynb, patched by cell name.

What changes from Run 1, and nothing else:
  1. run folder Extension_Study/runs/exp0/<timestamp>; no generator (GAN cells out, no 8 kHz audio);
  2. recordings: the Run 1 kept set restricted to those labelled for grading, timing or shape; tasks: those three;
  3. boundary sets: arm C (reproduction control: its out-of-fold predictions must equal Run 1's, embedded) and 60
     corrupted copies of arm C (missed beats, spurious beats, timing jitter; four levels each, five seeds;
     extstudy/corrupt.py and R1's jitter_cycles), each with its boundary quality against arm C;
  4. features, folds (embedded, child-grouped), five-seed MLP and random forest: Run 1's cells unchanged;
  5. integrity gate: kept subset, folds, arm C reproduced, repeat fit, pool check.
Curves, retention and the read-off are computed locally (notebooks/exp0_analysis.py) with Run 1's arms C and E.

Usage:  python build_exp0.py   (writes Experiment0_BeatErrorBudget.ipynb beside this file)
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
OUT_NB = HERE / 'Experiment0_BeatErrorBudget.ipynb'
PLAN = Path('E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md')
RUN1_DIR = PHD / 'Results' / 'A100 Results' / 'extracted' / 'runs' / 'run1' / '20261016_1516'
MODULES = ['__init__.py', 'corrupt.py']
EXP0_TASKS = ('grading', 'timing', 'shape')

sha = lambda b: hashlib.sha256(b).hexdigest()
lock = json.load(open(STUDY / 'prereg' / 'RUN1_LOCK.json'))
assert sha(RUN1_NB.read_bytes()) == lock['run1_notebook_file_sha256'], 'Extension_Run1.ipynb differs from its lock'
PLAN_SHA = sha(PLAN.read_bytes())
assert open(PLAN.with_suffix('.sha256')).read().split()[0] == PLAN_SHA, 'EXP0_PREREG.md changed after it was hashed'
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


# ------------------------------------------------------------------ embedded: corruption code, Run 1 arm C predictions
MODS = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in MODS.items()}
MOD_SHA = {m: sha(b) for m, b in MODS.items()}
o1 = np.load(RUN1_DIR / 'oof_predictions.npz', allow_pickle=False)
RUN1_C = {}
for t in EXP0_TASKS:
    p = '%s__pcg_seg__' % t
    RUN1_C[t] = dict(names=o1[p + 'names'].astype(str).tolist(), y=o1[p + 'y'].astype(int).tolist(),
                     mlp_pred=o1[p + 'mlp__pred'].astype(int).tolist(), mlp_proba=o1[p + 'mlp__proba'].astype(float).round(8).tolist(),
                     rf_pred=o1[p + 'rf__pred'].astype(int).tolist())
RUN1_C_B64 = base64.b64encode(zlib.compress(json.dumps(RUN1_C).encode(), 9)).decode()
RUN1_C_SHA = sha(RUN1_C_B64.encode())
OOF1_SHA = sha((RUN1_DIR / 'oof_predictions.npz').read_bytes())

# ------------------------------------------------------------------ 0. header
CELLS[0] = ('markdown', None, r'''# Experiment 0: the beat-error budget

How much beat-level error can a cycle segmentation carry before accurate gating loses its gain over equal-time windows
on murmur grading, timing and shape? Plan: `Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md` (hashed before any
corrupted set existed). Built by `extension_study/notebooks/build_exp0.py` from the locked Run 1 notebook, patched cell
by cell.

**What runs.** Arm C's annotated cycles, unchanged (reproduction control: its out-of-fold predictions must equal Run
1's) and corrupted in 60 controlled ways: missed beats (10, 20, 30, 50% of starts), spurious beats (10, 20, 30, 50%)
and timing jitter (SD 5, 10, 20, 40 ms), five seeds each. Each set goes through Run 1's feature extractor and
five-seed MLP on the child-grouped folds, for the recordings labelled for grading, timing or shape.

**How to run.** Runtime > Change runtime type > **A100 GPU** with High-RAM; leave `DRY_RUN = False`; Runtime > Run all.
Expected time about 1.5 to 2.5 hours; if the session drops, set `RESUME_DIR` to the run folder and run all again
(features and fits are cached).

**Afterwards.** Download `Extension_Study/runs/exp0/<timestamp>` (results.json, oof_predictions.npz,
exp0_boundaries.json, run_manifest.json); the curves and the read-off are computed locally.
''')

# ------------------------------------------------------------------ 1. flags and environment
patch('flags', "R1_RUN_DIR = '/content/drive/MyDrive/JBHI-Revision/R1_Gating_Campaign/20261003_1236'   # R1 run: its feature matrices are compared with this run's (reproduction check); None to skip",
      "R1_RUN_DIR = None\n"
      "EXP0_PLAN_SHA256 = '%s'   # Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md\n"
      "REQUIRE_GPU = 'A100'       # full run only: Run 1's GPU class, so arm C reproduces bit for bit\n"
      "EXP0_TASKS = ('grading', 'timing', 'shape')\n"
      "EXP0_LEVELS = {'del': (0.10, 0.20, 0.30, 0.50), 'ins': (0.10, 0.20, 0.30, 0.50), 'jit': (5, 10, 20, 40)}\n"
      "EXP0_SEEDS = (1, 2, 3, 4, 5)" % PLAN_SHA)
patch('env', "GAN_BATCH = 16 if (device.type == 'cpu' or 'T4' in GPU_NAME) else 32",
      "GAN_BATCH = None\n"
      "if (not DRY_RUN) and LIMIT_RECORDINGS == 0 and REQUIRE_GPU:\n"
      "    assert REQUIRE_GPU in GPU_NAME, 'Experiment 0 must run on an %s (Run 1 class) so arm C reproduces Run 1; this runtime has %s' % (REQUIRE_GPU, GPU_NAME)")
patch('env', "print('generator batch', GAN_BATCH, '| feature workers', N_PROC)", "print('feature workers', N_PROC)")

# ------------------------------------------------------------------ 2. resolve: no generator; exp0 folder
patch('resolve', """    if not MANUAL_CHECKPOINT:
        _targets['trail4_best.pt'] = 'file'
""", "")
i = idx('resolve')
txt = CELLS[i][2]
a, b = txt.index('    # ---- generator checkpoint ----'), txt.index('    # ---- output folder and run flags ----')
CELLS[i] = (CELLS[i][0], 'resolve', txt[:a] + '    GAN_CHECKPOINT, CKPT_SHA256, CKPT_HASHES = None, None, {}     # Experiment 0 runs no generator\n\n' + txt[b:])
patch('resolve', "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/run1'))",
      "OUT_ROOT = Path(MANUAL_OUTPUT_ROOT or (DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs/exp0'))")
patch('resolve', "run='extension_run1', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', prereg_sha256=PREREG_SHA256)",
      "run='exp0_beat_error_budget', ensemble_seeds=list(ENSEMBLE_SEEDS), groups='child', exp0_plan_sha256=EXP0_PLAN_SHA256)")
patch('resolve', "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH), ('GAN_CHECKPOINT', GAN_CHECKPOINT),",
      "    for k_, v_ in [('AUDIO_PATH', AUDIO_PATH), ('METADATA_PATH', METADATA_PATH),")
patch('resolve', "    print('paired wav+tsv files: %d | checkpoint sha256: %s' % (N_PAIRS, CKPT_SHA256))",
      "    print('paired wav+tsv files: %d' % N_PAIRS)")
drop('v_generator', '*Experiment 0: the generator architecture cell is not used.*')
drop('v_ckpt_load', '*Experiment 0: no generator checkpoint is loaded.*')
drop('s3_generator', '## Stage 3: not run\n\nExperiment 0 builds no synthetic ECG.')
patch('s2_audio', "    ECG_CHUNK_OK[k] = read_cache(CACHE_DIR / ('ecg_%03d.npz' % k), lambda z, names=names: True if ecg_chunk_matches(z, names) else None) is True",
      "    ECG_CHUNK_OK[k] = True       # no generator: the 8 kHz audio is never loaded")

# ------------------------------------------------------------------ 3. arm lists
patch('s4_arm_defs', """R1_ARMS = list(ARMS)                        # the kept set is decided by these six arms only (pinned to the R1 recordings)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
ARMS = R1_ARMS + S_ARMS + COV_ARMS""", """R1_ARMS = list(ARMS)
S_ARMS = ['springer_wpk', 'springer_pub']
COV_ARMS = ['springer_wpk_annot', 'equal_time_annot']
EXP0_TYPE_CODE = {'del': 1, 'ins': 2, 'jit': 3}
EXP0_CONDS = []                                                           # (tag, type, level, seed)
for _t, _levels in EXP0_LEVELS.items():
    for _li, _lv in enumerate(_levels):
        for _s in EXP0_SEEDS:
            _tag = '%s%03d_s%d' % (_t, int(round(_lv * 100)) if _t != 'jit' else int(_lv), _s)
            EXP0_CONDS.append((_tag, _t, _lv, 10000 * EXP0_TYPE_CODE[_t] + 100 * _li + _s))
COND_TAGS = [c[0] for c in EXP0_CONDS]
ARMS = ['pcg_seg'] + COND_TAGS                                            # arm C is the reproduction control""")
patch('s4_arm_defs', "print('arm definitions ready:', ARMS, '| arm configuration hash', ARM_CFG_HASH)",
      """ARM_LETTER.update({t: t for t in COND_TAGS})
ARM_LABEL.update({t: 'C corrupted ' + t for t in COND_TAGS})
print('Experiment 0: arm C plus %d corrupted copies (%s ...) | arm configuration hash %s' % (len(COND_TAGS), COND_TAGS[:3], ARM_CFG_HASH))""")

EXP0_DEFS = r'''
# ---------------- Experiment 0: corruption code and Run 1's arm C predictions (embedded) ----------------
import base64 as _b64x, zlib as _zx
_MOD_B64 = %(mod_b64)s
_MOD_SHA = %(mod_sha)s
_code_root = (Path('/content') if IN_COLAB else RUN_DIR) / 'exp0_code'
(_code_root / 'extstudy').mkdir(parents=True, exist_ok=True)
for _m, _b in _MOD_B64.items():
    _src = _b64x.b64decode(_b)
    assert hashlib.sha256(_src).hexdigest() == _MOD_SHA[_m], 'embedded module %%s corrupted' %% _m
    (_code_root / 'extstudy' / _m).write_bytes(_src)
for _k in [k for k in sys.modules if k == 'extstudy' or k.startswith('extstudy.')]:
    del sys.modules[_k]
sys.path.insert(0, str(_code_root))
from extstudy import corrupt as KOR
_C1 = '%(run1_c_b64)s'
assert hashlib.sha256(_C1.encode()).hexdigest() == '%(run1_c_sha)s', 'embedded Run 1 arm C predictions corrupted'
RUN1_C = json.loads(_zx.decompress(_b64x.b64decode(_C1)).decode())
RUN1_OOF_SHA256 = '%(oof1_sha)s'

def _exp0_worker(name):
    """Every corrupted copy of one recording's arm C cycles, with its boundary quality against arm C."""
    try:
        L4 = AUDIO_META[name]['L4']
        ref = CYC['pcg_seg'][name]
        out = {}
        for tag, typ, lv, seed in EXP0_CONDS:
            rng = KOR.rng_for(seed, name)
            if typ == 'del':
                cyc = KOR.delete_starts(ref, lv, rng)
            elif typ == 'ins':
                cyc = KOR.insert_starts(ref, lv, rng, segment_phases_samples, L4)
            else:
                cyc = jitter_cycles(ref, lv, TARGET_SR, rng, L4)
            bs = KOR.boundary_stats(cyc, ref, TARGET_SR)
            fb = len(cyc) < N_MIN_CYCLES
            if fb:
                cyc = equal_time_cycles(L4, TARGET_SR)
            out[tag] = (cyc, dict(bs, fallback=bool(fb)))
        return name, out, None
    except Exception as e:
        return name, None, '%%s: %%s' %% (type(e).__name__, e)
print('Experiment 0 code ready: %%d conditions | corrupt.py sha256 %%s' %% (len(EXP0_CONDS), _MOD_SHA['corrupt.py'][:16]))
''' % dict(mod_b64=json.dumps(MOD_B64), mod_sha=json.dumps(MOD_SHA), run1_c_b64=RUN1_C_B64, run1_c_sha=RUN1_C_SHA, oof1_sha=OOF1_SHA)
insert_after('ext_run1_defs', [('code', 'exp0_defs', EXP0_DEFS.strip('\n'))])

# ------------------------------------------------------------------ 4. Stage 4 for Experiment 0
replace_cell('s4_run', r'''
CYC = {arm: {} for arm in ARMS}
ARM_DIAG = {arm: {} for arm in ARMS}
ARM_FAILURES = []
with stage('S4 boundaries'):
    for _t in [t for t in list(TASKS) if t not in EXP0_TASKS]:
        del TASKS[_t]                                                          # Experiment 0: grading, timing and shape only
    _lab = lambda n: any(LAB_OF[n][TASKS[t]] != -1 for t in TASKS)
    if VALIDATION_APPLICABLE:
        _missing = [n for n in PINNED_KEEP if n not in set(NAMES_ALL)]
        assert not _missing, 'Run 1 recordings missing from this index: %s' % _missing[:5]
        KEEP = [n for n in PINNED_KEEP if _lab(n)]
    else:
        KEEP = [n for n in NAMES_ALL if _lab(n)]
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
    print('   recordings labelled for %s: %d | arm C built in %.0f s' % (list(TASKS), len(KEEP), time.time() - t0))
    t0 = time.time()
    with _pool(N_PROC) as ex:
        for name, payload, err in tqdm(ex.map(_exp0_worker, KEEP, chunksize=4), total=len(KEEP), desc='corrupted sets', leave=False):
            assert err is None, (name, err)
            for tag, (cyc, d) in payload.items():
                CYC[tag][name], ARM_DIAG[tag][name] = cyc, d
    for arm in ARMS:
        for n in KEEP:
            check_cycles(CYC[arm][n], AUDIO_META[n]['L4'], arm + ' ' + n, strict=False)
    EXP0_BOUNDARY = {}
    for tag, typ, lv, seed in EXP0_CONDS:
        rows = [ARM_DIAG[tag][n] for n in KEEP]
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        EXP0_BOUNDARY[tag] = dict(type=typ, level=lv, seed=seed, f1_median=med('f1'), insertion_median=med('insertion'),
                                  deletion_median=med('deletion'), offset_sd_ms_median=med('offset_sd_ms'),
                                  f1_pooled=float(2 * sum(r['matched'] for r in rows) / sum(r['n'] + r['n_ref'] for r in rows)),
                                  fallback_frac=float(np.mean([r['fallback'] for r in rows])))
    with open(RUN_DIR / 'exp0_boundaries.json', 'w') as fh:
        json.dump(dict(plan_sha256=EXP0_PLAN_SHA256, n_recordings=len(KEEP), conditions=EXP0_BOUNDARY), fh, indent=1)
    print('   %d corrupted sets built and checked in %.0f s' % (len(COND_TAGS), time.time() - t0))
    for typ in EXP0_LEVELS:
        for lv in EXP0_LEVELS[typ]:
            v = [EXP0_BOUNDARY[c[0]] for c in EXP0_CONDS if c[1] == typ and c[2] == lv]
            print('   %-3s %5s: boundary F1 median %.3f (range over seeds %.3f to %.3f) | offset SD %.1f ms | fallback %.2f%%' % (
                typ, lv, np.mean([x['f1_median'] for x in v]), min(x['f1_median'] for x in v), max(x['f1_median'] for x in v),
                np.mean([x['offset_sd_ms_median'] for x in v]), 100 * np.mean([x['fallback_frac'] for x in v])))
    DIAG_SUMMARY, GLOBAL_OFFSET, KEEP_PINNED_OK = {}, {}, (VALIDATION_APPLICABLE or None)
''')

# ------------------------------------------------------------------ 5. folds: detection is not part of Experiment 0
patch('s6_folds', "    assert 'detection' in TASKS, 'the detection task cannot be skipped'",
      "    assert (not VALIDATION_APPLICABLE) or set(TASKS) == set(EXP0_TASKS), 'Experiment 0 runs grading, timing and shape'")

# ------------------------------------------------------------------ 6. classification
i = idx('s7_run')
txt = CELLS[i][2]
a = txt.index('PAIRS = [')
b = txt.index("('pcg_seg', 'equal_time_annot')]") + len("('pcg_seg', 'equal_time_annot')]")
CELLS[i] = (CELLS[i][0], 's7_run', txt[:a] + 'PAIRS = []          # Experiment 0: comparisons are computed locally from oof_predictions.npz' + txt[b:])
patch('s7_run', "        r_, e_ = CV[task]['pcg_seg'], CV[task]['equal_time']", "        r_, e_ = CV[task]['pcg_seg'], CV[task][COND_TAGS[0]]     # Experiment 0: no arm E here")
patch('s7_run', "    _oracle_check('detection')", "    _oracle_check(list(TASKS)[0])")
patch('s7_run', "    for task in ['detection'] + _three[:1]:", "    for task in _three[:1]:")
patch('s7_run', "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Run 1 classifies only in the full, pre-registered run.')",
      "    raise SystemExit('Smoke test complete: stages 0 to 6 ran. Experiment 0 classifies only in the full run.')")
patch('s7_run', "fallback_share=S_FALLBACK, fallback_fail_share=S_FALLBACK_FAIL,",
      "exp0_boundaries=EXP0_BOUNDARY, exp0_plan_sha256=EXP0_PLAN_SHA256, run1_oof_sha256=RUN1_OOF_SHA256,")
patch('s7_run', "springer_cfg_hash=S_CFG_HASH, springer_variants=S_VARIANTS,", "")

# ------------------------------------------------------------------ 7. integrity gate
i8 = idx('s8_gate')
CELLS[i8 - 1] = ('markdown', None, '## Stage 8: integrity gate (Experiment 0)\n\nThe recordings are the Run 1 kept set restricted to the three '
                 'tasks; the folds equal the embedded child-grouped assignment; arm C reproduces Run 1 (MLP predictions identical, '
                 'probabilities within 1e-5, random forest identical); a repeat fit and the worker pool return identical probabilities.')
CELLS[i8] = ('code', 's8_gate', r'''
with stage('S8 integrity gate'):
    gate = dict(kind='integrity (Experiment 0)', applicable=bool(VALIDATION_APPLICABLE))
    if not VALIDATION_APPLICABLE:
        VALIDATION_OK = None
        gate.update(ok=None, decided_on='not applicable (reduced run)')
        print('integrity gate: not applicable in a reduced run')
    else:
        gate['folds'] = FOLD_CHECK
        gate['folds_ok'] = all(all(v.values()) for v in FOLD_CHECK.values()) and set(FOLD_CHECK) == set(TASKS)
        c_chk = {}
        for t in TASKS:
            r = CV[t]['pcg_seg']; ref = RUN1_C[t]
            same_rows = list(map(str, r['names'])) == ref['names'] and np.array_equal(np.asarray(r['y']), np.asarray(ref['y']))
            c_chk[t] = dict(same_rows=bool(same_rows),
                            mlp_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_mlp']), np.asarray(ref['mlp_pred']))),
                            mlp_proba_max_abs_diff=float(np.max(np.abs(np.asarray(r['proba_mlp'], float) - np.asarray(ref['mlp_proba'], float)))) if same_rows else None,
                            rf_pred_identical=bool(same_rows and np.array_equal(np.asarray(r['pred_rf']), np.asarray(ref['rf_pred']))))
        gate['run1_c_predictions'] = c_chk
        gate['run1_c_ok'] = all(v['mlp_pred_identical'] and v['rf_pred_identical'] and v['mlp_proba_max_abs_diff'] is not None
                                and v['mlp_proba_max_abs_diff'] < 1e-5 for v in c_chk.values())
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
        checks = [gate['folds_ok'], gate['run1_c_ok'], gate['repeat_fit_ok']]
        if MLP_POOL is not None:
            Xtr0_, Xte0_ = sc_.transform(Xm_[tr_]), sc_.transform(Xm_[te_])
            p_pool_, _ = MLP_POOL.submit(_mlpw.fit_job, Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], Xte0_, NCLS[t0_], ENSEMBLE_SEEDS[0]).result()
            m0_, _ = fit_mlp(Xtr0_, Fd_['y'][tr_], Fd_['groups'][tr_], NCLS[t0_], ENSEMBLE_SEEDS[0])
            gate['pool_vs_inprocess_max_abs_diff'] = float(np.max(np.abs(p_pool_ - mlp_proba(m0_, Xte0_))))
            gate['pool_ok'] = gate['pool_vs_inprocess_max_abs_diff'] < 1e-6
            checks.append(gate['pool_ok'])
        VALIDATION_OK = bool(all(checks))
        gate.update(ok=VALIDATION_OK, decided_on='integrity checks')
        print('integrity gate:', 'PASSED' if VALIDATION_OK else 'FAILED', {k: v for k, v in gate.items() if k not in ('folds', 'run1_c_predictions')})
        print('   arm C against Run 1:', gate['run1_c_predictions'])
    RESULTS['validation_gate'] = gate
    write_results()
    if MLP_POOL is not None:
        MLP_POOL.shutdown()
''')

# ------------------------------------------------------------------ 8. manifest
patch('s11_manifest', "        checkpoint=dict(path=str(GAN_CHECKPOINT), sha256=CKPT_SHA256, all_copies=CKPT_HASHES),",
      "        checkpoint=None, exp0=dict(plan_sha256=EXP0_PLAN_SHA256, corrupt_py_sha256=_MOD_SHA['corrupt.py'], n_conditions=len(COND_TAGS)),")
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
CELLS[iprov] = (CELLS[iprov][0], CELLS[iprov][1], ptxt.replace(old, "NOTEBOOK_SOURCE_SHA256 = '%s'   # filled in by build_exp0.py" % SRC_SHA))
cells = []
for kind, name, text in CELLS:
    assert '\u2014' not in text, 'em dash in cell %s' % name
    cells.append(new_markdown_cell(text) if kind == 'markdown' else new_code_cell(text, metadata={'name': name} if name else {}))
nb = new_notebook(cells=cells)
nb.metadata = dict(nb1['metadata'])
nb.metadata['colab'] = dict(nb1['metadata'].get('colab', {}), name='Experiment0_BeatErrorBudget.ipynb', machine_shape='hm', gpuType='A100')
nb.metadata['built_by'] = 'extension_study/notebooks/build_exp0.py (patches over the locked Extension_Run1.ipynb)'
nb.metadata['notebook_source_sha256'] = SRC_SHA
nb.metadata['embedded'] = dict(exp0_plan_sha256=PLAN_SHA, modules_sha256=MOD_SHA, run1_oof_sha256=OOF1_SHA)
tmp = Path(tempfile.mkdtemp())
for k, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % k)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('Experiment 0 notebook written: %s (%d cells) | source sha256 %s' % (OUT_NB, len(nb.cells), SRC_SHA[:16]))
