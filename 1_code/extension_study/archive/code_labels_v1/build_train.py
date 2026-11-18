#!/usr/bin/env python
"""
build_train.py: assembles Extension_Train_TW.ipynb, the Colab notebook that trains T and W on the paired pack
(prereg section 7) and freezes them. Paired data only: the notebook never reads CirCor.

The study modules are embedded (base64, SHA-256 checked) and written to /content/extstudy at run time; the label laws
are embedded and checked against the hash frozen in the pre-registration. The 200-beat label check gates real training:
the builder reads derived/labels_v1/check/check_beats.csv and embeds PASSED only when at least 95% of the panels carry
the verdict 'ok'; otherwise the notebook runs smoke mode only.

Usage:  python build_train.py
"""
import base64, hashlib, json, os, py_compile, tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PHD = STUDY.parent
OUT_NB = HERE / 'Extension_Train_TW.ipynb'
MODULES = ['__init__.py', 'config.py', 'frontend.py', 'augment.py', 'losses.py', 'model.py', 'robust_beats.py',
           'decoder.py', 'corpus.py', 'train.py']
LAWS_SHA_FROZEN = '394e830907073f18e615ed76a90c41d5539ff459b1c6bf4e7306969de2a631fc'
SEEDS = {'T': (11, 22, 33), 'W': (11, 22, 33)}

sha = lambda b: hashlib.sha256(b).hexdigest()
mods = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
for m, b in mods.items():
    assert 'circor' not in b.decode('utf-8').lower() or m == 'config.py', 'CirCor reference in %s' % m
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in mods.items()}
MOD_SHA = {m: sha(b) for m, b in mods.items()}
laws = json.load(open(PHD / 'derived' / 'labels_v1' / 'laws.json'))
assert laws['sha256'] == LAWS_SHA_FROZEN, 'label laws differ from the frozen ones'

# label check
chk = PHD / 'derived' / 'labels_v1' / 'check' / 'check_beats.csv'
import pandas as pd
C = pd.read_csv(chk, dtype={'verdict': str}).fillna({'verdict': ''})
v = C['verdict'].str.strip().str.lower()
n_done, n_ok = int((v != '').sum()), int((v == 'ok').sum())
LABEL_CHECK = 'PASSED' if n_done == len(C) and n_ok >= 0.95 * len(C) else ('FAILED' if n_done == len(C) else 'PENDING')
LABEL_CHECK_DETAIL = dict(status=LABEL_CHECK, panels=len(C), filled=n_done, ok=n_ok, file_sha256=sha(chk.read_bytes()))
print('label check:', LABEL_CHECK_DETAIL)

CELLS = []
md = lambda t: CELLS.append(new_markdown_cell(t.strip('\n')))
code = lambda name, t: CELLS.append(new_code_cell(t.strip('\n'), metadata={'name': name}))

md(r'''
# Extension: training and freezing T and W (paired data only)

Trains the ECG-supervised timing network **T** (S1, S2, R heads) and the waveform network **W** (ECG head) on the paired
training pack, three seeds each, as fixed in `prereg/PREREG_EXT.md` section 7, and freezes them (weights hashed, freeze
record written). Built by `extension_study/notebooks/build_train.py`; the study code is embedded and checked by SHA-256.
**This notebook never reads CirCor.**

**How to run.** Runtime > Change runtime type > A100 GPU, High-RAM. Leave `SMOKE = False` for the real training (it
runs only when the 200-beat label check has passed); `SMOKE = True` runs one short epoch per arm to test the pipeline
and saves nothing as frozen. Runtime > Run all. Expected time: about 30 minutes per model, six models.

**Inputs on Drive.** `MyDrive/JBHI-Revision/Extension_Study/corpus/paired_v1/` (the pack, with the ECG targets).
**Outputs on Drive.** `MyDrive/JBHI-Revision/Extension_Study/checkpoints/<timestamp>/`: one `.pt` per model with its
training history, `freeze.json` (weight hashes, configuration, code and data hashes), and `paired_test.json` (Gate A,
descriptive).
''')

code('flags', r'''
SMOKE = False                    # True: one short epoch per arm, nothing frozen (pipeline test)
KINDS = ('T', 'W')
SEEDS = %(seeds)s                 # three seeds per arm, fixed before training
LABEL_CHECK = %(label_check)r     # from derived/labels_v1/check/check_beats.csv at build time
LABEL_CHECK_DETAIL = %(label_check_detail)s
DRIVE_STUDY = '/content/drive/MyDrive/JBHI-Revision/Extension_Study'
if not SMOKE and LABEL_CHECK != 'PASSED':
    raise SystemExit('The 200-beat label check has not passed (%%s); real training is blocked. Set SMOKE = True to test the pipeline.' %% LABEL_CHECK)
print('SMOKE', SMOKE, '| kinds', KINDS, '| seeds', SEEDS, '| label check', LABEL_CHECK)
''' % dict(seeds=repr(SEEDS), label_check=LABEL_CHECK, label_check_detail=repr(LABEL_CHECK_DETAIL)))

md('## Environment: Drive, GPU, determinism')
code('env', r'''
import os, sys, json, time, hashlib, base64, shutil, errno, datetime
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
print('GPU', GPU, '| CPUs', os.cpu_count(), '| torch', torch.__version__)
''')

md('## Study code (embedded, SHA-256 checked) and label laws')
code('code', r'''
MOD_B64 = %(mod_b64)s
MOD_SHA = %(mod_sha)s
os.makedirs('/content/extstudy', exist_ok=True)
for m, b64 in MOD_B64.items():
    src = base64.b64decode(b64)
    assert hashlib.sha256(src).hexdigest() == MOD_SHA[m], 'module %%s corrupted' %% m
    open('/content/extstudy/' + m, 'wb').write(src)
if '/content' not in sys.path:
    sys.path.insert(0, '/content')
from extstudy import train as T, corpus as C, model as M
LAWS = %(laws)s
assert LAWS['sha256'] == '%(laws_sha)s'
print('modules written:', sorted(MOD_SHA), '| label laws', LAWS['sha256'][:16])
''' % dict(mod_b64=json.dumps(MOD_B64), mod_sha=json.dumps(MOD_SHA), laws=json.dumps(laws), laws_sha=LAWS_SHA_FROZEN))

md('## Training pack: copied to the Colab disk, every hash checked')
code('pack', r'''
def _retry(fn, *a, tries=6):
    for k in range(tries):
        try:
            return fn(*a)
        except OSError as e:
            if k == tries - 1:
                raise
            if e.errno in (errno.ENOTCONN, errno.ECONNABORTED, errno.EIO):
                drive.mount('/content/drive', force_remount=True)
            time.sleep(3 * (k + 1))
SRC = DRIVE_STUDY + '/corpus/paired_v1'
PACK = '/content/pack/paired_v1'
os.makedirs(PACK, exist_ok=True)
for f in ('manifest.json', 'records.csv', 'bin_medians.npy', 'events.npz', 'pcg_bp_1k_int16.npy',
          'ecg_manifest.json', 'ecg_records.csv', 'ecg_250_int16.npy'):
    if not (os.path.exists(os.path.join(PACK, f)) and os.path.getsize(os.path.join(PACK, f)) == _retry(os.path.getsize, os.path.join(SRC, f))):
        _retry(shutil.copyfile, os.path.join(SRC, f), os.path.join(PACK, f))
DATA = T.PairedData(PACK, LAWS)                     # checks every pack hash and the ECG manifest
print('pack loaded: %%d records, %%s' %% (len(DATA.R), {r: len(v) for r, v in DATA.roles.items()}))
print('pack manifest sha256', hashlib.sha256(open(os.path.join(PACK, 'manifest.json'), 'rb').read()).hexdigest())
''' % {})

md('## Train and freeze')
code('train', r'''
CFG = dict(T.CFG, workers=min(10, os.cpu_count() - 2), eval_workers=min(10, os.cpu_count() - 2))
if SMOKE:
    CFG.update(epochs=1, steps_per_epoch=20, val_max_records_per_dataset=5)
stamp = datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M') + ('_smoke' if SMOKE else '')
OUT = os.path.join(DRIVE_STUDY, 'checkpoints', stamp)
os.makedirs(OUT, exist_ok=True)
FREEZE = dict(created=None, smoke=SMOKE, gpu=GPU, cfg={k: v for k, v in CFG.items()}, modules_sha256=MOD_SHA,
              laws_sha256=LAWS['sha256'], label_check=LABEL_CHECK_DETAIL,
              pack_manifest_sha256=hashlib.sha256(open(os.path.join(PACK, 'manifest.json'), 'rb').read()).hexdigest(),
              ecg_manifest_sha256=hashlib.sha256(open(os.path.join(PACK, 'ecg_manifest.json'), 'rb').read()).hexdigest(),
              models={})
VAL = {}
for kind in KINDS:
    VAL[kind] = T.validation_set(DATA, kind, CFG)
    if kind == 'W':
        FREEZE['w_zero_output_val_l1'] = T.w_zero_l1(VAL[kind])
    for seed in (SEEDS[kind][:1] if SMOKE else SEEDS[kind]):
        print('==== %%s seed %%d ====' %% (kind, seed)); t0 = time.time()
        model, info = T.fit(DATA, kind, seed, DEV, CFG, val_items=VAL[kind])
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        fp = os.path.join(OUT, '%%s_seed%%d.pt' %% (kind, seed))
        torch.save(dict(state_dict=state, info=info, kind=kind, seed=seed), fp)
        FREEZE['models']['%%s_seed%%d' %% (kind, seed)] = dict(weights_sha256=T.state_sha256(state), file_sha256=hashlib.sha256(open(fp, 'rb').read()).hexdigest(),
                                                              best_val=info['best_val'], best_epoch=int(np.argmax([h['val_score'] for h in info['history']]) + 1),
                                                              minutes=(time.time() - t0) / 60)
        print('saved', fp, FREEZE['models']['%%s_seed%%d' %% (kind, seed)])
FREEZE['created'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5))).isoformat(timespec='seconds')
FREEZE['before_freeze_deadline'] = FREEZE['created'] < '2026-10-30T23:59:59+05:00'
json.dump(FREEZE, open(os.path.join(OUT, 'freeze.json'), 'w'), indent=1)
print('freeze record written:', os.path.join(OUT, 'freeze.json'), '| before the 30 Oct deadline:', FREEZE['before_freeze_deadline'])
''' % {})

md(r'''
## Gate A: paired test subjects (descriptive)

Beat-identity F1 of each frozen model on native 30 s crops of the paired test records (HFrEF community test set
included), per dataset, against a shifted null (decoded beats moved by half the median RR); kappa = (F1 - null) /
(1 - null). For W also its validation L1 against an all-zero output, and its readout kappa per validation dataset
(Panel G: W is reported as failed if any validation dataset has kappa below 0.5).
''')
code('gate_a', r'''
def crops(role, kind, seed=777):
    rng = np.random.default_rng(seed); c2 = dict(CFG, crop_frames=CFG['val_crop_frames']); out = []
    for d, groups in sorted(DATA.index[role].items()):
        for g in groups:
            for k in g:
                if DATA.events[k]['runs']:
                    out.append(T.make_sample(DATA, k, rng, c2, augment=False, warp_hr=None, kind=kind))
    return out

def kappa_eval(model, items, kind):
    from extstudy import decoder as D
    model.eval(); res = {}
    with torch.no_grad():
        for it in items:
            o = model(torch.tensor(it['spec'][None], device=DEV)).float().cpu().numpy()[0]
            if kind == 'T':
                p = 1 / (1 + np.exp(-o)); dec = D.decode_beats(D.beat_trace_from_heads(p[0], p[1]), 250, kind='event')
                truth = it['s1_times']
            else:
                dec = D.decode_beats(o[0], 250, kind='ecg'); truth = it['r_times']
            b = it['out_t0'] + dec['beats_s']
            f1 = T.beat_f1(truth, b, it['mask'], it['out_t0'])
            rr = np.median(np.diff(truth)) if len(truth) > 2 else 0.8
            f0 = T.beat_f1(truth, b + rr / 2, it['mask'], it['out_t0'])
            res.setdefault(it['info']['dataset'], []).append((f1, f0))
    out = {}
    for d, v in res.items():
        v = np.array(v, float); f1, f0 = np.nanmean(v[:, 0]), np.nanmean(v[:, 1])
        out[d] = dict(f1=float(f1), null=float(f0), kappa=float((f1 - f0) / (1 - f0)) if f0 < 1 else None, n=len(v))
    return out

GATE_A = {}
for kind in KINDS:
    test_items = crops('test', kind)
    for name, m in FREEZE['models'].items():
        if not name.startswith(kind):
            continue
        ck = torch.load(os.path.join(OUT, name + '.pt'), map_location=DEV)
        model = M.UNet1D(kind).to(DEV); model.load_state_dict(ck['state_dict'])
        assert T.state_sha256({k: v.cpu() for k, v in model.state_dict().items()}) == m['weights_sha256']
        GATE_A[name] = dict(test=kappa_eval(model, test_items, kind))
        if kind == 'W':
            GATE_A[name]['val_kappa'] = kappa_eval(model, VAL['W'], 'W')
            GATE_A[name]['w_failed_panel_g'] = any((v['kappa'] or 0) < 0.5 for v in GATE_A[name]['val_kappa'].values())
        print(name, json.dumps({d: round(v['kappa'], 3) if v['kappa'] is not None else None for d, v in GATE_A[name]['test'].items()}))
json.dump(GATE_A, open(os.path.join(OUT, 'paired_test.json'), 'w'), indent=1)
print('Gate A written:', os.path.join(OUT, 'paired_test.json'))
''')

# ------------------------------------------------------------------ write
nb = new_notebook(cells=CELLS)
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'},
               'accelerator': 'GPU', 'colab': {'name': 'Extension_Train_TW.ipynb', 'machine_shape': 'hm', 'gpuType': 'A100'},
               'built_by': 'extension_study/notebooks/build_train.py', 'modules_sha256': MOD_SHA, 'label_check': LABEL_CHECK_DETAIL}
tmp = Path(tempfile.mkdtemp())
for i, c in enumerate(nb.cells):
    assert '—' not in c.source
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % i); p.write_text(c.source, encoding='utf-8'); py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('wrote %s (%d cells) | label check %s' % (OUT_NB, len(nb.cells), LABEL_CHECK))
