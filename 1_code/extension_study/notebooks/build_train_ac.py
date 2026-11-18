#!/usr/bin/env python
"""
build_train_ac.py: assembles Extension_Train_AC.ipynb, the Colab notebook that trains the acoustic-label control arm
(kind A: the T architecture with S1 and S2 heads, no R head) on the paired pack with the acoustic-only labels of
labels_acoustic.py, and freezes it (Experiment 2; plan in 'Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md').
Paired data only: the notebook never reads CirCor.

Same pack (paired_v3 audio and splits), crops, augmentation, schedule, seeds and selection rule as T (build_train.py);
only the labels and the laws differ. The acoustic events file and the laws are read from Drive
(Extension_Study/corpus/labels_acoustic_v1/) and checked against the hashes embedded here; the label gate of the
acoustic labels (refcheck against PhysioNet/CinC 2016 training-a) is embedded and reported, not required (the plan).

Usage:  python build_train_ac.py   (writes Extension_Train_AC.ipynb beside this file)
"""
import base64, hashlib, json, os, py_compile, sys, tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PHD = STUDY.parent
OUT_NB = HERE / 'Extension_Train_AC.ipynb'
sys.path.insert(0, str(STUDY))
from extstudy.config import PACK
AC_VERSION = 'labels_acoustic_v1'
AC_DIR = PHD / 'derived' / AC_VERSION
PLAN = Path('E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md')
MODULES = ['__init__.py', 'config.py', 'frontend.py', 'augment.py', 'losses.py', 'model.py', 'robust_beats.py',
           'decoder.py', 'corpus.py', 'train.py']
SEEDS = {'A': (11, 22, 33)}

sha = lambda b: hashlib.sha256(b).hexdigest()
PLAN_SHA = sha(PLAN.read_bytes())
assert open(PLAN.with_suffix('.sha256')).read().split()[0] == PLAN_SHA, 'EXP2_PREREG.md changed after it was hashed'
mods = {m: (STUDY / 'extstudy' / m).read_bytes() for m in MODULES}
for m, b in mods.items():
    assert 'circor' not in b.decode('utf-8').lower() or m == 'config.py', 'CirCor reference in %s' % m
MOD_B64 = {m: base64.b64encode(b).decode() for m, b in mods.items()}
MOD_SHA = {m: sha(b) for m, b in mods.items()}
laws = json.load(open(AC_DIR / 'laws.json'))
LAWS_SHA = laws['sha256']
ev_man = json.load(open(AC_DIR / ('events_%s_manifest.json' % PACK)))
EVENTS_FILE, EVENTS_SHA = ev_man['file'], ev_man['sha256']
assert sha((AC_DIR / EVENTS_FILE).read_bytes()) == EVENTS_SHA, 'acoustic events file differs from its manifest'
gate_path = AC_DIR / 'refcheck_acoustic.json'
GATE = json.load(open(gate_path)) if gate_path.exists() else dict(status='NOT RUN')
pack_man = json.load(open(PHD / 'derived' / 'corpus' / PACK / 'manifest.json'))
PACK_MAN_SHA = sha((PHD / 'derived' / 'corpus' / PACK / 'manifest.json').read_bytes())

CELLS = []
md = lambda t: CELLS.append(new_markdown_cell(t.strip('\n')))
code = lambda name, t: CELLS.append(new_code_cell(t.strip('\n'), metadata={'name': name}))

md(r'''
# Extension: training and freezing the acoustic-label control arm A (paired data only)

Trains the control network **A** (the T architecture with S1 and S2 heads and no R head) on the paired training pack
with labels derived from the phonocardiogram alone (Springer segmenter, no electrocardiogram), three seeds, with the
crops, augmentation, schedule and selection rule of T, as fixed in `Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md`,
and freezes it (weights hashed, freeze record written). Built by `extension_study/notebooks/build_train_ac.py`; the
study code is embedded and checked by SHA-256. **This notebook never reads CirCor.**

**How to run.** Runtime > Change runtime type > A100 GPU, High-RAM. Leave `SMOKE = False`; Runtime > Run all.
Expected time: about 30 minutes per model, three models.

**Inputs on Drive.** `MyDrive/JBHI-Revision/Extension_Study/corpus/<PACK>/` (the pack of T) and
`MyDrive/JBHI-Revision/Extension_Study/corpus/labels_acoustic_v1/` holding `<EVENTS>` and `laws.json` (upload the two
files from `PhD Level Research/derived/labels_acoustic_v1/`; their hashes are checked).
**Outputs on Drive.** `MyDrive/JBHI-Revision/Extension_Study/checkpoints/<timestamp>_ac/`: one `.pt` per model with its
training history, `freeze.json`, and `paired_test.json` (Gate A, descriptive).
''')

code('flags', r'''
SMOKE = False                    # True: one short epoch, nothing frozen (pipeline test)
KINDS = ('A',)
SEEDS = %(seeds)s                 # three seeds, the seeds of T
EXP2_PLAN_SHA256 = '%(plan_sha)s'   # Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md
AC_VERSION = '%(ac_version)s'
EVENTS_FILE, EVENTS_SHA256 = '%(events_file)s', '%(events_sha)s'
LAWS_SHA256 = '%(laws_sha)s'
LABEL_GATE = %(gate)s            # acoustic labels against the hand-corrected training-a annotations: reported, not required (plan)
DRIVE_STUDY = '/content/drive/MyDrive/JBHI-Revision/Extension_Study'
print('SMOKE', SMOKE, '| kinds', KINDS, '| seeds', SEEDS, '| acoustic label gate', LABEL_GATE.get('status'))
''' % dict(seeds=repr(SEEDS), plan_sha=PLAN_SHA, ac_version=AC_VERSION, events_file=EVENTS_FILE, events_sha=EVENTS_SHA,
           laws_sha=LAWS_SHA, gate=repr(GATE)))

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

md('## Study code (embedded, SHA-256 checked)')
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
assert 'A' in M.HEADS and M.HEADS['A'] == ('s1', 's2') and T.HEADS_OF['A'] == ('s1', 's2')
print('modules written:', sorted(MOD_SHA))
''' % dict(mod_b64=json.dumps(MOD_B64), mod_sha=json.dumps(MOD_SHA)))

md('## Training pack and the acoustic labels: copied to the Colab disk, every hash checked')
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
SRC = DRIVE_STUDY + '/corpus/%(pack)s'
PACK = '/content/pack/%(pack)s'
os.makedirs(PACK, exist_ok=True)
for f in ('manifest.json', 'records.csv', 'bin_medians.npy', 'events.npz', 'pcg_bp_1k_int16.npy'):
    if not (os.path.exists(os.path.join(PACK, f)) and os.path.getsize(os.path.join(PACK, f)) == _retry(os.path.getsize, os.path.join(SRC, f))):
        _retry(shutil.copyfile, os.path.join(SRC, f), os.path.join(PACK, f))
assert hashlib.sha256(open(os.path.join(PACK, 'manifest.json'), 'rb').read()).hexdigest() == '%(pack_man_sha)s', 'the pack differs from the one T trained on'
AC_SRC = DRIVE_STUDY + '/corpus/' + AC_VERSION
AC = '/content/pack/' + AC_VERSION
os.makedirs(AC, exist_ok=True)
for f in (EVENTS_FILE, 'laws.json'):
    _retry(shutil.copyfile, os.path.join(AC_SRC, f), os.path.join(AC, f))
assert hashlib.sha256(open(os.path.join(AC, EVENTS_FILE), 'rb').read()).hexdigest() == EVENTS_SHA256, 'acoustic events file differs from the hash'
LAWS = json.load(open(os.path.join(AC, 'laws.json')))
assert LAWS['sha256'] == LAWS_SHA256, 'acoustic laws differ from the hash'
EV = dict(np.load(os.path.join(AC, EVENTS_FILE)))
DATA = T.PairedData(PACK, LAWS, with_ecg=False).replace_events(EV, LAWS)     # the pack's audio and splits; the acoustic labels
n_valid = int(sum(len(DATA.events[k]['vidx']) for k in DATA.events))
print('pack loaded: %%d records, %%s | acoustic labels: %%d beats, %%d valid; records in the training index: %%d' %% (
    len(DATA.R), {r: len(v) for r, v in DATA.roles.items()}, len(EV['r_s']), n_valid,
    sum(len(g) for d in DATA.index['train'].values() for g in d)))
''' % dict(pack=PACK, pack_man_sha=PACK_MAN_SHA))

md('## Train and freeze')
code('train', r'''
CFG = dict(T.CFG, workers=min(10, os.cpu_count() - 2), eval_workers=min(10, os.cpu_count() - 2))
if SMOKE:
    CFG.update(epochs=1, steps_per_epoch=20, val_max_records_per_dataset=5)
stamp = datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M') + '_ac' + ('_smoke' if SMOKE else '')
OUT = os.path.join(DRIVE_STUDY, 'checkpoints', stamp)
os.makedirs(OUT, exist_ok=True)
FREEZE = dict(created=None, smoke=SMOKE, gpu=GPU, cfg={k: v for k, v in CFG.items()}, modules_sha256=MOD_SHA,
              laws_sha256=LAWS['sha256'], label_check=dict(status='REPORTED, NOT REQUIRED (EXP2_PREREG.md)', acoustic_gate=LABEL_GATE),
              acoustic_events=dict(version=AC_VERSION, file=EVENTS_FILE, sha256=EVENTS_SHA256), exp2_plan_sha256=EXP2_PLAN_SHA256,
              pack_manifest_sha256=hashlib.sha256(open(os.path.join(PACK, 'manifest.json'), 'rb').read()).hexdigest(),
              models={})
VAL = {}
for kind in KINDS:
    VAL[kind] = T.validation_set(DATA, kind, CFG)
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
json.dump(FREEZE, open(os.path.join(OUT, 'freeze.json'), 'w'), indent=1)
print('freeze record written:', os.path.join(OUT, 'freeze.json'))
''' % {})

md(r'''
## Gate A: paired test subjects (descriptive)

Beat-identity F1 of each frozen model on native 30 s crops of the paired test records, per dataset, against a shifted
null (decoded beats moved by half the median RR); kappa = (F1 - null) / (1 - null). Read beside T's values in
`checkpoints/20261018_0908/paired_test.json`.
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
            p = 1 / (1 + np.exp(-o)); dec = D.decode_beats(D.beat_trace_from_heads(p[0], p[1]), 250, kind='event')
            truth = it['s1_times']
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
        ck = torch.load(os.path.join(OUT, name + '.pt'), map_location=DEV)
        model = M.UNet1D(kind).to(DEV); model.load_state_dict(ck['state_dict'])
        assert T.state_sha256({k: v.cpu() for k, v in model.state_dict().items()}) == m['weights_sha256']
        GATE_A[name] = dict(test=kappa_eval(model, test_items, kind), val_kappa=kappa_eval(model, VAL[kind], kind))
        print(name, json.dumps({d: round(v['kappa'], 3) if v['kappa'] is not None else None for d, v in GATE_A[name]['test'].items()}))
json.dump(GATE_A, open(os.path.join(OUT, 'paired_test.json'), 'w'), indent=1)
print('Gate A written:', os.path.join(OUT, 'paired_test.json'))
''')

for c in CELLS:
    if c.cell_type == 'markdown':
        c.source = c.source.replace('<PACK>', PACK).replace('<EVENTS>', EVENTS_FILE)
nb = new_notebook(cells=CELLS)
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'},
               'accelerator': 'GPU', 'colab': {'name': 'Extension_Train_AC.ipynb', 'machine_shape': 'hm', 'gpuType': 'A100'},
               'built_by': 'extension_study/notebooks/build_train_ac.py', 'modules_sha256': MOD_SHA, 'exp2_plan_sha256': PLAN_SHA,
               'acoustic_events_sha256': EVENTS_SHA, 'acoustic_laws_sha256': LAWS_SHA, 'label_gate': GATE}
tmp = Path(tempfile.mkdtemp())
for i, c in enumerate(nb.cells):
    assert '\u2014' not in c.source
    if c.cell_type == 'code':
        p = tmp / ('c%02d.py' % i); p.write_text(c.source, encoding='utf-8'); py_compile.compile(str(p), doraise=True)
nbformat.validate(nb)
nbformat.write(nb, str(OUT_NB))
print('wrote %s (%d cells) | plan %s | events %s | gate %s' % (OUT_NB, len(nb.cells), PLAN_SHA[:16], EVENTS_SHA[:16], GATE.get('status')))
