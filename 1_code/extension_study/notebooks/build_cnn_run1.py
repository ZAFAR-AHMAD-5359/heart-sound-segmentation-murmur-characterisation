#!/usr/bin/env python
"""
build_cnn_run1.py: assembles Extension_Run1_CNN.ipynb, the PCG-only CNN of Extension Run 1.

The R1 CNN builder (Revision-R1/notebooks/build_cnn_baseline.py) writes its notebook when imported, so its source is
executed with the output redirected to a temporary file (the R1 notebook on disk is never touched), checked to match
R1_CNN_Baseline.ipynb cell for cell, and then patched. Every patch asserts that its anchor occurs exactly once.

Run 1 changes (and nothing else):
  1. the fold files come from the newest full Extension Run 1 folder (JBHI-Revision/Extension_Study/runs/run1/), whose
     fold files carry child IDs (Patient ID union Additional ID) in `patient_ids`;
  2. the recording-to-patient map is replaced by recording-to-child, so the fold checks, the inner early-stopping
     split, and the bootstrap all group by child (twins never straddle a split);
  3. one declared seed (42), as in R1.
Usage:  python build_cnn_run1.py
"""
import hashlib, json, os, py_compile, re, sys, tempfile
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
R1_CNN_BUILDER = Path('E:/NUST MS Thesis/Final Draft/Submitted File/Revision-R1/notebooks/build_cnn_baseline.py')
R1_CNN_NB = R1_CNN_BUILDER.parent / 'R1_CNN_Baseline.ipynb'
OUT_NB = HERE / 'Extension_Run1_CNN.ipynb'

src = R1_CNN_BUILDER.read_text(encoding='utf-8')
anchor = "OUT_NB = HERE / 'R1_CNN_Baseline.ipynb'"
assert src.count(anchor) == 1
tmp_nb = Path(tempfile.mkdtemp()) / 'R1_CNN_Baseline_rebuilt.ipynb'
g = {'__file__': str(R1_CNN_BUILDER), '__name__': 'r1_cnn_builder'}
exec(compile(src.replace(anchor, 'OUT_NB = Path(%r)' % str(tmp_nb)), str(R1_CNN_BUILDER), 'exec'), g)
CELLS = g['CELLS']

ref = nbformat.read(str(R1_CNN_NB), as_version=4)
assert [c.source for c in ref.cells] == [c.source for c in CELLS], 'the R1 CNN builder no longer reproduces R1_CNN_Baseline.ipynb'
BASE_HASH = hashlib.sha256('\x00'.join(c.source for c in CELLS).encode('utf-8')).hexdigest()
print('R1 CNN base reproduced cell for cell (sha256 %s)' % BASE_HASH[:16])


def patch(old, new):
    hits = [i for i, c in enumerate(CELLS) if old in c.source]
    assert len(hits) == 1 and CELLS[hits[0]].source.count(old) == 1, 'anchor %r: %d cells' % (old[:70], len(hits))
    CELLS[hits[0]].source = CELLS[hits[0]].source.replace(old, new)


CELLS[0].source = '''# Extension Run 1: the PCG-only CNN under the child-grouped folds

Built by `extension_study/notebooks/build_cnn_run1.py` from the R1 CNN notebook, reproduced cell for cell and then
patched. The model, preprocessing, MFCC input, optimiser, schedule, batch size, epochs, augmentation, class balancing,
and the one declared seed (42) are those of R1. Two things change: the fold files come from the newest full
Extension Run 1 folder (`JBHI-Revision/Extension_Study/runs/run1/`), and every grouping (fold checks, inner
early-stopping split, bootstrap) uses children (Patient ID union Additional ID), so recordings of a child enrolled
twice never sit on both sides of a split. Outputs go to `<Run 1 folder>/cnn_baseline/`.'''

patch("    GATING_ROOT = os.path.join(DRIVE_ROOT, 'JBHI-Revision', 'R1_Gating_Campaign')",
      "    GATING_ROOT = os.path.join(DRIVE_ROOT, 'JBHI-Revision', 'Extension_Study', 'runs', 'run1')   # Extension Run 1 folders")
patch("    REC_PATIENT = dict(zip(df_rec['rec'], df_rec['patient_id'].astype(int)))",
      """    # Extension Run 1: group by child (connected components of Patient ID and Additional ID, labelled by the smallest
    # Patient ID), the grouping of the Run 1 fold files; fold checks, inner split and bootstrap all read this map
    _meta = pd.read_csv(CSV_PATH)
    _parent = {int(p): int(p) for p in _meta['Patient ID']}
    def _find(x):
        while _parent[x] != x:
            _parent[x] = _parent[_parent[x]]
            x = _parent[x]
        return x
    for _p, _a in zip(_meta['Patient ID'], _meta['Additional ID']):
        if pd.notna(_a) and int(_a) in _parent:
            _ra, _rb = _find(int(_p)), _find(int(_a))
            if _ra != _rb:
                _parent[max(_ra, _rb)] = min(_ra, _rb)
    CHILD_OF = {p: _find(p) for p in _parent}
    REC_PATIENT = {r: CHILD_OF[int(p)] for r, p in zip(df_rec['rec'], df_rec['patient_id'].astype(int))}
    print(f'grouping by child: {len(set(df_rec["patient_id"]))} patients -> {len(set(REC_PATIENT.values()))} children')""")

patch("""    if N_PAIRS != COMPLETE_PAIRS:
        print(f'  WARNING: dataset copy has {N_PAIRS} paired files, the complete copy has {COMPLETE_PAIRS}')
""", """    if N_PAIRS != COMPLETE_PAIRS:
        print(f'  WARNING: dataset copy has {N_PAIRS} paired files, the complete copy has {COMPLETE_PAIRS}')

    # Extension Run 1: stage the dataset on the Colab disk. Drive's mount dropped ("Transport endpoint is not connected")
    # under many parallel WAV reads; the copy runs one file at a time with retries and a forced remount, sizes are
    # checked, and every later read uses the local copy
    import time as _time, shutil as _shutil, errno as _errno
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
                _time.sleep(3 * (_k + 1))
    TRAINING_DATA_PATH_DRIVE = TRAINING_DATA_PATH
    if str(TRAINING_DATA_PATH).startswith('/content/drive'):
        _local = '/content/circor_local/training_data'
        os.makedirs(_local, exist_ok=True)
        _src = _drive_retry(lambda: sorted(f for f in os.listdir(TRAINING_DATA_PATH) if f.endswith(('.wav', '.tsv'))))
        _t0, _n_new = _time.time(), 0
        for _f in _src:
            _p, _q = os.path.join(TRAINING_DATA_PATH, _f), os.path.join(_local, _f)
            _size = _drive_retry(os.path.getsize, _p)
            if os.path.exists(_q) and os.path.getsize(_q) == _size:
                continue
            _drive_retry(_shutil.copyfile, _p, _q)
            assert os.path.getsize(_q) == _size, f'size mismatch after copy: {_f}'
            _n_new += 1
        _drive_retry(_shutil.copyfile, CSV_PATH, os.path.join(os.path.dirname(_local), 'training_data.csv'))
        TRAINING_DATA_PATH = _local
        CSV_PATH = os.path.join(os.path.dirname(_local), 'training_data.csv')
        print(f'  dataset staged on the Colab disk: {len(_src)} files ({_n_new} copied now) in {_time.time() - _t0:.0f} s -> {_local}')
""")

nb = nbformat.v4.new_notebook(cells=CELLS)
nb.metadata = dict(ref.metadata)
nb.metadata['colab'] = dict(ref.metadata['colab'], name='Extension_Run1_CNN.ipynb')
nb.metadata['built_by'] = 'extension_study/notebooks/build_cnn_run1.py (patches over build_cnn_baseline.py)'
nb.metadata['r1_cnn_base_sha256'] = BASE_HASH
nbformat.validate(nb)
tmpdir = Path(tempfile.mkdtemp())
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        p = tmpdir / ('cell_%02d.py' % i)
        p.write_text(c.source, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
    assert '\u2014' not in c.source and '\u2013' not in c.source, 'dash in cell %d' % i
nbformat.write(nb, str(OUT_NB))
print('wrote %s (%d cells)' % (OUT_NB, len(nb.cells)))
