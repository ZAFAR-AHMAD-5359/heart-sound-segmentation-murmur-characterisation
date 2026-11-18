"""Built Run 1 notebooks: expected cells, embedded hashes, safeguards. Run:  python -m pytest tests/test_run1_notebooks.py -q"""
import json, os
import nbformat
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, 'notebooks', 'Extension_Run1.ipynb')
NB_CNN = os.path.join(HERE, 'notebooks', 'Extension_Run1_CNN.ipynb')
pytestmark = pytest.mark.skipif(not os.path.exists(NB), reason='notebooks not built')


def _cells(path):
    nb = nbformat.read(path, as_version=4)
    return nb, {c.metadata.get('name'): c.source for c in nb.cells if c.cell_type == 'code'}


def test_run1_notebook():
    nb, c = _cells(NB)
    man = json.load(open(os.path.join(HERE, 'runs', 'folds_child', 'folds_manifest.json')))
    assert nb.metadata['embedded']['folds_sha256'] == {t: v['sha256'] for t, v in man.items()}
    assert nb.metadata['embedded']['pinned_keep_n'] == 3000
    for name in ('ext_run1_defs', 's4_run', 's6_folds', 's7_run', 's8_gate', 's9_s10_not_run', 's11_manifest'):
        assert name in c, name
    assert not any(k.startswith('s9_window') or k.startswith('s10_') for k in c if k)
    assert "raise SystemExit('Smoke test complete" in c['s7_run']                     # one scoring per arm
    assert 'ENSEMBLE_SEEDS = (42, 1042, 2042, 3042, 4042)' in c['flags']
    assert "Extension_Study/runs/run1" in c['resolve']
    assert 'CHILD_OF[PID_OF[n]]' in c['s5_features']
    assert 'R1_ARMS if len(CYC[arm][n]) < N_MIN_CYCLES' in c['s4_run']
    assert "mp_context=mp.get_context('spawn')" in c['s7_classifier_defs']               # parallel GPU workers
    assert "pool_vs_inprocess_max_abs_diff" in c['s8_gate'] and 'MLP_POOL.shutdown()' in c['s8_gate']


def test_cnn_notebook():
    nb = nbformat.read(NB_CNN, as_version=4)
    src = '\n'.join(c.source for c in nb.cells)
    assert "'Extension_Study', 'runs', 'run1'" in src
    assert 'REC_PATIENT = {r: CHILD_OF[int(p)]' in src
