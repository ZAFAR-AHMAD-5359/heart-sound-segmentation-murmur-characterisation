"""The CirCor firewall: training and selection code must never reach CirCor data."""
import os, re

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extstudy')
# Modules that may name CirCor: configuration, fold construction, and the statistics on CirCor predictions.
# (The planned infer_circor.py, r1_adapter.py, ledger.py, and metrics.py of CODE_DESIGN.md were implemented as
# infer.py, cycles.py, the run notebooks, and runs/ledger.csv; those modules are scanned and must stay clean.)
ALLOWED = {'config.py', 'splits.py', 'stats.py'}
ALLOWED_DIRS = {'springer'}
PATTERN = re.compile(r'circor_paths|training_data\.csv|CirCor|circor', re.IGNORECASE)


def test_no_circor_reference_in_training_code():
    offenders = []
    for d, _, files in os.walk(PKG):
        if os.path.basename(d) in ALLOWED_DIRS:
            continue
        for f in files:
            if f.endswith('.py') and f not in ALLOWED:
                txt = open(os.path.join(d, f), encoding='utf-8').read()
                code = '\n'.join(l for l in txt.splitlines() if not l.strip().startswith('#'))
                if PATTERN.search(code):
                    offenders.append(os.path.join(d, f))
    assert not offenders, 'CirCor referenced outside the allowed modules: %s' % offenders
