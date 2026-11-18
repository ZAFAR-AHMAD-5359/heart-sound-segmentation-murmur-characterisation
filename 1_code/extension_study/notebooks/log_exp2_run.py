"""Append the Run 3 receipt, its single scoring and its write-up to the ledger and the change log (append-only).
Usage: python log_exp2_run.py <stamp> [--dry]   (expects runs/run3/<stamp>, runs/exp2_analysis/exp2_results.json and EXP2_RESULTS.md)"""
import csv, hashlib, io, json, os, sys
from datetime import datetime, timezone, timedelta

ES = 'E:/NUST MS Thesis/PhD Level Research/extension_study'
STAMP = sys.argv[1]
DRY = '--dry' in sys.argv
RUN = os.path.join(ES, 'runs/run3', STAMP)
RES = os.path.join(ES, 'runs/exp2_analysis/exp2_results.json')
NOTE = os.path.join(ES, 'runs/exp2_analysis/EXP2_RESULTS.md')
LEDGER, CHANGES = os.path.join(ES, 'runs/ledger.csv'), os.path.join(ES, 'prereg/CHANGES_AFTER_FREEZE.md')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
now = lambda: datetime.now(timezone(timedelta(hours=5))).isoformat(timespec='seconds')
f = lambda d: '%+.2f [%+.2f, %+.2f]' % (d['point'], d['ci95'][0], d['ci95'][1])

R = json.load(open(RES))
r3 = json.load(open(os.path.join(RUN, 'results.json')))
assert r3['run']['run_flags']['exp2_plan_sha256'] == R['plan_sha256'] == 'f0b9fe2b9ddc76a2ffaa0c170cf13d0392e5e68f7a65a4dcdfbf190e5a31a184'
assert all(v['predictions_identical'] and v['rf_identical'] for v in R['reproduction_E'].values())
h, e, c = R['H1'], R['estimation'], R['composite']
gate_ok = (r3.get('validation_gate') or {}).get('ok')
nb = [x for x in os.listdir(RUN) if x.endswith('.ipynb')]
rows = [
    [now(), 'run3 %s' % STAMP, 'results received; gate %s (arm E reproduces Run 1; frozen A weights match the freeze record); before analysis' % ('PASSED' if gate_ok else 'see results.json'),
     'extension_study/runs/run3/%s/oof_predictions.npz' % STAMP, sha(os.path.join(RUN, 'oof_predictions.npz')),
     '%s; executed notebook %s (SHA-256 %s); fallback share %s' % (r3['run'].get('gpu', 'GPU n/a'), nb[0] if nb else 'n/a', sha(os.path.join(RUN, nb[0])) if nb else 'n/a',
                                                                    json.dumps(r3['run'].get('fallback_share')))],
    [now(), 'run3 %s' % STAMP, 'scored once; pre-registered H1 (T minus T_ac, two-sided 95%%): %s' % h['reading'],
     'extension_study/runs/exp2_analysis/exp2_results.json', sha(RES), 'T - T_ac %s; T_ac - E %s; T_ac - S %s; recovery T_ac %.2f' % (f(h), f(e['T_ac_minus_E']), f(e['T_ac_minus_S']), e['recovery_T_ac']['point'])],
    [now(), 'run3 %s' % STAMP, 'write-up', 'extension_study/runs/exp2_analysis/EXP2_RESULTS.md', sha(NOTE), 'copy in Next Study - Beat-Accurate Synthetic ECG/EXP2_RESULTS.md'],
]
change = ('Experiment 2 scored once (Run 3 `runs/run3/%s`, %s; gate %s: arm E reproduces Run 1 with identical MLP and random forest predictions on every task; '
          'the three frozen A weights match the freeze record; fallback share %s) with the pre-registered `exp2_analysis.py` (results SHA-256 %s) on the merged '
          'predictions of Runs 1, 2 and 3. Primary: T minus T_ac %s, reading fixed in the plan: %s. Estimation: T_ac composite %.1f [%.1f, %.1f] (T %.1f, S %.1f, '
          'S_pub %.1f, E %.1f, C %.1f); T_ac minus E %s; T_ac minus S %s; T_ac minus S_pub %s; recovery at matched coverage T_ac %.2f (T %.2f); random forest '
          'composite T_ac %.1f (T %.1f); detection macro-F1 T_ac %s. Write-up `runs/exp2_analysis/EXP2_RESULTS.md` (SHA-256 %s). The manuscript is reworded '
          'according to the reading, as the plan states' % (
              STAMP, r3['run'].get('gpu', 'GPU n/a'), 'PASSED' if gate_ok else 'see results.json', json.dumps(r3['run'].get('fallback_share')), sha(RES), f(h), h['reading'],
              c['timing_net_ac']['point'], c['timing_net_ac']['ci95'][0], c['timing_net_ac']['ci95'][1], c['timing_net']['point'], c['springer_wpk']['point'],
              c['springer_pub']['point'], c['equal_time']['point'], c['pcg_seg']['point'], f(e['T_ac_minus_E']), f(e['T_ac_minus_S']), f(e['T_ac_minus_S_pub']),
              e['recovery_T_ac']['point'], e['recovery_T']['point'], e['rf_composite']['timing_net_ac'], e['rf_composite']['timing_net'],
              ('%.1f' % R['detection']['timing_net_ac']['point']) if 'detection' in R else 'n/a', sha(NOTE)))
row = '| %s | %s | the plan fixed one scoring, the decision rule and every estimation contrast before any acoustic label or network existed; nothing changed after the result | the Experiment 2 results, as listed |' % (
    datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%d %H:%M'), change)
if DRY:
    for r in rows:
        print(r)
    print(row[:1500])
    sys.exit(0)
raw = open(LEDGER, 'rb').read()
nl = '\r\n' if b'\r\n' in raw else '\n'
assert raw.endswith(b'\n')
buf = io.StringIO()
w = csv.writer(buf, lineterminator=nl)
for r in rows:
    w.writerow(r)
open(LEDGER, 'ab').write(buf.getvalue().encode('utf-8'))
raw = open(CHANGES, 'rb').read()
nl2 = '\r\n' if b'\r\n' in raw else '\n'
assert raw.endswith(b'\n')
open(CHANGES, 'ab').write((row + nl2).encode('utf-8'))
print('ledger +3 rows; CHANGES +1 row')
