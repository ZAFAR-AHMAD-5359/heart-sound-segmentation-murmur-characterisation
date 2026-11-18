"""Render EXP2_RESULTS.md from exp2_results.json (numbers from the JSON, prose fixed here).
Usage: python write_exp2_results.py <exp2_results.json> <run3 run folder> <output .md> [--copy-to-next-study]"""
import json, os, shutil, hashlib, sys
ES = 'E:/NUST MS Thesis/PhD Level Research/extension_study'
NS = 'E:/NUST MS Thesis/Next Study - Beat-Accurate Synthetic ECG'
FR = os.path.join(ES, 'runs/run3_freeze/20261029_1824_ac')
TFR = os.path.join(ES, 'runs/run2_freeze/20261018_0908')
RES, RUN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(RES))
r3 = json.load(open(os.path.join(RUN, 'results.json')))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rel(p):
    try:
        return os.path.relpath(p, ES).replace(os.sep, '/')
    except ValueError:
        return p.replace(os.sep, '/')


est = lambda d, f='%+.1f': (f % d['point']) + ' [' + (f % d['ci95'][0]) + ', ' + (f % d['ci95'][1]) + ']'
pt3 = lambda d, f='%+.1f': ' / '.join(est(d['per_task'][t], f) for t in ('grading', 'timing', 'shape'))
c, e, h = R['composite'], R['estimation'], R['H1']
run_dir = r3['run'].get('run_dir', RUN)
secs = r3.get('stage_seconds') or {}
total_h = sum(secs.values()) / 3600 if secs else float('nan')
gate = r3.get('validation_gate') or {}
fb = r3['run'].get('fallback_share') or {}
dsum = r3.get('diagnostics_summary') or {}

L = ['# Experiment 2 results: the acoustic-label control arm T_ac (Run 3)', '']
L.append('Run 3 `%s` (%s; %.1f h of stage time). Integrity gate %s: arm E reproduces Run 1 (%s), the frozen A weights match the freeze record. '
         'Analysis by `notebooks/exp2_analysis.py`, run once, under the plan `EXP2_PREREG.md` (SHA-256 %s...), on the merged predictions of Runs 1, 2 '
         'and 3 with the shared child bootstrap (2,000 draws, seed 42). Fallback share: %s.' % (
             os.path.basename(run_dir.rstrip('/')), r3['run'].get('gpu', r3.get('gpu', 'GPU n/a')), total_h,
             'PASSED' if gate.get('ok') else 'see results.json',
             '; '.join('%s: MLP %s, RF %s, max |dp| %.1e' % (t, 'identical' if v['predictions_identical'] else 'DIFFERS', 'identical' if v['rf_identical'] else 'DIFFERS',
                                                            v['mlp_proba_max_abs_diff']) for t, v in R['reproduction_E'].items()),
             R['plan_sha256'][:8], json.dumps(fb)))
L.append('')
L.append('## Primary result (pre-registered): H1, T minus T_ac, two-sided 95%')
L.append('')
L.append('**T minus T_ac on the composite: %s. Reading fixed in the plan: %s.** Per task (grading / timing / shape): %s.' % (
    est(h, '%+.2f'), h['reading'], pt3(h)))
L.append('')
L.append('## Composite macro-F1 (MLP)')
L.append('')
L.append('| Arm | Composite [95% CI] |')
L.append('|---|---|')
for a, lab in (('equal_time', 'E'), ('springer_wpk', 'S'), ('springer_pub', 'S_pub'), ('timing_net', 'T'), ('timing_net_ac', 'T_ac'),
               ('waveform_net', 'W'), ('pcg_seg', 'C')):
    if a in c:
        L.append('| %s | %.1f [%.1f, %.1f] |' % (lab, c[a]['point'], *c[a]['ci95']))
L.append('')
L.append('## Estimation (no p values)')
L.append('')
L.append('| Contrast | Composite [95% CI] | Grading / timing / shape |')
L.append('|---|---|---|')
for k, lab in (('T_ac_minus_E', 'T_ac - E'), ('T_ac_minus_S', 'T_ac - S'), ('T_ac_minus_S_pub', 'T_ac - S_pub'), ('T_minus_E', 'T - E (Run 2)'), ('T_minus_S', 'T - S (Run 2)')):
    L.append('| %s | %s | %s |' % (lab, est(e[k]), pt3(e[k])))
rt, ra = e['recovery_T'], e['recovery_T_ac']
L.append('')
L.append('Recovery at matched coverage, (X_annot - E_annot)/(C - E_annot): T_ac %.2f %s (draws with a non-positive denominator: %d); T %.2f %s.' % (
    ra['point'], '[%.2f, %.2f]' % tuple(ra['ci95']) if ra['ci95'] else 'n.e.', ra['n_draws_denominator_not_positive'],
    rt['point'], '[%.2f, %.2f]' % tuple(rt['ci95']) if rt['ci95'] else 'n.e.'))
L.append('')
L.append('Per-task macro-F1 (MLP): ' + '; '.join('%s grading %.1f, timing %.1f, shape %.1f' % (a, R['per_task'][a]['grading'], R['per_task'][a]['timing'], R['per_task'][a]['shape'])
                                                 for a in ('equal_time', 'springer_wpk', 'timing_net', 'timing_net_ac', 'pcg_seg') if a in R['per_task']) + '.')
L.append('')
L.append('Random forest composite: ' + ', '.join('%s %.1f' % (a, v) for a, v in e['rf_composite'].items()) + '.')
if 'detection' in R:
    L.append('')
    L.append('Detection macro-F1 (MLP): ' + ', '.join('%s %.1f [%.1f, %.1f]' % (a, v['point'], *v['ci95']) for a, v in R['detection'].items()) + '.')
L.append('')
L.append('## Descriptive: labels, training beats, Gate A, boundary diagnostics')
L.append('')
d = R.get('descriptive', {})
if 'label_gate' in d:
    g = d['label_gate']
    L.append('Label gate (reported, not required): acoustic labels S1 %.2f%% [%.2f, %.2f], S2 %.2f%% [%.2f, %.2f] of marks inside the hand-corrected training-a sound widened by 20 ms '
             '(%d beats, %d records); ECG-derived labels v3: S1 %.2f%%, S2 %.2f%% (%d beats).' % (
                 100 * g['acoustic']['s1_ok'][0], 100 * g['acoustic']['s1_ok'][1], 100 * g['acoustic']['s1_ok'][2],
                 100 * g['acoustic']['s2_ok'][0], 100 * g['acoustic']['s2_ok'][1], 100 * g['acoustic']['s2_ok'][2],
                 g['acoustic']['beats'], g['acoustic']['records'], 100 * g['ecg_v3_reference']['s1_ok'][0], 100 * g['ecg_v3_reference']['s2_ok'][0],
                 g['ecg_v3_reference']['beats']))
if 'events' in d:
    L.append('Acoustic training beats: %s valid of %s candidate beats in %s records (T: 69,176 valid beats); lag laws fitted on %s beats of %s subjects.' % (
        format(d['events']['n_valid'], ','), format(d['events']['n_beats'], ','), d['events']['n_records_with_labels'],
        format(d['laws']['n_beats'], ',') if 'laws' in d else 'n/a', d['laws']['n_subjects'] if 'laws' in d else 'n/a'))
try:
    pa, pt = json.load(open(os.path.join(FR, 'paired_test.json'))), json.load(open(os.path.join(TFR, 'paired_test.json')))
    DS = ('hfref', 'senssmart', 'bsslab', 'ephnogram', 'foster')
    L.append('')
    L.append('Gate A, kappa on the held-out paired test records (descriptive):')
    L.append('')
    L.append('| Model | ' + ' | '.join(DS) + ' |')
    L.append('|---|' + '---|' * len(DS))
    for m in ('A_seed11', 'A_seed22', 'A_seed33'):
        L.append('| %s | ' % m + ' | '.join('%.3f' % pa[m]['test'][x]['kappa'] for x in DS) + ' |')
    for m in ('T_seed11', 'T_seed22', 'T_seed33'):
        L.append('| %s | ' % m + ' | '.join('%.3f' % pt[m]['test'][x]['kappa'] for x in DS) + ' |')
except Exception as ex:
    L.append('(Gate A table not available: %s)' % ex)
if dsum:
    L.append('')
    L.append('Boundary diagnostics against the annotation, medians over kept recordings (from results.json of Run 3):')
    L.append('')
    keys = sorted({k for v in dsum.values() for k in v.keys()})
    L.append('| Arm | ' + ' | '.join(keys) + ' |')
    L.append('|---|' + '---|' * len(keys))
    for a, v in dsum.items():
        L.append('| %s | ' % a + ' | '.join(('%.3f' % v[k]) if isinstance(v.get(k), float) else str(v.get(k, '')) for k in keys) + ' |')
L.append('')
L.append('## Reading, against the outcomes fixed in the plan')
L.append('')
lo, hi = h['ci95']
if lo > 0:
    L.append('- T above T_ac: the ECG teacher adds to what an acoustic teacher gives the same network; "electrocardiogram-supervised" names a mechanism. '
             'The difference of %s composite points bounds how much.' % est(h))
elif hi < 0:
    L.append('- T_ac above T: the acoustic teacher does at least as well; the ECG is a convenience in training, not a mechanism. The title and abstract '
             'are reworded as the plan states.')
else:
    L.append('- No difference shown at this precision (interval %s covers 0; no equivalence margin was set): the gain of T over Springer is not '
             'attributed to the ECG teacher; "electrocardiogram-supervised" describes how T was trained, not why it works, and the title and abstract '
             'are reworded as the plan states. A difference smaller than about 3 points was not expected to be shown.' % est(h))
L.append('- T_ac against Springer at its primary setting: %s (plan: T_ac below Springer would mean the acoustic labels are too poor to train the network; '
         'the gate shares say whether label quality or the absence of beat identity explains it).' % est(e['T_ac_minus_S']))
L.append('- Every arm and contrast is reported; nothing changed after the result was seen.')
L.append('')
L.append('## Boundaries of the claim')
L.append('')
L.append('CirCor audio, annotations and labels; nothing trained or selected on CirCor; no CirCor label or annotation entered the acoustic labels or the network. '
         'The acoustic teacher is the Springer segmenter, whose own model was trained on ECG-derived labels in its development, so the control removes the ECG '
         'from this study\'s training, not from the lineage of the acoustic method. Record selection and the usable spans are shared with T.')
L.append('')
L.append('## Files')
L.append('')
for f in sorted(os.listdir(RUN)):
    p = os.path.join(RUN, f)
    if os.path.isfile(p):
        L.append('- `%s/%s`: SHA-256 %s' % (rel(RUN), f, sha(p)))
L.append('- `%s`: SHA-256 %s' % (rel(RES), sha(RES)))
L.append('- Plan: `Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md`, SHA-256 %s.' % R['plan_sha256'])
L.append('')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
if '--copy-to-next-study' in sys.argv:
    shutil.copyfile(OUT, os.path.join(NS, 'EXP2_RESULTS.md'))
print('wrote', OUT, sha(OUT))
