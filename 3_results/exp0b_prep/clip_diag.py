"""Pre-run design diagnostic for Experiment 0b (boundaries only; no labels, features or classifiers): how often the
pre-registered 10 ms clipping fires on arm C's cycles of the 610 recordings, and the effective S2 shift."""
import pickle, sys, json
import numpy as np
sys.path.insert(0, 'E:/NUST MS Thesis/PhD Level Research/extension_study')
from extstudy import corrupt as K
PKL = 'C:/Users/ahmad/AppData/Local/Temp/claude/e--NUST-MS-Thesis-Final-Draft/15d5e8b9-9b00-4710-9b69-d0614bc76cda/scratchpad/run1_cycles/runs/run1/20261016_1516/cache/cycles_arms_cdee2.pkl'
OOF1 = 'E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz'
z = np.load(OOF1, allow_pickle=False)
names = sorted(set().union(*[set(z['%s__pcg_seg__names' % t].astype(str)) for t in ('grading', 'timing', 'shape')]))
o = pickle.load(open(PKL, 'rb'))['data']
C = {n: o[n][0] for n in names}
print('recordings', len(names), 'cycles', sum(len(v) for v in C.values()))
sys_ms = np.concatenate([[(c['s2'][0] - c['s1'][1]) / 4 for c in C[n]] for n in names])
dia_ms = np.concatenate([[(c['diastole'][1] - c['s2'][1]) / 4 for c in C[n]] for n in names])
print('arm C systole (S1 end to S2 onset) ms: median %.0f, IQR %.0f-%.0f, share < 110 ms %.3f' % (np.median(sys_ms), *np.percentile(sys_ms, [25, 75]), np.mean(sys_ms < 110)))
print('arm C diastole (S2 end to cycle end) ms: median %.0f, share < 110 ms %.3f' % (np.median(dia_ms), np.mean(dia_ms < 110)))
rows = {}
for typ, code in (('s2c', 4), ('s2s', 5)):
    for li, lv in enumerate((0.1, 0.2, 0.3)):
        for s in range(1, 6):
            seed = 10000 * code + 100 * li + s
            sel = K.select_recordings(names, lv, seed) if typ == 's2c' else set()
            mv = cl = n = 0; eff = {1: [], -1: []}; gt60 = []
            for nm in names:
                out, info = K.exp0b_cycles(C[nm], nm, typ, lv, seed, sel, None, 0)
                mv += info['moved']; cl += info['clipped']; n += info['n']
                if info['moved']:
                    d = [abs(a['s2'][0] - b['s2'][0]) / 4 for a, b in zip(out, C[nm]) if a['s2'] != b['s2']]
                    eff[info['sign']] += d
                off, npair = K.s2_onset_offset_ms(out, C[nm])
                if npair:
                    gt60.append(abs(off) > 60)
            r = rows.setdefault('%s_%.1f' % (typ, lv), dict(share_displaced=[], clipped_frac=[], eff_neg=[], eff_pos=[], gt60=[]))
            r['share_displaced'].append(mv / n); r['clipped_frac'].append(cl / max(1, mv))
            r['eff_neg'] += eff[-1]; r['eff_pos'] += eff[1]; r['gt60'].append(np.mean(gt60))
for k, r in rows.items():
    print('%s: cycles displaced %.3f (seed range %.3f-%.3f) | clipped %.2f | effective shift median earlier %.0f ms, later %.0f ms | S2 >60 ms off in %.3f of recordings' % (
        k, np.mean(r['share_displaced']), min(r['share_displaced']), max(r['share_displaced']), np.mean(r['clipped_frac']),
        np.median(r['eff_neg']), np.median(r['eff_pos']), np.mean(r['gt60'])))
