"""Experiment 0 analysis, as fixed in 'Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md': composite macro-F1
(MLP; grading, timing, shape) of every corrupted set with the shared child bootstrap, seed-averaged per level;
retention (X - E) / (C - E) against Run 1's arms C and E; the read-off of the thresholds and the Gate 1 targets.
Usage:  python notebooks/exp0_analysis.py <run1 oof_predictions.npz> <exp0 run folder> <output folder>
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

TASKS3 = ('grading', 'timing', 'shape')
SUFFIXES = ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba')


def merge(oof1, oof0):
    z1, z0 = np.load(oof1, allow_pickle=False), np.load(oof0, allow_pickle=False)
    out, check = {}, {}
    for t in TASKS3:
        c1, c0 = '%s__pcg_seg__' % t, '%s__pcg_seg__' % t
        for k in ('names', 'y', 'groups'):
            assert np.array_equal(z1[c1 + k], z0[c0 + k]), 'Experiment 0 rows differ from Run 1 on %s (%s)' % (t, k)
        same = bool(np.array_equal(z1[c1 + 'mlp__pred'], z0[c0 + 'mlp__pred']))
        diff = float(np.max(np.abs(z1[c1 + 'mlp__proba'].astype(float) - z0[c0 + 'mlp__proba'].astype(float))))
        check[t] = dict(predictions_identical=same, mlp_proba_max_abs_diff=diff)
        assert same and diff < 1e-5, 'Experiment 0 arm C does not reproduce Run 1 on %s' % t
        for tag in ('pcg_seg', 'equal_time'):
            for s in SUFFIXES:
                out['%s__%s__%s' % (t, tag, s)] = z1['%s__%s__%s' % (t, tag, s)]
        for k in z0.files:
            parts = k.split('__')
            if len(parts) >= 3 and parts[0] == t and parts[1] not in ('pcg_seg', 'equal_time') and '__'.join(parts[2:]) in SUFFIXES:
                out[k] = z0[k]
    return out, check


def analyse(merged, boundaries, levels):
    import tempfile
    tmp = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(tmp, 'oof_predictions.npz'), **merged)
    data, tasks, tags = stats.load_oof(os.path.join(tmp, 'oof_predictions.npz'))
    res = stats.analyse(data, tasks, tags)
    C, E = 'pcg_seg', 'equal_time'
    gain_point = {t: res['point'][(t, C)] - res['point'][(t, E)] for t in TASKS3}
    gain_comp = res['comp_point'][C] - res['comp_point'][E]
    out = dict(C=res['comp_point'][C], E=res['comp_point'][E], gain_composite=gain_comp, gain_per_task=gain_point, levels={})
    for typ, lvls in levels.items():
        for lv in lvls:
            conds = [c for c, b in boundaries.items() if b['type'] == typ and abs(b['level'] - lv) < 1e-9]
            comp_d = np.mean([res['comp_draws'][c] for c in conds], 0)
            comp_p = float(np.mean([res['comp_point'][c] for c in conds]))
            ret_d = (comp_d - res['comp_draws'][E]) / (res['comp_draws'][C] - res['comp_draws'][E])
            per_task = {}
            for t in TASKS3:
                pt = float(np.mean([res['point'][(t, c)] for c in conds]))
                dr = np.mean([res['draws'][(t, c)] for c in conds], 0)
                rt = (dr - res['draws'][(t, E)]) / (res['draws'][(t, C)] - res['draws'][(t, E)])
                per_task[t] = dict(macro_f1=pt, retention=(pt - res['point'][(t, E)]) / gain_point[t] if gain_point[t] else np.nan,
                                   retention_ci95=[float(np.nanpercentile(rt, 2.5)), float(np.nanpercentile(rt, 97.5))])
            out['levels']['%s_%s' % (typ, lv)] = dict(
                type=typ, level=lv, n_seeds=len(conds), composite=comp_p, composite_ci95=stats.interval(comp_d),
                retention=(comp_p - res['comp_point'][E]) / gain_comp if gain_comp else np.nan,
                retention_ci95=[float(np.nanpercentile(ret_d, 2.5)), float(np.nanpercentile(ret_d, 97.5))], per_task=per_task,
                boundary_f1_median=float(np.mean([boundaries[c]['f1_median'] for c in conds])),
                offset_sd_ms_median=float(np.mean([boundaries[c]['offset_sd_ms_median'] for c in conds])),
                fallback_frac=float(np.mean([boundaries[c]['fallback_frac'] for c in conds])))
    return out


def read_off(out, levels, thr=0.80):
    """The pre-registered read-off: per type the largest level, scanning upwards, before the first below thr."""
    ro = {}
    for typ, lvls in levels.items():
        last = None
        for lv in sorted(lvls):
            L = out['levels']['%s_%s' % (typ, lv)]
            ok = (all(L['per_task'][t]['retention'] >= thr for t in TASKS3) if typ == 'jit' else L['retention'] >= thr)
            if not ok:
                break
            last = lv
        ro[typ] = last
    f1 = [out['levels']['%s_%s' % (t, ro[t])]['boundary_f1_median'] for t in ('del', 'ins') if ro[t] is not None]
    return dict(threshold_missed=ro['del'], threshold_spurious=ro['ins'], threshold_jitter_sd_ms=ro['jit'],
                gate1_boundary_f1_target=max(f1) if len(f1) == 2 else None,
                note='missed or spurious threshold None: even the smallest level loses more than 20% of the gain; '
                     'the Gate 1 target is then the boundary F1 of arm C itself (1.0)')


def main(oof1, run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    b = json.load(open(os.path.join(run_dir, 'exp0_boundaries.json')))
    levels = {}
    for c in b['conditions'].values():
        levels.setdefault(c['type'], set()).add(c['level'])
    levels = {k: sorted(v) for k, v in levels.items()}
    merged, check = merge(oof1, os.path.join(run_dir, 'oof_predictions.npz'))
    out = analyse(merged, b['conditions'], levels)
    out['read_off'] = read_off(out, levels)
    out['arm_c_reproduction'] = check
    out['plan_sha256'] = b['plan_sha256']
    json.dump(out, open(os.path.join(out_dir, 'exp0_results.json'), 'w'), indent=1, default=float)
    print(json.dumps(out['read_off'], indent=1))
    for k, L in out['levels'].items():
        print('%-10s composite %5.1f  retention %5.2f [%5.2f, %5.2f]  boundary F1 %.3f  offset SD %.1f ms' % (
            k, L['composite'], L['retention'], L['retention_ci95'][0], L['retention_ci95'][1], L['boundary_f1_median'], L['offset_sd_ms_median']))
    return out


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
