"""Experiment 0b analysis, as fixed in 'Next Study - Beat-Accurate Synthetic ECG/EXP0B_PREREG.md': composite macro-F1
(MLP; grading, timing, shape) of every altered set with the shared child bootstrap (Experiment 0's conditions del010,
del020 and del030 in the same draws), seed-averaged per level; retention (X - E) / (C - E) against Run 1's arms C and E;
the primary statistic Delta = R(spread) - R(concentrated) for S2 misplacement, pooled over the three levels, and its
decision; the same Delta for missed beats and the total-failure curve (estimation only); the transfer check.
Usage:  python notebooks/exp0b_analysis.py <run1 oof_predictions.npz> <exp0 run folder> <exp0b run folder> <output folder>
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

PLAN_SHA = 'c63fa4394bd8b98097a6c2ed5e16e970864d8410afa58ced2f52c3bf50d286dc'
TASKS3 = ('grading', 'timing', 'shape')
SUFFIXES = ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba')
S2_LEVELS = (0.10, 0.20, 0.30)                   # s2c: share of recordings; s2s: share of every recording's cycles
DELC_LEVELS = (0.20, 0.40, 0.60)                 # half the beats missed in this share of recordings
DEL_LEVELS = (0.10, 0.20, 0.30)                  # Experiment 0: this share of beats missed in every recording
FAIL_LEVELS = (0.10, 0.20, 0.30)
EXP0_REUSED = re.compile(r'^del0(10|20|30)_s\d+$')
ARMS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 's2_analysis',
                        's2_between_recording_heterogeneity.csv')


def _reproduces(z1, zx, t, label):
    c = '%s__pcg_seg__' % t
    for k in ('names', 'y', 'groups'):
        assert np.array_equal(z1[c + k], zx[c + k]), '%s rows differ from Run 1 on %s (%s)' % (label, t, k)
    same = bool(np.array_equal(z1[c + 'mlp__pred'], zx[c + 'mlp__pred']))
    diff = float(np.max(np.abs(z1[c + 'mlp__proba'].astype(float) - zx[c + 'mlp__proba'].astype(float))))
    assert same and diff < 1e-5, '%s arm C does not reproduce Run 1 on %s' % (label, t)
    return dict(predictions_identical=same, mlp_proba_max_abs_diff=diff)


def merge(oof1, oof0, oof0b):
    z1, z0, zb = (np.load(p, allow_pickle=False) for p in (oof1, oof0, oof0b))
    out, check = {}, {'exp0': {}, 'exp0b': {}}
    for t in TASKS3:
        check['exp0'][t] = _reproduces(z1, z0, t, 'Experiment 0')
        check['exp0b'][t] = _reproduces(z1, zb, t, 'Experiment 0b')
        for tag in ('pcg_seg', 'equal_time'):
            for s in SUFFIXES:
                out['%s__%s__%s' % (t, tag, s)] = z1['%s__%s__%s' % (t, tag, s)]
        for z, keep in ((z0, lambda tag: bool(EXP0_REUSED.match(tag))), (zb, lambda tag: tag != 'pcg_seg')):
            for k in z.files:
                parts = k.split('__')
                if len(parts) >= 3 and parts[0] == t and keep(parts[1]) and '__'.join(parts[2:]) in SUFFIXES:
                    assert k not in out, 'tag %s occurs twice' % k
                    out[k] = z[k]
    return out, check


def _ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def analyse(merged, conds):
    """conds: {tag: dict(type, level, ...)} for every altered set in `merged`."""
    import tempfile
    tmp = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(tmp, 'oof_predictions.npz'), **merged)
    data, tasks, tags = stats.load_oof(os.path.join(tmp, 'oof_predictions.npz'))
    assert set(tags) == set(conds) | {'pcg_seg', 'equal_time'}, 'conditions and predictions differ'
    res = stats.analyse(data, tasks, tags)
    C, E = 'pcg_seg', 'equal_time'
    gain = res['comp_point'][C] - res['comp_point'][E]
    gain_t = {t: res['point'][(t, C)] - res['point'][(t, E)] for t in TASKS3}

    def level(typ, lv):
        cs = sorted(c for c, b in conds.items() if b['type'] == typ and abs(b['level'] - lv) < 1e-9)
        assert cs, 'no condition %s %s' % (typ, lv)
        cp = float(np.mean([res['comp_point'][c] for c in cs]))
        cd = np.mean([res['comp_draws'][c] for c in cs], 0)
        L = dict(type=typ, level=lv, conditions=cs, composite=cp, composite_ci95=stats.interval(cd),
                 R=(cp - res['comp_point'][E]) / gain, R_draws=(cd - res['comp_draws'][E]) / (res['comp_draws'][C] - res['comp_draws'][E]),
                 per_task={})
        for t in TASKS3:
            pt = float(np.mean([res['point'][(t, c)] for c in cs]))
            dr = np.mean([res['draws'][(t, c)] for c in cs], 0)
            L['per_task'][t] = dict(macro_f1=pt, R=(pt - res['point'][(t, E)]) / gain_t[t],
                                    R_draws=(dr - res['draws'][(t, E)]) / (res['draws'][(t, C)] - res['draws'][(t, E)]))
        for k in ('share_cycles_displaced', 's2_off_gt60_share', 'f1_median', 'fallback_frac'):
            v = [conds[c].get(k) for c in cs]
            L[k] = float(np.mean(v)) if all(x is not None for x in v) else None
        return L

    def contrast(spread, conc):
        """Delta = R(spread) - R(concentrated), per level and pooled over the levels, composite and per task."""
        per = [dict(spread_level=s['level'], concentrated_level=c['level'], R_spread=s['R'], R_concentrated=c['R'],
                    delta=s['R'] - c['R'], delta_ci95=_ci(s['R_draws'] - c['R_draws']),
                    per_task={t: dict(delta=s['per_task'][t]['R'] - c['per_task'][t]['R'],
                                      delta_ci95=_ci(s['per_task'][t]['R_draws'] - c['per_task'][t]['R_draws'])) for t in TASKS3})
               for s, c in zip(spread, conc)]
        pd_ = np.mean([s['R_draws'] - c['R_draws'] for s, c in zip(spread, conc)], 0)
        pooled = dict(delta=float(np.mean([p['delta'] for p in per])), delta_ci95=_ci(pd_),
                      per_task={t: dict(delta=float(np.mean([p['per_task'][t]['delta'] for p in per])),
                                        delta_ci95=_ci(np.mean([s['per_task'][t]['R_draws'] - c['per_task'][t]['R_draws']
                                                                for s, c in zip(spread, conc)], 0))) for t in TASKS3})
        return dict(per_level=per, pooled=pooled)

    s2c = [level('s2c', lv) for lv in S2_LEVELS]
    s2s = [level('s2s', lv) for lv in S2_LEVELS]
    delc = [level('delc', lv) for lv in DELC_LEVELS]
    dels = [level('del', lv) for lv in DEL_LEVELS]
    fail = [level('fail', lv) for lv in FAIL_LEVELS]
    prim = contrast(s2s, s2c)
    lo, hi = prim['pooled']['delta_ci95']
    prim['decision'] = ('concentration costs more' if lo > 0 else 'concentration costs less' if hi < 0 else 'no difference shown')
    prim['decision_rule'] = 'pooled Delta: 2.5th percentile above 0 -> costs more; 97.5th percentile below 0 -> costs less; otherwise no difference shown'
    out = dict(C=res['comp_point'][C], E=res['comp_point'][E], gain_composite=gain, gain_per_task=gain_t,
               primary_s2=prim, secondary_missed=contrast(dels, delc),
               levels={'%s_%s' % (L['type'], L['level']): L for L in s2c + s2s + delc + dels + fail})
    for L in out['levels'].values():
        L['R_ci95'] = _ci(L.pop('R_draws'))
        for t in TASKS3:
            L['per_task'][t]['R_ci95'] = _ci(L['per_task'][t].pop('R_draws'))
    out['displacement_check'] = [dict(level=lv, concentrated=out['levels']['s2c_%s' % lv]['share_cycles_displaced'],
                                      spread=out['levels']['s2s_%s' % lv]['share_cycles_displaced']) for lv in S2_LEVELS]
    return out


def transfer(out, arms_csv=ARMS_CSV):
    """The descriptive transfer check: share of recordings with |median S2 offset| above 60 ms against retention."""
    rows = [dict(source='Experiment 0b', name=k, s2_off_gt60_share=L['s2_off_gt60_share'], R=L['R'])
            for k, L in out['levels'].items() if L['type'] != 'del']
    if os.path.exists(arms_csv):
        import pandas as pd
        A = pd.read_csv(arms_csv)
        rows += [dict(source='Run 1 and Run 2 arms', name=r.arm, s2_off_gt60_share=float(r.share_recordings_s2_off_gt60), R=float(r.retention))
                 for r in A.itertuples()]
    return rows


def offsets_check(b, off):
    """Recompute every condition's S2 share from the per-recording offsets; it must equal the notebook's."""
    assert off['plan_sha256'] == b['plan_sha256']
    worst = 0.0
    for tag, c in b['conditions'].items():
        o = off['conditions'][tag]
        v = np.array([np.nan if x is None else x for x in o['offset_ms']], float)
        has = np.array(o['pairs']) > 0
        share = float(np.mean(np.abs(v[has]) > 60)) if has.any() else None
        assert (share is None) == (c['s2_off_gt60_share'] is None), tag
        if share is not None:
            worst = max(worst, abs(share - c['s2_off_gt60_share']))
        assert sum(o['selected']) == c['n_selected'], tag
        assert sum(o['displaced']) == c['cycles_displaced'] and sum(o['cycles']) == c['cycles'], tag
    assert worst < 1e-12, 'S2 shares do not recompute (%g)' % worst
    return dict(recomputed=True, max_abs_diff=worst)


def per_recording_share(off, conds):
    """Mean over recordings of the share of its cycles displaced, per S2 level (seed-averaged): the two patterns match
    on this to rounding; on the pooled share of cycles they match in expectation over the recording draw."""
    out = []
    for lv in S2_LEVELS:
        r = {}
        for typ in ('s2c', 's2s'):
            cs = [c for c, b in conds.items() if b['type'] == typ and abs(b['level'] - lv) < 1e-9]
            r[typ] = float(np.mean([np.mean(np.array(off['conditions'][c]['displaced']) / np.maximum(1, off['conditions'][c]['cycles']))
                                    for c in cs]))
        out.append(dict(level=lv, concentrated=r['s2c'], spread=r['s2s']))
    return out


def main(oof1, exp0_dir, exp0b_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    b0 = json.load(open(os.path.join(exp0_dir, 'exp0_boundaries.json')))
    bb = json.load(open(os.path.join(exp0b_dir, 'exp0b_boundaries.json')))
    off = json.load(open(os.path.join(exp0b_dir, 'exp0b_s2_offsets.json')))
    assert bb['plan_sha256'] == PLAN_SHA, 'Experiment 0b ran under a different plan'
    conds = {k: v for k, v in b0['conditions'].items() if EXP0_REUSED.match(k)}
    assert len(conds) == 15 and all(v['type'] == 'del' for v in conds.values())
    conds.update(bb['conditions'])
    merged, check = merge(oof1, os.path.join(exp0_dir, 'oof_predictions.npz'), os.path.join(exp0b_dir, 'oof_predictions.npz'))
    out = analyse(merged, conds)
    out['transfer'] = transfer(out)
    out['offsets_check'] = offsets_check(bb, off)
    out['displacement_check_per_recording'] = per_recording_share(off, bb['conditions'])
    out['arm_c_reproduction'] = check
    out['plan_sha256'] = bb['plan_sha256']
    json.dump(out, open(os.path.join(out_dir, 'exp0b_results.json'), 'w'), indent=1, default=float)
    p = out['primary_s2']
    print('PRIMARY (S2 misplacement): pooled Delta %+.3f [%+.3f, %+.3f] -> %s' % (p['pooled']['delta'], *p['pooled']['delta_ci95'], p['decision']))
    for r in p['per_level']:
        print('   level %.2f: R spread %.2f, R concentrated %.2f, Delta %+.3f [%+.3f, %+.3f]' % (
            r['spread_level'], r['R_spread'], r['R_concentrated'], r['delta'], *r['delta_ci95']))
    m = out['secondary_missed']['pooled']
    print('missed beats (estimation): pooled Delta %+.3f [%+.3f, %+.3f]' % (m['delta'], *m['delta_ci95']))
    for k, L in out['levels'].items():
        print('%-9s composite %5.1f  R %5.2f [%5.2f, %5.2f]  displaced %s  S2>60ms %s' % (
            k, L['composite'], L['R'], L['R_ci95'][0], L['R_ci95'][1],
            '-' if L['share_cycles_displaced'] is None else '%.3f' % L['share_cycles_displaced'],
            '-' if L['s2_off_gt60_share'] is None else '%.3f' % L['s2_off_gt60_share']))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:5])
