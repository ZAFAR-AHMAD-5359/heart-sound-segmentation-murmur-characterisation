"""Experiment 0c analysis, as fixed in 'Next Study - Beat-Accurate Synthetic ECG/EXP0C_PREREG.md': composite macro-F1
(MLP; grading, timing, shape) with the shared child bootstrap (Run 1's Springer and Run 2's W in the same draws),
seed-averaged per condition; retention (X - E) / (C - E) against Run 1's arms C and E; the primary Delta =
R(at random) - R(on F_S), averaged over the two failure forms, and its two-sided decision; the secondary contrasts
(real against stylised failure in the same recordings; inside plus outside against the whole arm); the descriptive
gain held by the failure sets and the net loss among failed and other recordings.
Usage:  python notebooks/exp0c_analysis.py <run1 oof_predictions.npz> <run2 oof_predictions.npz> <exp0c run folder> <output folder>
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

PLAN_SHA = 'b27aff856277db54f8ec3423e5180db150114c7f3e81c4f9a792be28cd88bcbd'
TASKS3 = ('grading', 'timing', 'shape')
SUFFIXES = ('names', 'y', 'groups', 'mlp__pred', 'mlp__proba')
FORMS = {'s2': ('s2rn', 's2fs'), 'total': ('etrn', 'etfs')}         # (at random, on F_S)
SEEDED = re.compile(r'^(s2fs|s2rn|etrn)_s\d+$')
CONTROLS = {'pcg_seg': ('run1', 'pcg_seg'), 's_all': ('run1', 'springer_wpk'), 'w_all': ('run2', 'waveform_net')}
REFS = {'S': 'springer_wpk', 'W': 'waveform_net'}


def group(tag):
    return tag.split('_s')[0] if SEEDED.match(tag) else tag


def _same(za, ka, zb, kb, t, label):
    pa, pb = '%s__%s__' % (t, ka), '%s__%s__' % (t, kb)
    for k in ('names', 'y', 'groups'):
        assert np.array_equal(za[pa + k], zb[pb + k]), '%s: rows differ on %s (%s)' % (label, t, k)
    same = bool(np.array_equal(za[pa + 'mlp__pred'], zb[pb + 'mlp__pred']))
    diff = float(np.max(np.abs(za[pa + 'mlp__proba'].astype(float) - zb[pb + 'mlp__proba'].astype(float))))
    rf = bool(np.array_equal(za[pa + 'rf__pred'], zb[pb + 'rf__pred']))
    assert same and rf and diff < 1e-5, '%s does not reproduce its run on %s' % (label, t)
    return dict(predictions_identical=same, rf_identical=rf, mlp_proba_max_abs_diff=diff)


def merge(oof1, oof2, oofc):
    z = dict(run1=np.load(oof1, allow_pickle=False), run2=np.load(oof2, allow_pickle=False))
    zc = np.load(oofc, allow_pickle=False)
    out, check = {}, {k: {} for k in CONTROLS}
    for t in TASKS3:
        for tag, (run, src) in CONTROLS.items():
            check[tag][t] = _same(zc, tag, z[run], src, t, 'Experiment 0c %s' % tag)
        for tag, run in (('pcg_seg', 'run1'), ('equal_time', 'run1'), ('springer_wpk', 'run1'), ('waveform_net', 'run2')):
            for s in SUFFIXES:
                out['%s__%s__%s' % (t, tag, s)] = z[run]['%s__%s__%s' % (t, tag, s)]
        for k in zc.files:
            parts = k.split('__')
            if len(parts) >= 3 and parts[0] == t and parts[1] not in CONTROLS and '__'.join(parts[2:]) in SUFFIXES:
                out[k] = zc[k]
    return out, check


def _ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def analyse(merged, cond_tags):
    import tempfile
    tmp = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(tmp, 'oof_predictions.npz'), **merged)
    data, tasks, tags = stats.load_oof(os.path.join(tmp, 'oof_predictions.npz'))
    assert set(tags) == set(cond_tags) | {'pcg_seg', 'equal_time', 'springer_wpk', 'waveform_net'}, 'conditions and predictions differ'
    res = stats.analyse(data, tasks, tags)
    C, E = 'pcg_seg', 'equal_time'
    gain = res['comp_point'][C] - res['comp_point'][E]
    gain_t = {t: res['point'][(t, C)] - res['point'][(t, E)] for t in TASKS3}

    def level(cs):
        cp = float(np.mean([res['comp_point'][c] for c in cs]))
        cd = np.mean([res['comp_draws'][c] for c in cs], 0)
        L = dict(conditions=sorted(cs), composite=cp, composite_ci95=stats.interval(cd), R=(cp - res['comp_point'][E]) / gain,
                 R_draws=(cd - res['comp_draws'][E]) / (res['comp_draws'][C] - res['comp_draws'][E]), per_task={})
        for t in TASKS3:
            pt = float(np.mean([res['point'][(t, c)] for c in cs]))
            dr = np.mean([res['draws'][(t, c)] for c in cs], 0)
            L['per_task'][t] = dict(macro_f1=pt, R=(pt - res['point'][(t, E)]) / gain_t[t],
                                    R_draws=(dr - res['draws'][(t, E)]) / (res['draws'][(t, C)] - res['draws'][(t, E)]))
        return L

    groups = {}
    for c in cond_tags:
        groups.setdefault(group(c), []).append(c)
    G = {g: level(cs) for g, cs in groups.items()}
    G.update({k: level([v]) for k, v in REFS.items()})

    def diff(a, b):
        """R(a) - R(b), composite and per task."""
        return dict(delta=G[a]['R'] - G[b]['R'], delta_ci95=_ci(G[a]['R_draws'] - G[b]['R_draws']),
                    per_task={t: dict(delta=G[a]['per_task'][t]['R'] - G[b]['per_task'][t]['R'],
                                      delta_ci95=_ci(G[a]['per_task'][t]['R_draws'] - G[b]['per_task'][t]['R_draws'])) for t in TASKS3})

    per_form = {f: diff(*FORMS[f]) for f in FORMS}
    pd_ = np.mean([G[r]['R_draws'] - G[s]['R_draws'] for r, s in FORMS.values()], 0)
    pooled = dict(delta=float(np.mean([per_form[f]['delta'] for f in FORMS])), delta_ci95=_ci(pd_),
                  per_task={t: dict(delta=float(np.mean([per_form[f]['per_task'][t]['delta'] for f in FORMS])),
                                    delta_ci95=_ci(np.mean([G[r]['per_task'][t]['R_draws'] - G[s]['per_task'][t]['R_draws']
                                                            for r, s in FORMS.values()], 0))) for t in TASKS3})
    lo, hi = pooled['delta_ci95']
    decision = ('placement on Springer\'s failures costs more' if lo > 0 else
                'placement on Springer\'s failures costs less' if hi < 0 else 'no difference shown')

    def decomposition(arm, inside, outside):
        li, lo_, la = 1 - G[inside]['R_draws'], 1 - G[outside]['R_draws'], 1 - G[arm]['R_draws']
        return dict(loss_inside=1 - G[inside]['R'], loss_inside_ci95=_ci(li), loss_outside=1 - G[outside]['R'], loss_outside_ci95=_ci(lo_),
                    loss_whole_arm=1 - G[arm]['R'], loss_whole_arm_ci95=_ci(la),
                    whole_minus_sum=(1 - G[arm]['R']) - (1 - G[inside]['R']) - (1 - G[outside]['R']), whole_minus_sum_ci95=_ci(la - li - lo_))

    out = dict(C=res['comp_point'][C], E=res['comp_point'][E], gain_composite=gain, gain_per_task=gain_t,
               primary=dict(per_form=per_form, pooled=pooled, decision=decision,
                            decision_rule='pooled Delta = R(at random) - R(on F_S) over the S2 and total failure forms; two-sided 95%'),
               secondary=dict(springer_real_vs_s2_stylised=diff('s_inside', 's2fs'), springer_real_vs_total_stylised=diff('s_inside', 'etfs'),
                              springer_decomposition=decomposition('S', 's_inside', 's_outside'),
                              w_decomposition=decomposition('W', 'w_inside', 'w_outside')),
               groups=G)
    for L in out['groups'].values():
        L['R_ci95'] = _ci(L.pop('R_draws'))
        for t in TASKS3:
            L['per_task'][t]['R_ci95'] = _ci(L['per_task'][t].pop('R_draws'))
    return out


def net_rates(merged, sets, off):
    """Descriptive. (1) Per task, the net gain of C over E (share wrong under E and right under C, minus the reverse) in
    each failure set and in the other recordings. (2) In each primary condition, the net loss against C among the
    selected recordings and among the rest, seed-averaged."""
    idx = {n: i for i, n in enumerate(off['names'])}
    gain_held, net_loss = {}, {}
    for t in TASKS3:
        names = merged['%s__pcg_seg__names' % t].astype(str)
        y = merged['%s__pcg_seg__y' % t]
        rc = merged['%s__pcg_seg__mlp__pred' % t] == y
        re_ = merged['%s__equal_time__mlp__pred' % t] == y
        for k, F in sets.items():
            m = np.isin(names, list(F))
            gain_held.setdefault(k, {})[t] = dict(
                inside=float(np.mean(rc[m] & ~re_[m]) - np.mean(~rc[m] & re_[m])), n_inside=int(m.sum()),
                outside=float(np.mean(rc[~m] & ~re_[~m]) - np.mean(~rc[~m] & re_[~m])), n_outside=int((~m).sum()))
        for g in ('s2fs', 's2rn', 'etfs', 'etrn'):
            tags = sorted(k.split('__')[1] for k in merged if k.startswith('%s__' % t) and k.endswith('__names') and group(k.split('__')[1]) == g)
            v = {'selected': [], 'rest': []}
            for tag in tags:
                rx = merged['%s__%s__mlp__pred' % (t, tag)] == y
                sel = np.array([off['conditions'][tag]['selected'][idx[n]] for n in names])
                for kk, m in (('selected', sel), ('rest', ~sel)):
                    v[kk].append(float(np.mean(rc[m] & ~rx[m]) - np.mean(~rc[m] & rx[m])))
            net_loss.setdefault(g, {})[t] = {kk: float(np.mean(x)) for kk, x in v.items()}
    return dict(gain_held=gain_held, net_loss=net_loss)


def main(oof1, oof2, run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    b = json.load(open(os.path.join(run_dir, 'exp0c_boundaries.json')))
    off = json.load(open(os.path.join(run_dir, 'exp0c_s2_offsets.json')))
    assert b['plan_sha256'] == PLAN_SHA and off['plan_sha256'] == PLAN_SHA, 'Experiment 0c ran under a different plan'
    conds = [k for k in b['conditions'] if k not in CONTROLS]
    assert len(conds) == 30, 'expected 30 sets, found %d' % len(conds)
    merged, check = merge(oof1, oof2, os.path.join(run_dir, 'oof_predictions.npz'))
    out = analyse(merged, conds)
    out['descriptive'] = net_rates(merged, {'F_S': b['f_s'], 'F_W': b['f_w']}, off)
    out['reproduction'] = check
    out['plan_sha256'] = b['plan_sha256']
    out['n_f_s'], out['n_f_w'] = len(b['f_s']), len(b['f_w'])
    out['boundaries'] = {g: dict(s2_off_gt60_share=float(np.mean([b['conditions'][c]['s2_off_gt60_share'] for c in b['conditions'] if group(c) == g])),
                                 fallback_frac=float(np.mean([b['conditions'][c]['fallback_frac'] for c in b['conditions'] if group(c) == g])))
                         for g in sorted({group(c) for c in b['conditions']})}
    json.dump(out, open(os.path.join(out_dir, 'exp0c_results.json'), 'w'), indent=1, default=float)
    p = out['primary']
    print('PRIMARY: pooled Delta %+.3f [%+.3f, %+.3f] -> %s' % (p['pooled']['delta'], *p['pooled']['delta_ci95'], p['decision']))
    for f, r in p['per_form'].items():
        print('   %-5s: Delta %+.3f [%+.3f, %+.3f]' % (f, r['delta'], *r['delta_ci95']))
    for g, L in sorted(out['groups'].items()):
        print('%-10s composite %5.1f  R %5.2f [%5.2f, %5.2f]' % (g, L['composite'], L['R'], *L['R_ci95']))
    for k in ('springer_decomposition', 'w_decomposition'):
        d = out['secondary'][k]
        print('%s: loss inside %.2f, outside %.2f, whole arm %.2f, whole - sum %+.2f [%+.2f, %+.2f]' % (
            k, d['loss_inside'], d['loss_outside'], d['loss_whole_arm'], d['whole_minus_sum'], *d['whole_minus_sum_ci95']))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:5])
