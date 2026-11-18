"""Exploratory follow-up to Experiment 0b (not pre-registered; written 21 Oct 2026 after the primary result was seen):
in the concentrated conditions, is the lost gain confined to the recordings chosen for failure, or does it reach the
exact recordings too (through training on altered features)? Per task and condition, relative to arm C (Run 1), among
the recordings chosen for failure and among the rest: net loss rate = share right under C and wrong under X minus share
wrong under C and right under X; the same for arm E (equal-time) as the reference for complete loss of gating.
Seed-averaged per level. Usage: python notebooks/exp0b_where_lost.py <run1 oof> <exp0b run folder> <output folder>
"""
import json, os, sys
import numpy as np

TASKS3 = ('grading', 'timing', 'shape')


def main(oof1, run_dir, out_dir):
    z1 = np.load(oof1, allow_pickle=False)
    zb = np.load(os.path.join(run_dir, 'oof_predictions.npz'), allow_pickle=False)
    off = json.load(open(os.path.join(run_dir, 'exp0b_s2_offsets.json')))
    b = json.load(open(os.path.join(run_dir, 'exp0b_boundaries.json')))['conditions']
    idx = {n: i for i, n in enumerate(off['names'])}
    rows = []
    for typ in ('s2c', 'delc', 'fail'):
        for lv in sorted({v['level'] for v in b.values() if v['type'] == typ}):
            tags = sorted(t for t, v in b.items() if v['type'] == typ and abs(v['level'] - lv) < 1e-9)
            for t in TASKS3:
                names = z1['%s__pcg_seg__names' % t].astype(str)
                y = z1['%s__pcg_seg__y' % t]
                rc = z1['%s__pcg_seg__mlp__pred' % t] == y
                re_ = z1['%s__equal_time__mlp__pred' % t] == y
                acc = {'chosen': [], 'rest': []}
                for tag in tags:
                    assert np.array_equal(zb['%s__%s__names' % (t, tag)].astype(str), names)
                    rx = zb['%s__%s__mlp__pred' % (t, tag)] == y
                    sel = np.array([off['conditions'][tag]['selected'][idx[n]] for n in names])
                    for k, m in (('chosen', sel), ('rest', ~sel)):
                        acc[k].append((float(np.mean(rc[m] & ~rx[m]) - np.mean(~rc[m] & rx[m])),
                                       float(np.mean(rc[m] & ~re_[m]) - np.mean(~rc[m] & re_[m])), int(m.sum())))
                for k in acc:
                    a = np.array(acc[k])
                    rows.append(dict(type=typ, level=lv, task=t, group=k, n_mean=float(a[:, 2].mean()),
                                     net_loss_x=float(a[:, 0].mean()), net_loss_x_seed_range=[float(a[:, 0].min()), float(a[:, 0].max())],
                                     net_loss_e_same_recordings=float(a[:, 1].mean())))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(out_dir, 'exp0b_where_lost.json'), 'w'), indent=1)
    for r in rows:
        print('%-4s %.1f %-7s %-6s n %5.0f | net loss X %+.3f (seeds %+.3f to %+.3f) | equal-time on the same recordings %+.3f' % (
            r['type'], r['level'], r['task'], r['group'], r['n_mean'], r['net_loss_x'], *r['net_loss_x_seed_range'], r['net_loss_e_same_recordings']))
    return rows


if __name__ == '__main__':
    main(*sys.argv[1:4])
