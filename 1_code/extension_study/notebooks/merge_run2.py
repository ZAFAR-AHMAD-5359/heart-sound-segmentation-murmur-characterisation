"""Merge Run 2's out-of-fold predictions (T, W, T_annot, W_annot) with Run 1's, then run the frozen stats.py with
t_frozen=True (H1 to H4, prereg section 3). Run 1's arms, arm E included, come from Run 1's file; Run 2's arm E must
equal Run 1's (it is the reproduction control) and is not merged.
Usage:  python notebooks/merge_run2.py <run1 oof_predictions.npz> <run2 oof_predictions.npz> <output folder>
"""
import hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import stats

STATS_SHA = 'b44d78a882f7574c'
TW_TAGS = ('timing_net', 'waveform_net', 'timing_net_annot', 'waveform_net_annot')
TASKS = ('detection', 'grading', 'timing', 'shape')


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def merge(oof1, oof2):
    z1, z2 = np.load(oof1, allow_pickle=False), np.load(oof2, allow_pickle=False)
    out = {k: z1[k] for k in z1.files}
    report = dict(e_reproduced={}, rows_aligned={})
    for t in TASKS:
        ref = '%s__equal_time__' % t
        for k in ('names', 'y', 'groups'):
            assert np.array_equal(z1[ref + k], z2[ref + k]), 'Run 2 rows differ from Run 1 on %s (%s)' % (t, k)
        same_pred = bool(np.array_equal(z1[ref + 'mlp__pred'], z2[ref + 'mlp__pred']) and np.array_equal(z1[ref + 'rf__pred'], z2[ref + 'rf__pred']))
        diff = float(np.max(np.abs(z1[ref + 'mlp__proba'].astype(float) - z2[ref + 'mlp__proba'].astype(float))))
        report['e_reproduced'][t] = dict(predictions_identical=same_pred, mlp_proba_max_abs_diff=diff)
        assert same_pred and diff < 1e-5, 'Run 2 arm E does not reproduce Run 1 on %s' % t
        for tag in TW_TAGS:
            pfx = '%s__%s__' % (t, tag)
            keys = [k for k in z2.files if k.startswith(pfx)]
            assert keys, 'Run 2 lacks %s' % pfx
            for k in ('names', 'y', 'groups'):
                assert np.array_equal(z2[pfx + k], z1[ref + k]), '%s rows differ from arm E' % pfx
            for k in keys:
                assert k not in out
                out[k] = z2[k]
        report['rows_aligned'][t] = True
    out['arms'] = np.array(sorted(set(str(a) for a in z1['arms']) | set(TW_TAGS)))
    return out, report


def main(oof1, oof2, out_dir):
    assert stats.__file__ and hashlib.sha256(open(stats.__file__, 'rb').read()).hexdigest().startswith(STATS_SHA), 'stats.py differs from the frozen copy'
    os.makedirs(out_dir, exist_ok=True)
    merged, report = merge(oof1, oof2)
    path = os.path.join(out_dir, 'oof_predictions.npz')
    np.savez_compressed(path, **merged)
    res = stats.run(out_dir, out_dir, t_frozen=True)
    report.update(run1_oof_sha256=sha_file(oof1), run2_oof_sha256=sha_file(oof2), merged_oof_sha256=sha_file(path),
                  stats_py_sha256=res['stats_py_sha256'])
    json.dump(report, open(os.path.join(out_dir, 'merge_report.json'), 'w'), indent=1)
    print(json.dumps(res['mlp']['hypotheses'], indent=1))
    return res, report


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
