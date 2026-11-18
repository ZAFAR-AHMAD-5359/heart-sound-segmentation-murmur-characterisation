"""Experiment 0c preparation (boundaries only; no label, feature or prediction is read): rebuilds the cycles of arms C,
Springer (S) and W from the Run 1 and Run 2 caches, checks them against the cycle hashes the runs recorded, and fixes
each arm's failure set among the 610 recordings of Experiments 0 and 0b: no cycle pairs with arm C within 150 ms, or
|median signed S2 onset offset| above 60 ms (the S2 follow-up's statistic). Writes the sets and the embedded cycles of
S and W for those recordings.
Usage:  python notebooks/exp0c_prep.py <run1 cycles dir> <audio cache dir> <output dir>
"""
import base64, hashlib, json, os, sys, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_phase_analysis as SP
from extstudy import corrupt as K

FAIL_MS = 60


def failure_set(cyc_x, cyc_c, names):
    out, stats = [], dict(no_pairs=0, off_gt60=0)
    for n in names:
        off, npair = K.s2_onset_offset_ms(cyc_x[n], cyc_c[n])
        if npair == 0:
            stats['no_pairs'] += 1
            out.append(n)
        elif abs(off) > FAIL_MS:
            stats['off_gt60'] += 1
            out.append(n)
    return sorted(out), stats


def pack(cyc_of, names):
    data = {n: [[int(x) for k in SP.PHASES for x in c[k]] for c in cyc_of[n]] for n in names}
    b64 = base64.b64encode(zlib.compress(json.dumps(data, sort_keys=True).encode(), 9)).decode()
    return b64, hashlib.sha256(b64.encode()).hexdigest()


def main(run1_cycles_dir, audio_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keep = np.load(os.path.join(SP.FOLDS, 'detection.npz'))['keep_names'].astype(str).tolist()
    cyc = SP.load_cycles(run1_cycles_dir, audio_dir, keep)
    rec1 = json.load(open(os.path.join(SP.RUN1, 'run_manifest.json')))['hashes']['feature_cycles_hashes']
    rec2 = json.load(open(os.path.join(SP.RUN2, 'run_manifest.json')))['hashes']['feature_cycles_hashes']
    verify = dict(C=(SP.cycles_hash(cyc['C'], keep), rec1['pcg_seg']), S=(SP.cycles_hash(cyc['S'], keep), rec1['springer_wpk']),
                  W=(SP.cycles_hash(cyc['W'], keep), rec2['waveform_net']))
    print('cycle hashes (recomputed, recorded):', verify)
    assert all(a == b for a, b in verify.values()), 'an arm does not reproduce its run'
    o1 = np.load(os.path.join(SP.RUN1, 'oof_predictions.npz'), allow_pickle=False)
    names = sorted(set().union(*[set(o1['%s__pcg_seg__names' % t].astype(str)) for t in SP.TASKS]))   # names only
    assert len(names) == 610
    sets = {}
    for k in ('S', 'W'):
        F, st = failure_set(cyc[k], cyc['C'], names)
        sets[k] = dict(recordings=F, n=len(F), share=len(F) / len(names), **st)
        print('%s fails in %d of %d recordings (%.3f): %s' % (k, len(F), len(names), len(F) / len(names), st))
    both = sorted(set(sets['S']['recordings']) & set(sets['W']['recordings']))
    print('both fail: %d | only S: %d | only W: %d' % (len(both), sets['S']['n'] - len(both), sets['W']['n'] - len(both)))
    emb = {}
    for k, key in (('S', 'springer_wpk'), ('W', 'waveform_net')):
        b64, h = pack(cyc[k], names)
        emb[k] = dict(arm=key, b64=b64, sha256=h, cycles_hash_610=SP.cycles_hash(cyc[k], names))
    out = dict(names=names, fail_ms=FAIL_MS, rule='no cycle paired with arm C within 150 ms, or |median signed S2 onset offset| > 60 ms',
               verify={k: dict(recomputed=a, recorded=b) for k, (a, b) in verify.items()}, sets=sets, both=both,
               cycles_hash_610_C=SP.cycles_hash(cyc['C'], names), embedded=emb)
    json.dump(out, open(os.path.join(out_dir, 'exp0c_sets.json'), 'w'))
    print('written', os.path.join(out_dir, 'exp0c_sets.json'))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:4])
