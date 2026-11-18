"""Experiment 0c (failure placed where Springer fails): failure sets, random recording sets of a fixed size, embedded
arm cycles, and the boundary set of one recording under each condition. Builds on corrupt.py, which stays as
Experiment 0b embedded it.

Conditions (typ): 's2' every cycle's S2 displaced by 100 ms in the selected recordings (Experiment 0b's 's2c' rule);
'fail' equal-time windows in the selected recordings; 'inside' another arm's own cycles in the selected recordings and
arm C's elsewhere; 'outside' the reverse; 'all' the other arm's cycles in every recording (reproduction control).
"""
import base64, hashlib, json, zlib
import numpy as np

from . import corrupt as K

PHASES = ('s1', 'systole', 's2', 'diastole')
FAIL_MS = 60


def cycles_hash(cyc_of, names):
    """The run notebooks' cycles_hash for cycle lists (boundaries as plain ints, '|' between recordings, 16 hex)."""
    h = hashlib.sha1()
    for n in names:
        h.update(json.dumps([[int(x) for k in PHASES for x in c[k]] for c in cyc_of[n]]).encode())
        h.update(b'|')
    return h.hexdigest()[:16]


def failure_set(cyc_x, cyc_c, names, fail_ms=FAIL_MS, sr=4000):
    """Recordings where arm X fails against arm C: no cycle pairs within 150 ms, or |median signed S2 onset offset|
    above fail_ms."""
    out = []
    for n in names:
        off, npair = K.s2_onset_offset_ms(cyc_x[n], cyc_c[n], sr)
        if npair == 0 or abs(off) > fail_ms:
            out.append(n)
    return sorted(out)


def select_count(names, k, seed):
    """The first k recordings of the seed's permutation of the sorted names."""
    names = sorted(names)
    perm = np.random.default_rng(int(seed)).permutation(len(names))
    return set(names[i] for i in perm[:int(k)])


def unpack(b64):
    """Embedded cycles ({name: [[8 ints] per cycle]}, zlib + base64) back to the run notebooks' cycle dicts."""
    data = json.loads(zlib.decompress(base64.b64decode(b64)).decode())
    return {n: [dict(cycle=i, **{k: (int(v[2 * j]), int(v[2 * j + 1])) for j, k in enumerate(PHASES)}) for i, v in enumerate(cyc)]
            for n, cyc in data.items()}


def exp0c_cycles(ref, name, typ, seed, selected, other, equal_time_fn, n_samples, sr=4000):
    """One recording's boundary set under an Experiment 0c condition, with its displacement counts.
    ref: arm C's cycles; other: the other arm's cycles of this recording (for 'inside', 'outside', 'all')."""
    if typ == 's2':
        return K.exp0b_cycles(ref, name, 's2c', 1.0, seed, selected, equal_time_fn, n_samples, sr)
    if typ == 'fail':
        return K.exp0b_cycles(ref, name, 'fail', 1.0, seed, selected, equal_time_fn, n_samples, sr)
    info = dict(n=len(ref), picked=0, moved=0, clipped=0, sign=0)
    use_other = {'inside': name in selected, 'outside': name not in selected, 'all': True}
    if typ not in use_other:
        raise ValueError('unknown Experiment 0c type %r' % typ)
    return K._renumber([dict(c) for c in (other if use_other[typ] else ref)]), info
