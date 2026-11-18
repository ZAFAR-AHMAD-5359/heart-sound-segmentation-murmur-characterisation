"""Training pack integrity (runs when derived/corpus/<config.PACK> exists). Run:  python -m pytest tests/test_corpus.py -q"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import corpus as C, frontend as F
from extstudy.config import LOCAL, PACK

FOLDER = os.path.join(LOCAL['phd'], 'derived', 'corpus', PACK)
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(FOLDER, 'manifest.json')), reason='pack not built')


@pytest.fixture(scope='module')
def pack():
    return C.load(FOLDER)                                     # checks every SHA-256 in the manifest


def test_roles_and_groups(pack):
    man, pcg, R, med, ev = pack
    assert R.groupby('group')['role'].nunique().max() == 1    # no subject in two roles
    assert set(R[R.dataset == 'cinc2016a']['role']) == {'train'}
    assert not ((R.dataset == 'ephnogram') & R.record.isin(['ECGPCG0001', 'ECGPCG0002'])).any()


def test_waveform_round_trip(pack):
    man, pcg, R, med, ev = pack
    rng = np.random.default_rng(1)
    for k in rng.choice(len(R), 5, replace=False):
        r = R.iloc[k]
        z = np.load(os.path.join(LOCAL['harmonised'], r.dataset, r.record + '.npz'))
        ref = F.bandpass(F.to_1k(z['pcg'], float(z['fs_pcg'])))
        got = C.record_signal(pcg, R, k)
        assert len(got) == len(ref)
        assert np.corrcoef(got, ref)[0, 1] > 0.999                # int16 clips the loudest 0.01% (spikes)
        # only samples above the 99.99th percentile can clip: at most ceil(n / 10^4) of them (one in a 10 s record)
        n = int(r.n_samples)
        assert np.sum(np.abs(pcg[int(r.offset):int(r.offset) + n]) >= 32767) <= int(np.ceil(n * 1e-4))
        assert np.allclose(med[k], F.bin_medians(got), atol=0.05)


def test_events_aligned(pack):
    man, pcg, R, med, ev = pack
    rid = ev['record_id']
    assert np.all(np.diff(rid) >= 0) and set(np.unique(rid)) <= set(R.record_id)
    dur = R.set_index('record_id').loc[rid, 'duration_s'].values
    v = ev['valid'].astype(bool)
    assert np.all(ev['r_next_s'][v] <= dur[v] + 1e-6)
    s1, s2, r, rn = ev['s1_s'][v], ev['s2_s'][v], ev['r_s'][v], ev['r_next_s'][v]
    assert np.all((s1 >= r) & (s2 > s1) & (s2 < rn))           # S1 after the beat, S2 inside the cycle
    counts = np.bincount(rid[v], minlength=len(R))
    assert np.array_equal(counts[R.record_id.values], R.n_valid.values)
