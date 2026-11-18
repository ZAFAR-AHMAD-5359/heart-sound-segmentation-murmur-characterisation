"""Training samples on the real pack: alignment of spectrogram, labels, ECG and mask, native and warped.
Run:  python -m pytest tests/test_train_samples.py -q   (needs derived/corpus/<config.PACK> with the ECG targets)"""
import json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import train as T, frontend as F, augment as A
from extstudy.config import LOCAL, LABELS, PACK as PACK_NAME

PACK = os.path.join(LOCAL['phd'], 'derived', 'corpus', PACK_NAME)
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(PACK, 'ecg_manifest.json')), reason='pack not built')


@pytest.fixture(scope='module')
def data():
    laws = json.load(open(os.path.join(LOCAL['phd'], 'derived', LABELS, 'laws.json')))
    return T.PairedData(PACK, laws)


def test_warped_heart_sounds_equal_the_native_spectrogram(data):
    rng = np.random.default_rng(7)
    checked = 0
    for k in [r for r in data.roles['train'] if data.events[r]['runs']][::150][:12]:
        s = T.make_sample(data, k, rng, warp_hr=150.0, augment=False, kind='T')
        src, dst = s['info']['knots']
        a = int(round(s['info']['start_s'] * F.FS))
        L = F.logpower(data.signal(k, a, min(int(data.R.n_samples[k]), a + 60000)).astype(np.float64)) - data.med[k][:, None]
        t0 = a / F.FS + T.FRAME_T0
        ev = data.events[k]
        for t in ev['s1_s'][(ev['s1_s'] > src[0] + 0.06) & (ev['s1_s'] < src[-1] - 0.06)][:5]:
            js = int(round((t - t0) / A.FRAME_S)); jt = int(round((A.map_times(t, src, dst) - s['out_t0']) / A.FRAME_S))
            if 5 <= jt < 2995 and 5 <= js < L.shape[1] - 5:
                assert np.allclose(s['spec'][:, jt - 5:jt + 6], L[:, js - 5:js + 6], atol=1e-5)
                assert s['peak_s1'][jt] or s['peak_s1'][jt - 1] or s['peak_s1'][jt + 1] or not s['mask'][jt]
                checked += 1
    assert checked >= 20


@pytest.mark.parametrize('warp', [None, 140.0])
def test_ecg_target_peaks_at_r(data, warp):
    rng = np.random.default_rng(11)
    vals = []
    for k in [r for r in data.roles['train'] if data.events[r]['runs']][::97][:15]:
        s = T.make_sample(data, k, rng, warp_hr=warp, augment=False, kind='W')
        j = np.round((s['r_times'] - s['out_t0']) / A.FRAME_S).astype(int)
        j = j[(j >= 2) & (j < len(s['ecg']) - 2)]
        j = j[s['mask'][j]]
        vals += [s['ecg'][i - 2:i + 3].max() for i in j]
    assert len(vals) > 50 and np.median(vals) > 0.8


def test_mask_cycles_carry_s1_targets(data):
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = T.make_sample(data, T.sample_record(data, rng), rng, kind='T')
        if s['mask'].sum() < 250:
            continue
        runs = np.flatnonzero(np.diff(np.r_[0, s['mask'].astype(int), 0]))
        for a, b in zip(runs[::2], runs[1::2]):
            if b - a > 300:                                                      # a labelled stretch longer than 1.2 s
                assert s['peak_s1'][a:b].sum() >= 1
