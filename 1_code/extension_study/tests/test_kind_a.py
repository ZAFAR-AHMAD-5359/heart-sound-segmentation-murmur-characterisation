"""Kind 'A' (the acoustic-label control arm): model shape and size, the loss over two heads, samples with S1 and S2
targets and no R target, and PairedData.replace_events on the local pack. T is unchanged by these additions.
Run:  python -m pytest tests/test_kind_a.py -q"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from extstudy import model as M, losses as L, train as T, infer as I

PACK = 'E:/NUST MS Thesis/PhD Level Research/derived/corpus/paired_v3'
LAWS = 'E:/NUST MS Thesis/PhD Level Research/derived/labels_v3/laws.json'


def test_a_model_has_two_heads_and_t_is_unchanged():
    a, t = M.UNet1D('A'), M.UNet1D('T')
    assert M.n_params(t) == 872387 and M.n_params(a) == 872387 - 65
    x = torch.randn(2, 24, 1001)
    assert a(x).shape == (2, 2, 1001) and t(x).shape == (2, 3, 1001)
    assert float(a.head.bias[0]) == -4.0


def test_t_loss_over_two_heads():
    logits = torch.zeros(1, 2, 100)
    tg = {h: torch.zeros(1, 100) for h in ('s1', 's2')}
    pk = {h: torch.zeros(1, 100, dtype=torch.bool) for h in ('s1', 's2')}
    tg['s1'][0, 10] = 1.0; pk['s1'][0, 10] = True
    total, parts = L.t_loss(logits, tg, pk, torch.ones(1, 100, dtype=torch.bool), heads=('s1', 's2'))
    assert set(parts) == {'s1', 's2'} and float(total) > 0


def test_infer_sigmoid_for_a():
    m = M.UNet1D('A').eval()
    spec = np.random.default_rng(0).standard_normal((24, 400)).astype(np.float32)
    tr = I.forward([('A_seed1', m)], spec, 'A')
    assert tr.shape == (2, 400) and tr.min() >= 0 and tr.max() <= 1


@pytest.mark.skipif(not (os.path.isdir(PACK) and os.path.exists(LAWS)), reason='local pack not present')
def test_sample_kind_a_and_replace_events():
    import json
    laws = json.load(open(LAWS))
    data = T.PairedData(PACK, laws, with_ecg=False)
    rng = np.random.default_rng(1)
    k = data.index['train']['ephnogram'][0][0]
    s = T.make_sample(data, k, rng, augment=False, kind='A')
    assert 'target_s1' in s and 'target_s2' in s and 'target_r' not in s and s['spec'].shape[0] == 24
    # replace_events with the pack's own events, every other record emptied: the index keeps only labelled records
    ev = dict(np.load(os.path.join(PACK, 'events.npz')))
    keep = ev['record_id'] % 2 == 0
    ev2 = {kk: v[keep] for kk, v in ev.items()}
    data.replace_events(ev2, laws)
    recs = [kk for d in data.index['train'].values() for g in d for kk in g]
    assert recs and all(kk % 2 == 0 for kk in recs)
    assert all(len(data.events[kk]['vidx']) == 0 for kk in range(len(data.R)) if kk % 2 == 1)
