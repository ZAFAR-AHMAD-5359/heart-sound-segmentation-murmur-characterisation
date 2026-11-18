"""Panel G check left out of the training notebook: W's validation L1 against the all-zero output (the notebook stored
only the all-zero value). Also, descriptive and post hoc, the readout kappa of the seed-averaged W trace, the form in
which W enters Run 2 (prereg section 7: CirCor traces averaged across seeds). Paired validation data only.
Usage:  python notebooks/check_w_val.py <checkpoint folder>
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import train as T, model as M, decoder as D
from extstudy.config import LOCAL, LABELS, PACK


def readout(trace, it):
    dec = D.decode_beats(trace, 250, kind='ecg')
    b = it['out_t0'] + dec['beats_s']
    truth = it['r_times']
    f1 = T.beat_f1(truth, b, it['mask'], it['out_t0'])
    rr = np.median(np.diff(truth)) if len(truth) > 2 else 0.8
    return f1, T.beat_f1(truth, b + rr / 2, it['mask'], it['out_t0'])


def main(folder):
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    freeze = json.load(open(os.path.join(folder, 'freeze.json')))
    laws = json.load(open(os.path.join(LOCAL['phd'], 'derived', LABELS, 'laws.json')))
    data = T.PairedData(os.path.join(LOCAL['phd'], 'derived', 'corpus', PACK), laws)
    val = T.validation_set(data, 'W', T.CFG)
    zero = T.w_zero_l1(val)
    print('validation crops %d | all-zero L1 %.6f (freeze.json %.6f)' % (len(val), zero, freeze['w_zero_output_val_l1']))
    assert abs(zero - freeze['w_zero_output_val_l1']) < 1e-9, 'validation set differs from the training run'
    names = [n for n in freeze['models'] if n.startswith('W')]
    traces = {n: [] for n in names}
    out = dict(all_zero_l1=zero, models={})
    for n in names:
        ck = torch.load(os.path.join(folder, n + '.pt'), map_location='cpu', weights_only=False)
        m = M.UNet1D('W'); m.load_state_dict(ck['state_dict']); m.eval()
        assert T.state_sha256(m.state_dict()) == freeze['models'][n]['weights_sha256']
        l1 = []
        with torch.no_grad():
            for it in val:
                o = m(torch.tensor(it['spec'][None])).float().numpy()[0, 0]
                traces[n].append(o)
                if it['mask'].any():
                    l1.append(np.abs(o[it['mask']] - it['ecg'][it['mask']]).mean())
        out['models'][n] = dict(val_l1=float(np.mean(l1)), beats_all_zero=bool(np.mean(l1) < zero))
        print('%s validation L1 %.6f (all-zero %.6f): %s' % (n, np.mean(l1), zero, 'beats it' if np.mean(l1) < zero else 'does NOT beat it'))
    res = {}
    for j, it in enumerate(val):
        avg = np.mean([traces[n][j] for n in names], axis=0)
        res.setdefault(it['info']['dataset'], []).append(readout(avg, it))
    ens = {}
    for d, v in sorted(res.items()):
        v = np.array(v, float); f1, f0 = np.nanmean(v[:, 0]), np.nanmean(v[:, 1])
        ens[d] = dict(f1=float(f1), null=float(f0), kappa=float((f1 - f0) / (1 - f0)), n=len(v))
    out['seed_average_val_kappa_post_hoc'] = ens
    print('seed-averaged W, validation readout kappa (post hoc, descriptive):',
          {d: round(x['kappa'], 3) for d, x in ens.items()})
    json.dump(out, open(os.path.join(folder, 'w_val_check.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    main(sys.argv[1])
