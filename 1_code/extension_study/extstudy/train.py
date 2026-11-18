"""Training of T and W on the paired pack (08 design, section 2.7; prereg section 7). Paired data only.

Samples: 12 s crops (3,000 frames). Dataset drawn by fixed shares, then a subject uniformly within the dataset, then a
record of that subject, then a crop start inside the record's labelled stretches. Waveform augmentation (noise, murmur-
like noise), log spectrogram with the recording's own bin medians, device tilt, then either a native crop or interval
warping to a target rate in 70 to 180 bpm, then frequency masking. Targets: Gaussian event heatmaps (T) or the ECG at
250 Hz (W), with the loss mask = valid cycles. Warping moves labels, ECG and mask with the same time map.

Selection: beat-identity F1 of the anchor event (T: S1 from the S1-minus-S2 trace; W: R from the synthetic ECG) through
the shared decoder, tolerance min(50 ms, 10% RR), inside the labelled stretches; equal weight over datasets and over
the native and the 110 to 170 bpm warped validation sets. Best state kept and re-scored at the end.
"""
import copy, hashlib, json, math, os, time
import numpy as np

from extstudy import augment as A, frontend as F, losses as LS

CFG = dict(crop_frames=3000, batch=64, lr=2e-3, weight_decay=1e-2, epochs=40, steps_per_epoch=300,
           p_warp=0.5, hr_range=(70.0, 180.0), p_noise=0.5, p_murmur=0.3, p_tilt=0.5, p_fmask=0.3,
           shares=dict(hfref=0.35, cinc2016a=0.15, ephnogram=0.15, senssmart=0.12, foster=0.12, bsslab=0.11),
           val_crop_frames=7500, val_crops_per_record=1, val_max_records_per_dataset=60, val_warp_hr=(110.0, 170.0),
           tolerance=(0.050, 0.10), grad_clip=1.0, workers=8, eval_workers=8)
CROP_SAMPLES = lambda frames: F.HOP * (frames - 1) + F.WIN                     # waveform samples behind `frames` frames
FRAME_T0 = F.WIN / 2 / F.FS                                                    # frame j centred at FRAME_T0 + j / 250
HEADS_OF = {'T': ('s1', 's2', 'r'), 'A': ('s1', 's2')}                          # event heads per kind (model.HEADS without W)
EVENT_KEY = {'s1': 's1_s', 's2': 's2_s', 'r': 'r_target_s'}


class PairedData:
    """The pack (and the ECG targets) in memory, with per-record valid cycles and labels."""

    def __init__(self, folder, laws, with_ecg=True):
        from extstudy import corpus as C
        self.man, self.pcg, self.R, self.med, ev = C.load(folder)
        self.laws = laws
        self.ecg = None
        if with_ecg:
            self.ecg_man, self.ecg, self.E = C.load_ecg(folder)
            assert self.E['record_id'].tolist() == self.R['record_id'].tolist()
        rid = ev['record_id']
        self.events = {}
        for k in range(len(self.R)):
            s = slice(np.searchsorted(rid, k), np.searchsorted(rid, k, side='right'))
            self.events[k] = {key: ev[key][s] for key in ('r_s', 'r_next_s', 's1_s', 's2_s', 'r_target_s', 'hr', 'valid')}
            self.events[k]['runs'] = [rn for rn in _valid_runs(self.events[k]) if len(rn) >= 2]
            self.events[k]['vidx'] = np.flatnonzero(self.events[k]['valid'].astype(bool))
        self.roles = {role: self.R.index[self.R.role == role].tolist() for role in ('train', 'val', 'test')}
        self.index = {}                                     # role -> dataset -> [records of each subject]
        for role, idx in self.roles.items():
            sub = self.R.loc[idx]
            self.index[role] = {d: [g.index.tolist() for _, g in s.groupby('group', sort=True)] for d, s in sub.groupby('dataset')}

    def replace_events(self, ev, laws):
        """Swap the labels (an events dict with record_id and the event keys, in the pack's record order) and the laws,
        keeping the audio; records without a valid beat under the new labels leave the sampling index, as records
        without one under the pack's labels never entered it (corpus.build keeps n_valid > 0 only)."""
        rid = np.asarray(ev['record_id'])
        order = np.argsort(rid, kind='stable')
        ev = {k: np.asarray(v)[order] for k, v in ev.items()}
        rid = ev['record_id']
        self.laws = laws
        self.events = {}
        for k in range(len(self.R)):
            s = slice(np.searchsorted(rid, k), np.searchsorted(rid, k, side='right'))
            self.events[k] = {key: ev[key][s] for key in ('r_s', 'r_next_s', 's1_s', 's2_s', 'r_target_s', 'hr', 'valid')}
            self.events[k]['runs'] = [rn for rn in _valid_runs(self.events[k]) if len(rn) >= 2]
            self.events[k]['vidx'] = np.flatnonzero(self.events[k]['valid'].astype(bool))
        has = {k: len(self.events[k]['vidx']) > 0 for k in self.events}
        self.index = {}
        for role, idx in self.roles.items():
            sub = self.R.loc[[k for k in idx if has[k]]]
            self.index[role] = {d: [g.index.tolist() for _, g in s.groupby('group', sort=True)] for d, s in sub.groupby('dataset')}
        return self

    def signal(self, k, a=0, b=None):
        r = self.R.iloc[k]
        b = int(r.n_samples) if b is None else b
        return self.pcg[int(r.offset) + a:int(r.offset) + b].astype(np.float32) / np.float32(r.scale)

    def ecg_signal(self, k):
        e = self.E.iloc[k]
        return self.ecg[int(e.ecg_offset):int(e.ecg_offset) + int(e.ecg_n)].astype(np.float32) / np.float32(8000.0)


def _valid_runs(ev):
    """Runs of consecutive valid beats whose cycles abut: list of index arrays."""
    v = ev['valid'].astype(bool)
    runs, cur = [], []
    for i in np.flatnonzero(v):
        if cur and (i != cur[-1] + 1 or abs(ev['r_s'][i] - ev['r_next_s'][cur[-1]]) > 1e-6):
            runs.append(np.array(cur)); cur = []
        cur.append(i)
    if cur:
        runs.append(np.array(cur))
    return runs


def make_sample(data, k, rng, cfg=CFG, warp_hr=None, start_s=None, augment=True, kind='T'):
    """One crop of record k. Returns dict(spec (24, n), mask (n,), targets/peaks per head (T) or ecg (W), info)."""
    n_out = cfg['crop_frames']
    ev = data.events[k]
    r = data.R.iloc[k]
    dur = int(r.n_samples) / F.FS
    runs = ev['runs']
    do_warp = warp_hr is not None or (augment and rng.random() < cfg['p_warp'])
    if do_warp and runs:
        run = runs[int(rng.integers(len(runs)))]
        hr_t = warp_hr if warp_hr is not None else rng.uniform(*cfg['hr_range'])
        need = int(math.ceil(n_out * A.FRAME_S * hr_t / 60.0)) + 2             # cycles that fill the output at hr_t
        j0 = int(rng.integers(0, max(1, len(run) - need)))
        w = run[j0:j0 + need + 1]                                               # beats; the last one closes the last cycle
        s1, s2 = ev['s1_s'][w], ev['s2_s'][w[:-1]]
        src, dst, hr_used = A.warp_map(s1, s2, ev['hr'][w[:-1]], hr_t, data.laws, r.dataset, rng)
        a_s = max(0.0, src[0] - 0.5)                                             # source waveform with margin
        b_s = min(dur, src[-1] + 0.5)
    else:
        hr_used = None
        span = n_out * A.FRAME_S
        if start_s is None:
            vi = ev['vidx']
            c = ev['r_s'][vi[int(rng.integers(len(vi)))]] if len(vi) else rng.uniform(0, max(0.0, dur - span))
            start_s = float(np.clip(c - rng.uniform(0, span), 0.0, max(0.0, dur - span)))
        a_s, b_s = start_s, min(dur, start_s + span + F.WIN / F.FS)
    a = int(round(a_s * F.FS / F.HOP)) * F.HOP                                   # start on the 4-sample grid
    b = min(int(r.n_samples), a + max(CROP_SAMPLES(8), int(round((b_s - a / F.FS) * F.FS))))
    x = data.signal(k, a, b).astype(np.float64)
    in_crop = (ev['s1_s'] >= a / F.FS) & (ev['s1_s'] < b / F.FS)
    info = dict(record_id=int(k), dataset=r.dataset, start_s=a / F.FS, warped=bool(hr_used is not None), hr_target=hr_used)
    if hr_used is not None:
        info['knots'] = (src, dst)
    if augment:
        if rng.random() < cfg['p_noise']:
            x, info['snr_db'] = A.add_noise(x, rng)
        if rng.random() < cfg['p_murmur']:
            v = np.zeros(len(in_crop), bool); v[ev['vidx']] = True; v &= in_crop
            x, info['murmur'] = A.murmur_noise(x, ev['s1_s'][v] - a / F.FS, ev['s2_s'][v] - a / F.FS, F.FS, rng)
    L = F.logpower(x) - data.med[k][:, None]
    if augment and rng.random() < cfg['p_tilt']:
        L, info['tilt'] = A.device_tilt(L, rng)
    t0 = a / F.FS + FRAME_T0                                                     # time of source frame 0
    if hr_used is not None:
        out_t0 = t0 + round((dst[0] - t0) / A.FRAME_S) * A.FRAME_S - 0.2         # start 0.2 s before the run
        spec, inside = A.warp_frames(L, t0, src, dst, n_out, out_t0)
        tmap = lambda tt: A.map_times(tt, src, dst)
    else:
        out_t0 = t0
        spec = np.zeros((L.shape[0], n_out), np.float32)
        m = min(n_out, L.shape[1]); spec[:, :m] = L[:, :m]
        inside = np.zeros(n_out, bool); inside[:m] = True
        tmap = lambda tt: np.asarray(tt, float)
    if augment and rng.random() < cfg['p_fmask']:
        spec, info['fmask'] = A.freq_mask(spec, rng)
    tt = out_t0 + np.arange(n_out) * A.FRAME_S
    mask = np.zeros(n_out, bool)
    vb = ev['vidx']
    vb = vb[(ev['r_next_s'][vb] > a / F.FS - 1.0) & (ev['r_s'][vb] < b / F.FS + 1.0)]   # beats near the source window only
    for i in vb:
        c0, c1 = tmap(ev['r_s'][i]), tmap(ev['r_next_s'][i])
        if hr_used is not None and not (src[0] <= ev['s1_s'][i] < src[-1] - 2 * A.KEEP_S + 1e-9):
            continue                                                             # only the warped cycles of the run
        mask |= (tt >= c0) & (tt < c1)
    mask &= inside
    out = dict(spec=spec.astype(np.float32), mask=mask, info=info, out_t0=out_t0)
    if kind in HEADS_OF:
        for h in HEADS_OF[kind]:
            key = EVENT_KEY[h]
            ok = vb[np.isfinite(ev[key][vb])]
            e = tmap(ev[key][ok])
            y, pk = LS.gaussian_targets(e, n_out, out_t0, LS.SIGMA_S[h])
            out['target_' + h], out['peak_' + h] = y, pk
        out['s1_times'] = tmap(ev['s1_s'][vb])
    else:
        ecg = data.ecg_signal(k)
        ts = np.interp(tt, dst, src) if hr_used is not None else tt              # source time behind each output frame
        out['ecg'] = np.interp(ts * 250.0, np.arange(len(ecg)), ecg).astype(np.float32)
        out['r_times'] = tmap(ev['r_s'][vb])
    return out


def sample_record(data, rng, cfg=CFG, role='train'):
    """Dataset by fixed share (among datasets present in the role), subject uniformly, then a record of the subject."""
    idx = data.index[role]
    ds = [d for d in cfg['shares'] if d in idx]
    p = np.array([cfg['shares'][d] for d in ds]); p = p / p.sum()
    groups = idx[ds[int(rng.choice(len(ds), p=p))]]
    recs = groups[int(rng.integers(len(groups)))]
    return int(recs[int(rng.integers(len(recs)))])


def batch(data, rng, kind, cfg=CFG, role='train'):
    items = [make_sample(data, sample_record(data, rng, cfg, role), rng, cfg, kind=kind) for _ in range(cfg['batch'])]
    out = dict(spec=np.stack([it['spec'] for it in items]), mask=np.stack([it['mask'] for it in items]))
    if kind in HEADS_OF:
        for h in HEADS_OF[kind]:
            out['target_' + h] = np.stack([it['target_' + h] for it in items])
            out['peak_' + h] = np.stack([it['peak_' + h] for it in items])
    else:
        out['ecg'] = np.stack([it['ecg'] for it in items])
    out['datasets'] = [it['info']['dataset'] for it in items]
    return out


# ------------------------------------------------------------------ validation
def validation_set(data, kind, cfg=CFG, seed=12345):
    """Fixed validation crops: per val record, native crops and crops warped to a rate in 110 to 170 bpm."""
    rng = np.random.default_rng(seed)
    c2 = dict(cfg, crop_frames=cfg['val_crop_frames'])
    items = []
    keep = []
    for d, groups in sorted(data.index['val'].items()):           # at most val_max_records_per_dataset per dataset
        recs = [k for g in groups for k in g if data.events[k]['runs']]
        sel = rng.permutation(len(recs))[:cfg['val_max_records_per_dataset']]
        keep += [recs[i] for i in sorted(sel)]
    for k in keep:
        for j in range(cfg['val_crops_per_record']):
            items.append(make_sample(data, k, rng, c2, augment=False, kind=kind))
            items.append(make_sample(data, k, rng, c2, warp_hr=rng.uniform(*cfg['val_warp_hr']), augment=False, kind=kind))
    return items


def beat_f1(truth_s, beats_s, mask, out_t0, tol=CFG['tolerance']):
    """F1 of decoded beats against true events inside the labelled stretches; tolerance min(50 ms, 10% of the local RR)."""
    t = np.sort(np.asarray(truth_s, float)); b = np.sort(np.asarray(beats_s, float))
    inside = lambda x: mask[np.clip(np.round((x - out_t0) / A.FRAME_S).astype(int), 0, len(mask) - 1)]
    t = t[inside(t)] if len(t) else t
    b = b[inside(b)] if len(b) else b
    if len(t) == 0:
        return np.nan
    rr = np.median(np.diff(t)) if len(t) > 1 else 0.8
    tolv = min(tol[0], tol[1] * rr)
    used, tp = set(), 0
    for x in t:
        j = np.flatnonzero(np.abs(b - x) <= tolv)
        j = [q for q in j if q not in used]
        if j:
            used.add(min(j, key=lambda q: abs(b[q] - x))); tp += 1
    return 2 * tp / max(1, len(t) + len(b))


def _decode_score(args):
    """One validation crop: decode the trace with the shared decoder and score it (runs in a worker process)."""
    from extstudy import decoder as D
    kind, o, truth, mask, out_t0, key = args
    if kind in HEADS_OF:
        p = 1 / (1 + np.exp(-o))
        dec = D.decode_beats(D.beat_trace_from_heads(p[0], p[1]), 250, kind='event')
    else:
        dec = D.decode_beats(o[0], 250, kind='ecg')
    return key, beat_f1(truth, out_t0 + dec['beats_s'], mask, out_t0)


def evaluate(model, items, kind, device, batch_size=16, workers=CFG['eval_workers']):
    """Selection score (equal weight over datasets, then over native and warped) and the per-dataset breakdown."""
    import torch
    model.eval()
    jobs = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            out = model(torch.tensor(np.stack([it['spec'] for it in chunk]), device=device)).float().cpu().numpy()
            for it, o in zip(chunk, out):
                truth = it['s1_times'] if kind in HEADS_OF else it['r_times']
                jobs.append((kind, o, truth, it['mask'], it['out_t0'],
                             (it['info']['dataset'], 'warped' if it['info']['warped'] else 'native')))
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_decode_score, jobs, chunksize=8))
    else:
        res = [_decode_score(j) for j in jobs]
    scores = {}
    for key, f1 in res:
        scores.setdefault(key, []).append(f1)
    per = {'%s/%s' % k: float(np.nanmean(v)) for k, v in scores.items()}
    ds = sorted({k[0] for k in scores})
    by_cond = {c: float(np.nanmean([np.nanmean(scores[(d, c)]) for d in ds if (d, c) in scores])) for c in ('native', 'warped')}
    return float(np.mean(list(by_cond.values()))), dict(per=per, by_condition=by_cond)


def w_zero_l1(items):
    """L1 of an all-zero ECG output on the validation crops (W must beat it; Panel G)."""
    v = [np.abs(it['ecg'][it['mask']]).mean() for it in items if it['mask'].any()]
    return float(np.mean(v))


# ------------------------------------------------------------------ training loop
class BatchStream:
    """Iterable of training batches; each DataLoader worker draws from its own seeded generator, so the stream is
    deterministic for a fixed seed and worker count."""

    def __init__(self, data, kind, cfg, seed, n_batches):
        self.data, self.kind, self.cfg, self.seed, self.n = data, kind, cfg, seed, n_batches

    def __iter__(self):
        import torch
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        for _ in range(wid, self.n, nw):
            yield batch(self.data, rng, self.kind, self.cfg)


try:
    import torch as _torch

    class StreamDataset(_torch.utils.data.IterableDataset):
        """Module-level wrapper (picklable for spawned workers; forked workers on Colab share the pack)."""

        def __init__(self, stream):
            super().__init__()
            self.stream = stream

        def __iter__(self):
            return iter(self.stream)
except ImportError:                                       # numpy-only use (tests of the sampler)
    StreamDataset = None


def _loader(data, kind, cfg, seed, n_batches):
    import torch
    stream = BatchStream(data, kind, cfg, seed, n_batches)
    if not cfg['workers']:
        return iter(stream)
    return iter(torch.utils.data.DataLoader(StreamDataset(stream), batch_size=None, num_workers=cfg['workers'],
                                            prefetch_factor=4))


def fit(data, kind, seed, device, cfg=CFG, log=print, val_items=None):
    import torch
    from extstudy import model as M
    torch.manual_seed(seed); np.random.seed(seed)
    assert kind == 'W' or tuple(M.HEADS[kind]) == tuple(HEADS_OF[kind]), 'head tables differ between model.py and train.py'
    model = M.UNet1D(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total = cfg['epochs'] * cfg['steps_per_epoch']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    val_items = val_items if val_items is not None else validation_set(data, kind, cfg)
    best, best_state, hist = -1.0, None, []
    for ep in range(cfg['epochs']):
        model.train(); t0 = time.time(); losses = []
        for b in _loader(data, kind, cfg, seed * 1000 + ep, cfg['steps_per_epoch']):
            x = torch.as_tensor(np.asarray(b['spec']), device=device)
            m = torch.as_tensor(np.asarray(b['mask']), device=device)
            out = model(x)
            if kind in HEADS_OF:
                tg = {h: torch.as_tensor(np.asarray(b['target_' + h]), device=device) for h in HEADS_OF[kind]}
                pk = {h: torch.as_tensor(np.asarray(b['peak_' + h]), device=device) for h in HEADS_OF[kind]}
                loss, _ = LS.t_loss(out, tg, pk, m, heads=HEADS_OF[kind])
            else:
                loss, _ = LS.w_loss(out[:, 0], torch.as_tensor(np.asarray(b['ecg']), device=device), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); sched.step()
            losses.append(float(loss))
        score, detail = evaluate(model, val_items, kind, device, workers=cfg['eval_workers'])
        hist.append(dict(epoch=ep + 1, loss=float(np.mean(losses)), val_score=score, seconds=time.time() - t0, **detail))
        log('epoch %2d  loss %.4f  val %.4f  (native %.3f, warped %.3f)  %.0f s' % (
            ep + 1, np.mean(losses), score, detail['by_condition']['native'], detail['by_condition']['warped'], time.time() - t0))
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    rescore, _ = evaluate(model, val_items, kind, device, workers=cfg['eval_workers'])
    assert abs(rescore - best) < 1e-9, 'best state does not re-score to its logged value'
    return model, dict(best_val=best, history=hist, seed=seed, kind=kind)


def state_sha256(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()
