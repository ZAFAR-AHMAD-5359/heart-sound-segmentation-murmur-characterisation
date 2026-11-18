"""Label-free inference of the frozen T and W models on one PCG recording (Run 2; prereg section 7).

Input path as in training: PCG resampled to 1 kHz, band-pass 25 to 400 Hz, log-power spectrogram with each bin's
median over the whole recording subtracted (frontend.logspec); frame j is centred at FRAME_T0 + j / 250 s.
Windows: 30 s (7,500 frames), the crop length of the validation and Gate A runs. A shorter recording is zero-padded to
7,500 frames as those crops were; a longer one runs in 30 s windows overlapping by 4 s and cross-faded linearly.
Seeds: the three frozen models of an arm are averaged (T: sigmoid probabilities; W: the ECG output), as the
pre-registration fixes (traces averaged across seeds, section 7).
Beats: the shared decoder (amendment 3); T decodes its S1 head minus its S2 head, W its synthetic ECG.
Nothing here reads a dataset or a label; the caller passes the audio.
"""
import hashlib, json, os
import numpy as np

from extstudy import frontend as F, model as M, decoder as D

FRAME_T0 = F.WIN / 2 / F.FS            # 0.016 s
FRAME_S = F.HOP / F.FS                 # 0.004 s
WIN_FRAMES, OVERLAP = 7500, 1000


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def state_sha256(state):
    """Same as train.state_sha256 (tests hold them equal); kept here so inference does not import the trainer."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def spectrogram(x, fs):
    """(24, frames) float32 model input for a whole recording sampled at fs."""
    return F.logspec(F.bandpass(F.to_1k(np.asarray(x, np.float64), fs)))


def load_models(folder, kind, device='cpu'):
    """The three frozen models of arm `kind` ('T' or 'W') from a checkpoint folder, each checked against freeze.json
    (file and weight hashes). Returns ([(name, model)], freeze record)."""
    import torch
    fr = json.load(open(os.path.join(folder, 'freeze.json')))
    assert not fr.get('smoke'), 'checkpoint folder holds a smoke run'
    out = []
    for name in sorted(fr['models']):
        if not name.startswith(kind + '_'):
            continue
        m = fr['models'][name]
        path = os.path.join(folder, name + '.pt')
        assert sha256_file(path) == m['file_sha256'], 'checkpoint file differs from freeze.json: %s' % name
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = M.UNet1D(kind)
        net.load_state_dict(ck['state_dict'])
        assert state_sha256(net.state_dict()) == m['weights_sha256'], 'weights differ from freeze.json: %s' % name
        out.append((name, net.eval().to(device)))
    assert len(out) == 3, 'expected three seeds for %s, found %d' % (kind, len(out))
    return out, fr


def windows(n, win=WIN_FRAMES, overlap=OVERLAP):
    """[(start, end)] frame windows covering n frames: one window when n <= win, else win-long windows every
    win - overlap frames with the last one aligned to the end."""
    if n <= win:
        return [(0, n)]
    starts = sorted(set(list(range(0, n - win, win - overlap)) + [n - win]))
    return [(s, s + win) for s in starts]


def _fade(a, b, n, overlap=OVERLAP):
    w = np.ones(b - a)
    if a > 0:
        k = min(overlap, b - a)
        w[:k] = np.linspace(0.0, 1.0, k + 2)[1:-1]
    if b < n:
        k = min(overlap, b - a)
        w[-k:] = np.minimum(w[-k:], np.linspace(1.0, 0.0, k + 2)[1:-1])
    return w


def forward(models, spec, kind, device='cpu'):
    """Seed-averaged traces (heads, frames) for a whole recording: T -> (s1, s2, r) probabilities, W -> (ecg,)."""
    import torch
    n = spec.shape[1]
    acc = np.zeros((len(M.HEADS[kind]), n))
    wsum = np.zeros(n)
    for a, b in windows(n):
        seg = np.zeros((spec.shape[0], WIN_FRAMES), np.float32)
        seg[:, :b - a] = spec[:, a:b]
        x = torch.tensor(seg[None], device=device)
        with torch.no_grad():
            outs = [net(x).float().cpu().numpy()[0, :, :b - a].astype(np.float64) for _, net in models]
        o = np.mean([1.0 / (1.0 + np.exp(-y)) for y in outs], 0) if kind != 'W' else np.mean(outs, 0)
        w = _fade(a, b, n)
        acc[:, a:b] += o * w
        wsum[a:b] += w
    return acc / wsum


def beats(trace, kind):
    """Beat times in seconds from the start of the recording (T: S1 events; W: R-peaks of the synthetic ECG), and the
    decoder's record summary."""
    if kind != 'W':                                    # T and the acoustic-label control A: S1 events from the S1-minus-S2 trace
        dec = D.decode_beats(D.beat_trace_from_heads(trace[0], trace[1]), 250, kind='event')
    else:
        dec = D.decode_beats(trace[0], 250, kind='ecg')
    return FRAME_T0 + np.asarray(dec['beats_s'], float), dec['rec']


def trace_hash(trace):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(trace, np.float32)).tobytes()).hexdigest()[:16]
