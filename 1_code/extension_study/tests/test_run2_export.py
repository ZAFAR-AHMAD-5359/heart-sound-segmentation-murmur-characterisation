"""Run 2 inference and cycle export. Run:  python -m pytest tests/test_run2_export.py -q"""
import ast, json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extstudy import cycles as C, infer as I

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN1_NB = os.path.join(HERE, 'notebooks', 'Extension_Run1.ipynb')


def _run1_defs():
    nb = json.load(open(RUN1_NB, encoding='utf-8'))
    s4 = {c.get('metadata', {}).get('name'): ''.join(c['source']) for c in nb['cells']}['s4_arm_defs']
    tree, lines = ast.parse(s4), s4.split('\n')
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = '\n'.join(lines[n.lineno - 1:n.end_lineno])
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('FUSION', 'AC_SOS'):
            out[n.targets[0].id] = '\n'.join(lines[n.lineno - 1:n.end_lineno])
    return out


def _module_defs():
    src = open(C.__file__, encoding='utf-8').read()
    tree, lines = ast.parse(src), src.split('\n')
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = '\n'.join(lines[n.lineno - 1:n.end_lineno])
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('FUSION', 'AC_SOS'):
            out[n.targets[0].id] = '\n'.join(lines[n.lineno - 1:n.end_lineno])
    return out


def test_copied_r1_definitions_are_verbatim():
    r1, mine = _run1_defs(), _module_defs()
    for k in ('FUSION', 'AC_SOS', 'check_cycles', 'acoustic_envelope', 'onset_50'):
        assert mine[k] == r1[k], k
    b = r1['arm_b_cycles'].replace('def arm_b_cycles(name, pk4):\n    x4 = AUDIO4[name]\n',
                                   'def b_cycles(x4, pk4):\n    """arm_b_cycles of the R1 notebook with the audio passed in '
                                   '(verbatim body; tests compare it with the source)."""\n', 1)
    b = b.replace("return check_cycles(cyc, L4, 'syn_fusion ' + name), d", "return check_cycles(cyc, L4, 'b_cycles'), d")
    assert mine['b_cycles'] == b


def test_state_hash_equals_the_trainer():
    from extstudy import train as T, model as M
    import torch
    torch.manual_seed(0)
    m = M.UNet1D('W')
    assert I.state_sha256(m.state_dict()) == T.state_sha256(m.state_dict())


def test_windows_cover_the_recording():
    assert I.windows(100) == [(0, 100)] and I.windows(7500) == [(0, 7500)]
    w = I.windows(20000)
    assert w[0][0] == 0 and w[-1][1] == 20000 and all(b - a == I.WIN_FRAMES for a, b in w)
    assert all(w[k + 1][0] < w[k][1] for k in range(len(w) - 1))          # consecutive windows overlap


def test_stitching_reproduces_a_position_independent_output():
    """A model that returns its first input bin: the cross-faded seed average must return that bin exactly."""
    import torch

    class FirstBin(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, :]
    rng = np.random.default_rng(1)
    for n in (900, 7500, 18321):
        spec = rng.standard_normal((24, n)).astype(np.float32)
        out = I.forward([('a', FirstBin()), ('b', FirstBin()), ('c', FirstBin())], spec, 'W')
        assert out.shape == (1, n) and np.allclose(out[0], spec[0], atol=1e-6)


def _synthetic(hr=110.0, n_beats=24, sr=4000, s1_lag=0.0, seed=0):
    """4 kHz PCG with S1 and S2 bursts (40 to 120 Hz noise, 60 ms) at known times; S2 at the paired-law interval."""
    import scipy.signal as sg
    rng = np.random.default_rng(seed)
    rr = 60.0 / hr
    s1 = 0.5 + rr * np.arange(n_beats)
    s2 = s1 + (C.S2_LAW_MS[0] + C.S2_LAW_MS[1] * hr) / 1000.0
    n = int((s1[-1] + rr + 0.5) * sr)
    x = 0.02 * rng.standard_normal(n)
    sos = sg.butter(4, (40, 120), btype='bandpass', fs=sr, output='sos')
    burst = sg.sosfiltfilt(sos, rng.standard_normal(int(0.06 * sr))) * np.hanning(int(0.06 * sr))
    burst /= np.abs(burst).max()
    for t, amp in [(t, 1.0) for t in s1] + [(t, 0.6) for t in s2]:
        i = int(t * sr)
        x[i:i + len(burst)] += amp * burst
    return x / np.abs(x).max(), s1, s2


def test_t_cycles_on_a_synthetic_recording():
    x, s1, s2 = _synthetic()
    tf = I.FRAME_T0 + I.FRAME_S * np.arange(int(len(x) / 4000 / I.FRAME_S))
    p_s2 = np.exp(-0.5 * ((tf[:, None] - (s2 + 0.03)[None, :]) / 0.012) ** 2).max(1)   # S2 head at the burst peak
    cyc, d = C.t_cycles(x, s1 + 0.03, p_s2)
    assert d['n_malformed'] == 0 and len(cyc) == len(s1) - 1 and d['frac_s2_acoustic'] == 1.0
    st = np.array([c['s1'][0] for c in cyc]) / 4000
    s2on = np.array([c['s2'][0] for c in cyc]) / 4000
    assert np.abs(st - s1[:-1]).max() < 0.03 and np.abs(s2on - s2[:-1]).max() < 0.03   # 50% onsets near the burst onsets


def test_t_cycles_prior_picks_the_s2_near_the_law_when_the_head_is_flat():
    x, s1, s2 = _synthetic(hr=90.0)
    tf = I.FRAME_T0 + I.FRAME_S * np.arange(int(len(x) / 4000 / I.FRAME_S))
    cyc, d = C.t_cycles(x, s1 + 0.03, np.full(len(tf), 0.2))
    s2on = np.array([c['s2'][0] for c in cyc]) / 4000
    assert len(cyc) == len(s1) - 1 and np.abs(s2on - s2[:-1]).max() < 0.03


def test_b_cycles_from_r_peaks_before_s1():
    x, s1, s2 = _synthetic(hr=100.0)
    cyc, d = C.b_cycles(x, ((s1 - 0.05) * 4000).astype(int))
    assert not d['fusion_fallback'] and d['n_malformed'] == 0 and len(cyc) == len(s1) - 1
    st = np.array([c['s1'][0] for c in cyc]) / 4000
    assert np.abs(st - s1[:-1]).max() < 0.03
