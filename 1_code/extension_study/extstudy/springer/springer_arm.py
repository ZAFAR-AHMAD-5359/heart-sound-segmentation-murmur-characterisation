"""
springer_arm.py -- adapter that turns springer_py output into R1-pipeline cycles (arm S).

R1 cycle format (build_gating_campaign.py, check_cycles B:1080): list of dicts
  {'cycle': i, 's1': (a, b), 'systole': (a, b), 's2': (a, b), 'diastole': (a, b)}
with int sample indices at 4 kHz, half-open, inside [0, L4] of AUDIO4[name] (analysed portion, max-abs normalised).

Pre-specified variants (to be frozen in the pre-registration):
  S_wpk : published sample2016 parameters, heart-rate search 40-220 bpm, highest local maximum, systole floor 0.10 s
  S_pub : published sample2016 parameters and published heart-rate rule (30-120 bpm, global maximum, 0.2 s)
  *_annot : same cycles, kept only when the S1 onset lies inside a TSV-annotated span (selection only, like arm E2)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import springer_py as sp

PHASE_KEYS = ("s1", "systole", "s2", "diastole")
VARIANTS = {"S_wpk": dict(hr_min=40, hr_max=220, sys_min_s=0.10, hr_method="peak"),
            "S_pub": dict(hr_min=30, hr_max=120, sys_min_s=0.20, hr_method="argmax")}
TARGET_SR = 4000


def states_to_cycles(states_1k, L4, sr_states=1000, sr_out=TARGET_SR):
    """Complete S1->systole->S2->diastole runs only (same state machine as parse_tsv_annotations)."""
    st = np.asarray(states_1k)
    ch = np.flatnonzero(np.diff(st)) + 1
    starts = np.concatenate([[0], ch]); ends = np.concatenate([ch, [len(st)]])
    runs = [(int(st[s]), int(s), int(e)) for s, e in zip(starts, ends)]
    k = sr_out / sr_states
    cyc, i = [], 0
    while i + 3 < len(runs):
        if [r[0] for r in runs[i:i + 4]] == [1, 2, 3, 4]:
            b = [min(L4, int(round(x * k))) for r in runs[i:i + 4] for x in (r[1], r[2])]
            c = dict(cycle=len(cyc), s1=(b[0], b[1]), systole=(b[2], b[3]), s2=(b[4], b[5]), diastole=(b[6], b[7]))
            if c["diastole"][1] > c["s1"][0]:
                cyc.append(c)
            i += 4
        else:
            i += 1
    return cyc


def check_cycles_strict(cyc, L):
    for c in cyc:
        seq = [v for kk in PHASE_KEYS for v in c[kk]]
        assert all(0 <= v <= L for v in seq), seq
        assert all(a <= b for a, b in zip(seq, seq[1:])), seq
    return cyc


def arm_s_cycles(x4, params, variant="S_wpk"):
    L4 = len(x4)
    states, info = sp.run_springer(np.asarray(x4, float), TARGET_SR, params, return_info=True, **VARIANTS[variant])
    cyc = check_cycles_strict(states_to_cycles(states, L4), L4)
    diag = dict(hr_bpm=float(info["hr"]), systolic_s=float(info["systolic_time"]), n_cycles=len(cyc),
                coverage=float(sum(c["diastole"][1] - c["s1"][0] for c in cyc) / L4) if L4 else np.nan,
                unassigned_frac=float(np.mean(states == 0)))
    return cyc, diag


def restrict_to_annotated(cyc, spans_samples):
    """spans_samples: list of (start, end) at 4 kHz from the TSV (state != 0). Selection only; boundaries untouched."""
    out = [c for c in cyc if any(a <= c["s1"][0] < b for a, b in spans_samples)]
    return [dict(c, cycle=i) for i, c in enumerate(out)]


if __name__ == "__main__":
    # units check on recordings already in the exploratory pilot (no new TSV exposure)
    import soundfile as sf, json, time
    CIRCOR = r"E:\NUST MS Thesis\Milestone-3\Dataset\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
    params = sp.load_sample2016_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_code", "sample2016"))
    for name in ("50781_PV", "85152_PV", "50721_MV"):
        x, fs = sf.read(os.path.join(CIRCOR, name + ".wav"))
        x4 = (x / (np.max(np.abs(x)) + 1e-10)).astype(np.float32)
        for v in VARIANTS:
            t = time.perf_counter(); cyc, d = arm_s_cycles(x4, params, v)
            durs = np.array([[c[k][1] - c[k][0] for k in PHASE_KEYS] for c in cyc]) / 4.0
            print(name, v, {k: round(val, 3) if isinstance(val, float) else val for k, val in d.items()},
                  "median phase ms", np.median(durs, axis=0).round(0).tolist(), f"{time.perf_counter()-t:.2f}s")
