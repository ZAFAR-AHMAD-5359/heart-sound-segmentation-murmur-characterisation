"""Recompute the nine class-median systolic Welch PSDs and the 9x9 Bhattacharyya
matrix for Figure 2, from the same 135 curated CirCor recordings and with the
same code path as PCG_Spectral_Overlap_Analysis.ipynb (Complete Thesis Book/PSD
Analysis). Output: figures_r1/out/fig2_psd_data.npz plus a printed check of the
recomputed Bhattacharyya matrix against spectral_analysis_results.json.

Reproduced settings: fs 4000 Hz, 4th-order Butterworth band-pass 20-800 Hz
(sosfiltfilt), systolic segments = Springer TSV label 2 (min 50 ms) concatenated,
Welch (hann, nperseg 512, noverlap 256, nfft 2048, density), dB = 10 log10(P+1e-14),
class curve = per-frequency MEDIAN over recordings, groups larger than 15 sampled
with DataFrame.sample(15, random_state=42) in metadata order (as in the notebook).
"""
import os, json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch, butter, sosfiltfilt

ROOT = r"E:/NUST MS Thesis/Complete Thesis Book/PSD Analysis"
DATA = os.path.join(ROOT, "PSD_Analysis_CirCor")
AUDIO = os.path.join(DATA, "audio")
META = os.path.join(DATA, "metadata.csv")
REF_JSON = os.path.join(ROOT, "PSD Anlaysis-Results", "figures", "spectral_analysis_results.json")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

FS, FMIN, FMAX_PLOT, FMAX_LOAD = 4000, 20, 500, 800
NFFT, NPERSEG, NOVERLAP, MAX_N, MIN_SYS_S = 2048, 512, 256, 15, 0.05

meta = pd.read_csv(META)
meta["wav_path"] = meta["filename"].apply(lambda f: os.path.join(AUDIO, f))
meta["tsv_path"] = meta["tsv_file"].apply(lambda f: os.path.join(AUDIO, f) if pd.notna(f) and f != "" else "")
meta["wav_exists"] = meta["wav_path"].apply(os.path.exists)
assert meta["wav_exists"].all(), "missing WAV files"


def get_group(key, max_n=MAX_N):
    mask = meta["analysis_groups"].apply(lambda s: key in str(s).split("|"))
    sub = meta[mask & meta["wav_exists"]].copy()
    if max_n and len(sub) > max_n:
        sub = sub.sample(max_n, random_state=42)
    return sub


def bandpass(y, fs=FS, lo=FMIN, hi=FMAX_LOAD, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, y)


def load_systolic(wav_path, tsv_path, fs=FS, min_dur=MIN_SYS_S):
    # soundfile.read at the native 4000 Hz gives the same float32 [-1, 1] samples that
    # librosa.load(sr=4000) returned in the notebook (no resampling is involved).
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    assert sr == fs, (wav_path, sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    y = y - y.mean()
    if tsv_path and os.path.exists(tsv_path):
        tsv = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "label"])
        segs = []
        for _, row in tsv[tsv["label"] == 2].iterrows():
            if row["end"] - row["start"] < min_dur:
                continue
            s = max(0, int(row["start"] * fs)); e = min(len(y), int(row["end"] * fs))
            if e > s:
                segs.append(y[s:e])
        if segs:
            return np.concatenate(segs)
    n = int(3.0 * fs); c = len(y) // 2
    seg = y[max(0, c - n // 2): c + n // 2]
    return np.pad(seg, (0, max(0, n - len(seg))))


def welch_db(y, fs=FS):
    f, p = welch(y, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, scaling="density")
    return f, 10 * np.log10(p + 1e-14)


def bhattacharyya(p_db, q_db):
    p = 10 ** (p_db / 10); p = p / (p.sum() + 1e-20)
    q = 10 ** (q_db / 10); q = q / (q.sum() + 1e-20)
    return float(np.sum(np.sqrt(p * q)))


GROUPS = [  # (metadata key, task, class label)
    ("grade_I", "grade", "I/VI"), ("grade_II", "grade", "II/VI"), ("grade_III", "grade", "III/VI"),
    ("shape_plateau", "shape", "Plateau"), ("shape_decrescendo", "shape", "Decrescendo"), ("shape_diamond", "shape", "Diamond"),
    ("timing_holosystolic", "timing", "Holosystolic"), ("timing_early", "timing", "Early-systolic"), ("timing_mid", "timing", "Mid-systolic"),
]

freqs, medians, q25s, q75s, ns, files = None, [], [], [], [], []
for key, task, label in GROUPS:
    rows = get_group(key)
    psds = []
    for _, r in rows.iterrows():
        y = bandpass(load_systolic(r["wav_path"], r["tsv_path"]))
        f, p = welch_db(y)
        psds.append(p)
    arr = np.vstack(psds)
    freqs = f
    medians.append(np.median(arr, 0)); q25s.append(np.percentile(arr, 25, 0)); q75s.append(np.percentile(arr, 75, 0))
    ns.append(len(rows)); files.append("|".join(rows["filename"].tolist()))
    print(f"{label:<15} n={len(rows)}")

medians, q25s, q75s = np.array(medians), np.array(q25s), np.array(q75s)
m = (freqs >= FMIN) & (freqs <= FMAX_PLOT)
bc = np.array([[bhattacharyya(medians[i][m], medians[j][m]) for j in range(9)] for i in range(9)])

ref = json.load(open(REF_JSON))
ref_bc = np.array(ref["bhattacharyya"]["matrix"])
diff = np.abs(bc - ref_bc)
print("\nBhattacharyya check against spectral_analysis_results.json")
print("  max |difference| = %.6f   (values off-diagonal: %.4f to %.4f)" % (diff.max(), bc[~np.eye(9, dtype=bool)].min(), bc[~np.eye(9, dtype=bool)].max()))
print("  reference group sizes:", ref["group_sizes"])

np.savez(os.path.join(OUT, "fig2_psd_data.npz"), freqs=freqs, median_db=medians, q25_db=q25s, q75_db=q75s,
         n=np.array(ns), labels=np.array([g[2] for g in GROUPS]), tasks=np.array([g[1] for g in GROUPS]),
         files=np.array(files), bc=bc, bc_reference=ref_bc)
print("saved", os.path.join(OUT, "fig2_psd_data.npz"))
