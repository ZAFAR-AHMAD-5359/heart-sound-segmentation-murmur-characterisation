"""Figure 4: the bounded tanh output head clips synthetic R-peaks.

Example: EPHNOGRAM test chunk ECGPCG0060_chunk078 (subject S019, exercise), the chunk
the original notebook selected (tallest recorded R-peak below 4.5 SD: 4.49 SD at
sample 2500). Panel A: context around that R-peak; panel B: 150 ms detail.

Data provenance (two paths, chosen automatically):
  1. If the Trail-02 checkpoint trial2_checkpoint_epoch_150.pt is found (CKPT below or
     the environment variable TRAIL02_CKPT), the synthetic trace is regenerated with the
     original generator code (trail02_generator.py) and both panels carry both traces.
  2. Otherwise (the checkpoint lives only on the Colab Drive):
     - the recorded ECG comes from the raw chunk (z-scored, polarity-fixed, as in the
       notebook), so panel A shows the recorded trace with the tanh bounds and the
       amplitude above the bound shaded, without a synthetic trace;
     - panel B re-plots the synthetic trace from the vector paths embedded in the
       current figure PDF (fig_zoomed_rpeak_clipping.pdf), calibrated with its tick
       marks; the recorded path extracted the same way is checked against the raw
       chunk and must agree to within 0.03 SD.
No signal is synthesised or approximated from a raster.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks
import ehj_style as st

RAW = r"E:/NUST MS Thesis/Refine-Chunked/ECGPCG0060_chunk078.npz"
PDF = r"E:/NUST MS Thesis/Final Draft/Manuescript-In Progress/Advanced-Figures/Figures_Clipping/fig_zoomed_rpeak_clipping.pdf"
CKPT = os.environ.get("TRAIL02_CKPT") or next(iter(
    [p for pat in (r"D:/NUST-Drive Data/**/trial2_checkpoint_epoch_150.pt",
                   r"E:/NUST MS Thesis/**/trial2_checkpoint_epoch_150.pt")
     for p in glob.glob(pat, recursive=True)]), None)
FS = 8000
st.apply()

# ---- recorded ECG from the raw chunk, notebook preprocessing -------------------------
d = np.load(RAW, allow_pickle=True)
ecg = d["ecg"].astype(np.float32)
if np.abs(ecg.min()) > np.abs(ecg.max()):
    ecg = -ecg
rec = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
pcg = d["pcg"].astype(np.float32)
pcg = (pcg - pcg.mean()) / (pcg.std() + 1e-8)
peaks, _ = find_peaks(rec, height=1.0, distance=int(0.3 * FS))
pk = int(peaks[np.argmax(rec[peaks])])
print("chunk %s | subject %s | %s | R-peak sample %d, %.2f SD" % (os.path.basename(RAW), d["subject_id"], d["scenario"], pk, rec[pk]))

ctx_s, ctx_e = max(0, pk - FS // 2), min(len(rec), pk + FS // 2)
det_s, det_e = pk - int(0.075 * FS), pk + int(0.075 * FS)
tA = (np.arange(ctx_s, ctx_e) - pk) / FS * 1000
tB = (np.arange(det_s, det_e) - pk) / FS * 1000

# ---- the traces plotted in the current figure, read back from its vector PDF -----------
# Used as a provenance cross-check of the regenerated trace (and as the fallback source
# for panel B when the checkpoint is absent).
paths = {}
if os.path.exists(PDF):
    import fitz
    page = fitz.open(PDF)[0]
    # calibration from the tick marks of the original panel (b): x 0 ms -> 349.44 pt,
    # 150 ms -> 491.69 pt; y 0 -> 111.18 pt, 16.23 pt per SD (PDF y grows downward)
    X0, X150, Y0, YPU = 349.44, 491.69, 111.18, 16.23
    for dr in page.get_drawings():
        nseg = sum(1 for it in dr["items"] if it[0] == "l")
        if nseg > 300 and dr.get("fill") is None:
            c = dr["color"]
            key = "recorded" if (c[0] < 0.2 and c[2] < 0.2) else "synthetic"
            pts = [dr["items"][0][1]] + [it[2] for it in dr["items"] if it[0] == "l"]
            xy = np.array([[p.x, p.y] for p in pts])
            paths[key] = ((xy[:, 0] - X0) / (X150 - X0) * 150.0 - 75.0, (Y0 - xy[:, 1]) / YPU)
    rx, ry = paths["recorded"]
    dev = np.abs(np.interp(rx, tB, rec[det_s:det_e]) - ry)
    print("PDF-extracted recorded path vs raw chunk: max deviation %.3f SD (calibration check)" % dev.max())
    assert dev.max() < 0.03, "calibration check failed"

# ---- synthetic trace ---------------------------------------------------------------
syn_full = None
if CKPT and os.path.exists(CKPT):
    import torch
    from trail02_generator import ImprovedGenerator
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("generator_state_dict") or ck.get("model_state_dict") or ck.get("state_dict") or ck
    gen = ImprovedGenerator()
    missing, unexpected = gen.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    gen.eval()
    with torch.no_grad():
        x = torch.from_numpy(pcg)[None, None, :]
        sc = torch.zeros(1, 3); sc[0, 1] = 1.0          # exercise, as in the notebook
        syn_full = gen(x, sc).numpy().squeeze().astype(float)
    print("synthetic trace regenerated from %s (epoch %s); max |output| %.3f" % (CKPT, ck.get("epoch") if isinstance(ck, dict) else "?", np.abs(syn_full).max()))
    synA, synB, tBg = syn_full[ctx_s:ctx_e], syn_full[det_s:det_e], tB
    if "synthetic" in paths:
        gx, gy = paths["synthetic"]
        dev = np.abs(np.interp(gx, tB, synB) - gy)
        print("regenerated synthetic trace vs the trace plotted in the current figure: max deviation %.3f SD" % dev.max())
else:
    assert "synthetic" in paths, "neither the checkpoint nor the current PDF is available"
    tBg, synB = paths["synthetic"]
    synA = None
    print("checkpoint not found: panel B synthetic trace taken from the vector paths of the current PDF (%d vertices)" % len(synB))

# ---- figure ------------------------------------------------------------------------
W, H = st.W_DOUBLE, 2.75
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], left=0.062, right=0.985, top=0.905, bottom=0.30, wspace=0.20)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
LW_R, LW_S, BOUND = 1.2, 1.2, dict(color=st.GREY1, lw=0.8, ls=(0, (4, 2.2)))
SHADE = dict(color=st.GREY2, alpha=0.45, lw=0)

# A: context
recA = rec[ctx_s:ctx_e]
axA.axvspan(tB[0], tB[-1], color=st.BAND, lw=0, zorder=0)
for y in (-1, 1):
    axA.axhline(y, zorder=1, **BOUND)
axA.fill_between(tA, 1.0, recA, where=recA > 1.0, zorder=2, **SHADE)
axA.plot(tA, recA, color=st.BLUE, lw=LW_R, zorder=3)
if synA is not None:
    axA.plot(tA, synA, color=st.AMBER, lw=LW_S, zorder=4)
axA.set_xlim(tA[0], tA[-1])
axA.set_ylim(-2.0, 5.0)
axA.set_yticks([-2, -1, 0, 1, 2, 3, 4, 5])
axA.set_xticks([-300, -150, 0, 150, 300, 450])
axA.set_xlabel("Time relative to the R-peak (ms)")
axA.set_ylabel("Amplitude (SD)")
axA.text(tA[-1], 1.08, "tanh bound +1", ha="right", va="bottom", fontsize=8, color=st.GREY1)
axA.text(tA[-1], -1.10, "tanh bound −1", ha="right", va="top", fontsize=8, color=st.GREY1)
axA.text(0.5 * (tB[0] + tB[-1]), -1.92, "B", ha="center", va="bottom", fontsize=8, fontweight="bold", color=st.GREY1)

# B: detail
recB = rec[det_s:det_e]
syn_on_grid = np.interp(tB, tBg, synB)
axB.axhline(1.0, zorder=1, **BOUND)
axB.fill_between(tB, 1.0, recB, where=recB > 1.0, zorder=2, **SHADE)
axB.plot(tB, recB, color=st.BLUE, lw=LW_R, zorder=3)
axB.plot(tBg, synB, color=st.AMBER, lw=LW_S, zorder=4)
axB.set_xlim(-75, 75)
axB.set_ylim(-2.0, 5.0)
axB.set_yticks([-2, -1, 0, 1, 2, 3, 4, 5])
axB.set_xticks([-75, -50, -25, 0, 25, 50, 75])
axB.set_xlabel("Time relative to the R-peak (ms)")
ipk = int(np.argmax(recB))
axB.annotate("recorded peak\n%.1f SD" % recB[ipk], xy=(tB[ipk], recB[ipk]), xytext=(-70, 4.35),
             ha="left", va="center", fontsize=8, color=st.INK, linespacing=1.1,
             arrowprops=dict(arrowstyle="-", color=st.GREY1, lw=0.6, shrinkA=0, shrinkB=2))
flat = tBg[synB >= 0.985]
axB.annotate("synthetic held\nat %.2f" % synB.max(), xy=(np.median(flat), 1.0), xytext=(28, 2.7),
             ha="left", va="center", fontsize=8, color=st.INK, linespacing=1.1,
             arrowprops=dict(arrowstyle="-", color=st.GREY1, lw=0.6, shrinkA=0, shrinkB=2))

handles = [Line2D([], [], color=st.BLUE, lw=LW_R, label="Recorded"),
           Line2D([], [], color=st.AMBER, lw=LW_S, label="Synthetic"),
           Line2D([], [], label="tanh bounds", **BOUND),
           Patch(facecolor=st.GREY2, alpha=0.45, label="recorded amplitude above the bound")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), columnspacing=1.6)

st.add_panel_letters(fig, [(axA, "A"), (axB, "B")])
st.save(fig, "fig4_tanh_clipping")
