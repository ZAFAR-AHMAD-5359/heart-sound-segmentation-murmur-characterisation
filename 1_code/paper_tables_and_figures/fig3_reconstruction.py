"""Figure 3: synthetic ECG against the recorded ECG, held-out EPHNOGRAM chunk
ECGPCG0001_chunk181 (subject S023, exercise). Re-plotted from the arrays saved by the
original figure notebook (Draft-1/COLAB_IEEE_FIGURES.ipynb, cell "Save raw data for the
best sample"): ecg_ground_truth.npy (z-scored, polarity-fixed recorded ECG) and
pred_Proposed.npy (generator output), both 64 000 samples at 8 kHz.
Window selection is the notebook's rule: the 2 s window (stride 0.5 s) that holds the
tallest R-peak; panel B expands 400 ms centred on that R-peak.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import ehj_style as st

SRC = r"E:/NUST MS Thesis/Draft-1/Images/EUSIPCO_2026_Figures/raw_predictions"
FS = 8000
st.apply()

rec = np.load(os.path.join(SRC, "ecg_ground_truth.npy")).astype(float)
syn = np.load(os.path.join(SRC, "pred_Proposed.npy")).astype(float)
meta = json.load(open(os.path.join(SRC, "metrics.json")))
m = meta["metrics"]["Proposed"]
print("chunk:", meta["fname"], "| subject", meta["subject"], "| metrics (whole 8 s chunk):",
      "PSNR %.2f dB, r %.3f, SSIM %.3f" % (m["psnr"], m["corr"], m["ssim"]))

# --- window selection, identical to the notebook ---------------------------------
win = 2 * FS
best_s, best_pk = 0, -np.inf
for s in range(0, len(rec) - win, FS // 2):
    pk = rec[s:s + win].max()
    if pk > best_pk:
        best_pk, best_s = pk, s
s, e = best_s, best_s + win
t = np.arange(win) / FS
rp = int(np.argmax(rec[s:e]))
zh = int(0.2 * FS)
zs, ze = max(0, rp - zh), min(win, rp + zh)
tz = np.arange(ze - zs) / FS * 1000
print("2 s window starts at %.2f s; R-peak at %.3f s inside it; zoom length %.0f ms"
      % (s / FS, rp / FS, (ze - zs) / FS * 1000))

# metrics on the plotted 2 s segment, for the README (the quoted values are chunk-level)
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None
a, b = rec[s:e], syn[s:e]
psnr_seg = 20 * np.log10(np.max(np.abs(a)) / np.sqrt(np.mean((a - b) ** 2)))
print("2 s segment: PSNR %.2f dB, r %.3f, SSIM %s" % (psnr_seg, pearsonr(a, b)[0],
      "%.3f" % ssim(a, b, data_range=a.max() - a.min()) if ssim else "(scikit-image not installed)"))

# --- figure --------------------------------------------------------------------------
# heights in inches: top margin (panel letters), axes, bottom block (ticks, axis label,
# gap, colour key); the page height follows from them, so no blank strip is left
TOP_IN, AX_IN, BOT_IN = 0.26, 1.47, 0.575
W, H = st.W_DOUBLE, TOP_IN + AX_IN + BOT_IN
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], left=0.062, right=0.982,
                      top=1 - TOP_IN / H, bottom=BOT_IN / H, wspace=0.16)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

LW_R, LW_S = 1.3, 1.2
axA.axvspan(zs / FS, ze / FS, color=st.BAND, lw=0, zorder=0)
axA.plot(t, rec[s:e], color=st.BLUE, lw=LW_R, zorder=2)
axA.plot(t, syn[s:e], color=st.AMBER, lw=LW_S, zorder=3)
axA.set_xlim(0, 2)
axA.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axA.set_xlabel("Time (s)")
axA.set_ylabel("Amplitude (SD)")

axB.plot(tz, rec[s:e][zs:ze], color=st.BLUE, lw=LW_R, zorder=2)
axB.plot(tz, syn[s:e][zs:ze], color=st.AMBER, lw=LW_S, zorder=3)
axB.set_xlim(0, 400)
axB.set_xticks([0, 100, 200, 300, 400])
axB.set_xlabel("Time (ms)")

lo = min(rec[s:e].min(), syn[s:e].min()) - 0.4
hi = max(rec[s:e].max(), syn[s:e].max()) + 0.4
for ax in (axA, axB):
    ax.set_ylim(lo, hi)
    ax.set_yticks([-4, -2, 0, 2, 4])
axB.tick_params(labelleft=False)

# one colour key for both panels, KEY_GAP_PT below the lowest axis label
KEY_GAP_PT = 6.0
fig.canvas.draw()
r = fig.canvas.get_renderer()
label_bottom = min(ax.xaxis.label.get_window_extent(r).y0 for ax in (axA, axB))
y_key = (label_bottom - KEY_GAP_PT * fig.dpi / 72) / (H * fig.dpi)
leg = fig.legend(handles=[Line2D([], [], color=st.BLUE, lw=LW_R, label="Recorded"),
                          Line2D([], [], color=st.AMBER, lw=LW_S, label="Synthetic")],
                 loc="upper center", ncol=2, bbox_to_anchor=(0.5, y_key), columnspacing=2.0,
                 borderpad=0.0, borderaxespad=0.0)
fig.canvas.draw()
gap_pt = (label_bottom - max(t.get_window_extent(r).y1 for t in leg.get_texts())) * 72 / fig.dpi
print("gap between the axis label and the colour legend: %.1f pt" % gap_pt)
assert gap_pt >= 4.0, "colour legend closer than 4 pt to the axis label"

st.add_panel_letters(fig, [(axA, "A"), (axB, "B")])
st.save(fig, "fig3_reconstruction")
