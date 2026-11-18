"""Figure 2: spectral overlap of systolic phonocardiogram spectra across the nine
grade, shape, and timing classes (103 murmur-present recordings in 134 group
slots, drawn from the curated CirCor set of 135; its 20 murmur-absent recordings enter
no curve and no coefficient).

Panel A: nine class-median Welch PSD curves, 20 to 500 Hz, one hue family per task
(blues = grade, ambers = shape, greys = timing) with linestyle as the second encoding
and direct labels at the right end; the 100 to 300 Hz core band is shaded.
Panel B: the 36 pairwise Bhattacharyya coefficients as a lower-triangle matrix.

Data: figures_r1/out/fig2_psd_data.npz, written by fig2_psd_compute.py from the same
recordings and code path as the original notebook; its Bhattacharyya matrix matches
spectral_analysis_results.json to 6 decimals, so the curves are the published ones.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
import ehj_style as st

st.apply()
D = np.load(os.path.join(st.OUT, "fig2_psd_data.npz"))
f, P, labels, tasks, BC = D["freqs"], D["median_db"], list(D["labels"]), list(D["tasks"]), D["bc"]
m = (f >= 20) & (f <= 500)
f, P = f[m], P[:, m]

NAMES = {"I/VI": "Grade I/VI", "II/VI": "Grade II/VI", "III/VI": "Grade III/VI"}
name = [NAMES.get(l, l) for l in labels]

# one hue family per task, light to dark within the family; linestyle per task
FAMILY = {
    "grade":  (["#6f9bd1", "#4475b3", "#184f95"], "-"),
    "shape":  (["#eda100", "#b37600", "#7a4f00"], (0, (4.0, 2.0))),
    "timing": (["#9e9e9e", "#6e6e6e", "#3d3d3d"], (0, (1.0, 1.7))),
}
col, ls = [], []
for task in ("grade", "shape", "timing"):
    idx = [i for i, t in enumerate(tasks) if t == task]
    for k, i in enumerate(idx):
        col.append(FAMILY[task][0][k]); ls.append(FAMILY[task][1])
order = [i for task in ("grade", "shape", "timing") for i in range(9) if tasks[i] == task]
assert order == list(range(9))

# ---------------------------------------------------------------------------------
W, H = st.W_DOUBLE, 3.65
fig = plt.figure(figsize=(W, H))
# panel B: 8 x 8 lower triangle, cells CELL_W x CELL_H mm (wide enough for "0.987" at
# 8 pt), labelled along the diagonal so the empty upper-right half carries the class
# names and no label competes with panel A. Widths are measured so both panels fit.
PT2MM = 25.4 / 72


def text_mm(txt):
    return TextPath((0, 0), txt, size=8, prop=st.font()).get_extents().width * PT2MM


CELL_W, CELL_H, WMM = 8.2, 7.4, 180.0
reach_B = max((k + 0.12) * CELL_W + text_mm(name[k]) for k in range(9))   # from the left edge of B
B_LEFT = WMM - 1.5 - reach_B
A_LEFT, LAB_AT = 0.072 * WMM, 517.0            # panel A: left edge (mm), label start (Hz)
A_W = (B_LEFT - 4.0 - max(text_mm(nm) for nm in name) - A_LEFT) / (1 + (LAB_AT - 500.0) / 480.0)
axA = fig.add_axes([A_LEFT / WMM, 0.125, A_W / WMM, 0.815])
bw = 8 * CELL_W / WMM
bh = 8 * CELL_H / (H * 25.4)
axB = fig.add_axes([B_LEFT / WMM, 0.86 - bh, bw, bh])
print("panel A axes %.1f mm wide; panel B starts at %.1f mm, cells %.1f x %.1f mm" % (A_W, B_LEFT, CELL_W, CELL_H))

# ---- A: PSD curves ----------------------------------------------------------------
axA.axvspan(100, 300, color="#efefef", lw=0, zorder=0)
ylo, yhi = -78, -40
for i in range(9):
    axA.plot(f, P[i], color=col[i], lw=1.2, ls=ls[i], zorder=3 if tasks[i] == "grade" else 2)
axA.set_xlim(20, 500)
axA.set_ylim(ylo, yhi)
axA.set_xticks([20, 100, 200, 300, 400, 500])
axA.set_yticks([-75, -65, -55, -45])
axA.set_xlabel("Frequency (Hz)")
axA.set_ylabel("PSD (dB/Hz)")
axA.text(200, yhi - 0.6, "100 to 300 Hz core band", ha="center", va="top", fontsize=8, color=st.GREY1)

# direct labels at the right end, pushed apart so they cannot overlap
y_end = np.array([P[i][-1] for i in range(9)])
y_anchor = np.array([P[i][f >= 480].mean() for i in range(9)])
ax_h_in = H * 0.815
gap = 9.5 / 72 / ax_h_in * (yhi - ylo)          # 9.5 pt in data units
srt = np.argsort(y_anchor)
y_lab = y_anchor[srt].copy()
for k in range(1, 9):
    if y_lab[k] - y_lab[k - 1] < gap:
        y_lab[k] = y_lab[k - 1] + gap
if y_lab[-1] > yhi - gap / 2:
    y_lab[-1] = yhi - gap / 2
    for k in range(7, -1, -1):
        if y_lab[k + 1] - y_lab[k] < gap:
            y_lab[k] = y_lab[k + 1] - gap
yl = np.empty(9); yl[srt] = y_lab
for i in range(9):
    axA.plot([500, 513], [y_end[i], yl[i]], color=col[i], lw=0.8, ls="-", clip_on=False, zorder=4)
    axA.text(LAB_AT, yl[i], name[i], ha="left", va="center", fontsize=8, color=st.INK, clip_on=False)

# a three-entry key for the linestyle (task) encoding; class names are on the curves
axA.legend(handles=[Line2D([], [], color=st.GREY1, lw=1.2, ls=FAMILY["grade"][1], label="grade"),
                    Line2D([], [], color=st.GREY1, lw=1.2, ls=FAMILY["shape"][1], label="shape"),
                    Line2D([], [], color=st.GREY1, lw=1.2, ls=FAMILY["timing"][1], label="timing")],
           loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, handlelength=2.4, labelspacing=0.35,
           borderaxespad=0.2)

# ---- B: Bhattacharyya lower triangle ---------------------------------------------
n = 9
cmap = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#184f95"])
vmin, vmax = 0.93, 1.0
# one vector rectangle per cell (keeps the PDF free of raster content)
for i in range(1, n):
    for j in range(i):
        v = BC[i, j]
        axB.add_patch(Rectangle((j - 0.5, i - 1.5), 1, 1, facecolor=cmap((v - vmin) / (vmax - vmin)),
                                edgecolor="white", lw=0.4))
        assert text_mm("%.3f" % v) <= CELL_W - 0.8, "coefficient wider than its cell"
        axB.text(j, i - 1, "%.3f" % v, ha="center", va="center", fontsize=8,
                 color="white" if v > 0.975 else st.INK)
# class names on the diagonal: each names the row to its left and the column below it
for k in range(n):
    axB.text(k - 0.5 + 0.12, k - 1, name[k], ha="left", va="center", fontsize=8, color=st.INK, clip_on=False)
axB.set_xticks([]); axB.set_yticks([])
for sp in axB.spines.values():
    sp.set_visible(False)
axB.set_xlim(-0.5, n - 1.5); axB.set_ylim(n - 1.5, -0.5)
axB.text(n - 1.5, -1.0, "Bhattacharyya coefficient", ha="right", va="center", fontsize=8,
         color=st.GREY1, clip_on=False)

off = np.array([BC[i, j] for i in range(n) for j in range(i)])
print("BC off-diagonal range %.3f to %.3f (n = %d pairs)" % (off.min(), off.max(), len(off)))

st.add_panel_letters(fig, [(axA, "A"), (axB, "B")], dy_pt=3.0, same_height=True)
st.save(fig, "fig2_psd")
