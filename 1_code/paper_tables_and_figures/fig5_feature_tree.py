"""Figure 5: the 464-feature vector as a tree, drawn with matplotlib.

Counts are taken from the released extractor (Submission/materials/code/src/features.py,
function extract_all_v4_features) and checked in this script:
  recording-level 196 = 160 + 26 + 6 + 4
  soft detection   10 = 3 + 2 + 2 + 3
  phase-specific  258 = 125 per-cycle quantities x (median, SD across cycles) + 8
The signal-processing footer quotes the CONFIG block of the same file.
Leaf text is wrapped to the measured box width, so no word can break or overflow.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
import ehj_style as st

st.apply()
FP, FPB = st.font(), st.font(bold=True)      # Arial, set once in ehj_style
PT2MM = 25.4 / 72


def text_w(s, size=8, bold=False):
    """Rendered width of one line in mm."""
    if not s:
        return 0.0
    return TextPath((0, 0), s, size=size, prop=FPB if bold else FP).get_extents().width * PT2MM


def wrap(s, width_mm, size=8):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_w(trial, size) <= width_mm or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    assert all(text_w(l, size) <= width_mm + 0.01 for l in lines), ("word too long for box", s, width_mm)
    return "\n".join(lines)


# ---- content -----------------------------------------------------------------------
REC = [("40 MFCC: mean, SD, min, max", 160),
       ("13 delta MFCC: mean, SD", 26),
       ("Spectral centroid, bandwidth, rolloff, flatness", 6),
       ("Zero-crossing rate and RMS: mean, SD", 4)]
SOFT = [("Adaptive SNR in three bands", 3),
        ("Spectral and pitch stability", 2),
        ("Harmonic ratio and energy variance", 2),
        ("Frequency-band energy ratios", 3)]
CYCLE = [("Phase MFCC, 13 × 4 phases", 52),
         ("MFCC distances between phases", 4),
         ("Systolic envelope timing, shape, 10 envelope samples", 20),
         ("Phase amplitude, SNR, energy", 13),
         ("Phase-boundary RMS and continuity", 4),
         ("Cross-phase energy ratios", 6),
         ("Wavelet band energies, systole", 7),
         ("Systolic subphase energies", 7),
         ("Phase durations and heart rate", 5),
         ("Diastolic descriptors", 7)]
VAR = ("Cycle-to-cycle variability", 8)
NODE = ("125 per-cycle quantities, each summarised as its median and SD across cycles", 250)

n_rec, n_soft, n_cycle = sum(c for _, c in REC), sum(c for _, c in SOFT), sum(c for _, c in CYCLE)
n_phase = 2 * n_cycle + VAR[1]
total = n_rec + n_soft + n_phase
print("recording-level %d | soft detection %d | per-cycle %d x 2 + %d = %d | total %d"
      % (n_rec, n_soft, n_cycle, VAR[1], n_phase, total))
assert (n_rec, n_soft, n_cycle, n_phase, total) == (196, 10, 125, 258, 464), "feature counts do not add up"

FOOTER = ("Signal processing: 4 kHz sampling; recording-level MFCC from a 2048-sample FFT and phase MFCC from a "
          "512-sample FFT; envelopes from a 4th-order band-pass (detection 20 to 800 Hz, characterisation 50 to 400 Hz), "
          "rectification, 10 Hz low-pass, and 50 ms Gaussian smoothing; wavelet Daubechies-4 with 5 levels; "
          "systolic subphases early 0 to 40%, mid 20 to 80%, late 60 to 100% of systole.")

# ---- geometry (mm) ---------------------------------------------------------------
WMM = 180.0
LINE = dict(color=st.OUTLINE, lw=0.6, solid_capstyle="butt")
PAD, CNT_W, LH, IND, GAP = 1.6, 6.0, 3.3, 3.0, 1.4   # text pad, count column, line height, indent, leaf gap

# layout plan computed first so the figure height follows the content
COLS = [(2.0, 43.0, "Recording-level\n196 features", REC),
        (48.0, 41.0, "Soft detection\n10 features", SOFT),
        (92.0, 86.0, "Phase-specific\n258 features", None)]


def leaf_height(lines):
    return LH * lines + 2.9


def prep(items, box_w):
    out = []
    for text, cnt in items:
        t = wrap(text, box_w - 2 * PAD - CNT_W)
        out.append((t, cnt, leaf_height(t.count("\n") + 1)))
    return out


rec_items = prep(REC, COLS[0][1] - IND)
soft_items = prep(SOFT, COLS[1][1] - IND)
node_text = wrap(NODE[0], COLS[2][1] - IND - 2 * PAD - CNT_W)
node_h = leaf_height(node_text.count("\n") + 1)
cw = (COLS[2][1] - 2 * IND - 2.0) / 2
cyc1, cyc2 = prep(CYCLE[:5], cw), prep(CYCLE[5:], cw)
# equalise row heights across the two phase columns so the rows align
rows = [max(a[2], b[2]) for a, b in zip(cyc1, cyc2)]
cyc1 = [(t, c, h) for (t, c, _), h in zip(cyc1, rows)]
cyc2 = [(t, c, h) for (t, c, _), h in zip(cyc2, rows)]
var_text = wrap(VAR[0], COLS[2][1] - IND - 2 * PAD - CNT_W)
var_h = leaf_height(var_text.count("\n") + 1)

RH, GH, TOP, ROOT_GAP, GROUP_GAP = 8.5, 10.0, 2.5, 8.0, 3.0
stack_h = max(sum(h for *_, h in rec_items) + GAP * (len(rec_items) - 1),
              sum(h for *_, h in soft_items) + GAP * (len(soft_items) - 1),
              node_h + 3.0 + sum(rows) + GAP * (len(rows) - 1) + 2.0 + var_h)
footer_lines = wrap(FOOTER, WMM - 6.0).count("\n") + 1
FOOT_H = 3.67 * footer_lines + 0.8     # 8 pt lines at 1.3 spacing, plus the rule gap
HMM = TOP + RH + ROOT_GAP + GH + GROUP_GAP + stack_h + 4.0 + FOOT_H + 1.5
fig = plt.figure(figsize=(WMM * st.MM, HMM * st.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WMM); ax.set_ylim(0, HMM); ax.axis("off")


def box(x, y_top, w, h, text, count=None, bold=False, fill="white", size=8):
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                facecolor=fill, edgecolor=st.OUTLINE, lw=0.6, zorder=2))
    yc = y_top - h / 2
    if count is None:
        ax.text(x + w / 2, yc, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=st.INK, zorder=3, linespacing=1.15)
    else:
        ax.text(x + PAD, yc, text, ha="left", va="center", fontsize=size, color=st.INK, zorder=3, linespacing=1.15)
        ax.text(x + w - PAD, yc, str(count), ha="right", va="center", fontsize=size, fontweight="bold",
                color=st.INK, zorder=3)


def stack(x, y_top, w, items):
    y, centres = y_top, []
    for text, cnt, h in items:
        box(x, y, w, h, text, cnt)
        centres.append(y - h / 2)
        y -= h + GAP
    return centres, y + GAP


# root and bus
RW = 54.0
rx, ry = (WMM - RW) / 2, HMM - TOP
box(rx, ry, RW, RH, "464 features per recording", bold=True, size=9, fill="#ededed")
GY = ry - RH - ROOT_GAP
bus_y = ry - RH - ROOT_GAP / 2
ax.plot([rx + RW / 2, rx + RW / 2], [ry - RH, bus_y], **LINE)
ax.plot([COLS[0][0] + COLS[0][1] / 2, COLS[2][0] + COLS[2][1] / 2], [bus_y, bus_y], **LINE)
for x, w, title, _ in COLS:
    ax.plot([x + w / 2, x + w / 2], [bus_y, GY], **LINE)
    box(x, GY, w, GH, title, bold=True, size=8.5, fill="#ededed")

# recording-level and soft-detection columns
top = GY - GH - GROUP_GAP
for (x, w, _, _), items in zip(COLS[:2], (rec_items, soft_items)):
    centres, _ = stack(x + IND, top, w - IND, items)
    ax.plot([x + IND / 2, x + IND / 2], [GY - GH, centres[-1]], **LINE)
    for yc in centres:
        ax.plot([x + IND / 2, x + IND], [yc, yc], **LINE)

# phase-specific column: node (250) with two leaf columns, then the variability leaf
x, w = COLS[2][0], COLS[2][1]
box(x + IND, top, w - IND, node_h, node_text, NODE[1])
node_c = top - node_h / 2
lt = top - node_h - 3.0
c1, b1 = stack(x + 2 * IND, lt, cw, cyc1)
c2, b2 = stack(x + 2 * IND + cw + 2.0, lt, cw, cyc2)
sp1 = x + IND + IND / 2
ax.plot([sp1, sp1], [top - node_h, c1[-1]], **LINE)
for yc in c1:
    ax.plot([sp1, x + 2 * IND], [yc, yc], **LINE)
sp2 = x + 2 * IND + cw + 1.0
ax.plot([sp2, sp2], [top - node_h, c2[-1]], **LINE)
for yc in c2:
    ax.plot([sp2, x + 2 * IND + cw + 2.0], [yc, yc], **LINE)
vy = min(b1, b2) - 2.0
box(x + IND, vy, w - IND, var_h, var_text, VAR[1])
var_c = vy - var_h / 2
ax.plot([x + IND / 2, x + IND / 2], [GY - GH, var_c], **LINE)
for yc in (node_c, var_c):
    ax.plot([x + IND / 2, x + IND], [yc, yc], **LINE)

# footer
fy = FOOT_H + 1.5
ax.plot([3, WMM - 3], [fy, fy], color=st.GREY2, lw=0.6)
ax.text(3, fy - 1.2, wrap(FOOTER, WMM - 6.0), ha="left", va="top", fontsize=8, color=st.GREY1, linespacing=1.3)

print("figure %.0f x %.0f mm" % (WMM, HMM))
st.save(fig, "fig5_feature_tree")
