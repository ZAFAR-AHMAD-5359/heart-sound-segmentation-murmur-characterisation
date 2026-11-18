"""Figure 6: external detection check, Michigan Heart Sound and Murmur Library.

One row per recording (23), a stem from 0 to the maximum murmur probability across the
8 s chunks of that recording, and the decision at the 0.5 threshold (any-chunk rule).
Rows are grouped as murmur types represented in training, murmur types absent from
training, and no murmur; inside a group they run from the highest probability down.

Data. Probabilities and decisions come from the per-case file of the locked run
(percase_michigan_20260615T132950.csv: p_present_max, detection_true, detection_pred);
finding strings and the grouping come from the label file locked before the run
(michigan_labels_locked.csv: finding, murmur_present, detection_target, timing_target).
Every row, the three group counts, and the row order of the figure this script
replaces are asserted below, so the script stops if the data and the drawing differ.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
import ehj_style as st

PERCASE = r"E:/NUST MS Thesis/Final Draft/Coding-Extension/Michigan_Validation/Run Notebook-2/percase_michigan_20260615T132950.csv"
LABELS = r"E:/NUST MS Thesis/Final Draft/Validation/Michigan/michigan_labels_locked.csv"
THRESHOLD = 0.5
DAGGER = chr(0x2020)

GROUPS = [  # (group id, heading)
    ("represented", "Murmur types represented in training"),
    ("absent", "Murmur types absent from training"),
    ("none", "No murmur"),
]
X_LABEL = "Maximum murmur probability across 8 s chunks"
LEGEND_CORRECT, LEGEND_MISSED = "correct", "missed"
LEGEND_THRESHOLD = "decision threshold %.1f" % THRESHOLD
LEGEND_DAGGER = DAGGER + " an added heart sound accompanies the murmur"
# an added heart sound next to the murmur: these words in the locked finding string
ADDED_SOUND = ("click", "s3", "s4", "split s2", "single s2", "opening snap")

# what the figure must contain (read off the figure this script replaces)
EXPECTED_ROWS = {
    "represented": ["06", "07", "09", "15", "23", "11", "22", "12", "21"],
    "absent": ["08", "10", "17", "13", "16"],
    "none": ["04", "01", "02", "03", "05", "14", "18", "19", "20"],
}
EXPECTED_COUNTS = {"represented": (4, 9), "absent": (4, 5), "none": (9, 9)}
EXPECTED_DAGGER = {"10", "11", "12", "13", "21", "22", "23"}

# ---- data ---------------------------------------------------------------------------
run = pd.read_csv(PERCASE, dtype={"id": str})
lab = pd.read_csv(LABELS, dtype={"id": str})
assert len(run) == 23 and len(lab) == 23 and sorted(run["id"]) == sorted(lab["id"]) == ["%02d" % i for i in range(1, 24)]
df = run[["id", "detection_true", "detection_pred", "p_present_max"]].merge(
    lab[["id", "finding", "murmur_present", "detection_target", "timing_target"]], on="id", validate="one_to_one")
df["murmur"] = df["murmur_present"].astype(str).str.upper() == "TRUE"
assert (df["detection_true"] == df["detection_target"]).all(), "run labels differ from the locked labels"
assert ((df["detection_true"] == "Present") == df["murmur"]).all()
assert ((df["p_present_max"] >= THRESHOLD) == (df["detection_pred"] == "Present")).all(), "decision differs from the threshold rule"
df["correct"] = df["detection_pred"] == df["detection_true"]


def group_of(r):
    if not r["murmur"]:
        return "none"
    return "absent" if r["timing_target"] == "OUT_OF_LABEL_SPACE" else "represented"


df["group"] = df.apply(group_of, axis=1)
df["dagger"] = [m and any(w in f.lower() for w in ADDED_SOUND) for m, f in zip(df["murmur"], df["finding"])]
df = df.sort_values(["p_present_max", "id"], ascending=[False, True])

rows = {g: df[df["group"] == g] for g, _ in GROUPS}
counts = {g: (int(rows[g]["correct"].sum()), len(rows[g])) for g, _ in GROUPS}
for g, _ in GROUPS:
    assert list(rows[g]["id"]) == EXPECTED_ROWS[g], (g, list(rows[g]["id"]))
    assert counts[g] == EXPECTED_COUNTS[g], (g, counts[g])
assert set(df.loc[df["dagger"], "id"]) == EXPECTED_DAGGER, sorted(df.loc[df["dagger"], "id"])
tp = int(((df["detection_true"] == "Present") & df["correct"]).sum())
tn = int(((df["detection_true"] == "Absent") & df["correct"]).sum())
assert (tp, int(df["murmur"].sum()), tn, int((~df["murmur"]).sum())) == (8, 14, 9, 9), "sensitivity 8/14 and specificity 9/9 not reproduced"
for g, head in GROUPS:
    print("%s: %d / %d correct" % (head, counts[g][0], counts[g][1]))
    for _, r in rows[g].iterrows():
        print("   %s  p_max %.3f  %-7s -> %-7s %s  %s%s" % (r["id"], r["p_present_max"], r["detection_true"], r["detection_pred"],
                                                       "correct" if r["correct"] else "missed ", r["finding"], " " + DAGGER if r["dagger"] else ""))

# ---- geometry (mm) ------------------------------------------------------------------
st.apply()
PT2MM = 25.4 / 72
FP, FPB = st.font(), st.font(bold=True)
CAP_EM = TextPath((0, 0), "H", size=100, prop=FP).get_extents().height / 100.0


def text_mm(s, size=8, bold=False):
    return TextPath((0, 0), s, size=size, prop=FPB if bold else FP).get_extents().width * PT2MM


def label_of(r):
    return "%s  %s%s" % (r["id"], r["finding"], " " + DAGGER if r["dagger"] else "")


WMM, MARG = 180.0, 2.0
ROW, HEAD, GAP, TOP = 3.9, 4.3, 2.4, 1.5          # row pitch, heading pitch, gap between groups, top margin
LAB_GAP = 1.6                                      # between a row label and the axis
X0, X1 = -0.03, 1.04                               # probability axis limits
lab_w = max([text_mm(label_of(r)) for _, r in df.iterrows()] + [text_mm(h, bold=True) for _, h in GROUPS])
cnt_w = max(text_mm("%d / %d" % c, 9, True) for c in counts.values())
AX_L = MARG + lab_w + LAB_GAP
AX_R = WMM - MARG - cnt_w - 2.5
AX_W = AX_R - AX_L
PLOT_H = sum(HEAD + ROW * len(rows[g]) for g, _ in GROUPS) + GAP * (len(GROUPS) - 1) + 1.0
KEY_GAP_PT, KEY_H, BOTTOM = 6.0, 2.9, 1.5          # axis label to key (at least 4 pt), key height, bottom margin


def mm_to_x(mm):
    return X0 + (mm - AX_L) / AX_W * (X1 - X0)


def draw(bot_mm):
    hmm = TOP + PLOT_H + bot_mm
    fig = plt.figure(figsize=(WMM * st.MM, hmm * st.MM))
    ax = fig.add_axes([AX_L / WMM, bot_mm / hmm, AX_W / WMM, PLOT_H / hmm])
    ax.set_xlim(X0, X1)
    ax.set_ylim(PLOT_H, 0)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(X_LABEL)
    ax.plot([THRESHOLD, THRESHOLD], [0, PLOT_H], color=st.GREY1, lw=0.8, ls=(0, (4, 2.2)), zorder=1,
            solid_capstyle="butt", dash_capstyle="butt")
    base_off = CAP_EM * 8 * PT2MM / 2              # baseline below the row centre, 8 pt text
    y = 0.0
    for g, head in GROUPS:
        yc = y + HEAD / 2
        ax.text(mm_to_x(AX_L - LAB_GAP), yc + base_off, head, ha="right", va="baseline", fontsize=8,
                fontweight="bold", color=st.INK, clip_on=False)
        ax.text(mm_to_x(WMM - MARG), yc + CAP_EM * 9 * PT2MM / 2, "%d / %d" % counts[g], ha="right", va="baseline",
                fontsize=9, fontweight="bold", color=st.INK, clip_on=False)
        y += HEAD
        if g == "none":
            ax.add_patch(Rectangle((X0, y - 0.2), X1 - X0, ROW * len(rows[g]) + 0.4, facecolor=st.BAND,
                                   edgecolor="none", zorder=0))
        for _, r in rows[g].iterrows():
            yc = y + ROW / 2
            p = float(r["p_present_max"])
            col = st.BLUE if r["correct"] else st.AMBER
            if p > 0:
                ax.plot([0, p], [yc, yc], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
            if r["correct"]:
                ax.plot([p], [yc], marker="o", ms=5.0, mfc=st.BLUE, mec="white", mew=0.5, ls="none", zorder=3, clip_on=False)
            else:
                ax.plot([p], [yc], marker="x", ms=5.2, mec=st.AMBER, mew=1.8, ls="none", zorder=3, clip_on=False)
            ax.text(mm_to_x(AX_L - LAB_GAP), yc + base_off, label_of(r), ha="right", va="baseline", fontsize=8,
                    color=st.INK, clip_on=False)
            y += ROW
        y += GAP
    return fig, ax, hmm


# first pass: measure how far the axis label reaches below the axis, then size the bottom block
fig, ax, hmm = draw(20.0)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
px = fig.dpi / 25.4
label_reach = 20.0 - ax.xaxis.label.get_window_extent(rend).y0 / px      # mm from the axis line to the label bottom
plt.close(fig)
BOT = label_reach + KEY_GAP_PT * PT2MM + KEY_H + BOTTOM
fig, ax, hmm = draw(BOT)

# key: one row, centred, KEY_GAP_PT below the axis label; the dagger entry is plain text
ov = fig.add_axes([0, 0, 1, 1])
ov.set_xlim(0, WMM)
ov.set_ylim(hmm, 0)
ov.axis("off")
ov.patch.set_alpha(0.0)
SYM, SYM_GAP, SEP, LINE_W = 3.0, 1.0, 5.5, 7.0
entries = [("circle", LEGEND_CORRECT), ("cross", LEGEND_MISSED), ("line", LEGEND_THRESHOLD), ("text", LEGEND_DAGGER)]
widths = [(LINE_W if k == "line" else 0.0 if k == "text" else SYM) + (0.0 if k == "text" else SYM_GAP) + text_mm(s)
          for k, s in entries]
total = sum(widths) + SEP * (len(entries) - 1)
assert total <= WMM - 2 * MARG, "legend row wider than the figure"
yk = hmm - BOTTOM - KEY_H / 2
x = (WMM - total) / 2
for (k, s), w in zip(entries, widths):
    if k == "circle":
        ov.plot([x + SYM / 2], [yk], marker="o", ms=5.0, mfc=st.BLUE, mec="white", mew=0.5, ls="none")
        tx = x + SYM + SYM_GAP
    elif k == "cross":
        ov.plot([x + SYM / 2], [yk], marker="x", ms=5.2, mec=st.AMBER, mew=1.8, ls="none")
        tx = x + SYM + SYM_GAP
    elif k == "line":
        ov.plot([x, x + LINE_W], [yk, yk], color=st.GREY1, lw=0.8, ls=(0, (4, 2.2)), dash_capstyle="butt")
        tx = x + LINE_W + SYM_GAP
    else:
        tx = x
    ov.text(tx, yk + CAP_EM * 8 * PT2MM / 2, s, ha="left", va="baseline", fontsize=8, color=st.INK)
    x += w + SEP

fig.canvas.draw()
rend = fig.canvas.get_renderer()
label_bottom = ax.xaxis.label.get_window_extent(rend).y0
key_top = max(t.get_window_extent(rend).y1 for t in ov.texts)
gap_pt = (label_bottom - key_top) * 72 / fig.dpi
print("gap between the axis label and the legend row: %.1f pt" % gap_pt)
assert gap_pt >= 4.0, "legend row closer than 4 pt to the axis label"

# no two text items may overlap
texts = [t for t in fig.findobj(plt.Text) if t.get_text().strip() and t.get_visible()]
bbs = [t.get_window_extent(rend) for t in texts]
n_over = 0
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        a, b = bbs[i], bbs[j]
        if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
            n_over += 1
            print("OVERLAP: %r and %r" % (texts[i].get_text(), texts[j].get_text()))
print("text items %d | overlaps %d | rows %d | axis %.1f mm wide" % (len(texts), n_over, len(df), AX_W))
assert n_over == 0
st.save(fig, "fig6_michigan_external")
