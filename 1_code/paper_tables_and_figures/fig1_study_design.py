"""Figure 1 (revision R1): study design. Sources of the cardiac-phase boundaries (left), the shared CirCor pipeline
(centre), and the outcomes (right). All text lives in the TEXT dictionary; counts come from the analysis outputs
(cohort: Run 1 results.json; training pack: derived/corpus/paired_v3/manifest.json, 2,113 records, 48.3 h).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import ehj_style as S

W_MM, H_MM = 180.0, 104.0
NAME = "fig1_study_design"
TEXT = {
    "left_head": "Sources of the cardiac-phase boundaries",
    "trained_head": "Trained without CirCor data",
    "none_head": "No training",
    "paired": ("Six adult datasets with simultaneous", "PCG and ECG: 2,113 records, 48.3 h"),
    "T": ("T", "Timing network", "S1, S2, and R-peak heads"),
    "W": ("W", "Waveform network", "ECG reconstruction head"),
    "gan": ("A, B", "Adversarial synthetic ECG", "R-peaks; fixed or refined phases"),
    "spr": ("S", "Springer segmenter", "published parameters"),
    "E": ("E", "Equal-time windows", "control"),
    "D": ("D", "Annotated S1 onsets", "fixed phase fractions"),
    "C": ("C", "Expert-reviewed annotation", "reference"),
    "centre_head": "Shared CirCor pipeline",
    "circor": ("CirCor DigiScope", "3,000 recordings, 816 participants"),
    "phases": ("Four phases per cycle", "S1, systole, S2, diastole"),
    "features": ("464 features per recording", "one extractor for every arm"),
    "clf": ("Five-seed MLP ensemble", "random forest as secondary"),
    "cv": ("Participant-grouped ten-fold", "cross-validation"),
    "right_head": "Outcomes",
    "o1": ("Composite macro-F1", "grading, timing, shape;", "pre-registered H1 to H4"),
    "o2": ("Recovery fraction", "share of the expert gain,", "matched coverage"),
    "o3": ("Detection macro-F1", "against a PCG-only CNN"),
    "o4": ("Error budget", "expert boundaries corrupted", "in controlled ways"),
}
ARM_COLOUR = {"T": S.BLUE, "W": S.AMBER, "S": S.GREY1}


def box(ax, x, y, w, h, lines, tag=None, tag_colour=None, bold_first=True, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill, edgecolor=S.OUTLINE, lw=0.6, zorder=2))
    tx = x + 2.0
    if tag:
        ax.add_patch(FancyBboxPatch((x + 1.6, y + h / 2 - 2.6), 9.0, 5.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=tag_colour or "#ededed", edgecolor="none", zorder=3))
        ax.text(x + 6.1, y + h / 2, tag, ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if tag_colour in (S.BLUE, S.GREY1) else S.INK, zorder=4)
        tx = x + 12.6
    n = len(lines)
    lh = 3.6
    y0 = y + h / 2 - (n - 1) * lh / 2
    for i, s in enumerate(lines):
        ax.text(tx, y0 + i * lh, s, ha="left", va="center", fontsize=8, fontweight="bold" if (i == 0 and bold_first) else "normal", color=S.INK, zorder=4)


def arrow(ax, p, q, colour=S.GREY1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=colour, zorder=1, shrinkA=0, shrinkB=0))


def head(ax, x, y, s):
    ax.text(x, y, s, ha="left", va="center", fontsize=8, fontweight="bold", color=S.INK)


def main():
    S.apply()
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(H_MM, 0)
    ax.axis("off")

    # ---------------- left column: boundary sources
    L, LW = 2.0, 64.0
    head(ax, L, 3.5, TEXT["left_head"])
    ax.text(L, 9.0, TEXT["trained_head"], ha="left", va="center", fontsize=8, style="italic", color=S.INK)
    box(ax, L, 12.0, LW, 9.5, TEXT["paired"], bold_first=False, fill="#f4f4f4")
    yT, yW = 25.5, 35.5
    box(ax, L + 6, yT, LW - 6, 8.5, TEXT["T"][1:], tag="T", tag_colour=ARM_COLOUR["T"])
    box(ax, L + 6, yW, LW - 6, 8.5, TEXT["W"][1:], tag="W", tag_colour=ARM_COLOUR["W"])
    for yy in (yT, yW):
        ax.plot([L + 3, L + 3, L + 6], [21.5, yy + 4.25, yy + 4.25], color=S.GREY1, lw=0.8, zorder=1)
    box(ax, L, 46.5, LW, 8.5, TEXT["gan"][1:], tag="A, B", tag_colour="#ededed")
    box(ax, L, 57.0, LW, 8.5, TEXT["spr"][1:], tag="S", tag_colour=ARM_COLOUR["S"])
    ax.text(L, 70.5, TEXT["none_head"], ha="left", va="center", fontsize=8, style="italic", color=S.INK)
    box(ax, L, 73.5, LW, 8.5, TEXT["E"][1:], tag="E", tag_colour="#ededed")
    box(ax, L, 84.0, LW, 8.5, TEXT["D"][1:], tag="D", tag_colour="#ededed")
    box(ax, L, 94.5, LW, 8.5, TEXT["C"][1:], tag="C", tag_colour="#ededed")

    # bracket from every source into the phase box
    BX = L + LW + 3.0
    tops = [yT + 4.25, yW + 4.25, 50.75, 61.25, 77.75, 88.25, 98.75]
    for yy in tops:
        ax.plot([L + LW, BX], [yy, yy], color=S.GREY1, lw=0.8, zorder=1)
    ax.plot([BX, BX], [tops[0], tops[-1]], color=S.GREY1, lw=0.8, zorder=1)

    # ---------------- centre column: shared pipeline
    CX, CW = 78.0, 50.0
    head(ax, CX, 3.5, TEXT["centre_head"])
    ycirc, yph, yfe, ycl, ycv = 12.0, 30.0, 48.0, 66.0, 84.0
    for yy, key in ((ycirc, "circor"), (yph, "phases"), (yfe, "features"), (ycl, "clf"), (ycv, "cv")):
        box(ax, CX, yy, CW, 10.0, TEXT[key], fill="#f4f4f4" if key == "circor" else "white")
    for a, b in ((ycirc + 10, yph), (yph + 10, yfe), (yfe + 10, ycl), (ycl + 10, ycv)):
        arrow(ax, (CX + CW / 2, a), (CX + CW / 2, b))
    mid = yph + 5.0
    ax.plot([BX, BX + 4.0], [mid, mid], color=S.GREY1, lw=0.8, zorder=1)
    arrow(ax, (BX + 4.0, mid), (CX, mid))
    ax.plot([BX, BX], [mid, tops[0]], color=S.GREY1, lw=0.8, zorder=1)

    # ---------------- right column: outcomes
    RX, RW = 136.0, 42.0
    head(ax, RX, 3.5, TEXT["right_head"])
    ys = (12.0, 30.0, 48.0, 66.0)
    for yy, key in zip(ys, ("o1", "o2", "o3", "o4")):
        box(ax, RX, yy, RW, 13.0, TEXT[key])
    JX = CX + CW + 3.0
    ax.plot([CX + CW, JX], [ycv + 5, ycv + 5], color=S.GREY1, lw=0.8, zorder=1)
    ax.plot([JX, JX], [ys[0] + 6.5, ycv + 5], color=S.GREY1, lw=0.8, zorder=1)
    for yy in ys[:3]:
        arrow(ax, (JX, yy + 6.5), (RX, yy + 6.5))
    # error budget: corrupted copies of C re-enter the shared pipeline
    ax.plot([RX + RW / 2, RX + RW / 2], [ys[3] + 13.0, 98.75], color=S.GREY2, lw=0.8, ls=(0, (3, 1.5)), zorder=1)
    ax.plot([L + LW, RX + RW / 2], [98.75, 98.75], color=S.GREY2, lw=0.8, ls=(0, (3, 1.5)), zorder=0)

    S.save(fig, NAME)


if __name__ == "__main__":
    main()
