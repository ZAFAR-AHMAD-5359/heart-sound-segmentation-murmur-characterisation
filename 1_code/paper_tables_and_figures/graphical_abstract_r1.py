"""Graphical abstract, revision R1: 180 x 125 mm landscape, Arial, no text below 8 pt. Left: design; right: composite
macro-F1 of the key arms with 95% intervals and the recovery fractions; bottom: three findings. Values are read from
06_results/run2_scoring/stats_results.json.
"""
import json, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import ehj_style as S

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PKG, '06_results', 'run2_scoring', 'stats_results.json')
NAME = "graphical_abstract_r1"
W_MM, H_MM = 180.0, 125.0
ARMS = [("E", "equal_time", "Equal-time", S.GREY2, "o"), ("A", "syn_fixed", "Synthetic ECG, fixed", S.GREY2, "o"), ("B", "syn_fusion", "Synthetic ECG, refined", S.GREY2, "o"),
        ("S", "springer_wpk", "Springer", S.GREY1, "^"), ("T", "timing_net", "Timing network", S.BLUE, "s"),
        ("W", "waveform_net", "Reconstruction net", S.AMBER, "D"), ("C", "pcg_seg", "Expert-reviewed", S.INK, "o")]


def box(ax, x, y, w, h, lines, fill="white", bold_first=True, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill, edgecolor=S.OUTLINE, lw=0.6, zorder=2))
    n = len(lines)
    for i, s in enumerate(lines):
        ax.text(x + 2.0, y + h / 2 + (i - (n - 1) / 2) * 3.7, s, ha="left", va="center", fontsize=size,
                fontweight="bold" if (bold_first and i == 0) else "normal", color=S.INK, zorder=3)


def main():
    S.apply()
    m = json.load(open(SRC))["mlp"]
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(H_MM, 0)
    ax.axis("off")
    ax.text(3, 6.5, "ECG-supervised cardiac-phase gating for paediatric murmur grade, timing, and shape",
            fontsize=10, fontweight="bold", color=S.INK, va="center")
    ax.text(3, 12.5, "Pre-registered comparison on CirCor DigiScope: 3,000 recordings, 816 participants, participant-grouped ten-fold cross-validation",
            fontsize=8, color=S.INK, va="center")

    # ---------------- left: design
    box(ax, 3, 18, 62, 10, ["Phonocardiogram", "one stethoscope channel"], fill="#f4f4f4")
    ax.add_patch(FancyArrowPatch((34, 28), (34, 33), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=S.GREY1))
    ax.text(3, 37, "Cardiac-phase boundaries from", fontsize=8, fontweight="bold", color=S.INK, va="center")
    srcs = [("Equal-time windows (control)", S.GREY2), ("Springer segmenter", S.GREY1), ("Timing network, trained with ECGs", S.BLUE),
            ("Reconstruction network, trained with ECGs", S.AMBER), ("Expert-reviewed annotation", S.INK)]
    for i, (s, c) in enumerate(srcs):
        y = 42 + i * 5.6
        ax.add_patch(FancyBboxPatch((3, y - 1.8), 3.6, 3.6, boxstyle="round,pad=0,rounding_size=0.6", facecolor=c, edgecolor="none", zorder=3))
        ax.text(9, y, s, fontsize=8, color=S.INK, va="center")
    ax.text(9, 42 + 5 * 5.6, "networks: six adult PCG-ECG datasets,", fontsize=8, color=S.INK, va="center", style="italic")
    ax.text(9, 42 + 5 * 5.6 + 4.2, "no paediatric labels", fontsize=8, color=S.INK, va="center", style="italic")
    ax.add_patch(FancyArrowPatch((34, 77), (34, 82), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=S.GREY1))
    box(ax, 3, 82, 62, 13, ["Same 464 features, same classifier", "grade, timing, and shape:", "composite macro-F1"])

    # ---------------- right: results
    axr = fig.add_axes([0.585, 0.355, 0.39, 0.47])
    n = len(ARMS)
    e, c = m["composite"]["equal_time"]["point"], m["composite"]["pcg_seg"]["point"]
    axr.axvline(e, color=S.GREY2, lw=0.6, ls=(0, (1, 2)))
    axr.axvline(c, color=S.GREY2, lw=0.6, ls=(0, (3, 1.5)))
    for i, (a, key, lab, col, mk) in enumerate(ARMS):
        d = m["composite"][key]
        y = n - 1 - i
        axr.errorbar(d["point"], y, xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]], fmt="none", ecolor=col, elinewidth=1.0, capsize=1.6)
        axr.plot(d["point"], y, ls="none", marker=mk, ms=5.0, mfc=col, mec="white", mew=0.6)
    axr.set_yticks(range(n))
    axr.set_yticklabels([lab for _, _, lab, _, _ in ARMS][::-1])
    axr.set_ylim(-0.6, n - 0.4)
    axr.set_xlabel("Composite macro-F1 (%), 95% interval")
    rec = m["recovery_fraction"]
    ax.text(80, 96.5, "Share of the expert gain recovered, at matched coverage:", fontsize=8, fontweight="bold", color=S.INK, va="center", ha="left")
    ax.text(80, 100.6, "timing network %.0f%%, reconstruction network %.0f%%, Springer %.0f%%" % (
        100 * rec["timing_net_annot"]["point"], 100 * rec["waveform_net_annot"]["point"], 100 * rec["springer_wpk_annot"]["point"]),
        fontsize=8, color=S.INK, va="center", ha="left")

    # ---------------- bottom: findings
    find = ["Beat timing carried the benefit: waveform reconstruction gave no measurable further gain",
            "About two thirds of the expert gain at matched coverage, without a second sensor",
            "Gains seen below 100 bpm, to 125 for the waveform network; faster rates and infants inconclusive"]
    for i, s in enumerate(find):
        ax.add_patch(FancyBboxPatch((3 + i * 59.0, 106), 56.0, 15.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor="#eef3fa" if i == 0 else "#f4f4f4", edgecolor=S.OUTLINE, lw=0.6))
        words, lines, cur = s.split(), [], ""
        for w_ in words:
            if len(cur) + len(w_) + 1 > 34:
                lines.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        lines.append(cur)
        for j, l in enumerate(lines):
            ax.text(5 + i * 59.0, 110 + j * 3.7, l, fontsize=8, color=S.INK, va="center", fontweight="bold" if i == 0 else "normal")
    S.save(fig, NAME)


if __name__ == "__main__":
    main()
