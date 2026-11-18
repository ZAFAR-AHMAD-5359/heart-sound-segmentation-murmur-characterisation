"""Figure 3 (revision R1): macro-F1 by source of the cardiac-phase boundaries, composite and per task, with 95%
participant-bootstrap intervals. Values are read from 06_results/run2_scoring/stats_results.json (the frozen stats.py run).
"""
import json, os
import matplotlib.pyplot as plt
import ehj_style as S

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PKG, '06_results', 'run2_scoring', 'stats_results.json')
NAME = "fig3_results"
ARMS = [("E", "equal_time", "E, equal-time"), ("E2", "equal_time_matched", "E2, equal pseudo-cycles"), ("A", "syn_fixed", "A, synthetic ECG, fixed"), ("B", "syn_fusion", "B, synthetic ECG, refined"),
        ("S", "springer_wpk", "S, Springer"), ("S_pub", "springer_pub", "Springer, published rule"), ("T", "timing_net", "T, timing network"),
        ("W", "waveform_net", "W, waveform network"), ("D", "pcg_seg_fixed", "D, annotated S1, fixed phases"), ("C", "pcg_seg", "C, expert-reviewed")]
STYLE = {"T": (S.BLUE, "s", 4.4), "W": (S.AMBER, "D", 4.0), "S": (S.GREY1, "^", 4.8), "S_pub": (S.GREY1, "^", 4.8),
         "C": (S.INK, "o", 4.2)}
PANELS = [("composite", "Composite"), ("grading", "Grading"), ("timing", "Timing"), ("shape", "Shape")]


def main():
    S.apply()
    m = json.load(open(SRC))["mlp"]
    get = lambda panel, key: m["composite"][key] if panel == "composite" else m["per_task"]["%s/%s" % (panel, key)]
    fig = plt.figure(figsize=(S.W_DOUBLE, 78 * S.MM))
    gs = fig.add_gridspec(1, 4, left=0.245, right=0.985, bottom=0.17, top=0.88, wspace=0.22, width_ratios=[1.25, 1, 1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    n = len(ARMS)
    for ax, (panel, title) in zip(axes, PANELS):
        e, c = get(panel, "equal_time")["point"], get(panel, "pcg_seg")["point"]
        ax.axvline(e, color=S.GREY2, lw=0.6, ls=(0, (1, 2)), zorder=1)
        ax.axvline(c, color=S.GREY2, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
        for i, (a, key, lab) in enumerate(ARMS):
            y = n - 1 - i
            d = get(panel, key)
            colour, marker, ms = STYLE.get(a, (S.GREY2, "o", 3.8))
            ax.errorbar(d["point"], y, xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]], fmt="none", ecolor=colour,
                        elinewidth=0.9, capsize=1.4, capthick=0.9, zorder=2)
            ax.plot(d["point"], y, ls="none", marker=marker, ms=ms, mfc=colour, mec="white", mew=0.6, zorder=3)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for _, _, lab in ARMS][::-1] if ax is axes[0] else [])
        if ax is not axes[0]:
            ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=8, loc="left", pad=3)
        lo = min(get(panel, k)["ci95"][0] for _, k, _ in ARMS)
        hi = max(get(panel, k)["ci95"][1] for _, k, _ in ARMS)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xticks([t for t in ax.get_xticks() if lo - pad <= t <= hi + pad])
    fig.text(0.615, 0.035, "Macro-F1 (%)", ha="center", va="center", fontsize=8)
    S.add_panel_letters(fig, [(ax, l) for ax, l in zip(axes, "ABCD")], same_height=True)
    S.save(fig, NAME)


if __name__ == "__main__":
    main()
