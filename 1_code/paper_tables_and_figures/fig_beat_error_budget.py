"""Beat-error budget with the automatic segmenters placed on it (three panels, 180 mm wide).

Curves: arm C's annotated cycles degraded in controlled ways (Experiment 0: missed beats, spurious beats, start jitter;
Experiment 0b: S2 misplaced in every cycle of a share of recordings, equal-time windows in a share of recordings, and
the same S2 error spread over all recordings), as the share of the annotation's gain over equal-time windows that is
kept, (X - E) / (C - E), seed-averaged, with 95% participant-bootstrap intervals. Points: Springer, T and W at matched
coverage, (X_annot - E_annot) / (C - E_annot), placed at their own boundary agreement with the annotation.
Every value is read from the analysis outputs listed in SOURCES; nothing is typed in by hand.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ehj_style as S

RUNS = "E:/NUST MS Thesis/PhD Level Research/extension_study/runs"
SOURCES = dict(exp0=RUNS + "/exp0_analysis/exp0_results.json", exp0b=RUNS + "/exp0b_analysis/exp0b_results.json",
               rec=RUNS + "/run2_scoring/stats_results.json", f1=RUNS + "/exp0_analysis/arms_on_curve_matched_coverage.csv",
               sd=RUNS + "/s2_analysis/s2_variability_per_recording.csv", fail=RUNS + "/s2_analysis/s2_between_recording_heterogeneity.csv")
NAME = "fig_beat_error_budget"
ARMS = [("Springer", "springer_wpk_annot", "S_annot", "S", S.GREY1, "^"),
        ("T", "timing_net_annot", "T_annot", "T", S.BLUE, "s"),
        ("W", "waveform_net_annot", "W_annot", "W", S.AMBER, "D")]
CURVE = S.GREY2


def load():
    e0 = json.load(open(SOURCES["exp0"]))["levels"]
    e0b = json.load(open(SOURCES["exp0b"]))["levels"]
    rec = json.load(open(SOURCES["rec"]))["mlp"]["recovery_fraction"]
    f1 = pd.read_csv(SOURCES["f1"]).set_index("arm")
    sd = pd.read_csv(SOURCES["sd"]).groupby("arm")["start_sd"].median()
    fail = pd.read_csv(SOURCES["fail"]).set_index("arm")["share_recordings_s2_off_gt60"]
    lvl = lambda d, k: (d[k]["retention"] if "retention" in d[k] else d[k]["R"],
                        d[k]["retention_ci95"] if "retention_ci95" in d[k] else d[k]["R_ci95"])
    curves = dict(
        missed=[(1.0, 1.0, None)] + [(e0["del_%s" % v]["boundary_f1_median"], *lvl(e0, "del_%s" % v)) for v in (0.1, 0.2, 0.3, 0.5)],
        spurious=[(1.0, 1.0, None)] + [(e0["ins_%s" % v]["boundary_f1_median"], *lvl(e0, "ins_%s" % v)) for v in (0.1, 0.2, 0.3, 0.5)],
        jitter=[(0.0, 1.0, None)] + [(e0["jit_%s" % v]["offset_sd_ms_median"], *lvl(e0, "jit_%s" % v)) for v in (5, 10, 20, 40)],
        conc=[(0.0, 1.0, None)] + [(100 * e0b["s2c_%s" % v]["s2_off_gt60_share"], *lvl(e0b, "s2c_%s" % v)) for v in (0.1, 0.2, 0.3)],
        total=[(0.0, 1.0, None)] + [(100 * e0b["fail_%s" % v]["s2_off_gt60_share"], *lvl(e0b, "fail_%s" % v)) for v in (0.1, 0.2, 0.3)],
        spread=[(100 * e0b["s2s_%s" % v]["s2_off_gt60_share"], *lvl(e0b, "s2s_%s" % v)) for v in (0.1, 0.2, 0.3)])
    arms = []
    for label, key, f1key, armkey, colour, marker in ARMS:
        r = rec[key]
        arms.append(dict(label=label, colour=colour, marker=marker, R=r["point"], ci=r["ci95"], f1=float(f1.loc[f1key, "boundary_f1_median"]),
                         sd=float(sd[armkey]), fail=100 * float(fail[armkey])))
    return curves, arms


def draw_curve(ax, pts, ls, marker, label):
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ax.plot(x, y, ls=ls, color=CURVE, lw=1.2, zorder=2, label=label)
    for p in pts[1:]:
        ax.errorbar(p[0], p[1], yerr=[[p[1] - p[2][0]], [p[2][1] - p[1]]], fmt="none", ecolor=CURVE, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=2)
    ax.plot(x[1:], y[1:], ls="none", marker=marker, ms=3.2, mfc="white", mec=CURVE, mew=0.9, zorder=3)
    ax.plot(x[:1], y[:1], ls="none", marker="o", ms=3.2, mfc=S.INK, mec="white", mew=0.6, zorder=4)


def draw_arms(ax, arms, xkey, offsets, dodge=None):
    """dodge: {label: shift in x units}, a display offset only (disclosed in the legend text of the caller)."""
    for a in arms:
        x = a[xkey] + (dodge or {}).get(a["label"], 0.0)
        ax.errorbar(x, a["R"], yerr=[[a["R"] - a["ci"][0]], [a["ci"][1] - a["R"]]], fmt="none", ecolor=a["colour"], elinewidth=0.9,
                    capsize=1.6, capthick=0.9, zorder=5)
        ax.plot(x, a["R"], ls="none", marker=a["marker"], ms=5.2 if a["marker"] == "^" else 4.6, mfc=a["colour"], mec="white", mew=0.7, zorder=6)
        dx, dy, ha = offsets[a["label"]]
        ax.annotate(a["label"], (x, a["R"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8, color=S.INK, zorder=7)


def frame(ax, xlabel):
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0.8, color=CURVE, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.set_xlabel(xlabel)


def main():
    S.apply()
    curves, arms = load()
    fig = plt.figure(figsize=(S.W_DOUBLE, 85 * S.MM))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.975, bottom=0.285, top=0.94, wspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    draw_curve(axA, curves["missed"], "-", "o", "Missed beats")
    draw_curve(axA, curves["spurious"], (0, (3, 1.5)), "s", "Spurious beats")
    draw_arms(axA, arms, "f1", {"Springer": (5, 0, "left"), "T": (5, 0, "left"), "W": (-5, 0, "right")})
    axA.set_xlim(1.005, 0.64)
    axA.set_xticks([1.0, 0.9, 0.8, 0.7])
    frame(axA, "Cycle-start F1 against the annotation")
    axA.set_ylabel("Share of the annotation's gain kept")
    axA.annotate("Annotation", (1.0, 1.0), xytext=(4, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=S.INK)

    draw_curve(axB, curves["jitter"], "-", "o", "Start jitter")
    draw_arms(axB, arms, "sd", {"Springer": (5, 0, "left"), "T": (-5, 0, "right"), "W": (5, 0, "left")}, dodge={"T": -0.7, "W": 0.7})
    axB.set_xlim(-1, 42)
    axB.set_xticks([0, 10, 20, 30, 40])
    frame(axB, "SD of start offset in a recording (ms)")

    draw_curve(axC, curves["conc"], "-", "o", "S2 failed in some recordings")
    draw_curve(axC, curves["total"], (0, (3, 1.5)), "s", "Equal-time in some recordings")
    sp = curves["spread"]
    axC.errorbar([p[0] for p in sp], [p[1] for p in sp], yerr=[[p[1] - p[2][0] for p in sp], [p[2][1] - p[1] for p in sp]], fmt="none",
                 ecolor=CURVE, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=2)
    axC.plot([p[0] for p in sp], [p[1] for p in sp], ls="none", marker="v", ms=3.4, mfc="white", mec=CURVE, mew=0.9, zorder=3,
             label="Same S2 error, spread")
    draw_arms(axC, arms, "fail", {"Springer": (-5, 0, "right"), "T": (5, 0, "left"), "W": (-5, 0, "right")})
    axC.set_xlim(-1.5, 32)
    axC.set_xticks([0, 10, 20, 30])
    frame(axC, "Recordings with S2 > 60 ms off (%)")
    for ax in (axA, axB, axC):
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.19), handlelength=2.4, borderaxespad=0.0, labelspacing=0.35)

    S.add_panel_letters(fig, [(axA, "A"), (axB, "B"), (axC, "C")], same_height=True)
    S.save(fig, NAME)
    with open(os.path.join(S.OUT, NAME + "_values.json"), "w") as fh:
        json.dump(dict(sources=SOURCES, curves={k: [[p[0], p[1], p[2]] for p in v] for k, v in curves.items()}, arms=arms), fh, indent=1)


if __name__ == "__main__":
    main()
