"""Shared visual system for the EHJ-DH R1 figures (Figures 1 to 6).

One matplotlib style for every figure: Arial (DejaVu Sans only as a glyph fallback),
8 to 10 pt text at final size, 0.6 pt axes, 1 to 1.5 pt data lines, no chart titles,
top and right spines removed, fixed categorical palette (blue, amber, then two greys),
bold panel letters outside the axes, and export as vector PDF (Type 42 fonts) plus
600 dpi PNG.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

MM = 1 / 25.4
W_SINGLE = 88 * MM      # single column, inches
W_DOUBLE = 180 * MM     # double column, inches

BLUE, AMBER, GREY1, GREY2 = "#184f95", "#eda100", "#4d4d4d", "#9e9e9e"
CATEGORICAL = [BLUE, AMBER, GREY1, GREY2]
INK = "#222222"          # text and axes
BAND = "#ededed"         # light shading for highlighted intervals
OUTLINE = GREY1          # the one outline colour for diagram boxes (Figures 1 and 5)

# The journal names Gill Sans, Myriad Pro, Arial or Helvetica; Arial is installed here.
FONT_FAMILY = ["Arial", "DejaVu Sans"]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "axes.unicode_minus": True,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "path.simplify": True,
    "path.simplify_threshold": 0.05,
}


def apply():
    plt.rcParams.update(RC)
    used = os.path.basename(findfont(FontProperties(family=FONT_FAMILY)))
    assert used.lower().startswith("arial"), "Arial not found, matplotlib would fall back to %s" % used


def font(bold=False):
    """FontProperties of the figure font, for text measurement (TextPath)."""
    return FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal")


def add_panel_letters(fig, items, dx_pt=0.0, dy_pt=3.0, size=10, same_height=False):
    """Place bold panel letters at the top-left OUTSIDE each axes.

    items: list of (axes, letter). The letter's left edge is aligned with the left
    edge of the axes' tight bounding box (so it sits left of the y-axis label) and
    its baseline is dy_pt above the axes top edge. With same_height=True every letter
    takes the height of the highest one, so the letters share one baseline.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = []
    for ax, letter in items:
        bb = ax.get_tightbbox(r)
        x_disp = bb.x0 + dx_pt * fig.dpi / 72
        # above everything that belongs to the axes (labels drawn outside it included)
        y_disp = max(ax.get_window_extent(r).y1, bb.y1) + dy_pt * fig.dpi / 72
        pos.append((x_disp, y_disp, letter))
    if same_height:
        top = max(p[1] for p in pos)
        pos = [(x, top, letter) for x, _, letter in pos]
    for x_disp, y_disp, letter in pos:
        x, y = inv.transform((x_disp, y_disp))
        fig.text(x, y, letter, fontsize=size, fontweight="bold", ha="left", va="bottom", color=INK)


def texts_in(fig):
    """Every rendered text string in the figure (for the wording scan)."""
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if s and s.strip():
            out.append(s)
    return out


def check_bounds(fig, name):
    """Report any text that lies outside the figure canvas (nothing may be clipped)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0 and bb.height == 0:
            continue
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            bad.append(t.get_text())
    print("[%s] text outside canvas: %s" % (name, bad if bad else "none"))
    return bad


def ink_margins(fig, name, min_mm=0.5):
    """Distance in mm from the drawn content (first non-white pixel of the rendered
    canvas) to each canvas edge: left, right, top, bottom. Nothing may touch an edge."""
    import numpy as np
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    ink = (img < 250).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    px = 25.4 / fig.dpi
    m = [cols[0] * px, (ink.shape[1] - 1 - cols[-1]) * px, rows[0] * px, (ink.shape[0] - 1 - rows[-1]) * px]
    flag = "" if min(m) >= min_mm else "   <-- closer than %.1f mm to an edge" % min_mm
    print("[%s] clearance to canvas edge (mm): left %.1f, right %.1f, top %.1f, bottom %.1f%s"
          % (name, m[0], m[1], m[2], m[3], flag))
    return m


def save(fig, name):
    """Export <name>.pdf (vector) and <name>.png (600 dpi) at the fixed figure size,
    and dump every text string to <name>_text.txt for the wording scan."""
    check_bounds(fig, name)
    ink_margins(fig, name)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, name + "." + ext), bbox_inches=None, facecolor="white")
    with open(os.path.join(OUT, name + "_text.txt"), "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(texts_in(fig)))
    w, h = fig.get_size_inches()
    print("[%s] saved %.0f x %.1f mm  ->  %s" % (name, w * 25.4, h * 25.4, OUT))
