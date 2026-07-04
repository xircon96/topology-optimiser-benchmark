"""
Generate Figure 1: the four benchmark design domains with loads and supports.
Drawn to match the boundary conditions actually used in topopt_extended.py:
  - mbb            : symmetry (roller) on left edge, roller at bottom-right,
                     downward point load at top-left.
  - cantilever     : left edge fully clamped, downward load at right-edge middle.
  - bridge         : pin at bottom-left, roller at bottom-right,
                     downward point load at bottom-centre.
  - tip_cantilever : left edge fully clamped, downward load at bottom-right.
UK English. Consistent visual key across panels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import numpy as np

# visual key
C_DOMAIN = "#e9edf2"
C_EDGE = "#2b2b2b"
C_LOAD = "#d62728"
C_SUPPORT = "#1f77b4"


def domain(ax, w, h, title):
    ax.add_patch(Rectangle((0, 0), w, h, facecolor=C_DOMAIN,
                           edgecolor=C_EDGE, lw=1.6, zorder=1))
    ax.set_xlim(-0.22 * w, 1.18 * w)
    ax.set_ylim(-0.30 * h, 1.30 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def load_arrow(ax, x, y, h, label="P"):
    L = 0.28 * h
    ax.add_patch(FancyArrow(x, y + L, 0, -L, width=0.0,
                            head_width=0.07 * h, head_length=0.10 * h,
                            length_includes_head=True, color=C_LOAD, zorder=4))
    ax.text(x + 0.04 * h, y + L * 0.6, label, color=C_LOAD,
            fontsize=10, fontweight="bold")


def clamp(ax, x0, x1, y0, y1):
    """Hatched clamped edge (fully fixed)."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=C_SUPPORT, hatch="////", lw=1.4, zorder=3))


def pin(ax, x, y, h):
    """Triangle = pinned support (both directions fixed)."""
    s = 0.10 * h
    ax.fill([x, x - s, x + s], [y, y - 1.4 * s, y - 1.4 * s],
            color=C_SUPPORT, zorder=4)


def roller(ax, x, y, h, vertical=False):
    """Triangle with a line = roller (one direction fixed)."""
    s = 0.10 * h
    ax.fill([x, x - s, x + s], [y, y - 1.4 * s, y - 1.4 * s],
            color=C_SUPPORT, zorder=4)
    ax.plot([x - s, x + s], [y - 1.7 * s, y - 1.7 * s],
            color=C_SUPPORT, lw=1.6, zorder=4)


fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
W, H = 3.0, 1.0   # aspect roughly matches the 3:1 meshes

# --- MBB (half beam) ---
ax = axes[0]; domain(ax, W, H, "(a) Half MBB beam")
load_arrow(ax, 0, H, H, "P")
# left edge symmetry (rollers, x fixed)
for yy in np.linspace(0.12 * H, 0.88 * H, 4):
    roller(ax, 0, yy, H)
ax.text(-0.20 * W, 0.5 * H, "symmetry", color=C_SUPPORT, fontsize=8,
        rotation=90, va="center")
# bottom-right roller (y fixed)
roller(ax, W, 0, H)

# --- Cantilever (mid-edge load) ---
ax = axes[1]; domain(ax, W, H, "(b) Cantilever (mid load)")
clamp(ax, -0.06 * W, 0, 0, H)
load_arrow(ax, W, 0.5 * H, H, "P")

# --- Bridge ---
ax = axes[2]; domain(ax, W, H, "(c) Bridge")
pin(ax, 0, 0, H)
roller(ax, W, 0, H)
load_arrow(ax, 0.5 * W, 0, H, "P")
ax.text(0, -0.20 * H, "pin", color=C_SUPPORT, fontsize=8, ha="center")
ax.text(W, -0.20 * H, "roller", color=C_SUPPORT, fontsize=8, ha="center")

# --- Tip-loaded cantilever ---
ax = axes[3]; domain(ax, W, H, "(d) Tip-loaded cantilever")
clamp(ax, -0.06 * W, 0, 0, H)
load_arrow(ax, W, 0, H, "P")

# shared legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=C_DOMAIN, edgecolor=C_EDGE, label="Design domain"),
    Line2D([0], [0], color=C_LOAD, lw=2, marker="v", markersize=8,
           label="Applied load P"),
    Patch(facecolor="none", edgecolor=C_SUPPORT, hatch="////",
          label="Clamped edge (fixed)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_SUPPORT,
           markersize=11, label="Pin / roller support"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("fig1_benchmarks.png", dpi=200, bbox_inches="tight")
fig.savefig("fig1_benchmarks.pdf", bbox_inches="tight")
print("saved fig1_benchmarks.png and .pdf")
