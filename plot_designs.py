"""
plot_designs.py
---------------
Render the saved final designs as a grid for visual comparison.
Reads designs_<bc>.npz produced by experiment.py.

Usage: python plot_designs.py mbb
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NELX, NELY = 120, 40   # must match the experiment configuration


def main(bc="mbb"):
    data = np.load(f"designs_{bc}.npz")
    keys = sorted(data.files)
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 1.6 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for ax, k in zip(axes, keys):
        img = data[k].reshape(NELX, NELY).T
        ax.imshow(-img, cmap="gray", vmin=-1, vmax=0)
        ax.set_title(k, fontsize=8)
        ax.axis("off")
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    out = f"designs_{bc}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mbb")
