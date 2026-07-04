"""
metrics.py
----------
Metrics for comparing topology-optimisation outcomes beyond final compliance.
The central question of the study is whether different optimisers reach
genuinely different designs, so we need measures of topological difference,
not just performance.
"""

from __future__ import annotations
import numpy as np


def grayness(xPhys):
    """0 = perfectly black/white, 1 = all intermediate. Mean of 4 x (1-x)."""
    return float(4.0 * np.mean(xPhys * (1.0 - xPhys)))


def density_rmse(xa, xb):
    """Root-mean-square difference between two density fields."""
    return float(np.sqrt(np.mean((xa - xb) ** 2)))


def binary_iou(xa, xb, thresh=0.5):
    """Intersection-over-union of the thresholded (solid) regions.
    1.0 = identical layout, 0.0 = no overlap."""
    a = xa >= thresh
    b = xb >= thresh
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def solid_fraction(xPhys, thresh=0.5):
    return float(np.mean(xPhys >= thresh))


def hole_count(xPhys, nelx, nely, thresh=0.5):
    """Number of connected void regions (a topology descriptor).
    Uses a simple flood fill on the thresholded void field."""
    void = (xPhys.reshape(nelx, nely) < thresh)
    seen = np.zeros_like(void, dtype=bool)
    count = 0
    stack = []
    for i in range(nelx):
        for j in range(nely):
            if void[i, j] and not seen[i, j]:
                count += 1
                stack.append((i, j))
                seen[i, j] = True
                while stack:
                    ci, cj = stack.pop()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < nelx and 0 <= nj < nely
                                and void[ni, nj] and not seen[ni, nj]):
                            seen[ni, nj] = True
                            stack.append((ni, nj))
    return count


def summarise(results, nelx, nely):
    """Given a list of run results (same problem), return per-run descriptors
    and a pairwise IoU matrix between the optimisers' final designs."""
    rows = []
    for r in results:
        x = r["xPhys"]
        rows.append({
            "optimiser": r["optimiser"],
            "seed": r.get("seed", 0),
            "compliance": r["compliance"],
            "converged_at": r["converged_at"],
            "grayness": grayness(x),
            "solid_fraction": solid_fraction(x),
            "holes": hole_count(x, nelx, nely),
        })
    n = len(results)
    iou = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = binary_iou(results[i]["xPhys"], results[j]["xPhys"])
            iou[i, j] = iou[j, i] = v
    return rows, iou
