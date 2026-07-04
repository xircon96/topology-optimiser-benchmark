"""
mma.py
------
Self-contained implementation of Svanberg's Method of Moving Asymptotes (MMA)
for a single volume constraint, sufficient for compliance-minimisation
topology optimisation. Follows the standard formulation (Svanberg, 1987) with
the dual subproblem solved by a primal-dual interior point step.

This is a compact educational implementation, not the full GCMMA. It is
adequate for the benchmark comparison and avoids any external dependency.
"""

from __future__ import annotations
import numpy as np


class MMAState:
    """Holds the moving-asymptote history between iterations."""

    def __init__(self, n):
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.n = n


def mma_sub(x, dfdx, gx, dgdx, xmin, xmax, state,
            iteration, move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7):
    """One MMA step for objective f(x) with one constraint g(x) <= 0.

    x      : current design (n,)
    dfdx   : objective gradient (n,)
    gx     : constraint value (scalar), g <= 0 feasible
    dgdx   : constraint gradient (n,)
    Returns the new design x.
    """
    n = len(x)
    xval = x.copy()
    eeen = np.ones(n)

    # --- update asymptotes ---
    if state.low is None or state.xold2 is None:
        state.low = xval - asyinit * (xmax - xmin)
        state.upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - state.xold1) * (state.xold1 - state.xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        state.low = xval - factor * (state.xold1 - state.low)
        state.upp = xval + factor * (state.upp - state.xold1)
        lowmin = xval - 10.0 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10.0 * (xmax - xmin)
        state.low = np.clip(state.low, lowmin, lowmax)
        state.upp = np.clip(state.upp, uppmin, uppmax)

    # --- move limits / bounds for this subproblem ---
    alfa = np.maximum.reduce([xmin, state.low + 0.1 * (xval - state.low),
                              xval - move * (xmax - xmin)])
    beta = np.minimum.reduce([xmax, state.upp - 0.1 * (state.upp - xval),
                              xval + move * (xmax - xmin)])

    # --- build the convex approximation coefficients ---
    ux1 = state.upp - xval
    xl1 = xval - state.low
    p0 = np.maximum(dfdx, 0.0) * ux1 ** 2
    q0 = np.maximum(-dfdx, 0.0) * xl1 ** 2
    p1 = np.maximum(dgdx, 0.0) * ux1 ** 2
    q1 = np.maximum(-dgdx, 0.0) * xl1 ** 2
    b1 = (p1 / ux1 + q1 / xl1).sum() - gx

    # --- solve dual: maximise w.r.t. single multiplier lam >= 0 by bisection ---
    def primal(lam):
        plam = np.maximum(p0 + lam * p1, 1e-30)
        qlam = np.maximum(q0 + lam * q1, 1e-30)
        xnew = (np.sqrt(plam) * state.low + np.sqrt(qlam) * state.upp) / \
               (np.sqrt(plam) + np.sqrt(qlam))
        return np.clip(xnew, alfa, beta)

    def dual_grad(lam):
        xnew = primal(lam)
        return (p1 / (state.upp - xnew) + q1 / (xnew - state.low)).sum() - b1

    lo, hi = 0.0, 1e9
    # ensure sign change; if constraint slack at lam=0, solution is lam=0
    if dual_grad(0.0) < 0:
        xnew = primal(0.0)
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if dual_grad(mid) > 0:
                lo = mid
            else:
                hi = mid
        xnew = primal(0.5 * (lo + hi))

    state.xold2 = state.xold1
    state.xold1 = xval.copy()
    return xnew
