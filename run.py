"""
run.py
------
Driver for a single topology optimisation run. The loop is identical for every
optimiser. To avoid cutting any optimiser short during a slow warm-up, the run
uses a FIXED ITERATION HORIZON and detects convergence post hoc, rather than
stopping early. Convergence iteration is reported but does not truncate the run.
"""

from __future__ import annotations
import numpy as np
from topopt_core import TopOptProblem
from optimisers import OPTIMISERS


def detect_convergence(compliance, window=10, tol=1e-3):
    """First iteration at which the relative change over `window` steps stays
    below `tol`. Returns len(compliance) if never reached."""
    c = np.asarray(compliance)
    for i in range(window, len(c)):
        rel = abs(c[i - window] - c[i]) / (abs(c[i - window]) + 1e-12)
        if rel < tol:
            return i
    return len(c)


def run_optimisation(problem, optimiser_name, maxiter=300, seed=0,
                     x0=None, opt_kwargs=None, verbose=False):
    rng = np.random.default_rng(seed)
    n = problem.nelx * problem.nely

    if x0 is None:
        x = np.full(n, problem.volfrac)
    elif x0 == "random":
        # feasible random start: perturb then renormalise to the volume
        x = np.clip(problem.volfrac + 0.1 * (rng.random(n) - 0.5), 0.01, 0.99)
        x *= problem.volfrac / x.mean()
        x = np.clip(x, 0.01, 0.99)
    else:
        x = x0.copy()

    opt_kwargs = opt_kwargs or {}
    optimiser = OPTIMISERS[optimiser_name](problem, **opt_kwargs)
    optimiser._rng = rng  # available for symmetry-breaking init

    xPhys = problem.filt.filter_density(x)
    history = {"compliance": [], "change": [], "vol": []}

    for it in range(maxiter):
        c, dc, vol = problem.analyse(xPhys)
        dv = np.ones(n) / n
        dc = problem.filt.filter_sensitivity(dc)
        dv = problem.filt.filter_sensitivity(dv)

        xold = x.copy()
        x = optimiser.step(x, dc, dv)
        xPhys = problem.filt.filter_density(x)

        history["compliance"].append(float(c))
        history["change"].append(float(np.abs(x - xold).max()))
        history["vol"].append(float(vol))
        if verbose and it % 25 == 0:
            print(f"  [{optimiser_name}] it={it:3d} c={c:9.3f} "
                  f"vol={vol:.3f}")

    conv_it = detect_convergence(history["compliance"])
    grayness = float(4.0 * np.mean(xPhys * (1.0 - xPhys)))
    return {
        "optimiser": optimiser_name,
        "xPhys": xPhys,
        "compliance": history["compliance"][-1],
        "converged_at": conv_it,
        "grayness": grayness,
        "final_volume": float(xPhys.mean()),
        "history": history,
        "seed": seed,
    }


if __name__ == "__main__":
    prob = TopOptProblem(nelx=120, nely=40, volfrac=0.4,
                         penal=3.0, rmin=2.0, bc="mbb")
    print("method  final_c   conv@   vol     grayness")
    for name, kw in [("OC", {}), ("GD", {"lr": 0.5}), ("MMA", {})]:
        r = run_optimisation(prob, name, maxiter=300, opt_kwargs=kw)
        print(f"{name:5s}  {r['compliance']:8.3f}  {r['converged_at']:4d}   "
              f"{r['final_volume']:.4f}  {r['grayness']:.4f}")
