"""
================================================================================
 EXTENDED STUDIES for "Optimiser-induced non-uniqueness in density-based
 topology optimisation".
================================================================================

 This module ADDS simulations and data to the base study. It imports the
 validated core from topopt_study.py (the file that already runs), so keep
 both files in the same directory.

 New studies provided here:
   A. Mesh-independence sweep        (does the optimiser ranking survive mesh refinement?)
   B. Filter-radius sensitivity      (length-scale / minimum-feature effect)
   C. Volume-fraction sweep          (does the GD penalty depend on volfrac?)
   D. Penalisation continuation      (does ramping the SIMP penalty help GD?)
   E. Hyperparameter sensitivity     (GD learning rate, MMA move limit)
   F. Multi-benchmark comparison     (MBB, cantilever, bridge, mid-cantilever)
   G. Statistical significance       (bootstrap CIs, Mann-Whitney U, Cliff's delta)
   H. Computational cost             (wall-time, solves, efficiency frontier)

 Each study writes charts (PNG), data (CSV) and LaTeX tables (.tex) into
 ./outputs_extended/.

 Run:  python topopt_extended.py

 IMPORTANT NOTES BEFORE YOU TRUST THE NUMBERS
   1. The new boundary conditions (bridge, tip_cantilever) are defined here for
      the first time. Sanity-check them once: run a single OC case and confirm
      the structure looks physically sensible before relying on the data.
   2. Default sweep sizes are modest so a first run completes quickly. Scale up
      the seed counts and mesh sizes (see CONFIG_EXT) for publication-strength
      statistics. Runtime grows roughly linearly with seeds and with the number
      of elements.
   3. Validate absolute compliance against a published 88-line reference before
      quoting figures, exactly as for the base study.

 UK English throughout.
================================================================================
"""

from __future__ import annotations
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Validated core (must be in the same directory)
from topopt_study import (
    TopOptProblem, run_optimisation, OPTIMISERS,
    binary_iou, density_rmse, hole_count, detect_convergence, PALETTE,
)


# ----------------------------------------------------------------------------
# CONFIGURATION  (edit here; defaults are modest for a quick first run)
# ----------------------------------------------------------------------------
CONFIG_EXT = {
    "outdir": "outputs_extended",
    "maxiter": 160,

    # base sweep reused by the statistics and cost studies
    "base_benchmark": "mbb",
    "base_nelx": 100, "base_nely": 40,
    "base_volfrac": 0.4, "base_penal": 3.0, "base_rmin": 2.0,
    "base_seeds": [0, 1, 2, 3, 4, 5],          # raise to 20+ for the paper

    # A. mesh-independence
    "mesh_list": [(60, 20), (90, 30), (120, 40), (150, 50)],  # add (180,60) etc.
    "mesh_seeds": [0, 1, 2],
    "mesh_ref_nelx": 120,        # filter radius is scaled to keep it physically constant

    # B. filter radius
    "rmin_list": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    "rmin_seeds": [0, 1, 2],

    # C. volume fraction
    "volfrac_list": [0.2, 0.3, 0.4, 0.5, 0.6],
    "volfrac_seeds": [0, 1, 2],

    # D. penalisation continuation
    "continuation_seeds": [0, 1, 2, 3],
    "penal_final": 3.0,
    "penal_steps": 40,           # iterations over which penal ramps 1 -> penal_final

    # E. hyperparameter sensitivity
    "gd_lr_list": [0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
    "mma_move_list": [0.05, 0.1, 0.2, 0.3, 0.5],
    "hyper_seeds": [0, 1, 2],

    # F. multi-benchmark
    "benchmarks": ["mbb", "cantilever", "bridge", "tip_cantilever"],
    "bench_nelx": 100, "bench_nely": 40,
    "bench_seeds": [0, 1, 2, 3],

    # optimiser hyperparameters used everywhere unless overridden
    "optimisers": {"OC": {}, "GD": {"lr": 0.5}, "MMA": {}},

    # statistics
    "bootstrap_n": 10000,
}


# ----------------------------------------------------------------------------
# NEW BOUNDARY CONDITIONS  (set f and free on an already-built problem)
# ----------------------------------------------------------------------------
def apply_bc(prob, bc):
    """Overwrite prob.f and prob.free for a benchmark not built natively by
    TopOptProblem. Node (i, j) has index (nely+1)*i + j, with j=0 at the top;
    x-dof = 2*node, y-dof = 2*node + 1."""
    nelx, nely, ndof = prob.nelx, prob.nely, prob.ndof
    f = np.zeros((ndof, 1))
    dofs = np.arange(ndof)

    def node(i, j):
        return (nely + 1) * i + j

    if bc == "bridge":
        # simply supported beam: pin bottom-left, roller bottom-right,
        # downward point load at the bottom-centre (deck load).
        bl = node(0, nely)
        br = node(nelx, nely)
        mid = node(nelx // 2, nely)
        f[2 * mid + 1, 0] = -1.0
        fixed = np.array([2 * bl, 2 * bl + 1, 2 * br + 1])
    elif bc == "tip_cantilever":
        # left edge fully clamped, downward load at the BOTTOM of the right edge
        # (a tip-loaded cantilever; distinct from the native mid-edge cantilever).
        tip = node(nelx, nely)
        f[2 * tip + 1, 0] = -1.0
        fixed = np.arange(0, 2 * (nely + 1))
    else:
        raise ValueError(f"apply_bc does not handle '{bc}'")

    prob.f = f
    prob.free = np.setdiff1d(dofs, fixed)
    return prob


def make_problem(nelx, nely, volfrac, penal, rmin, bc):
    """Build a problem for any benchmark, native or extended."""
    if bc in ("mbb", "cantilever"):
        return TopOptProblem(nelx, nely, volfrac, penal=penal, rmin=rmin, bc=bc)
    prob = TopOptProblem(nelx, nely, volfrac, penal=penal, rmin=rmin, bc="mbb")
    return apply_bc(prob, bc)


# ----------------------------------------------------------------------------
# CUSTOM DRIVER: SIMP with penalisation continuation
# ----------------------------------------------------------------------------
def run_with_continuation(problem, name, maxiter, seed, penal_final,
                          penal_steps, opt_kwargs=None):
    """Same loop as the base driver, but the SIMP penalty is ramped linearly
    from 1.0 to penal_final over the first penal_steps iterations. Returns the
    same result dictionary shape as run_optimisation."""
    rng = np.random.default_rng(seed)
    n = problem.nelx * problem.nely
    x = np.clip(problem.volfrac + 0.1 * (rng.random(n) - 0.5), 0.01, 0.99)
    x *= problem.volfrac / x.mean()
    x = np.clip(x, 0.01, 0.99)
    opt = OPTIMISERS[name](problem, **(opt_kwargs or {}))
    if hasattr(opt, "_rng"):
        opt._rng = rng
    xPhys = problem.filt.filter_density(x)
    hist = {"compliance": [], "change": [], "vol": [], "grayness": []}
    for it in range(maxiter):
        frac = min(1.0, it / max(1, penal_steps))
        problem.penal = 1.0 + frac * (penal_final - 1.0)
        c, dc, vol = problem.analyse(xPhys)
        dv = np.ones(n) / n
        dc = problem.filt.filter_sensitivity(dc)
        dv = problem.filt.filter_sensitivity(dv)
        xold = x.copy()
        x = opt.step(x, dc, dv)
        xPhys = problem.filt.filter_density(x)
        hist["compliance"].append(float(c))
        hist["change"].append(float(np.abs(x - xold).max()))
        hist["vol"].append(float(vol))
        hist["grayness"].append(float(4 * np.mean(xPhys * (1 - xPhys))))
    problem.penal = penal_final
    return {"optimiser": name, "seed": seed, "xPhys": xPhys,
            "compliance": hist["compliance"][-1],
            "converged_at": detect_convergence(hist["compliance"]),
            "grayness": hist["grayness"][-1],
            "final_volume": float(xPhys.mean()), "history": hist}


def run_timed(problem, name, maxiter, seed, opt_kwargs=None):
    """Wrap the base driver with a wall-clock timer."""
    t0 = time.perf_counter()
    r = run_optimisation(problem, name, maxiter=maxiter, seed=seed,
                         x0="random", opt_kwargs=opt_kwargs)
    r["wall_time"] = time.perf_counter() - t0
    r["fe_solves"] = maxiter
    return r


# ----------------------------------------------------------------------------
# STATISTICS HELPERS
# ----------------------------------------------------------------------------
def bootstrap_ci(samples, n_boot=10000, ci=95, rng=None):
    """Percentile bootstrap confidence interval for the mean."""
    rng = rng or np.random.default_rng(0)
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 2:
        m = float(samples.mean()) if len(samples) else float("nan")
        return m, m, m
    means = np.array([rng.choice(samples, len(samples), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(samples.mean()), float(lo), float(hi)


def cliffs_delta(a, b):
    """Cliff's delta effect size in [-1, 1]. Positive means a tends to exceed b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    n = len(a) * len(b)
    return float((gt - lt) / n) if n else 0.0


def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U p-value; returns nan if samples too small."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


# ----------------------------------------------------------------------------
# LATEX TABLE WRITER
# ----------------------------------------------------------------------------
def write_latex_table(path, caption, label, header, rows, col_format=None):
    """Emit a booktabs LaTeX table. header is a list of column titles; rows is
    a list of lists of cell strings."""
    ncol = len(header)
    col_format = col_format or ("l" + "r" * (ncol - 1))
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# ============================================================================
# STUDY A: MESH INDEPENDENCE
# ============================================================================
def study_mesh_independence(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rows = []          # (nelx, nely, ndof, optimiser, mean_c, sd_c, mean_gray)
    iou_oc_mma = []    # within-mesh OC vs MMA agreement at seed 0
    for (nelx, nely) in cfg["mesh_list"]:
        # scale the filter radius so the physical length is mesh-independent
        rmin = cfg["base_rmin"] * nelx / cfg["mesh_ref_nelx"]
        ndof = 2 * (nelx + 1) * (nely + 1)
        seed0_designs = {}
        for name in names:
            cs, gs = [], []
            for seed in cfg["mesh_seeds"]:
                prob = make_problem(nelx, nely, cfg["base_volfrac"],
                                    cfg["base_penal"], rmin, cfg["base_benchmark"])
                r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                     seed=seed, x0="random",
                                     opt_kwargs=cfg["optimisers"][name])
                cs.append(r["compliance"]); gs.append(r["grayness"])
                if seed == cfg["mesh_seeds"][0]:
                    seed0_designs[name] = r["xPhys"]
            rows.append([nelx, nely, ndof, name,
                         np.mean(cs), np.std(cs), np.mean(gs)])
            print(f"[mesh] {nelx}x{nely} {name}: c={np.mean(cs):.2f}")
        if "OC" in seed0_designs and "MMA" in seed0_designs:
            iou_oc_mma.append((ndof,
                               binary_iou(seed0_designs["OC"], seed0_designs["MMA"])))

    # chart: compliance vs DOFs (per optimiser)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in names:
        sub = [r for r in rows if r[3] == name]
        ndofs = [r[2] for r in sub]; mc = [r[4] for r in sub]; sc = [r[5] for r in sub]
        ax1.errorbar(ndofs, mc, yerr=sc, marker="o", capsize=4,
                     color=PALETTE[name], label=name)
    ax1.set_xscale("log"); ax1.set_xlabel("Degrees of freedom")
    ax1.set_ylabel("Final compliance (J)")
    ax1.set_title("Mesh independence of compliance")
    ax1.legend(title="Optimiser"); ax1.grid(alpha=0.3)
    if iou_oc_mma:
        nd = [x[0] for x in iou_oc_mma]; iv = [x[1] for x in iou_oc_mma]
        ax2.plot(nd, iv, "o-", color="#555")
        ax2.set_ylim(0, 1); ax2.set_xscale("log")
        ax2.set_xlabel("Degrees of freedom")
        ax2.set_ylabel("IoU (OC vs MMA, seed 0)")
        ax2.set_title("Cross-optimiser agreement vs mesh\n(non-uniqueness persists if < 1)")
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extA_mesh_independence.png"), dpi=150)
    plt.close(fig)

    write_csv(os.path.join(out, "extA_mesh.csv"),
              ["nelx", "nely", "ndof", "optimiser", "mean_c", "sd_c", "mean_gray"],
              [[a, b, c, d, f"{e:.3f}", f"{f:.3f}", f"{g:.3f}"]
               for a, b, c, d, e, f, g in rows])
    write_latex_table(
        os.path.join(out, "extA_mesh.tex"),
        "Mesh-independence study: final compliance (mean over seeds) at each "
        "resolution, with the filter radius scaled to keep the physical "
        "length scale constant.",
        "tab:mesh",
        ["$n_x \\times n_y$", "DOF"] + names,
        _pivot_rows(rows, cfg["mesh_list"], names))
    return rows


def _pivot_rows(rows, mesh_list, names):
    """Helper: one table row per mesh, one column per optimiser (mean_c)."""
    out = []
    for (nelx, nely) in mesh_list:
        ndof = 2 * (nelx + 1) * (nely + 1)
        cells = [f"${nelx}\\times{nely}$", ndof]
        for name in names:
            match = [r for r in rows if r[0] == nelx and r[3] == name]
            cells.append(f"{match[0][4]:.2f}" if match else "--")
        out.append(cells)
    return out


# ============================================================================
# STUDY B: FILTER RADIUS SENSITIVITY
# ============================================================================
def study_filter_radius(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rows = []
    for rmin in cfg["rmin_list"]:
        for name in names:
            cs, gs = [], []
            for seed in cfg["rmin_seeds"]:
                prob = make_problem(cfg["base_nelx"], cfg["base_nely"],
                                    cfg["base_volfrac"], cfg["base_penal"],
                                    rmin, cfg["base_benchmark"])
                r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                     seed=seed, x0="random",
                                     opt_kwargs=cfg["optimisers"][name])
                cs.append(r["compliance"]); gs.append(r["grayness"])
            rows.append([rmin, name, np.mean(cs), np.std(cs), np.mean(gs)])
            print(f"[rmin] {rmin} {name}: c={np.mean(cs):.2f} gray={np.mean(gs):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in names:
        sub = [r for r in rows if r[1] == name]
        rr = [r[0] for r in sub]
        ax1.errorbar(rr, [r[2] for r in sub], yerr=[r[3] for r in sub],
                     marker="o", capsize=4, color=PALETTE[name], label=name)
        ax2.plot(rr, [r[4] for r in sub], "o-", color=PALETTE[name], label=name)
    ax1.set_xlabel("Filter radius $r_{min}$ (elements)")
    ax1.set_ylabel("Final compliance (J)")
    ax1.set_title("Compliance vs filter radius"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Filter radius $r_{min}$ (elements)")
    ax2.set_ylabel("Grayness"); ax2.set_title("Discreteness vs filter radius")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extB_filter_radius.png"), dpi=150)
    plt.close(fig)

    write_csv(os.path.join(out, "extB_filter.csv"),
              ["rmin", "optimiser", "mean_c", "sd_c", "mean_gray"],
              [[a, b, f"{c:.3f}", f"{d:.3f}", f"{e:.3f}"] for a, b, c, d, e in rows])
    return rows


# ============================================================================
# STUDY C: VOLUME FRACTION SWEEP
# ============================================================================
def study_volume_fraction(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rows = []
    for vf in cfg["volfrac_list"]:
        for name in names:
            cs = []
            for seed in cfg["volfrac_seeds"]:
                prob = make_problem(cfg["base_nelx"], cfg["base_nely"], vf,
                                    cfg["base_penal"], cfg["base_rmin"],
                                    cfg["base_benchmark"])
                r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                     seed=seed, x0="random",
                                     opt_kwargs=cfg["optimisers"][name])
                cs.append(r["compliance"])
            rows.append([vf, name, np.mean(cs), np.std(cs)])
            print(f"[volfrac] {vf} {name}: c={np.mean(cs):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in names:
        sub = [r for r in rows if r[1] == name]
        vfs = [r[0] for r in sub]
        ax1.errorbar(vfs, [r[2] for r in sub], yerr=[r[3] for r in sub],
                     marker="o", capsize=4, color=PALETTE[name], label=name)
    ax1.set_xlabel("Volume fraction"); ax1.set_ylabel("Final compliance (J)")
    ax1.set_title("Compliance vs volume fraction"); ax1.legend(); ax1.grid(alpha=0.3)
    # relative gap of each method to the best (lowest) compliance at each volfrac
    for name in names:
        gaps = []
        for vf in cfg["volfrac_list"]:
            at_vf = [r[2] for r in rows if r[0] == vf]
            best = min(at_vf)
            this = [r[2] for r in rows if r[0] == vf and r[1] == name][0]
            gaps.append(100 * (this - best) / best)
        ax2.plot(cfg["volfrac_list"], gaps, "o-", color=PALETTE[name], label=name)
    ax2.set_xlabel("Volume fraction")
    ax2.set_ylabel("Compliance gap to best (%)")
    ax2.set_title("Relative penalty of each optimiser vs volume fraction")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extC_volume_fraction.png"), dpi=150)
    plt.close(fig)

    write_csv(os.path.join(out, "extC_volfrac.csv"),
              ["volfrac", "optimiser", "mean_c", "sd_c"],
              [[a, b, f"{c:.3f}", f"{d:.3f}"] for a, b, c, d in rows])
    return rows


# ============================================================================
# STUDY D: PENALISATION CONTINUATION
# ============================================================================
def study_penal_continuation(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rows = []   # (optimiser, scheme, mean_c, sd_c, mean_gray, sd_gray)
    for name in names:
        for scheme in ("fixed", "continuation"):
            cs, gs = [], []
            for seed in cfg["continuation_seeds"]:
                prob = make_problem(cfg["base_nelx"], cfg["base_nely"],
                                    cfg["base_volfrac"], cfg["penal_final"],
                                    cfg["base_rmin"], cfg["base_benchmark"])
                if scheme == "fixed":
                    r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                         seed=seed, x0="random",
                                         opt_kwargs=cfg["optimisers"][name])
                else:
                    r = run_with_continuation(prob, name, cfg["maxiter"], seed,
                                              cfg["penal_final"],
                                              cfg["penal_steps"],
                                              opt_kwargs=cfg["optimisers"][name])
                cs.append(r["compliance"]); gs.append(r["grayness"])
            rows.append([name, scheme, np.mean(cs), np.std(cs),
                         np.mean(gs), np.std(gs)])
            print(f"[continuation] {name} {scheme}: c={np.mean(cs):.2f} "
                  f"gray={np.mean(gs):.3f}")

    x = np.arange(len(names)); w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for k, scheme in enumerate(("fixed", "continuation")):
        cvals = [next(r for r in rows if r[0] == n and r[1] == scheme)[2] for n in names]
        cerr = [next(r for r in rows if r[0] == n and r[1] == scheme)[3] for n in names]
        gvals = [next(r for r in rows if r[0] == n and r[1] == scheme)[4] for n in names]
        gerr = [next(r for r in rows if r[0] == n and r[1] == scheme)[5] for n in names]
        ax1.bar(x + (k - 0.5) * w, cvals, w, yerr=cerr, capsize=4,
                label=scheme, alpha=0.8)
        ax2.bar(x + (k - 0.5) * w, gvals, w, yerr=gerr, capsize=4,
                label=scheme, alpha=0.8)
    for ax, ttl, yl in [(ax1, "Compliance: fixed vs continuation", "Final compliance (J)"),
                        (ax2, "Grayness: fixed vs continuation", "Grayness")]:
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title(ttl)
        ax.set_ylabel(yl); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extD_continuation.png"), dpi=150)
    plt.close(fig)

    write_csv(os.path.join(out, "extD_continuation.csv"),
              ["optimiser", "scheme", "mean_c", "sd_c", "mean_gray", "sd_gray"],
              [[a, b, f"{c:.3f}", f"{d:.3f}", f"{e:.3f}", f"{f:.3f}"]
               for a, b, c, d, e, f in rows])
    return rows


# ============================================================================
# STUDY E: HYPERPARAMETER SENSITIVITY
# ============================================================================
def study_hyperparameter(cfg):
    out = cfg["outdir"]
    gd_rows, mma_rows = [], []
    for lr in cfg["gd_lr_list"]:
        cs, gs = [], []
        for seed in cfg["hyper_seeds"]:
            prob = make_problem(cfg["base_nelx"], cfg["base_nely"],
                                cfg["base_volfrac"], cfg["base_penal"],
                                cfg["base_rmin"], cfg["base_benchmark"])
            r = run_optimisation(prob, "GD", maxiter=cfg["maxiter"], seed=seed,
                                 x0="random", opt_kwargs={"lr": lr})
            cs.append(r["compliance"]); gs.append(r["grayness"])
        gd_rows.append([lr, np.mean(cs), np.std(cs), np.mean(gs)])
        print(f"[GD lr] {lr}: c={np.mean(cs):.2f}")
    for mv in cfg["mma_move_list"]:
        cs, gs = [], []
        for seed in cfg["hyper_seeds"]:
            prob = make_problem(cfg["base_nelx"], cfg["base_nely"],
                                cfg["base_volfrac"], cfg["base_penal"],
                                cfg["base_rmin"], cfg["base_benchmark"])
            r = run_optimisation(prob, "MMA", maxiter=cfg["maxiter"], seed=seed,
                                 x0="random", opt_kwargs={"move": mv})
            cs.append(r["compliance"]); gs.append(r["grayness"])
        mma_rows.append([mv, np.mean(cs), np.std(cs), np.mean(gs)])
        print(f"[MMA move] {mv}: c={np.mean(cs):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar([r[0] for r in gd_rows], [r[1] for r in gd_rows],
                 yerr=[r[2] for r in gd_rows], marker="o", capsize=4,
                 color=PALETTE["GD"])
    ax1.set_xscale("log"); ax1.set_xlabel("GD learning rate")
    ax1.set_ylabel("Final compliance (J)")
    ax1.set_title("GD sensitivity to learning rate"); ax1.grid(alpha=0.3)
    ax2.errorbar([r[0] for r in mma_rows], [r[1] for r in mma_rows],
                 yerr=[r[2] for r in mma_rows], marker="s", capsize=4,
                 color=PALETTE["MMA"])
    ax2.set_xlabel("MMA move limit"); ax2.set_ylabel("Final compliance (J)")
    ax2.set_title("MMA sensitivity to move limit"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extE_hyperparameter.png"), dpi=150)
    plt.close(fig)

    write_csv(os.path.join(out, "extE_gd_lr.csv"),
              ["lr", "mean_c", "sd_c", "mean_gray"],
              [[a, f"{b:.3f}", f"{c:.3f}", f"{d:.3f}"] for a, b, c, d in gd_rows])
    write_csv(os.path.join(out, "extE_mma_move.csv"),
              ["move", "mean_c", "sd_c", "mean_gray"],
              [[a, f"{b:.3f}", f"{c:.3f}", f"{d:.3f}"] for a, b, c, d in mma_rows])
    return gd_rows, mma_rows


# ============================================================================
# STUDY F: MULTI-BENCHMARK
# ============================================================================
def study_multibenchmark(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rows = []
    per_bench_designs = {}
    for bc in cfg["benchmarks"]:
        seed0 = {}
        for name in names:
            cs = []
            for seed in cfg["bench_seeds"]:
                prob = make_problem(cfg["bench_nelx"], cfg["bench_nely"],
                                    cfg["base_volfrac"], cfg["base_penal"],
                                    cfg["base_rmin"], bc)
                r = run_optimisation(prob, name, maxiter=cfg["maxiter"],
                                     seed=seed, x0="random",
                                     opt_kwargs=cfg["optimisers"][name])
                cs.append(r["compliance"])
                if seed == cfg["bench_seeds"][0]:
                    seed0[name] = r["xPhys"]
            rows.append([bc, name, np.mean(cs), np.std(cs)])
            print(f"[bench] {bc} {name}: c={np.mean(cs):.2f}")
        per_bench_designs[bc] = seed0

    # grouped bar: compliance per optimiser per benchmark
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cfg["benchmarks"])); w = 0.8 / len(names)
    for k, name in enumerate(names):
        vals = [next(r for r in rows if r[0] == bc and r[1] == name)[2]
                for bc in cfg["benchmarks"]]
        err = [next(r for r in rows if r[0] == bc and r[1] == name)[3]
               for bc in cfg["benchmarks"]]
        ax.bar(x + (k - (len(names) - 1) / 2) * w, vals, w, yerr=err,
               capsize=3, color=PALETTE[name], label=name, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cfg["benchmarks"])
    ax.set_ylabel("Final compliance (J)")
    ax.set_title("Optimiser comparison across benchmarks")
    ax.legend(title="Optimiser"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extF_multibenchmark.png"), dpi=150)
    plt.close(fig)

    # design strips per benchmark (seed 0)
    nb = len(cfg["benchmarks"])
    fig2, axes = plt.subplots(nb, len(names),
                              figsize=(3.2 * len(names), 1.5 * nb))
    axes = np.atleast_2d(axes)
    for i, bc in enumerate(cfg["benchmarks"]):
        for j, name in enumerate(names):
            img = per_bench_designs[bc][name].reshape(cfg["bench_nelx"],
                                                      cfg["bench_nely"]).T
            ax = axes[i, j]
            ax.imshow(-img, cmap="gray", vmin=-1, vmax=0)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(bc, fontsize=10)
            if i == 0:
                ax.set_title(name, color=PALETTE[name], fontweight="bold")
    fig2.suptitle("Final designs by benchmark and optimiser (seed 0)")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(os.path.join(out, "extF_designs_by_benchmark.png"), dpi=150)
    plt.close(fig2)

    write_csv(os.path.join(out, "extF_multibenchmark.csv"),
              ["benchmark", "optimiser", "mean_c", "sd_c"],
              [[a, b, f"{c:.3f}", f"{d:.3f}"] for a, b, c, d in rows])
    write_latex_table(
        os.path.join(out, "extF_multibenchmark.tex"),
        "Final compliance (mean $\\pm$ s.d. over seeds) for each optimiser on "
        "each benchmark.",
        "tab:multibench",
        ["Benchmark"] + names,
        [[bc] + [f"{next(r for r in rows if r[0]==bc and r[1]==n)[2]:.2f} $\\pm$ "
                 f"{next(r for r in rows if r[0]==bc and r[1]==n)[3]:.2f}"
                 for n in names] for bc in cfg["benchmarks"]])
    return rows


# ============================================================================
# STUDY G + H: STATISTICS AND COST (share one base sweep)
# ============================================================================
def study_statistics_and_cost(cfg):
    out = cfg["outdir"]
    names = list(cfg["optimisers"])
    rng = np.random.default_rng(0)
    by_opt_c = {n: [] for n in names}
    by_opt_t = {n: [] for n in names}
    by_opt_conv = {n: [] for n in names}
    prob_kwargs = (cfg["base_nelx"], cfg["base_nely"], cfg["base_volfrac"],
                   cfg["base_penal"], cfg["base_rmin"], cfg["base_benchmark"])
    for name in names:
        for seed in cfg["base_seeds"]:
            prob = make_problem(*prob_kwargs)
            r = run_timed(prob, name, cfg["maxiter"], seed,
                          opt_kwargs=cfg["optimisers"][name])
            by_opt_c[name].append(r["compliance"])
            by_opt_t[name].append(r["wall_time"])
            by_opt_conv[name].append(r["converged_at"])
            print(f"[base] {name} seed {seed}: c={r['compliance']:.2f} "
                  f"t={r['wall_time']:.2f}s")

    # ---- G. statistics: bootstrap CI forest plot + pairwise test table ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(names))
    for i, name in enumerate(names):
        m, lo, hi = bootstrap_ci(by_opt_c[name], cfg["bootstrap_n"], rng=rng)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                    color=PALETTE[name], capsize=5, markersize=9)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_xlabel("Final compliance (J), mean with 95% bootstrap CI")
    ax.set_title("Compliance estimates with uncertainty")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extG_bootstrap_ci.png"), dpi=150)
    plt.close(fig)

    stat_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = mannwhitney_p(by_opt_c[a], by_opt_c[b])
            d = cliffs_delta(by_opt_c[a], by_opt_c[b])
            stat_rows.append([f"{a} vs {b}",
                              f"{np.mean(by_opt_c[a]):.2f}",
                              f"{np.mean(by_opt_c[b]):.2f}",
                              ("%.4f" % p) if p == p else "n/a",
                              f"{d:.2f}"])
    write_csv(os.path.join(out, "extG_pairwise_tests.csv"),
              ["pair", "mean_A", "mean_B", "mannwhitney_p", "cliffs_delta"],
              stat_rows)
    write_latex_table(
        os.path.join(out, "extG_pairwise_tests.tex"),
        "Pairwise comparison of final compliance between optimisers: "
        "Mann-Whitney $U$ two-sided $p$-value and Cliff's $\\delta$ effect size. "
        "Small seed counts give coarse $p$-values; increase the seed count for "
        "the final analysis.",
        "tab:stats",
        ["Comparison", "Mean A", "Mean B", "$p$", "Cliff's $\\delta$"],
        stat_rows)

    # ---- H. cost: bar of time per FE solve + efficiency scatter ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tps = [np.mean(by_opt_t[n]) / cfg["maxiter"] for n in names]
    tps_e = [np.std(by_opt_t[n]) / cfg["maxiter"] for n in names]
    ax1.bar(names, tps, yerr=tps_e, capsize=5,
            color=[PALETTE[n] for n in names], alpha=0.8, edgecolor="k")
    ax1.set_ylabel("Wall-time per FE solve (s)")
    ax1.set_title("Per-iteration cost"); ax1.grid(alpha=0.3, axis="y")
    for n in names:
        # effective time to converge = time per solve * iterations to converge
        eff_t = (np.mean(by_opt_t[n]) / cfg["maxiter"]) * np.mean(by_opt_conv[n])
        ax2.scatter(eff_t, np.mean(by_opt_c[n]), s=130, color=PALETTE[n],
                    edgecolor="k", label=n)
    ax2.set_xlabel("Estimated time to convergence (s)")
    ax2.set_ylabel("Final compliance (J)")
    ax2.set_title("Efficiency frontier (lower-left is better)")
    ax2.legend(title="Optimiser"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "extH_cost.png"), dpi=150)
    plt.close(fig)

    cost_rows = []
    for n in names:
        cost_rows.append([
            n,
            f"{np.mean(by_opt_t[n]):.3f}",
            f"{np.mean(by_opt_t[n]) / cfg['maxiter'] * 1000:.2f}",
            f"{np.mean(by_opt_conv[n]):.1f}",
            f"{np.mean(by_opt_c[n]):.2f}",
        ])
    write_csv(os.path.join(out, "extH_cost.csv"),
              ["optimiser", "total_time_s", "ms_per_solve",
               "mean_conv_iter", "mean_compliance"],
              cost_rows)
    write_latex_table(
        os.path.join(out, "extH_cost.tex"),
        "Computational cost: total wall-time for the fixed horizon, time per "
        "finite-element solve, mean iterations to convergence, and mean final "
        "compliance.",
        "tab:cost",
        ["Optimiser", "Total (s)", "ms/solve", "Conv. iter", "Compliance"],
        cost_rows)

    # ---- summary table across all base results ----
    summ_rows = []
    for n in names:
        cs = by_opt_c[n]
        summ_rows.append([n, f"{np.mean(cs):.2f}", f"{np.std(cs):.2f}",
                          f"{np.min(cs):.2f}", f"{np.max(cs):.2f}",
                          f"{np.mean(by_opt_conv[n]):.1f}"])
    write_latex_table(
        os.path.join(out, "extG_summary.tex"),
        "Summary of final compliance across seeds for each optimiser on the "
        "base benchmark.",
        "tab:summary",
        ["Optimiser", "Mean", "s.d.", "Min", "Max", "Conv. iter"],
        summ_rows)
    return by_opt_c, by_opt_t, by_opt_conv


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = CONFIG_EXT
    os.makedirs(cfg["outdir"], exist_ok=True)

    # toggle studies on or off here
    studies = {
        "A_mesh": True,
        "B_filter": True,
        "C_volfrac": True,
        "D_continuation": True,
        "E_hyperparameter": True,
        "F_multibenchmark": True,
        "GH_stats_cost": True,
    }

    print("=" * 70)
    print(" EXTENDED TOPOLOGY-OPTIMISATION STUDIES")
    print(" Output directory:", cfg["outdir"])
    print("=" * 70)

    if studies["A_mesh"]:
        print("\n[A] Mesh independence ...");           study_mesh_independence(cfg)
    if studies["B_filter"]:
        print("\n[B] Filter radius ...");                study_filter_radius(cfg)
    if studies["C_volfrac"]:
        print("\n[C] Volume fraction ...");              study_volume_fraction(cfg)
    if studies["D_continuation"]:
        print("\n[D] Penalisation continuation ...");    study_penal_continuation(cfg)
    if studies["E_hyperparameter"]:
        print("\n[E] Hyperparameter sensitivity ...");   study_hyperparameter(cfg)
    if studies["F_multibenchmark"]:
        print("\n[F] Multi-benchmark ...");              study_multibenchmark(cfg)
    if studies["GH_stats_cost"]:
        print("\n[G+H] Statistics and cost ...");        study_statistics_and_cost(cfg)

    print("\nAll requested studies complete. Charts, CSVs and LaTeX tables are "
          "in ./" + cfg["outdir"] + "/")


if __name__ == "__main__":
    main()
