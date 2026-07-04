"""
experiment.py
-------------
The study harness: runs each optimiser across multiple random seeds on a
chosen benchmark, then reports distributions of compliance and convergence,
and the pairwise topological agreement between final designs.

This is the core data-collection script for the paper. Everything is
reproducible: fix the seed list and the run will reproduce exactly.

Usage:
    python experiment.py
Outputs:
    results_<bc>.csv         per-run descriptors
    iou_<bc>.csv             pairwise IoU between mean designs (seed 0)
    designs_<bc>.npz         all final density fields for later plotting
"""

from __future__ import annotations
import csv
import numpy as np
from topopt_core import TopOptProblem
from run import run_optimisation
from metrics import summarise, binary_iou, density_rmse

# ---- experiment configuration (edit here) --------------------------------
BENCHMARK = "mbb"            # "mbb" or "cantilever"
NELX, NELY = 120, 40
VOLFRAC = 0.4
PENAL, RMIN = 3.0, 2.0
MAXITER = 300
SEEDS = list(range(8))       # 8 seeds per optimiser; raise for the final paper
OPT_CONFIG = {
    "OC": {},
    "GD": {"lr": 0.5},
    "MMA": {},
}
# --------------------------------------------------------------------------


def main():
    prob = TopOptProblem(NELX, NELY, VOLFRAC, penal=PENAL, rmin=RMIN,
                         bc=BENCHMARK)
    all_results = []
    designs = {}

    for name, kw in OPT_CONFIG.items():
        for seed in SEEDS:
            # random feasible start so seeds actually differ
            r = run_optimisation(prob, name, maxiter=MAXITER, seed=seed,
                                 x0="random", opt_kwargs=kw)
            all_results.append(r)
            designs[f"{name}_seed{seed}"] = r["xPhys"]
            print(f"{name:4s} seed {seed}: c={r['compliance']:8.3f} "
                  f"conv@{r['converged_at']:3d} gray={r['grayness']:.3f}")

    # per-run descriptor table
    rows, _ = summarise(all_results, NELX, NELY)
    with open(f"results_{BENCHMARK}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # distribution summary per optimiser
    print("\n--- compliance distribution (across seeds) ---")
    print(f"{'opt':5s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s} "
          f"{'conv_mean':>9s}")
    for name in OPT_CONFIG:
        cs = [r["compliance"] for r in all_results if r["optimiser"] == name]
        cv = [r["converged_at"] for r in all_results if r["optimiser"] == name]
        print(f"{name:5s} {np.mean(cs):9.3f} {np.std(cs):8.3f} "
              f"{np.min(cs):9.3f} {np.max(cs):9.3f} {np.mean(cv):9.1f}")

    # cross-optimiser topological agreement (seed 0 designs)
    print("\n--- pairwise IoU / density-RMSE between optimisers (seed 0) ---")
    names = list(OPT_CONFIG)
    base = {n: designs[f"{n}_seed0"] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a:4s} vs {b:4s}: IoU={binary_iou(base[a], base[b]):.3f}  "
                  f"RMSE={density_rmse(base[a], base[b]):.3f}")

    # within-optimiser reproducibility (IoU across seeds, same optimiser)
    print("\n--- within-optimiser IoU across seed pairs (reproducibility) ---")
    for name in OPT_CONFIG:
        ds = [designs[f"{name}_seed{s}"] for s in SEEDS]
        ious = [binary_iou(ds[i], ds[j])
                for i in range(len(ds)) for j in range(i + 1, len(ds))]
        print(f"  {name:4s}: mean IoU={np.mean(ious):.3f} "
              f"(min {np.min(ious):.3f})")

    np.savez_compressed(f"designs_{BENCHMARK}.npz", **designs)
    print(f"\nSaved results_{BENCHMARK}.csv and designs_{BENCHMARK}.npz")


if __name__ == "__main__":
    main()
