"""
Export result figures in SVG, PDF and TIFF.
- Data-driven plots are regenerated as TRUE VECTOR (PDF, SVG) from the saved
  CSV/NPZ data, plus a 600 dpi TIFF, so they are sharp at any size.
- Figures whose full plot data was not saved (convergence histories, the
  agreement matrix, bootstrap samples) are converted from the published PNG
  into the three formats, so they match the manuscript exactly.
UK English. Colours match the paper palette.
"""
import os, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "figs_export"
os.makedirs(OUT, exist_ok=True)
PALETTE = {"OC": "#1f77b4", "GD": "#d62728", "MMA": "#2ca02c"}
NAMES = ["OC", "GD", "MMA"]

def save_all(fig, stem):
    """Save a matplotlib figure as vector PDF, vector SVG and 600 dpi TIFF."""
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".tiff"), dpi=600,
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

# ---------- A. mesh independence (compliance vs DOF) ----------
rows = read_csv("outputs_extended/extA_mesh.csv")
fig, ax = plt.subplots(figsize=(7, 5))
for n in NAMES:
    sub = [r for r in rows if r["optimiser"] == n]
    nd = [int(r["ndof"]) for r in sub]
    mc = [float(r["mean_c"]) for r in sub]
    sc = [float(r["sd_c"]) for r in sub]
    ax.errorbar(nd, mc, yerr=sc, marker="o", capsize=4, color=PALETTE[n], label=n)
ax.set_xscale("log"); ax.set_xlabel("Degrees of freedom")
ax.set_ylabel("Final compliance (J)")
ax.set_title("Mesh independence of compliance")
ax.legend(title="Optimiser"); ax.grid(alpha=0.3)
save_all(fig, "figA_mesh_independence")

# ---------- B. filter radius (compliance + grayness) ----------
rows = read_csv("outputs_extended/extB_filter.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for n in NAMES:
    sub = [r for r in rows if r["optimiser"] == n]
    rr = [float(r["rmin"]) for r in sub]
    ax1.errorbar(rr, [float(r["mean_c"]) for r in sub],
                 yerr=[float(r["sd_c"]) for r in sub], marker="o", capsize=4,
                 color=PALETTE[n], label=n)
    ax2.plot(rr, [float(r["mean_gray"]) for r in sub], "o-", color=PALETTE[n], label=n)
ax1.set_xlabel("Filter radius $r_{min}$ (elements)"); ax1.set_ylabel("Final compliance (J)")
ax1.set_title("Compliance vs filter radius"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_xlabel("Filter radius $r_{min}$ (elements)"); ax2.set_ylabel("Grayness")
ax2.set_title("Discreteness vs filter radius"); ax2.legend(); ax2.grid(alpha=0.3)
save_all(fig, "figB_filter_radius")

# ---------- C. volume fraction (compliance + relative gap) ----------
rows = read_csv("outputs_extended/extC_volfrac.csv")
vfs = sorted(set(float(r["volfrac"]) for r in rows))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for n in NAMES:
    sub = [r for r in rows if r["optimiser"] == n]
    v = [float(r["volfrac"]) for r in sub]
    ax1.errorbar(v, [float(r["mean_c"]) for r in sub],
                 yerr=[float(r["sd_c"]) for r in sub], marker="o", capsize=4,
                 color=PALETTE[n], label=n)
ax1.set_xlabel("Volume fraction"); ax1.set_ylabel("Final compliance (J)")
ax1.set_title("Compliance vs volume fraction"); ax1.legend(); ax1.grid(alpha=0.3)
for n in NAMES:
    gaps = []
    for vf in vfs:
        at = [float(r["mean_c"]) for r in rows if float(r["volfrac"]) == vf]
        best = min(at)
        this = [float(r["mean_c"]) for r in rows
                if float(r["volfrac"]) == vf and r["optimiser"] == n][0]
        gaps.append(100 * (this - best) / best)
    ax2.plot(vfs, gaps, "o-", color=PALETTE[n], label=n)
ax2.set_xlabel("Volume fraction"); ax2.set_ylabel("Compliance gap to best (%)")
ax2.set_title("Relative penalty vs volume fraction"); ax2.legend(); ax2.grid(alpha=0.3)
save_all(fig, "figC_volume_fraction")

# ---------- D. continuation (compliance + grayness, fixed vs continuation) ----------
rows = read_csv("outputs_extended/extD_continuation.csv")
x = np.arange(len(NAMES)); w = 0.36
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for k, scheme in enumerate(("fixed", "continuation")):
    cv = [float(next(r for r in rows if r["optimiser"]==n and r["scheme"]==scheme)["mean_c"]) for n in NAMES]
    ce = [float(next(r for r in rows if r["optimiser"]==n and r["scheme"]==scheme)["sd_c"]) for n in NAMES]
    gv = [float(next(r for r in rows if r["optimiser"]==n and r["scheme"]==scheme)["mean_gray"]) for n in NAMES]
    ge = [float(next(r for r in rows if r["optimiser"]==n and r["scheme"]==scheme)["sd_gray"]) for n in NAMES]
    ax1.bar(x + (k - 0.5)*w, cv, w, yerr=ce, capsize=4, label=scheme, alpha=0.8)
    ax2.bar(x + (k - 0.5)*w, gv, w, yerr=ge, capsize=4, label=scheme, alpha=0.8)
for ax, ttl, yl in [(ax1, "Compliance: fixed vs continuation", "Final compliance (J)"),
                    (ax2, "Grayness: fixed vs continuation", "Grayness")]:
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_title(ttl)
    ax.set_ylabel(yl); ax.legend(); ax.grid(alpha=0.3, axis="y")
save_all(fig, "figD_continuation")

# ---------- E. hyperparameter (GD lr + MMA move) ----------
gd = read_csv("outputs_extended/extE_gd_lr.csv")
mm = read_csv("outputs_extended/extE_mma_move.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar([float(r["lr"]) for r in gd], [float(r["mean_c"]) for r in gd],
             yerr=[float(r["sd_c"]) for r in gd], marker="o", capsize=4, color=PALETTE["GD"])
ax1.set_xscale("log"); ax1.set_xlabel("GD learning rate"); ax1.set_ylabel("Final compliance (J)")
ax1.set_title("GD sensitivity to learning rate"); ax1.grid(alpha=0.3)
ax2.errorbar([float(r["move"]) for r in mm], [float(r["mean_c"]) for r in mm],
             yerr=[float(r["sd_c"]) for r in mm], marker="s", capsize=4, color=PALETTE["MMA"])
ax2.set_xlabel("MMA move limit"); ax2.set_ylabel("Final compliance (J)")
ax2.set_title("MMA sensitivity to move limit"); ax2.grid(alpha=0.3)
save_all(fig, "figE_hyperparameter")

# ---------- F. multibench bar ----------
rows = read_csv("outputs_extended/extF_multibenchmark.csv")
benches = []
for r in rows:
    if r["benchmark"] not in benches: benches.append(r["benchmark"])
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(benches)); w = 0.8/len(NAMES)
for k, n in enumerate(NAMES):
    vals = [float(next(r for r in rows if r["benchmark"]==b and r["optimiser"]==n)["mean_c"]) for b in benches]
    err = [float(next(r for r in rows if r["benchmark"]==b and r["optimiser"]==n)["sd_c"]) for b in benches]
    ax.bar(x + (k-(len(NAMES)-1)/2)*w, vals, w, yerr=err, capsize=3,
           color=PALETTE[n], label=n, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(benches); ax.set_ylabel("Final compliance (J)")
ax.set_title("Optimiser comparison across benchmarks")
ax.legend(title="Optimiser"); ax.grid(alpha=0.3, axis="y")
save_all(fig, "figF_multibenchmark")

# ---------- H. cost (per-solve + efficiency frontier) ----------
rows = read_csv("outputs_extended/extH_cost.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
tps = [float(next(r for r in rows if r["optimiser"]==n)["ms_per_solve"]) for n in NAMES]
ax1.bar(NAMES, tps, color=[PALETTE[n] for n in NAMES], alpha=0.8, edgecolor="k")
ax1.set_ylabel("Wall-time per FE solve (ms)"); ax1.set_title("Per-iteration cost")
ax1.grid(alpha=0.3, axis="y")
for n in NAMES:
    r = next(r for r in rows if r["optimiser"]==n)
    eff = float(r["ms_per_solve"])/1000.0 * float(r["mean_conv_iter"])
    ax2.scatter(eff, float(r["mean_compliance"]), s=130, color=PALETTE[n],
                edgecolor="k", label=n)
ax2.set_xlabel("Estimated time to convergence (s)"); ax2.set_ylabel("Final compliance (J)")
ax2.set_title("Efficiency frontier (lower-left is better)")
ax2.legend(title="Optimiser"); ax2.grid(alpha=0.3)
save_all(fig, "figH_cost")

# ---------- base distribution (from results.csv): compliance box + strip ----------
rows = read_csv("outputs_study/results.csv")
data = {n: [float(r["final_compliance"]) for r in rows if r["optimiser"]==n] for n in NAMES}
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([data[n] for n in NAMES], tick_labels=NAMES, patch_artist=True, widths=0.5)
for patch, n in zip(bp["boxes"], NAMES):
    patch.set_facecolor(PALETTE[n]); patch.set_alpha(0.4)
for i, n in enumerate(NAMES):
    ys = data[n]; xs = np.random.normal(i+1, 0.04, len(ys))
    ax.scatter(xs, ys, color=PALETTE[n], s=28, zorder=3, edgecolor="k", lw=0.4)
ax.set_ylabel("Final compliance (J)")
ax.set_title("Final-compliance distribution across seeds")
ax.grid(alpha=0.3, axis="y")
save_all(fig, "fig_distributions")

print("VECTOR regenerated:", sorted(set(f.split('.')[0] for f in os.listdir(OUT))))
