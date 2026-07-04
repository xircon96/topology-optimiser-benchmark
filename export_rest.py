import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "figs_export"
PALETTE = {"OC": "#1f77b4", "GD": "#d62728", "MMA": "#2ca02c"}

def save_all(fig, stem):
    fig.savefig(os.path.join(OUT, stem+".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem+".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem+".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

# ---- design grid from npz (vector axes/labels; raster density imagery) ----
d = np.load("outputs_study/designs.npz")
NAMES = ["OC", "GD", "MMA"]; seeds = [0, 1, 2, 3]
nelx, nely = 100, 40
# need compliance annotations: read results.csv
import csv
res = {}
with open("outputs_study/results.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        res[(r["optimiser"], int(r["seed"]))] = float(r["final_compliance"])
fig, axes = plt.subplots(len(NAMES), len(seeds), figsize=(3.2*len(seeds), 1.5*len(NAMES)))
for i, n in enumerate(NAMES):
    for j, s in enumerate(seeds):
        key = f"{n}_seed{s}"
        img = d[key].reshape(nelx, nely).T
        ax = axes[i, j]
        ax.imshow(-img, cmap="gray", vmin=-1, vmax=0, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0: ax.set_ylabel(n, fontsize=12, color=PALETTE[n], fontweight="bold")
        if i == 0: ax.set_title(f"seed {s}", fontsize=10)
        c = res.get((n, s))
        if c is not None:
            ax.text(0.97, 0.06, f"c={c:.0f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
fig.suptitle("Final designs (MBB benchmark, vol=0.4)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_all(fig, "fig_design_grid")
print("regenerated design grid as vector")

# ---- convert PNG-only figures into PDF/SVG/TIFF (raster-embedded) ----
png_only = {
    "fig_convergence":            "outputs_study/fig1_convergence.png",
    "fig_change_grayness":        "outputs_study/fig2_change_grayness.png",
    "fig_agreement":              "outputs_study/fig4_agreement.png",
    "fig_compliance_vs_grayness": "outputs_study/fig6_compliance_vs_grayness.png",
    "fig_bootstrap_ci":           "outputs_extended/extG_bootstrap_ci.png",
    "fig_multibench_gallery":     "outputs_extended/extF_designs_by_benchmark.png",
}
for stem, src in png_only.items():
    im = Image.open(src).convert("RGB")
    dpi = im.info.get("dpi", (150, 150))
    # PDF and TIFF directly from PIL (preserves raster at native resolution)
    im.save(os.path.join(OUT, stem+".pdf"), "PDF", resolution=dpi[0])
    im.save(os.path.join(OUT, stem+".tiff"), "TIFF", dpi=(300, 300), compression="tiff_lzw")
    # SVG: embed the PNG as a base64 image so the file is a valid standalone SVG
    import base64
    with open(src, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    w, h = im.size
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
           f'<image width="{w}" height="{h}" '
           f'xlink:href="data:image/png;base64,{b64}"/></svg>')
    with open(os.path.join(OUT, stem+".svg"), "w") as fh:
        fh.write(svg)
print("converted PNG-only figures:", list(png_only))
