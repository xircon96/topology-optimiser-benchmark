# Optimiser-induced non-uniqueness in density-based topology optimisation

This folder holds the code and the results for the topology optimisation
benchmark study. Everything runs in pure Python using only NumPy, SciPy and
Matplotlib. No commercial finite element software and no external optimisation
library are needed. UK English is used throughout.

## Folder layout

- `code/` contains the full implementation.
- `results/data/` contains the numerical results as CSV files and the
  publication tables as LaTeX fragments.
- `results/figures_svg/` contains every figure as a scalable SVG.

## Code files

- `topopt_study.py` is the complete base study in one file. It holds the SIMP
  finite element physics, the three optimisers behind one shared interface, the
  self-contained Method of Moving Asymptotes, the fixed-horizon driver with
  post-hoc convergence detection, the topology-difference metrics, the
  seed-sweep experiment and the plotting routines.
- `topopt_extended.py` holds the additional studies, namely mesh independence,
  filter-radius sensitivity, the volume-fraction sweep, penalisation
  continuation, hyperparameter sensitivity, the multi-benchmark comparison, the
  statistical tests and the computational-cost analysis. It imports the core
  from `topopt_study.py`, so keep the two files together.
- `topopt_core.py`, `optimisers.py`, `mma.py`, `run.py`, `metrics.py`,
  `experiment.py`, `plot_designs.py` are the earlier modular version of the same
  base study, kept for reference.
- `make_fig1.py` draws the benchmark schematic.
- `export_figs.py` and `export_rest.py` regenerate the result figures as vector
  PDF and SVG plus a high-resolution raster copy from the saved data.

## Results

The CSV files in `results/data/` are the raw numbers behind each study, for
example `extA_mesh.csv` for the mesh-independence sweep and
`extG_pairwise_tests.csv` for the statistical comparison. The matching `.tex`
files are the same content formatted as journal tables. The SVG figures in
`results/figures_svg/` are the plots that appear in the paper.

## Reproducing

1. `python code/topopt_study.py` runs the base study and writes its figures.
2. `python code/topopt_extended.py` runs the extended studies and writes theirs.
3. `python code/make_fig1.py` draws the benchmark schematic.

## Honest notes on the numbers

The default seed counts and mesh sizes are deliberately modest so that a first
run finishes quickly. For publication-strength statistics, raise the seed counts
to twenty or more and increase the mesh sizes in the configuration block at the
top of each script. The close agreement between the Optimality Criteria and the
Method of Moving Asymptotes is a strong internal consistency check, but the
absolute compliance values should still be validated against a published
88-line reference implementation before they are quoted, as the manuscript's
limitations section states.
