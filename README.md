# Full-scale rerun: updated figures, code and data

This package contains the updated figures, code and numerical results from the
full-scale rerun of the topology optimisation benchmark study (iteration
horizon 160; base comparison and statistics at 20 seeds; parameter sweeps at
6 seeds). Use it to update the manuscript.

## figures/
Each result figure is provided in four formats: SVG and PDF are true vector for
the data-driven plots (mesh, filter, volfrac, continuation, hyperparameter,
multibench, cost); the agreement heatmap, bootstrap plot and multibench design
gallery are raster embedded because their full plot data was not cached. TIFF is
600 dpi (300 dpi for the raster-embedded ones); PNG is a screen preview.

Figure name  ->  manuscript figure:
  fig_bootstrap          -> Fig. 2  (bootstrap confidence intervals, base comparison)
  fig_agreement          -> Fig. 3  (cross- and within-optimiser IoU)
  fig_design_grid        -> Fig. 4  (final designs, 4 representative seeds)
  fig_mesh               -> Fig. 5  (mesh independence)
  fig_filter             -> Fig. 6  (filter radius sensitivity)
  fig_volfrac            -> Fig. 7  (volume fraction sweep)
  fig_continuation       -> Fig. 8  (fixed vs continuation penalisation)
  fig_hyperparameter     -> Fig. 9  (GD learning rate, MMA move limit)
  fig_cost               -> Fig. 10 (computational cost)
  fig_multibench         -> Fig. 11 (compliance across four benchmarks)
  fig_multibench_gallery -> Fig. 12 (final designs by benchmark and optimiser)

Note: Fig. 1 (the benchmark schematic) is a static drawing that does not change
with seeds and is not included here; keep the existing one.

## code/
  topopt_study.py     - base study (SIMP solver, OC/GD/MMA behind one interface,
                        self-contained MMA, driver, metrics, plotting).
                        Base comparison set to 20 seeds.
  topopt_extended.py  - mesh, filter, volfrac, continuation, hyperparameter,
                        multibenchmark, statistics and cost studies. Imports the
                        core from topopt_study.py; keep the two files together.
                        Includes a fix so each benchmark carries its own identity
                        (apply_bc sets prob.bc), which the caching relies on.

Run: `python code/topopt_study.py` then `python code/topopt_extended.py`.
Depends only on numpy, scipy, matplotlib.

## data/
Raw numerical results as CSV, one per study:
  base_results_20seeds.csv, extA_mesh.csv, extB_filter.csv, extC_volfrac.csv,
  extD_continuation.csv, extE_gd_lr.csv, extE_mma_move.csv,
  extF_multibenchmark.csv, extG_pairwise_tests.csv, extH_cost.csv

## FULLSCALE_NUMBERS.txt
A plain-text summary of every key number for updating the manuscript text and
tables, including the statistics change: at 20 seeds the OC vs MMA difference is
now significant (p = 0.0028, Cliff's delta = 0.56), so the "statistically
indistinguishable" wording is replaced by "small but significant advantage for
MMA".
