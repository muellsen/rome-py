# Analysis and reproducibility

Experiment drivers call package code; numerical updates belong in `src/`.
The canonical manuscript is `paper/manuscript.qmd`. Numerical outputs under
`analysis/outputs/` are ignored by Git and accompanied by provenance sidecars.
`pipeline.py` records source/input/output hashes, settings, and the environment;
rendering verifies the recorded numerical dependencies before building the paper.

## Current manuscript workflow

Install the package with `python -m pip install -e ".[test,benchmark]"`.
The original R/C ROME comparator additionally requires R, Rcpp, and mvtnorm.
Quarto and TeX are required for the manuscript PDF.

```bash
make sources check rome-reference
python analysis/fetch_sources.py --scope data
Rscript analysis/export_glass.R
make algorithm-template
make figure1-complete
make revision-r4
make optimized-cv
make optimized-figures
make pdf manuscript
```

- **ROME comparison:** `figure1_complete.py` runs the four original designs,
  nine original size combinations, ten replications per setting, and 100
  penalties per path. The three implementations are original R/C ROME,
  fixed-scale Python CD, and concomitant Python CD. `figure1_accuracy.py`
  checks returned solutions separately. Approximate returns remain labeled;
  native ROME's stopping criterion is not claimed to match the Python checks.
- **Unconstrained R4:** `revision_r4.py` compares concomitant CD with c-lasso
  Path-Alg and Douglas–Rachford at matched objective normalization and common
  accuracy requirements. Prepared solve, setup, and total times are separate.
  Read [the R4 audit](classo-r4-audit.md) before interpreting the comparison.
- **Prediction:** `optimized_real_parallel.py` runs ten outer holdouts and
  five-fold inner CV on glass and B. subtilis, using separate model-specific
  null anchors. Filtering and transformations use only the relevant training
  fold. Both fixed-scale and concomitant models retain an unpenalized intercept.
- **Assembly:** `optimized_assemble.py` reads saved results to produce manuscript
  figures and tables; `render_optimized.py` verifies their provenance and records
  the rendered output. See [the full protocol](revision-protocol.md).

The optional AMP comparison requires separately supplied unpublished source
materials; they are not redistributed. It compares the matched B. subtilis
problem instance, with the source's 500-gene correlation filter and parameter
mapping, rather than adding another prediction method to the CV experiment.

```bash
make optional-amp-check
make amp
make revision-amp
make optimized-figures
make pdf manuscript
```

## Additional diagnostics

- `make python-scale-timing`: paired fixed/concomitant timings with the same
  Python coefficient kernel on four declared validation datasets.
- `make scale-strategies`: three small fixtures comparing MM, scalar-root, and
  joint-prox scale updates at common accuracy.
- `make scale-path-benchmark`: 36 paired MM/root datasets and 30 penalties,
  preserving unsuccessful points and unattempted tails. The accompanying
  report is `scale-path-report.qmd`.
- `make implementation-audit`: pinned native ROME versus Python source and
  numerical checks, including stopping rules and isolated coordinate updates.
- `python analysis/export_submission.py`: source-only GitHub archive and audit.

Recorded timing kernels and historical manifests remain unchanged. Updating
README text does not require rerunning numerical experiments. Do not replace a
historical manifest to conceal changes to a numerical dependency.

## Historical pilots and audits

The commands below document earlier studies. They are retained for provenance;
they do not replace the current three-implementation Figure 1 computation.

## Implemented pilot

`small_scale_experiment.py` runs the dense prototype in fixed-scale and joint
MM1/MM8 modes, with matched coefficient optimality at the final estimated scale.
It saves raw timings, complete input arrays and solutions, summaries and a
Quarto table with a provenance sidecar. `make pilot` runs it. The initial weak-
penalty failure and first completed MM1 run are retained under outputs/.
See algorithm-design.md for the scale majorizer and stopping rules.

## R4 audit and timing

`r4_benchmark.py` compares prepared MM-CD, DR, and Path-Alg solves under a
common external accuracy gate. Read classo-r4-audit.md before interpreting
timings; stock and adjusted numerical settings are labeled separately.
`make r4` saves setup, prepared solver and total-call timings plus failures.
`make tables` assembles both equally important main comparisons. Old numerical
outputs are verified against their actual solver/driver dependencies while the
original complete source snapshot remains unchanged in their manifests; editing
manuscript prose alone does not require rerunning timed experiments.

Run `analysis/export_glass.R` with base R, then
`python analysis/convergence_amp_experiment.py` for sweep trajectories and the
AMP/real-data feasibility pilot. Inputs and the reference AMP snapshot are
hashed in `sources/data/registry.json`; all runs, including failed AMP attempts,
are retained in `analysis/outputs/convergence-amp-summary.json`. The opt-in
tracer instruments the production kernel and checks reproduction; it is never
used for timing. See `rome-differences.md` and `amp-overlap-audit.md` for scope.

Run make diagnostics amp for the reference AISTATS preprocessing diagnostic:
B. subtilis is reduced to 500 genes by the exact greedy correlation filter.
The primary pilot fits the filter on training X only. amp_source_protocol.py
separately matches the source's global predictor filter, split seed and 80%
training fraction, and audits support calibration across 120 settings. Both
AMP and CD receive identical filtered matrices and objective parameters.

Build the reference with python analysis/install_rome_reference.py and run
python analysis/rome_paper_experiment.py for protocol validation, or pass
--full for the full declared replication grid. ROME's native stopping rule and
joint KKT certification differ; inspect pointwise accuracy and unattempted tails
before interpreting times. No partial path is an eligible full-path speed result.
python analysis/r4_path_scaling.py varies n at p=2n and p at n=100, with fixed
accuracy and no DR tolerance/budget search.

Run make convergence after rome-validation and diagnostics to obtain sigma and
beta trajectories on the original compound-Gaussian and multivariate-t2 AR
cases, together with the glass and correlation-filtered B. subtilis cases.
All state-copy traces are excluded from timing studies.

Run python analysis/audit_incomplete_paths.py to diagnose the seven first failed
ROME-validation points. It preserves the historical outputs, uses verified warm
traces, compares scale batching with fixed-scale continuation, and constructs
independent feasible dual bounds and boundary lasso-LAD LP solutions.
Results and provenance are under outputs/incomplete-path-audit/.
The explanatory report is incomplete-path-audit.qmd. This is a diagnostic,
not a replacement timing benchmark or certification of unattempted path tails.
