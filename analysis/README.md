# Analysis

The original-paper protocol is pinned in configs/rome.json. The ROME track
uses its four original predictor distributions and 100-penalty paths, comparing
only the native original rome algorithm and the concomitant extension. The
default driver runs a declared validation subset; --full requests all nine
dimensions and 100 independent replications. This full study has not completed.
The separate R4 track starts with Path-Alg size scaling. Drivers call package
code rather than placing solver updates in the manuscript.

`pipeline.py` provides capture/verify primitives for generated files. Its smoke
command exercises the artifact chain. Render verifies hashes and records the
actual Quarto command, environment, source hashes, and resulting document.
Outputs are ignored by Git; publish a reviewed reproducibility bundle for a
submission. Preserve manifests with that bundle. No freeze/cache is used.

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
