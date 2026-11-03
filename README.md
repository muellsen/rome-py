# ROME for Python (`rome-py`)

A research Python package for sparse Huber regression with exact coefficient
coordinate descent, supporting both fixed scale and concomitant scale estimation
by majorization–minimization. The estimator follows Combettes–Müller; the exact
coefficient-search principle follows Kim–Loh–Basu and their R package rome.
The accompanying manuscript targets a short JCGS note.

## Install and use

The distribution is named `rome-py`; import it as `rome_py`. Python 3.10 or
later is required. From this checkout:

    python -m pip install .
    python -m pip install -e ".[benchmark,test]"   # development and experiments

The previous `concomitant_huber` imports remain supported; both names use the
same numerical implementation. Recorded experiment scripts retain their original
imports and solver source hashes.

The runtime package uses NumPy, Numba and SciPy. First use compiles the
numerical kernel.

    import numpy as np
    from rome_py import HuberCD
    from rome_py.path import huber_path

    rng = np.random.default_rng(71)
    X = rng.normal(size=(80, 12))
    y = 2 * X[:, 0] + rng.normal(size=80)

    joint = HuberCD(penalty=0.25).fit(X, y)
    fixed = HuberCD(penalty=0.25, estimate_scale=False,
                    initial_scale=1.0).fit(X, y)
    predictions = joint.predict(X)

    # Absolute penalties of the averaged perspective objective; no preprocessing.
    path = huber_path(X, y, [0.5, 0.35, 0.25], initial_scale=1.0)
    print(joint.status_, joint.scale_, path.status)

The objective is mean sigma*h_rho(residual/sigma) + a*sigma +
lambda0*||beta||1. Coefficient subproblems use threshold rho*sigma and penalty
lambda0*sigma. HuberCD supports an unpenalized intercept; huber_path currently
omits an intercept. Neither function standardizes data. The default a=0.5 is
a scale convention, not the Gaussian-consistency constant.

This is version 0.0.1, an experimental dense lasso implementation. Sparse
matrices and pathwise strong rules are not implemented. The reference
HuberCD reports zero-scale states as uncertified; FastHuberCD adds a boundary
LAD-lasso solver with a perspective dual certificate.
The model_selection module supplies training transforms and intercept-aware
warm paths; repeated nested CV is implemented by the analysis driver. Nonconvergence and unattempted path tails are explicit.

## Repository

| Location | Responsibility |
|:--|:--|
| src/rome_py/ | Public Python interface |
| src/concomitant_huber/ | Numerical implementation retained for reproducibility and compatible imports |
| tests/ | Scalar-update, joint-optimum, path, tracing, audit and provenance checks |
| analysis/ | Experiment drivers, configurations and artifact assembly |
| analysis/outputs/ | Ignored numerical artifacts and provenance sidecars |
| paper/manuscript.qmd | Canonical manuscript |
| paper/algorithm.qmd | Exact pseudocode; changes from ROME marked in blue |
| sources/ | Portable source registries, attribution and checksum-verified downloads |
| skills/ | Repository guidance for ROME and c-lasso R4 work |

## Focused manuscript workflow

The writing and author-manuscript layout follow Fast computation of latent
correlations (Yoon, Müller and Gaynanova). The note presents the ROME objective,
the Combettes–Müller perspective formulation, MM building blocks and exact
coordinate updates, followed by two primary timing tracks and real applications.

Base R with Rcpp and mvtnorm is required for the original R/C ROME comparator.
Quarto and a TeX installation are required for the manuscript PDF.

    make sources check rome-reference
    python analysis/fetch_sources.py --scope data
    Rscript analysis/export_glass.R
    make algorithm-template
    make revision-timings
    make revision-r4
    make revision-real

The first track uses all 36 original design/size settings and ten replications
per setting, explicitly reduced from the original 100 for this revision.
Original ROME, fixed-scale Python CD, and concomitant Python CD are compared. Four single-threaded
workers run independent paired paths, with process CPU times recorded.
Failures, numerical accuracy and incomplete tails are retained.

The second track compares both c-lasso R4 solvers, Path-Alg and
Douglas–Rachford, against the same unconstrained perspective objective.
It includes moderate scaling and a same-instance coefficient/scale trajectory.
Prepared solve, setup and total costs are stored separately.

The applications use ten outer holdouts and five-fold inner CV for glass and
B. subtilis. All preprocessing and the 500-gene correlation filter are fitted
inside the relevant training fold. Fixed and joint fits use the same numerical
kernel and an unpenalized intercept.

The AMP comparison is restricted to the matched B. subtilis source-protocol
instance, lambda_AMP=2.5 and omega=14/500. Its unpublished reference source is
not redistributed. See sources/amp-reference.json for checksums.

    make optional-amp-check
    make amp                 # reconstruct the source-protocol inputs
    make revision-amp        # time only the matched instance
    make revision-figures
    make pdf manuscript

The figure and rendering stages verify numerical input/output hashes. See
analysis/revision-protocol.md for the exact conventions. Numerical outputs and
per-case/per-fold checkpoints are ignored, with provenance sidecars recording
sources, inputs, configuration, environment and failures.

Earlier validation, timing, convergence and AMP-feasibility audits remain
available through their existing analysis scripts and Make targets; they no
longer define the main manuscript. The public CI still builds the package and
the smaller public primary report. It does not claim to reproduce unpublished
AMP inputs or run the full local research computation.

## Submission status

The package, tests, primary workflow and Quarto sources are organized for
GitHub at https://github.com/muellsen/rome-py.
The current revision uses the user-selected ten-replication ROME study.
Robust path certification and final submission checks remain research work. Downloaded papers, templates,
datasets, local drafts and build products are excluded from the Git submission.

Project code is GPL-3.0-only; third-party materials retain their own licenses.
See LICENSE, THIRD_PARTY.md and CITATION.cff.

### Optimized solver and model-specific CV

```python
from rome_py import FastHuberCD
model = FastHuberCD(penalty=0.1, rho=1.345, scale_penalty=0.355082274)
model.fit(X, y)
```

`FastHuberCD` preserves exact coefficient minimization and the perspective
objective. It adds verified affine-root shortcuts, reusable buffers, provisional
working sets, and safeguarded support Newton polishing. Set `newton_every=0`
to disable polishing. Zero-scale optima require an LP-based perspective dual
certificate; no scale floor is imposed. The original `HuberCD` remains the
reference implementation.

`analysis/optimized_real_parallel.py` reruns the real-data CV with separately
optimized fixed-scale and concomitant null-model penalty anchors in each
training fold. The original results in `analysis/outputs/revision-real` remain
unchanged. `analysis/optimized_timing_validation.py` runs four declared fresh
full-path checks against native ROME; these checks supplement, rather than
replace, the original 360-path experiment. Render the updated manuscript with
`analysis/render_optimized.py` after `analysis/optimized_assemble.py`.

`make python-scale-timing` adds five paired timing repeats per validation dataset,
using the same optimized Python kernel for fixed scale and concomitant fitting.
Both modes use the same absolute penalties and coefficient tolerance. These
checks isolate implementation-language differences while retaining any change
in supports, curvature, and boundary-solve cost caused by estimating scale.

### Three scale-update options

```python
from rome_py import FastHuberCD, update_scale

models = {
    method: FastHuberCD(scale_update=method, penalty=0.1).fit(X, y)
    for method in ("mm", "exact", "prox")
}
block = update_scale(residuals, scale=1.0, method="exact")
```

The three options share the production exact coefficient loop and the same
joint objective. `mm` takes at most `scale_mm_steps` tangent-majorizer steps;
`exact` solves the monotone scalar scale equation numerically; `prox` iterates
observation-wise joint residual/scale proxs and common-scale projection using
Douglas--Rachford splitting. One prox/projection is not treated as an exact
scale update. `scale_inner_tol` defaults to half the outer tolerance,
`scale_max_iter=2000` bounds root/splitting iterations, and `prox_gamma=1`
is the normalized splitting step size. Every accepted block update is a
descent step; full coefficient/scale accuracy remains necessary for convergence.
`scale_inner_iterations_` and `inexact_scale_updates_` expose the inner work.
`update_scale` returns its block certificate separately from regression fitting.
Fixed-scale mode bypasses all three scale strategies.

MM remains the default while selection is pending. The unchanged reference
`HuberCD` and recorded MM benchmark kernels remain available. The three-option
validation uses `make scale-strategies`; its small fixtures do not select a
winner. The current manuscript presents all three subroutines together in
`paper/scale-options.qmd`; after selection, two will move to the Appendix while
all software options remain supported.

The moderate MM/root path pilot uses `make scale-path-benchmark`: 36 paired
datasets from the four ROME designs, three sizes, three replications and
30 shared absolute penalties. Independent checks gate all reported complete
paths; failed and unattempted points are preserved. The standalone report is
`analysis/scale-path-report.qmd`; render it with
`quarto render analysis/scale-path-report.qmd --to html`, then record its
provenance with `python analysis/scale_path_report.py --record-html`.
Raw checkpoints, paired timing summaries, failure audits, scale-block replay
and coefficient/scale convergence figures are under
`analysis/outputs/scale-path-benchmark/`. These optimized MM/root results are
separate from the historical 360-path manuscript baseline.

`make figure1-complete` reruns all360 datasets with native ROME, fixed-scale
Python CD and concomitant Python CD. The package helper
`rome_py.qualified_path.qualified_path` visits every penalty,
retains the lower-original-objective iterate or LAD-lasso candidate when
uncertified, and continues with explicit approximate status. It reports
internal stopping flags and independent feasible-dual objective bounds.
The separate Figure1 audit applies independent strict accuracy checks;
approximate returns remain in the timing curves and are marked. Raw
historical results are preserved. Compilation and external audits are excluded
from timing; fallback and continuation work are included.

`analysis/implementation_audit.py` and `analysis/implementation-audit.qmd`
audit the pinned ROME C implementation against the Numba kernels, including
stopping rules, screening, measured boundary-LP cost, and independent scalar
coordinate checks. Native ROME is kept unchanged in the timing study;
coordinate-level discrepancies are documented separately from whole-path
accuracy and performance.
