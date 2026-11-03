# adaHuber 1.1 and perspective Huber estimation

The source archive and checksum are recorded in `sources/adahuber.json`.
`install_adahuber_reference.py` builds the unchanged release in the ignored
project-local R library. The installation log and provenance record the compiler,
R dependencies, archive, relevant sources, and installed shared library.

## Formulations

The intended adaptive-lasso calibration solves mean min(r^2/tau^2,1) = log(np)/n.
With tau=rho*sigma, it has the perspective scale equation when
`a = rho^2*log(np)/(2*n)`. This algebra identifies a scale equation, not the
complete penalized estimator. The corresponding coefficient step requires
`eta=sigma*lambda0`. The stock package retains its supplied eta.

The archived C++ functions `adaHuberLasso`, `adaHuberLassoList`, and the final
CV refit calculate the residual and then form `resSq` from `square(Y)`.
The resulting threshold calibration uses centered responses. The source is not
changed in this comparison. Its native threshold is reported as sigma=tau/rho,
with rho fixed explicitly. This does not assert that the package returns a
separate noise-scale estimate under that name.

## Numerical design

One selected ROME Compound-N instance uses n=p=100, seed 71, the original sparse
coefficient vector and Gaussian response errors. Shared preprocessing centers
predictors and response and uses sample-SD predictor scaling, matching the R
wrapper. Three absolute penalties are 0.6, 0.4 and 0.2 of the concomitant
intercept-only null maximum. These are three regularization settings on one
dataset, not three independent datasets. Each setting has three timing repeats.

The matched fits start at zero coefficients and the null-model scale.
Native adaHuber retains its own MAD-based threshold initialization.
The native supplied penalty is initial_scale*lambda0. This fixed, pre-fit rule
avoids choosing a native penalty using the desired final solution. Prepared
native calls are checked against the exported R function at every setting.
The high-accuracy reference has tolerance 1e-11 and an independently constructed
feasible dual bound. Common acceptance uses coefficient/scale residual 1e-5
and relative objective difference 1e-8, as in the c-lasso comparison.

The matched adapter is new R orchestration, not native adaHuber. It alternates
a scalar perspective scale solve and at most 100 calls to the unchanged
compiled `lamm` coefficient step, with eta=sigma*lambda0. It uses joint stopping.
The block step is a majorizer update, not an exact Huber coordinate update.
The main CD comparator uses MM and disables optional support Newton steps.

R and Python timing blocks alternate across penalty settings. R method order
alternates across repeats. Prepared R timings exclude shared standardization,
intercept-matrix construction and package loading. Python timings call the
public estimator on the same transformed inputs, so its validation and
intercept-matrix construction remain charged. Native R public-call times are
also saved. Compilation, process startup, file exchange, reference fitting,
plotting, traces and post-fit quality audits are outside the timers. Scale roots,
R adapter loops and internal optimality checks remain charged. Thread settings
include the Apple vecLib limit. R CPU times have millisecond resolution.

Exploratory runs remain in ignored output directories and do not enter the
paper's timing summaries. The reported benchmark uses the existing c-lasso
accuracy requirements and records the residuals and objective differences for
every return. An iteration limit is not labeled convergence.

## Reproduction

```bash
python analysis/install_adahuber_reference.py
python analysis/adahuber_benchmark.py
```

Saved outputs include input matrices, every repeated solution and timing,
reference and dual-bound diagnostics, native/public equivalence checks, R
session information, and untimed trajectories. A native runtime is not an
accuracy-qualified time for the perspective objective when its joint checks fail.
