# Perspective-Huber RAMP overlap and comparison protocol

The relevant AISTATS manuscript is the local anonymous AISTATS 2027 draft
“Perspective M-Estimation Meets Approximate Message Passing: Adaptive Scale,
Robustness, and Sparse High-Dimensional Estimation”, from perspective-amp.
This is a local draft, not a verified published AISTATS article. The reference
solver and derivation are snapshotted under sources/amp-aistats2027, with hashes.
The earlier sparse robust AMP literature includes Bradic (2016), which does not
by itself supply the concomitant scale algorithm implemented in that draft.

## Overlap

Both methods target a sparse Huber perspective objective and its scale score
condition. Thus the estimator, scale calibration and robust-sparsity motivation
cannot be claimed as differences. The AMP draft studies effective-score
calibration, Onsager corrections and Gaussian-design state evolution. This note
studies finite-sample optimization by exact coordinate minimization with an MM
scale block. State evolution at finite iteration horizons does not establish
convergence on arbitrary correlated real-data matrices.

## Exact normalization

Our X has empirical column second moment one, and our objective averages losses.
Use A=X/sqrt(n), B=sqrt(n)*beta for AMP's approximately N(0,1/n) design convention.
Then A*B=X*beta and multiply our objective by n: the AMP summed-loss penalty is
lambda_AMP=sqrt(n)*lambda0 on B. gamma=1, c=rho and the same scale constant a
are supplied to both methods. We use the analytic Gaussian consistency constant
rather than the Monte Carlo approximation in some draft scripts. The earlier
runtime pilots retain their original a=1/2; these are different experiments.

## Necessary fixed-point check

The reference implementation uses a fixed omega, whereas q is the empirical
soft-threshold divergence. Its fixed-point correspondence requires q=omega,
positive scale, and convergence of all state variables. Support stability alone
is insufficient. We therefore evaluate the returned beta and sigma against the
original residual y-X*beta, computing joint KKT residuals and objective gap to
CD. Native AMP convergence is recorded separately. No thresholding, polishing,
whitening, damping or reference warm start is applied.

We test prespecified support counts 3 and 10 in two synthetic cases, plus one
explicitly reference-informed support diagnostic (not a tuning procedure).
For glass and B. subtilis, counts 3,10,20 are crossed with three prespecified
penalty fractions .4,.6,.8. Each return receives the same external accuracy gate
as the c-lasso study. Failed runs remain in JSON. This initial grid is an accuracy/feasibility audit.
The additional source-protocol audit finds matching fixed points at penalty2.5,
support14 and penalty3,support6. Only those settings receive five-repeat fresh
prepared implementation timings, with a common external accuracy gate. The
timers exclude calibration search and keep AMP history construction and scalar
bisection. Thus the timings are conditional on known valid calibration.

## Real data

Glass uses PbO, n=180 and the original selected 486 wavelengths. B. subtilis
uses the TREX riboflavin MATLAB example (71 samples, 4088 candidate genes),
with the AISTATS deterministic greedy correlation filter retaining 500 genes.
The main pilot estimates this filter from training predictors only; the same
selected columns are supplied to AMP, fixed CD and concomitant CD. An additional
source-protocol diagnostic uses the original global predictor-only filter,
80% training split and seed20261006, with AMP penalties 2.5,3,4,5 and support
counts 1..30. These are feasibility/calibration sweeps, not test-set tuning. A single seeded split uses 120 and 50 training rows,
respectively; all transformations are learned on training rows, and response
centering provides a fixed training intercept. Three penalties are reported
without choosing the best test result. No claim of predictive superiority or
biological gene discovery follows from this exploratory split.
