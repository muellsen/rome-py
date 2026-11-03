# c-lasso R4 fairness audit

Audited installed c-lasso 1.0.11. Exact source snapshots and hashes are in
sources/classo-1.0.11-audit/. The benchmark imports the installed package;
these snapshots identify its numerical implementation, not the current online
main branch. The paper retains both main comparisons: fixed-scale versus concomitant CD,
and concomitant MM-CD versus all supported c-lasso R4 optimizers.

## Formulation and dispatch

- compact_func.Classo accepts Path-Alg and DR for R4; other requested methods
  silently become DR. P-PDS and PF-PDS do not implement R4 in this release.
- pathlasso_R4 unconditionally sets pb.type='DR'. It cannot benchmark the two
  distinct algorithms by changing a path-API label. We use single-lambda calls.
- Classo_R4 silently redirects sufficiently small normalized lambda (<1e-5)
  to Path-Alg. The benchmark checks this threshold before timing.
- c-lasso uses the doubled Huber convention H=2*h, and the objective
  sigma*sum H(r/sigma)+e*sigma+lambda_classo*||beta||1. Dividing by 2n maps
  to mean sigma*h(r/sigma)+a*sigma+lambda0*||beta||1 with
  e=2*n*a and lambda_classo=2*n*lambda0. We use a=1/2 and e=n, rho=1.345,
  absolute lambda, unit penalty factors, and no automatic rho scaling.
- A single all-zero constraint row means C beta=0 imposes no restrictions.
  This avoids an empty-matrix matrix_rank failure in the stock projection helper.
  An identity-projector diagnostic was tested separately and leaves the small
  stock DR solution unchanged. The main timing report retains the stock dense
  projection representation to avoid conflating that optimization with metric tuning.
- Intercepts are disabled for all methods in this initial comparison, with no
  intercept in the generating model. This avoids differences in intercept
  handling: R4's constructor prepends ones to C when intercept=True, so a zero
  row would then constrain the intercept, and the mean-shift path branch uses
  a sqrt(2) rescaling that needs a separate intercept-equivalence audit.

## Accuracy

Equal numerical tolerance values do not mean equal accuracy. MM-CD checks
joint stationarity. DR stops on changes of splitting iterates; its cubic solver
has an absolute 1e-6 residual threshold and a 20-step safeguard. The path method
uses parameter-dependent knot tracing, not the same stopping tolerance.

The path implementation builds AtA=P.T@P+eps_L2*I with eps_L2=1e-3 by default.
In a small independent audit this produced a coefficient KKT error around
2.6e-5; setting eps_L2=0 via the existing constructor argument reduced it to
roundoff. We report stock Path-Alg and the explicitly labeled eps0 diagnostic.
A fast run that fails the common accuracy gate is not a qualified speed result.

Every returned fit is checked outside the timer against the same objective,
relative objective gap <=1e-8 to a high-accuracy reference, and joint residual
<=1e-5. The coefficient residual is the continuous proximal KKT mapping
||beta-soft(beta-gradient,lambda0)||_infinity at unit step. This avoids assigning
a full nonzero-coordinate sign violation to arbitrarily tiny DR coefficients.
The scale residual is |1/2 - mean(clip(r/sigma,-rho,rho)^2)/2|. No coefficients
are thresholded or polished to make methods pass. The stock DR iterate-change stop did not return within 20,000 iterations at
three tolerance settings in the first benchmark calibration. The retained
initial attempt records that failure. For the main benchmark, only its stopping
condition is replaced by this common external gate, checked every ten updates;
all DR update equations and scalar proximal tolerances remain unchanged.
The objective reference and external diagnostic checks are prepared outside
timing, but the checks used to stop DR remain charged to its solver time.
No stock-native-stop speed claim is made for this adjusted run.

## Timing boundaries

Imports and JIT compilation occur once outside timed solves. Matrices, response,
preprocessing, native parameter initialization and dense setup are prepared
before each timed solve. Cold setup and prepared numerical solve times are
reported separately; their sum is also saved. External diagnostics are excluded.
All methods receive identical standardized X, y, rho, and absolute lambda.
Each method retains its native scale initialization; no cross-method warm start
is used. All coefficient initializations are zero. BLAS thread counts are one.

- MM-CD: time the warmed _solve kernel, excluding estimator validation and
  conversion. Its in-kernel residual allocation, objective trace, and full KKT
  checks remain charged to the solver.
- DR: prepare problem_R4 and compute_param outside timing. The latter constructs
  the projection and dense Q1/Q2/QA matrices using inverses. The numerical DR updates are unchanged. A guarded in-memory copy
  disables only the native iterate-change stop and inserts the external gate;
  its already-completed compute_param call becomes a no-op. regpath remains False, so its ordinary return/scale is kept.
- Path-Alg: perform the exact Classo_R4 mean-shift augmentation outside timing,
  construct problem_R3, and prepare parameters_for_update (including the large
  Gram matrix and initial inverse) outside timing. The unchanged Classo_R3 and
  solve_path_Conc functions run inside the timer. A temporary factory returns
  the one freshly prepared parameter object, with an assertion that it is used
  once; it is never reused across repetitions. Scalar conversion and final
  interpolation are included. Some O(np) normalization inside solve_path_Conc
  remains included: these are prepared-solve times, not isolated arithmetic
  loop times. Lambda-dependent augmentation/setup cannot be treated as a
  one-time cost for an entire R4 path.

Each repetition rebuilds fresh mutable state. Method order alternates. Setup is
not discarded from the reproducibility record. DR's stock implementation still
performs dense identity projection products even when C=0. An identity-action
variant is checked for numerical equivalence in the tests, but is not used in
the main timing table.

## Interpretation limits

The comparison measures these implementations, not intrinsic complexity of
abstract algorithms: MM-CD is Numba-compiled; stock c-lasso DR contains Python
loops for scalar proximal operations. A publication must identify this language
asymmetry. The pilot has two seeds and three repeats per setting, single lambda,
positive scales, and independent Gaussian design columns. Correlation, full
regularization paths, intercept equivalence, zero-scale solutions, and larger
replication counts remain work for the final benchmark.

## DR metric calibration

The stock metric c=(d/||A||_2)^2 is dimension dependent. In the first 80x120
Gaussian calibration case, a 3000-update diagnostic still had scale near 3.01
and joint residual about 0.18, versus the reference scale near 0.864. This is
slow progress, not merely a native stopping-test discrepancy. A fixed c=1
passed the common objective/stationarity gate in that case; c=0.1 did not
pass within 5000 updates. We use c=1 consistently across all subsequent
settings. We retain stock-metric diagnostics with a stated 1000-update budget
and do not treat their budget-exhaustion times as convergence times. A failed
budget is not evidence that the method cannot eventually converge.
