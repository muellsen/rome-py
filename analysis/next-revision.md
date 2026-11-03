# Penalty grids and solver efficiency after the first full revision

The recorded revision remains intact. Do not substitute a different grid or
solver into its existing provenance manifests.

## Model-specific CV grids

For each inner training fold, optimize the intercept-only model separately
for fixed and concomitant scale. At a positive-scale joint null solution
(b0,sigma0), use lambda_max = ||X' psi_rho((y-b0)/sigma0)||_infinity/n.
For fixed scale use its optimized null intercept and the fixed sigma.
Use common logarithmic fractions with separate model-specific anchors.
Keep the joint lambda0 constant inside each fit; only the inner exact-coordinate
penalty sigma*lambda0 changes with scale. Retain training-only preprocessing.

A new grid does not cure uncertified iterates. Select its lower endpoint and
resolution from inner-fold diagnostics, never held-out errors. Report boundary
optima and failed points explicitly. Rerun all outer refits after revising CV.
Keep the shared-absolute-lambda ROME simulation as its separate protocol.

## Implementation audit

Current solver.py performs an O(np) activation scan and another O(np) full KKT
scan every sweep, plus exact-coordinate work. _diagnostics recomputes the same
Huber score for every column. Each active coordinate creates partial residual,
knots, jumps, and argsort arrays. The coefficient root search sorts up to 2n
breakpoints. Scale MM is O(Bn), so its direct cost alone cannot explain a large
path-level slowdown. Active coefficient count, sorting cost, sweep count and
scale shrinkage must be measured separately on the saved difficult cases.

Priorities:
1. Reuse score and scratch buffers; test an exact affine-root fast path that
   verifies that the candidate crosses no kink before skipping sorting.
2. Work on a small active set between full KKT checks. A changed scale makes
   prior inactivity provisional; require a full current-scale certificate
   before accepting any solution. No unproved permanent screening.
3. Compare certified sweep counts before/after; preserve the fixed-scale and
   joint objectives, stopping thresholds, and zero-scale extension.
4. If sweeps still dominate, test safeguarded acceleration with objective
   acceptance separately from low-level kernel improvements. Preserve the
   unaccelerated MM/CD baseline and exact scalar subroutine.
5. Only then repeat the model-specific CV paths and paired timing suite.

Do not promise fixed-scale speed parity: joint scale changes coefficient
curvature and active sets, and the current 136/360 full-path completion result
is evidence of convergence limitations in addition to implementation overhead.
