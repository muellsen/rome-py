---
name: classo-r4
description: Audit or benchmark c-lasso concomitant Huber R4 against an unconstrained MM coordinate descent implementation, matching perspective objectives and numerical accuracy.
---

# c-lasso R4

Sources: https://github.com/Leo-Simpson/c-lasso and
https://c-lasso.readthedocs.io/en/latest/description.html.
Inspect classo/solve_R4.py, solve_R3.py, compact_func.py, and solver.py in the
actual installed or pinned source. Record package version, commit where known,
and hashes. Older documentation inconsistently swaps R2/R3 labels; inspect
formulation flags and equations rather than relying on labels alone.

R4 is concomitant Huber: huber=True and concomitant=True. Source inspected
2026-10-19 exposes Path-Alg and DR. Path-Alg uses a mean-shift augmentation;
pathlasso_R4 can force DR internally. Record the method actually executed,
not merely the requested label. Do not present PF-PDS as an R4 baseline without
verifying an implementation that solves the same objective.

For this project remove constraints, verifying whether the selected API accepts
an empty constraint matrix or requires an equivalent zero constraint. Match
Huber normalization, scale linear term e, rho and scale_rho behavior, absolute
lambda versus lambda/lambdamax conventions, intercept, standardization,
penalty factors, and zero-scale treatment. Derive an algebraic mapping to
mean sigma*h(r/sigma) + a*sigma + lambda0*||beta||1; do not copy parameter values
between differently normalized objectives.

Compare primal objectives and joint optimality at matched tolerance before
runtime. Under nonuniqueness assess fitted values/objectives rather than
requiring identical coefficients. Save method, iterations, failures, scale,
residuals, timings, hardware/thread settings, seeds and source hashes.
Read analysis/derivation-audit.md for the supplied draft's unresolved conventions.

Start computational scaling with Path-Alg: vary n at fixed p/n and p at fixed n.
Declare numerical settings and budgets once, retain all failures, and separate
setup, prepared solve and total cost. Extend to DR using the same axes and
accuracy gates; do not substitute repeated tolerance/iteration-budget searches
for a scaling experiment. Stock Gram correction and the objective-matched
zero-correction variant must remain distinctly labeled.
