---
name: rome
description: Port or extend yk748/rome exact Huber coordinate descent in Python, including concomitant MM scale estimation and its JCGS reproducibility workflow.
---

# rome and concomitant Huber

Use https://github.com/yk748/rome and arXiv:2510.13715 as primary sources.
Inspect R/rome.adaptive.R, R/cv.rome.R, prediction methods, src/rome.c,
src/rome_fit.c, and src/util.c at a recorded commit before assuming behavior.
The checked source snapshot was 99d6f07e2f5daa262768387701af7769dfbc2212;
verify the working version rather than silently using main.
DESCRIPTION declares GPL-3. Preserve attribution and review licensing before
publishing a translation. README scope can lag the paper and source.

The project extends rome with MM concomitant scale estimation, preserving
exact Huber coefficient coordinate minimization and the path/CV/prediction
interface. A quadratic surrogate minimizer is not an exact Huber minimizer.
Follow the Combettes–Müller perspective objective and its closed zero-scale
extension. Use canonical lambda0 for the joint penalty, explicitly map any
lambda(sigma) convention and any multiplication of the coefficient subproblem.
Read analysis/derivation-audit.md in this project before reusing the Gemini draft.
Check scale factors, residual bookkeeping, descent, and joint KKT conditions.
Original draft timing and recovery claims have no validated provenance.

Keep package solvers in src/, drivers in analysis/, canonical text in
paper/manuscript.qmd. Figures/tables read saved outputs with verified manifests.
Use analysis/pipeline.py capture/verify helpers; save seed, source commit/hash,
configuration, environment, command, solver diagnostics and output hashes.
The two experiment tracks are original-paper settings and unconstrained R4.
Read the project's c-lasso R4 skill when matching those baselines.

The original-paper experiment compares only original fixed-scale rome and the
concomitant extension; exclude ILAMM, hqreg and glmnet from that track.
Use the original four predictor distributions, sparse coefficient vector and
Gaussian response noise. The Section4_1.R 100-point path ends at .05 of the
fixed-model maximum; Section3.2 prose gives a different default .001.
Separate protocol-validation subsets from the full 100-replication grid.
Never rank a partial joint path as a completed path or equate the native
R iterate-change tolerance to a joint KKT tolerance.

For the optional AISTATS Perspective-Huber RAMP comparison on B. subtilis,
apply its deterministic greedy correlation filter retaining 500 genes and
give every compared solver the same retained matrix. Distinguish training-only
filtering from the source workflow's global predictor-only filter. Match AMP's
summed-loss normalization with A=X/sqrt(n), B=sqrt(n)*beta and
lambda_AMP=sqrt(n)*lambda0. Check q=omega, joint KKT and objective gap; native
AMP convergence alone does not certify the same estimator.

Use sign/min expressions for the Huber score. Algorithm variables use mathematical notation; initial scale is sigma^(0), with rho reserved for the Huber threshold. Pseudocode uses mathematical assignments and control statements; exact coordinate minimization has a separate numbered subroutine.

The focused revision follows the writing and 12-point author-manuscript layout
of Yoon, Müller and Gaynanova, Fast computation of latent correlations, and uses
ROME's algorithm2e ruled/vlined algorithm style. The user explicitly selected
10 replications per original simulation cell for this revision. The matched
B. subtilis AMP instance (lambda_AMP=2.5, omega=14/500, exact source filter/split)
belongs in the real-data section; broad AMP calibration searches do not.
Real-data CV compares fixed and joint estimators with the same Python kernel,
training-only preprocessing and per-inner-fold correlation filtering.

Optimized revision: retain solver.py as the reference. optimized.py verifies
local affine coordinate roots, checks full current-scale KKT before termination,
and optionally uses objective-safeguarded support Newton polishing. boundary.py
solves a zero-scale LAD-lasso LP and verifies a feasible perspective dual plus
a primal-dual gap; tiny scale alone is not a certificate. Revised CV uses
model-specific intercept-only null anchors and can continue after a failed grid
point, recording that failure rather than silently truncating the entire tail.

Scale strategies are retained side by side: mm, exact scalar root, and local
joint-prox/consensus DR. Keep the original MM benchmark kernels immutable.
Use the common coefficient loop in scale_solver.py for controlled comparisons.
Scale block zero optimality is not a joint regression certificate. Proximal
shadows need a descent guard and original-block accuracy diagnostics; never
label one prox/projection an exact minimizer. Keep all three in software;
method selection is deferred until broader evidence supports one main-text
option, with the other two then documented in the Appendix.
