# Contribution and source audit

Source: Kim, Loh and Basu, arXiv:2510.13715v3 (14 August 2026), and
rome commit 99d6f07e2f5daa262768387701af7769dfbc2212. Read alongside the
Combettes–Müller perspective formulation; the estimator is existing work.

| Original paper | Retained | Difference in this note | Current limitation |
|---|---|---|---|
| Sections 2.1–2.2, exact scalar kink sweep | Exact Huber coefficient minimization | Scale-dependent knots and penalty are rebuilt at every scale | Dense lasso implementation, not the full elastic-net/adaptive package |
| Section 2.2 and Section 5, concomitant estimation left for future work | Robust sparse regression motivation | MM scale block integrated with exact coefficient sweeps | Interior KKT certification only |
| Section 2.3, Proposition 2.1 | Fixed-scale CD reference | Joint descent proved by touching majorizers | Original O(1/k) bound is not a joint-rate theorem |
| Section 3.1, cheap zero-coordinate checks | Current residual optimality checks | Recheck at the new scale before skipping sorting | No transfer of scale-stale screening decisions |
| Sections 3.2–3.3, paths and sequential strong rules | Warm-start paths implemented | Penalty conversion changes with scale; zero-coordinate checks are refreshed | No joint strong-rule or CV implementation yet |
| Section 4.1, four correlated/heavy-tail design families | Numerical stress-testing motivation | Current pilots isolate scale cost and joint-solver accuracy | 100-penalty protocol-validation paths added; full 100-replication grid remains unfinished |
| Section 4.2, convergence comparisons without screening/warm starts | Record objective across sweeps | Also record sigma, coefficient paths and joint KKT | Diagnostic figures are not runtime comparisons |
| Section 4.5, glass spectra and PbO | Same data and columns 15:500 | Training-only transforms and fitting; lasso pilot | Original alpha=.3 elastic net, 50 splits/5-fold CV not reproduced |

## Why it matters

An absolute Huber threshold encodes response units. The perspective estimator
separates a dimensionless capping constant from a fitted residual scale. Its
coefficient penalty is fixed in the joint objective; multiplying the coefficient
subproblem by sigma induces both delta=rho*sigma and lambda_inner=lambda0*sigma.
Updating the threshold alone changes the target estimator. Computation is the
contribution: combine an existing exact scalar solver with an inexpensive,
provably descending scale majorizer and verify the resulting joint optimum.
Neither the perspective estimator nor the mean-shift representation is new.
No claim of a new statistical recovery theorem or first use of MM is made.

## Executable glass protocol discrepancy

The pinned code_paper/Section4_5.R scales the complete data before splitting.
In its repeated-split experiment, CV uses training rows, but fit_rome calls
rome.adaptive with all scaled_x/scaled_y before evaluating held-out rows.
The glmnet branch instead fits training rows and uses alpha=.2 despite the
surrounding alpha=.3 declaration. These are statements about this snapshot,
not a claim about unobserved runs underlying the publication. We do not reuse
its reported prediction errors. Our one-split, three-penalty diagnostic uses
training-only preprocessing and fitting; it is not a reproduction of Table 2.

## Updated experiment scope

The original-paper track compares only original fixed-scale rome and the
concomitant extension, not ILAMM, hqreg or glmnet. The native R/C reference is
built from the pinned source. The distributional design and 100-point path
follow Section4.1 and its script; one paired validation realization at three
dimensions is explicitly distinguished from the full 100-replication grid.
The script's minimum penalty fraction is .05, whereas Section3.2 prose
describes a default .001. Native rome eps1e-4 is preserved, with external
accuracy reported; it is not equated to our joint KKT tolerance.
An initial eps1e-8 native attempt was interrupted before a return and retained.

R4 now starts with a seven-size Path-Alg scaling experiment, using fixed
settings, fresh state, separated setup costs and a common accuracy gate.
The earlier DR budgets and adjustments remain supplementary diagnostics.
