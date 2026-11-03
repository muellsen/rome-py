# Derivation audit (open)

Original Gemini draft is preserved in sources/private/ with a SHA-256 register.
Its benchmark numbers and guarantees are unverified.

1. The canonical joint penalty is lambda0 * ||beta||1. The user clarified that
   lambda may implicitly depend on sigma. A written lambda(sigma)*sigma penalty
   is equivalent only when that product is lambda0. A constant lambda times
   sigma*||beta||1 is not jointly convex in general. Track the convention
   through the objective, scale derivative, and coefficient subproblem.
2. For mean sigma*h(r/sigma), the coordinate Lipschitz bound is
   ||x_j||²/(n*sigma). Multiplying the coefficient subproblem by sigma permits
   using ||x_j||²/n with the correspondingly rescaled penalty sigma*lambda0.
   This explains one legitimate appearance of lambda0*sigma in inner updates.
3. A gradient update at current beta uses the full current residual. With
   partial residuals, solve for the absolute coordinate; do not also add the old
   coefficient as though the gradient were evaluated at current beta.
4. Minimizing a quadratic MM surrogate is not rome's exact Huber coordinate
   minimization. Preserve the requested exact solver unless a change is justified.
5. The draft's fixed-point scale equation is not an exact scale minimizer in
   one evaluation. Derive an MM bound for the canonical objective, retain the
   scale constant consistently, and distinguish stationarity from an iteration.
6. Handle the closed perspective at sigma=0; an arbitrary floor changes the
   feasible problem and must not be silently imposed on only one benchmark.
7. Reassess screening when scale changes. Validate full joint optimality.
8. c-lasso R4 baselines are Path-Alg and DR; do not use the draft's PF-PDS label
   without establishing that it actually solves the same concomitant problem.

Open: pin Combettes–Müller equation and source version, map c-lasso normalization,
validate MM descent, derive stopping criteria, and implement independent checks.

## Implemented correction

The positive-scale algorithm is now derived in algorithm-design.md and implemented
in src/concomitant_huber/solver.py. Independent optimizer tests validate the
coordinate minimizer and a joint solution; scale-MM tangency and descent are
tested. Boundary certification, c-lasso normalization verification, and the
full publication convergence proof remain open.
