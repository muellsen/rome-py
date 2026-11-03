# Exact coefficient CD with a scale MM step

## Objective and normalization

For h_rho(z) = z²/2 when |z| <= rho and rho|z|-rho²/2 otherwise,
minimize

    F(b, sigma) = mean[sigma h_rho((y - b0 - X b)/sigma)]
                  + a sigma + lambda0 ||b||1,  sigma >= 0.

The intercept is not penalized. At sigma=0 the closed perspective is
rho*mean|r|. The formulation follows Combettes and Müller (2020), equation
(1.2), with a equal to their additive scale constant, augmented by the lasso
penalty. Source: https://pcombet.math.ncsu.edu/ejs1.pdf.
The scale is an M-scale; it is not automatically the noise standard deviation.
Normal-consistency calibration would set 2a = E[min(Z²,rho²)] for Z~N(0,1).
The pilot uses a=1/2 explicitly, not a normal-consistency calibration.

## A genuine global MM scale update

The identity

    sigma h_rho(r/sigma) = min_o [(r-o)²/(2 sigma) + rho |o|]

has minimizer o_t = soft(r, rho*t) at current scale t>0. Holding these
shifts fixed gives a global scale majorizer

    Q(sigma | t) = mean[(r-o_t)²]/(2 sigma) + a sigma
                   + rho mean|o_t| + lambda0 ||b||1.

It touches F at t. Therefore its minimizer is

    sigma_new = sqrt(mean[min(r², rho²*t²)]/(2a)).

The three inequalities F(b,sigma_new) <= Q(sigma_new|t) <= Q(t|t) = F(b,t)
prove descent. This is one MM step, not an exact scale-block minimization.
It costs O(n), with no sorting, matrix inversion or inner convergence loop.

## Preserve exact coefficient minimization

For fixed positive sigma, multiplying the coefficient objective by sigma
converts it to

    mean h_(rho*sigma)(r) + (sigma*lambda0) ||b||1.

Thus rome's piecewise-affine derivative/kink sweep can be reused with
threshold rho*sigma and penalty sigma*lambda0. The scale dependence in this
inner penalty is required; the joint objective penalty remains lambda0.
The new implementation follows the kink-sweep principle independently of
rome's C code. It first tests the exact zero-coordinate KKT condition, then
walks positive breakpoints in the descent direction and interpolates the
piecewise-affine derivative root. This is exact up to floating-point error,
rather than one proximal-gradient or local Newton step.

Take a bounded batch of scale MM steps and one cyclic coefficient sweep per
iteration. The default batch cap is eight, stopping the scale batch early when
its stationarity residual is <= tol/2. The one-step variant remains available
for comparison. Every step satisfies the same descent inequality; there is no
extrapolation or unverified Newton correction. The cap is configurable as
scale_mm_steps. Do not
fully solve the coefficient problem between scale updates. Refresh all zero-
coordinate checks at the current scale; a skipped coordinate can be activated
on the next sweep. Full KKT diagnostics follow every sweep. No unverified
strong screening rule is carried over across scale changes.

## Stopping and scope

For sigma>0 let psi=clip(r/sigma,-rho,rho). Stop only when the maximum
coefficient/intercept KKT violation and |a - mean(psi²)/2| are each <= tol.
Fixed-scale mode uses the identical solver and only the coefficient criterion.
Coordinate and scale steps both decrease F; objective values are bounded below.
This alone is not a global convergence theorem at the zero-scale boundary.
On positive-scale compact level sets bounded away from zero, standard block-MM
arguments can be developed; a publication-ready convergence proof is pending.

The prototype handles dense finite designs and lasso penalties, with an optional
unpenalized intercept. It does not yet implement regularization paths, sparse
matrices, adaptive/elastic-net penalties or boundary optimality certificates.
If scale reaches zero it returns boundary_uncertified with converged_=False;
it never silently imposes a floor and declares the R4 problem solved.

## Pilot interpretation

Use the same compiled kernels for both methods, with compilation separately
reported. Freeze scale at the estimated joint optimum for the main baseline;
then the coefficient solution must agree. A fixed-at-initial-scale control
is also recorded but solves a different problem. Record KKT residuals, sweeps,
seed-level paired ratios, individual timings and raw inputs. A negligible scale
step does not guarantee a negligible total overhead: changing scale can require
more coefficient sweeps. Do not select settings or tolerances based on timing.

## Pilot revisions

The initial penalty fraction 0.15 failed its positive-scale KKT check and is
retained in pilot-initial-failure.json. The pilot was changed to fraction 0.4
for positive-scale evaluation. The first completed MM1 pilot found measurable
overhead, not speed parity; its outputs and exact solver/driver snapshots are
preserved in analysis/outputs/mm1-pilot/. The next pilot compares MM1 and
MM8 on the same data, parameters and accuracy. These revisions are motivated
by convergence diagnostics and algorithm structure; all timings are retained.
