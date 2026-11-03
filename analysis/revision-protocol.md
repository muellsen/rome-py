# Focused JCGS revision

The canonical paper is paper/manuscript.qmd. The writing and author-manuscript
layout follow Yoon, Müller and Gaynanova, Fast computation of latent correlations.
Algorithms use the ruled/vlined algorithm2e environment of the ROME source.
The prior broad manuscript is retained as a superseded private source record.

## Primary computations

1. ROME Section 4.1: four original design distributions, nine original n,p cells,
   original executable-script sparse vector and 100-penalty grid; only original
   R/C ROME and concomitant MM-CD. Ten replications per cell were explicitly
   chosen by the user for this revision, rather than the paper's 100.
   Four workers run independent pairs with one BLAS thread each.
   Numerical CPU times exclude startup, compilation and serialization.
   Native ROME tolerance and joint KKT tolerance are reported separately.
   A 4000-sweep point budget and 120-second path safeguard are fixed.
   Failures and partial paths are never eligible full-path speed results.
2. R4: independent Gaussian designs, moderate sample/feature scaling, two seeds,
   three fresh repetitions, same perspective objective. Both supported c-lasso
   algorithms are present: mean-shift Path-Alg (zero Gram correction) and DR
   (c=1). Audited native update equations are unchanged. DR stops using a
   common external accuracy gate, charged to its solve time. Setup and solve
   costs are separate. A same-instance trajectory compares CD/DR iterates with
   the matched Path-Alg endpoint.
3. Applications: ten outer holdouts per dataset, five-fold inner CV, fixed and
   joint Huber lasso, identical Python exact coordinate kernel and intercept
   convention. Filtering, centering and standardization are training-only.
   Glass training/test sizes are 120/60; B. subtilis sizes are 50/21.
   The correlation filter retains 500 genes in each appropriate training fold.
   The final grid has 50 fractions from 1 to .001. An initial grid ending at
   .05 was often selected at its boundary by inner CV; those exploratory runs
   remain archived. The expanded grid follows the ROME path default.
   No held-out score selects a penalty or preprocessing rule.
4. AMP: retain the source-protocol B. subtilis instance with global unsupervised
   correlation filtering, n=57, p=500, lambda_AMP=2.5, omega=14/500. This is a
   matched numerical problem, not another CV prediction estimate.

## Commands

    make algorithm-template
    make revision-timings
    make revision-r4
    make revision-real
    make revision-figures
    make pdf manuscript

All numerical stages have outputs and provenance sidecars under
analysis/outputs/revision-*. Per-case timing and per-fold CV checkpoints allow
resumption without silently dropping failures. The rendering workflow verifies
numerical input hashes; old AMP source is required only for reproducing that
specific comparison. Broad AMP feasibility searches and earlier audits stay
outside the main text. The ROME timing track and c-lasso R4 track have equal
standing in the main paper.

## Implementation qualification

The timing experiment compares native ROME screening with current-scale full
checks in the extension, as documented in Algorithm 1. Different native
stopping rules, compiled languages and numbers of coefficient sweeps mean that
this experiment does not establish uniform scale-estimation overhead.
The R4 study compares matched objectives and accuracy instead.

## Optimized revision

The reference solver and the 360-path/R4/AMP baseline outputs remain intact.
The new backend uses verified exact local affine roots, reusable buffers,
provisional working sets with full KKT certification, and optional support
Newton polishing every five sweeps for supports at most 128. Newton directions
use a relative 1e-10 ridge, Armijo constant 1e-4, and at most 24 halvings. The
objective is unchanged. A scale below 1e-10 times the response standard deviation
triggers a boundary LP, never automatic acceptance or a numerical scale floor.
A rescaled feasible perspective dual and relative primal-dual gap at most1e-6
certify a zero-scale return.

Revised CV uses 50 fractions from1 to.001, with each model's own optimized null
intercept (and joint scale) for its lambda_max. Positive-scale iteration budget
is20,000; both models use the same optimized backend and stopping thresholds.
Optimized paths record failed points but continue certifying later penalties.
Numerical CV ties within1e-10 times max(1,abs(minimum pooled MSE)) select the
largest penalty. All preprocessing/filtering remains training-only. Original
shared-anchor CV outputs and superseded optimized pilot runs are retained.

## Paired Python scale-cost timing

The four saved optimized validation datasets (seed71) each receive five fresh
paired timing repeats. Both modes use solve_warm with rho=.5, fixed/initial
sigma=1, the original100 absolute penalties, no intercept/standardization,
coefficient tolerance1e-6,4000 sweeps per point, a120-second path safeguard,
check_every=5,newton_every=5, and scale_mm_steps=8. The joint fit additionally
requires positive-scale stationarity or a feasible boundary dual and gap.
Both are independently rechecked outside the timer. Compilation, input loading,
serialization and post-fit verification are untimed; internal diagnostics and
boundary LPs are timed. Alternate method order across repeats and datasets.
Report medians on common fully certified repeats, median paired CPU ratios,
all failures, sweep counts and boundary counts. Timing repeats of a fixed
dataset are not counted as independent simulation replications. The original
R/C-versus-Python experiment and the c-lasso R4 comparison remain separate.

## Three interchangeable scale blocks (selection pending)

The public FastHuberCD exposes scale_update=mm/exact/prox. A common production
coefficient loop is generated with only its scale block replaced. Existing
optimized.solve_warm and solver.py remain unchanged to preserve historical
MM measurements. Scale computations normalize r and sigma by max(abs(r)).
The zero-residual and nonnegative right-derivative boundary cases are handled
explicitly. No scale floor is used. The exact option bisects the original
scale equation. The proximal option uses observation-wise joint perspective
proxs and DR projection onto fixed residuals and a common scale. Its lifted
objective is n times the averaged normalized block objective, so local proxs
use prox_gamma, not prox_gamma/n. Each local prox uses a scalar projection-root
solve with relative interval tolerance1e-14 and at most80 halvings.

Both root and proximal options report the original block stationarity and a
feasible-dual normalized gap. Finite MM batches may remain inner-inexact.
Truncated proximal updates return only their best feasible descent shadow.
An inner limit never substitutes for full outer joint optimality. A zero-scale
block minimizer still needs the existing regression boundary dual certificate.

Initial validation: three declared45-by8 fixtures, three timing repeats per
method, common starts/penalty/kernel, outer tolerance1e-7, inner tolerance5e-8,
MM batch8, root/DR limit2000, normalized prox_gamma1, no support Newton
polishing. Independent reference fits use the original solver at1e-9.
External requirements are joint residual1e-7 and relative objective gap1e-8.
No winner is selected from this validation. Broader accuracy-qualified path
studies, including zero-scale cases, precede moving a selected method to the
main paper and the two alternatives to the Appendix.

## Moderate paired MM / scalar-root decision study

Before method selection, run `make scale-path-benchmark`: all four original
ROME designs, dimensions (100,100), (100,500), (500,500), and seeds71--73.
These are36 independent paired datasets, not timing repetitions. Each path
contains30 absolute penalties lambda_max(fixed scale1) times .05^(k/30),
k=1,...,30; both methods receive precisely the same grid and starts.
Use rho=.5, the existing Gaussian-consistency scale constant, no intercept
or standardization, joint tolerance1e-6, scalar inner tolerance5e-7, MM batch8,
root limit2000, maximum4000 sweeps per point, checks every5 sweeps, and
identical optional support Newton polishing every5 sweeps. Both methods use
the common production coefficient loop and the same certified boundary LP
fallback. This compares interchangeable scale blocks in the optimized solver;
it does not isolate unaccelerated MM from all other scale-changing steps.

Compile/warm JIT and the LP backend before timing; run paths serially and
alternate method order across consecutive cases. Prepared CPU/wall timings
include internal certificates and boundary LPs, exclude input preparation,
serialization and external certification. The120-second path safeguard is
soft and checked between penalties. A failed point stops that path; failures
and unattempted tails are retained. Only independently qualified complete
pairs enter paired time ratios. Report completion for every dataset as well
as objectives, fitted values, coefficient/scale differences, sweep counts,
inner iterations and LP time. Under nonuniqueness, coefficient agreement is
descriptive, not an equivalence requirement.

Four separate diagnostics use seed71, n=p=100, one case per design, and
the eleventh penalty, starting cold. Prefix solves at declared sweep budgets
provide joint residuals against standalone prefix CPU time (three timing
repetitions); these are restarted prefixes, not a continuous instrumented
clock. A recording-only clone must reproduce production results exactly,
then its stored residual/scale inputs are replayed through the original
scale routine to measure isolated block work. Replay timings are diagnostic
and are not subtracted from or added to primary path CPU. Full provenance,
checkpoints and raw solutions live under outputs/scale-path-benchmark.

## Complete three-implementation Figure1 rerun

Run all360 saved original-paper datasets, with ten replications per cell and
the original100 absolute penalties. Native R/C ROME is rerun with its original
eps1e-4 stopping rule; fixed-scale Python and joint Python use the common
optimized coefficient routines, tolerance1e-6,4000 sweeps, working sets and
support polishing every5 sweeps. Joint scale updates are MM batches of8.
This new experiment is separate from all historical truncated paths and from
the moderate MM/root pilot. Solves run serially with single-threaded libraries;
fixed/joint execution order alternates by dataset. No path wall-budget tail
is counted as visited: every penalty receives a solver or LAD-lasso call.

An uncertified joint iterate is compared with the LAD-lasso boundary solution
under the original perspective objective; return whichever is smaller.
At sigma=0, LAD loss is rho*mean(abs(r)), and its conventional penalty is
lambda0/rho. Failed boundary certification does not silently certify the
candidate. Continue to the next penalty with the returned coefficients. If
the preceding scale is zero, try the boundary LP first; when uncertified,
restart positive-scale CD from scale1 to avoid permanently locking the path
on an uncertified boundary. The iterate cap is not relabeled convergence.

All three timing curves use all360 complete returned paths. Timings include
candidate selection, boundary LP calls and continuation. Exclude compilation,
input loading, process startup, serialization and independent final audits.
External checks distinguish strict certificates from normalized feasible-dual
gap <=1e-4 and from larger unresolved bounds. Hollow Python markers identify
cells with any externally uncertified path. The native ROME curve is expressly
not accuracy-matched; all independent stationarity checks are saved. New
checkpoints and provenance are under outputs/figure1-complete.
