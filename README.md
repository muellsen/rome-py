# ROME for Python (`rome-py`)

**Sparse robust M-estimation with fixed or jointly estimated scale.**

ROME for Python implements exact coordinate descent for lasso-penalized Huber
regression and extends it to concomitant scale estimation using the
Combettes–Müller perspective formulation. The fixed-scale algorithm follows
Kim, Loh, and Basu's [ROME paper](https://arxiv.org/abs/2510.13715) and
[R implementation](https://github.com/yk748/rome).

The package provides fitted coefficients, an unpenalized intercept, predictions,
regularization paths, and tools for training-only preprocessing and
cross-validation. Both formulations use the same compiled coefficient updates;
concomitant fitting adds an MM scale update. The repository also contains the
Quarto manuscript and reproducible experiments for a short JCGS note.

- [Problem formulations](#problem-formulations)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Regularization paths](#regularization-paths)
- [Cross-validation](#cross-validation)
- [Optimization and numerical accuracy](#optimization-and-numerical-accuracy)
- [Research and reproducibility](#research-and-reproducibility)
- [References](#references)

## Problem formulations

Let $X\in\mathbb{R}^{n\times p}$ contain the predictors, let
$y\in\mathbb{R}^n$ be the response, and write

$$
r_i(b,\beta)=y_i-b-x_i^\top\beta.
$$

Here $b\in\mathbb{R}$ is an unpenalized intercept and
$\beta\in\mathbb{R}^p$ is the coefficient vector. Setting
`fit_intercept=False` fixes $b=0$. For $\rho>0$, the Huber loss and score are

$$
h_\rho(u)=
\begin{cases}
\tfrac12 u^2, & |u|\leq\rho,\\
\rho|u|-\tfrac12\rho^2, & |u|>\rho,
\end{cases}
\qquad
\psi_\rho(u)=\operatorname{sign}(u)\min\{|u|,\rho\}.
$$

### Fixed-scale sparse Huber regression

For a specified scale $s_0>0$, the package solves

$$
\underset{b,\beta}{\operatorname{minimize}}\quad
\frac{s_0}{n}\sum_{i=1}^{n}
h_\rho\!\left(\frac{r_i(b,\beta)}{s_0}\right)
+\lambda_0\|\beta\|_1.
$$

Multiplying by $s_0$ gives the equivalent ROME lasso objective

$$
\underset{b,\beta}{\operatorname{minimize}}\quad
\frac1n\sum_{i=1}^{n}h_\delta\!\left(r_i(b,\beta)\right)
+\eta\|\beta\|_1,
\qquad \delta=\rho s_0,\quad \eta=s_0\lambda_0.
$$

Use `estimate_scale=False` and `initial_scale=s0`. At $s_0=1$, the package's
`penalty` equals the penalty $\eta$ in this ROME objective. The original ROME
package also supports elastic-net penalties; this implementation currently
supports the lasso penalty.

### Concomitant sparse Huber regression

For $a>0$, the scale is estimated jointly with the coefficients:

$$
\underset{b,\beta,\,\sigma\geq0}{\operatorname{minimize}}\quad
\frac1n\sum_{i=1}^{n}\widetilde h_\rho
\!\left(\sigma,r_i(b,\beta)\right)
+a\sigma+\lambda_0\|\beta\|_1,
$$

where the closed perspective is

$$
\widetilde h_\rho(\sigma,r)=
\begin{cases}
\sigma h_\rho(r/\sigma), & \sigma>0,\\
\rho|r|, & \sigma=0,\\
+\infty, & \sigma<0.
\end{cases}
$$

Use `estimate_scale=True`, the default. At $\sigma=0$, the objective reduces to
LAD-lasso with loss $\rho\|y-b\mathbf{1}-X\beta\|_1/n$.
`FastHuberCD` can return a zero-scale solution after checking a perspective
dual certificate; no positive scale floor is imposed.

The parameter correspondence is:

| Mathematical quantity | Python parameter | Estimator default |
|:--|:--|:--|
| $\lambda_0$ | `penalty` | `0.05` |
| $\rho$ | `rho` | `1.345` |
| $a$ | `scale_penalty` | `0.5` |
| Fixed scale $s_0$, or starting scale $\sigma^{(0)}$ | `initial_scale` | `None` (automatic initialization) |
| Estimate $\sigma$ | `estimate_scale` | `True` |
| Estimate $b$ | `fit_intercept` | `True` |

`penalty` is an **absolute penalty for the averaged objective**, not a fraction
of a maximum penalty. The joint parameter $\lambda_0$ stays fixed during a fit.
At a positive current scale $\sigma$, the equivalent coefficient subproblem
uses $\delta=\rho\sigma$ and $\eta=\sigma\lambda_0$; those two quantities change
when the scale changes.

The default $a=0.5$ specifies a scale convention. For Gaussian scale calibration,
choose

$$
a_{\mathrm G}(\rho)=\tfrac12\mathbb E\!\left[\min\{Z^2,\rho^2\}\right],
\qquad Z\sim\mathcal N(0,1).
$$

For example, $a_{\mathrm G}(1.345)\approx0.355082274$. This choice concerns the
population scale equation under Gaussian residuals; it does not remove the
finite-sample effect of coefficient penalization.

## Installation

Python 3.10 or later is required. Install from the repository:

```bash
git clone https://github.com/muellsen/rome-py.git
cd rome-py
python -m pip install .
```

Access to the repository is currently required; a PyPI release has not been
published. For development and the c-lasso comparisons:

```bash
python -m pip install -e ".[test,benchmark]"
```

The distribution name is `rome-py`; the import name is `rome_py`. NumPy, SciPy,
and Numba are runtime dependencies. The first fit compiles numerical kernels,
so exclude that first call when measuring steady-state solver time.

## Getting started

### Generate data

The following examples use a sparse linear model with contaminated responses.
The last 20 observations are held out for prediction.

```python
import numpy as np
from rome_py import FastHuberCD

rng = np.random.default_rng(71)
X = rng.normal(size=(120, 20))
beta = np.zeros(X.shape[1])
beta[:3] = [2.0, -1.5, 1.0]
y = 0.5 + X @ beta + rng.normal(size=len(X))
y[:6] += 8.0

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]
penalty = 0.1
rho = 1.345
a = 0.5
```

### Fit with a fixed scale

```python
fixed = FastHuberCD(
    penalty=penalty,
    rho=rho,
    estimate_scale=False,
    initial_scale=1.0,
).fit(X_train, y_train)

print(fixed.status_)
print(fixed.intercept_, fixed.coef_)
print("Fixed scale:", fixed.scale_)
```

### Estimate coefficients and scale jointly

```python
joint = FastHuberCD(
    penalty=penalty,
    rho=rho,
    scale_penalty=a,
    estimate_scale=True,
    scale_update="mm",
    initial_scale=1.0,
).fit(X_train, y_train)

if not joint.converged_:
    raise RuntimeError(f"Fit requires attention: {joint.status_}")

print("Estimated scale:", joint.scale_)
print("Intercept:", joint.intercept_)
print("Coefficients:", joint.coef_)
predictions = joint.predict(X_test)
print("Test MSE:", np.mean((y_test - predictions) ** 2))
```

Neither estimator standardizes predictors or the response automatically.
`HuberCD` offers the reference coefficient-CD/MM implementation with the same
objective and estimator parameters; `FastHuberCD` adds verified coordinate-root
shortcuts, optional support Newton steps, and zero-scale certification.
The estimator interface provides `fit` and `predict`; full scikit-learn
estimator compatibility is not implemented.

## Regularization paths

`standardized_path` fits a descending sequence of absolute penalties with warm
starts and an unpenalized intercept. Despite its name, it does **not** transform
the inputs. Use `TrainingTransform` to learn centering and scaling from the
training observations, then apply those same transformations to new data.

```python
from rome_py.model_selection import (
    TrainingTransform,
    null_penalty_anchor,
    standardized_path,
)

transform = TrainingTransform.fit(X_train, y_train)
Xt = transform.design(X_train)
yt = transform.response(y_train)

anchor = null_penalty_anchor(Xt, yt, rho=rho, a=a, estimate_scale=True)
fractions = np.geomspace(1.0, 0.1, 20)
penalties = anchor["lambda_max"] * fractions

path = standardized_path(
    Xt, yt, penalties,
    rho=rho, a=a,
    estimate_scale=True,
    initial_scale=anchor["scale"],
    backend="polished",
)

print(path["status"])
print("Coefficient array, including intercept:", path["coefficients"].shape)
print("Scale path:", path["scales"])
```

The null-model anchor uses the intercept-only solution of the chosen model.
For a fixed-scale path, pass `estimate_scale=False, fixed_scale=1.0` to
`null_penalty_anchor` and `estimate_scale=False, initial_scale=1.0` to
`standardized_path`. Fixed and concomitant models generally have different
maximum penalties.

Rows of `path["coefficients"]` correspond to penalties; the first column is the
intercept. Scales are in transformed-response units. Multiply them by
`transform.y_scale` to express them in the original response units. Predictions
at a chosen path point are obtained as follows:

```python
k = 10  # An illustrative path point, not a CV-selected model.
if path["status"][k] != "converged":
    raise RuntimeError("Chosen path point did not meet the accuracy checks")
predictions = transform.predict(X_test, path["coefficients"][k])
```

The lower-level `rome_py.path.huber_path` omits the intercept and preprocessing,
and stops after an unsuccessful point. Unattempted points contain `NaN` and
have status `not_attempted`. The research helper
`rome_py.qualified_path.qualified_path` visits every penalty and supplies
objective bounds and explicit labels for approximate returns.

## Cross-validation

The package supplies preprocessing, null anchors, and warm paths; it does not
yet provide a single cross-validation estimator. This three-fold example
selects a penalty fraction using training data only. Each fold estimates its
own transformations and null-model anchor. The test set stays untouched until
the final prediction.

```python
folds = np.array_split(np.random.default_rng(19).permutation(len(y_train)), 3)
cv_fractions = np.array([0.8, 0.4, 0.2, 0.1])
squared_error = np.zeros(len(cv_fractions))

for validation in folds:
    training = np.setdiff1d(np.arange(len(y_train)), validation)
    tr = TrainingTransform.fit(X_train[training], y_train[training])
    Xf = tr.design(X_train[training])
    yf = tr.response(y_train[training])
    null = null_penalty_anchor(Xf, yf, rho=rho, a=a)
    fitted = standardized_path(
        Xf, yf, null["lambda_max"] * cv_fractions,
        rho=rho, a=a, initial_scale=null["scale"], backend="polished",
    )
    for j, status in enumerate(fitted["status"]):
        if status != "converged":
            raise RuntimeError(f"CV fit did not converge at fraction {cv_fractions[j]}")
        prediction = tr.predict(X_train[validation], fitted["coefficients"][j])
        squared_error[j] += np.sum((y_train[validation] - prediction) ** 2)

best = int(np.argmin(squared_error))
final_path = standardized_path(
    Xt, yt, [anchor["lambda_max"] * cv_fractions[best]],
    rho=rho, a=a, initial_scale=anchor["scale"], backend="polished",
)
if final_path["status"][0] != "converged":
    raise RuntimeError("Final refit did not converge")
predictions = transform.predict(X_test, final_path["coefficients"][0])
print("Selected penalty fraction:", cv_fractions[best])
print("Test MSE:", np.mean((y_test - predictions) ** 2))
```

For high-dimensional gene-expression data, `TrainingTransform.fit(X, y,
keep=500)` also learns a greedy correlation filter on the training predictors.
Fit that filter separately within each CV training fold. The manuscript's
B. subtilis comparison uses this training-only procedure for prediction; the
matched AMP experiment uses the separate source-paper filtering protocol.

## Optimization and numerical accuracy

Coefficient updates minimize the scalar Huber-plus-lasso objective exactly.
For the concomitant formulation, the default `scale_update="mm"` applies
majorization–minimization. At a current positive scale $t$ and fixed residuals,
its scale step is

$$
\sigma^+=\left(
\frac{1}{2an}\sum_{i=1}^{n}\min\{r_i^2,\rho^2t^2\}
\right)^{1/2}.
$$

The scale term $a\sigma$ is included in the majorizer. A finite batch of MM
steps decreases the scale objective; it is not an exact minimization claim.
Two other scale updates are available with the same coefficient loop:

| `scale_update` | Scale computation |
|:--|:--|
| `"mm"` (default) | Tangent-majorizer steps |
| `"exact"` | Numerical solution of the monotone scalar scale equation |
| `"prox"` | Joint residual/scale proximity operators with common-scale projection, iterated by Douglas–Rachford splitting |

```python
alternative = FastHuberCD(
    penalty=penalty, rho=rho, scale_penalty=a, scale_update="exact",
).fit(X_train, y_train)
```

Fixed-scale fitting bypasses scale updates. One joint proximity operation and
projection is not an exact scale minimizer.

The main `FastHuberCD` numerical controls are explicit:

| Parameter | Default | Meaning |
|:--|:--|:--|
| `tol` | `1e-7` | Coefficient and scale optimality tolerance |
| `max_iter` | `2000` | Maximum outer iterations |
| `scale_mm_steps` | `8` | Maximum MM steps per scale update |
| `check_every` | `5` | Interval between full optimality checks; additional checks occur near termination |
| `newton_every` | `5` | Support Newton interval; `0` disables these steps |
| `scale_inner_tol` | `None` | Uses `tol / 2` when unspecified |
| `scale_max_iter` | `2000` | Iteration limit for scalar-root or splitting scale updates |
| `prox_gamma` | `1.0` | Proximal splitting step size |
| `certify_boundary` | `True` | Attempt a certified zero-scale solution if positive-scale fitting is unsuccessful |

Inspect `converged_` and `status_` before using a fit. `FastHuberCD` reports
`converged` for a positive-scale fit passing its optimality checks,
`boundary_certified` for a certified zero-scale solution, and `max_iter` or
`boundary_uncertified` otherwise. The coefficient and scale residuals are
available as `coef_kkt_` and `scale_kkt_`; `objective_history_` records the
objective values. With fixed scale, the scale optimality condition is not a
stopping requirement, and the reported objective includes the constant
$a s_0$ omitted from the fixed-scale formulation above.

Version 0.0.1 supports dense arrays and lasso penalties. Sparse-matrix input,
elastic-net penalties, equality constraints, and classification formulations
are not implemented. The earlier `concomitant_huber` imports remain compatible
and refer to the same numerical implementation as `rome_py`.

## Research and reproducibility

The main experiments compare:

1. Original ROME in R/C, fixed-scale Python CD, and concomitant Python CD on
   ROME's four simulation designs, with ten replications per setting for this
   revision.
2. Concomitant CD and c-lasso's unconstrained R4 algorithms, Path-Alg and
   Douglas–Rachford, using matched objectives and common accuracy checks.
3. Fixed-scale and concomitant prediction on glass and B. subtilis with nested
   model selection, plus AMP on the matched B. subtilis problem instance.

Timings distinguish compilation, setup, and solver work. Numerical outputs
have provenance sidecars recording settings, inputs, source hashes, and
accuracy checks. Approximate solutions remain labeled in timing summaries.
No general speed ranking is claimed by the package API.

| Location | Contents |
|:--|:--|
| [`src/rome_py/`](src/rome_py/) | Public Python interface |
| [`src/concomitant_huber/`](src/concomitant_huber/) | Numerical implementation and compatible earlier imports |
| [`analysis/`](analysis/) | Experiment drivers and reproducibility tools |
| [`paper/manuscript.qmd`](paper/manuscript.qmd) | Canonical Quarto manuscript |
| [`sources/`](sources/) | Source registries, attribution, and download checksums |
| [`tests/`](tests/) | Numerical and provenance checks |

See [the analysis guide](analysis/README.md) for experiment commands and
[CONTRIBUTING.md](CONTRIBUTING.md) for development conventions. Downloaded
papers, datasets, private AMP sources, numerical outputs, and generated
manuscripts are excluded from Git. Some optional experiments require separately
supplied unpublished materials. The source distribution and wheel contain the
library; the repository contains the research workflow.

## References

- Kim, Y., Loh, P.-L., and Basu, S. (2025).
  [Exact Coordinate Descent for High-Dimensional Regularized Huber Regression](https://arxiv.org/abs/2510.13715).
  Original ROME algorithm and simulation designs.
- Combettes, P. L., and Müller, C. L. (2020).
  [Perspective maximum likelihood-type estimation via proximal decomposition](https://pcombet.math.ncsu.edu/ejs1.pdf).
  *Electronic Journal of Statistics*, 14, 207–238. Perspective M-estimation.
- Combettes, P. L., and Müller, C. L. (2018).
  [Perspective functions: Proximal calculus and applications in high-dimensional statistics](https://doi.org/10.1016/j.jmaa.2016.12.021).
  *Journal of Mathematical Analysis and Applications*, 457, 1283–1306.
- Friedman, J., Hastie, T., and Tibshirani, R. (2010).
  [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://doi.org/10.18637/jss.v033.i01).
  *Journal of Statistical Software*, 33(1), 1–22.
- Simpson, L., Combettes, P. L., and Müller, C. L. (2020).
  [c-lasso: a Python package for constrained sparse and robust regression and classification](https://arxiv.org/abs/2011.00898).
  R4 comparison and the formulation-first structure used in this README.

Please cite the methodological references relevant to your analysis and the
software metadata in [CITATION.cff](CITATION.cff). Project code is licensed
under [GPL-3.0-only](LICENSE); see [THIRD_PARTY.md](THIRD_PARTY.md) for attribution
and third-party licenses.
