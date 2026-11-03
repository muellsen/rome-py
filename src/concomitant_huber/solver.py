"""Exact coefficient CD and mean-shift MM for the closed Huber perspective.

The kink-sweep principle follows Kim, Loh and Basu (arXiv:2510.13715).
This implementation is written independently of the R/C implementation.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from numba import njit


@njit
def _coordinate(u, x, delta, penalty):
    """Minimize mean h_delta(u-x*b) + penalty*abs(b), exactly up to roundoff."""
    n = len(u)
    gradient = 0.0
    for i in range(n):
        gradient -= x[i] * min(delta, max(-delta, u[i])) / n
    if abs(gradient) <= penalty:
        return 0.0
    direction = 1.0 if gradient < 0 else -1.0
    derivative = direction * gradient + penalty
    slope = 0.0
    knots = np.empty(2 * n)
    jumps = np.empty(2 * n)
    count = 0
    for i in range(n):
        if x[i] == 0:
            continue
        center = u[i] / (direction * x[i])
        width = delta / abs(x[i])
        left, right = center - width, center + width
        weight = x[i] ** 2 / n
        if left <= 0 < right:
            slope += weight
        if left > 0:
            knots[count], jumps[count] = left, weight
            count += 1
        if right > 0:
            knots[count], jumps[count] = right, -weight
            count += 1
    order = np.argsort(knots[:count])
    current = 0.0
    for k in order:
        gap = knots[k] - current
        if slope > 0 and derivative + slope * gap >= 0:
            return direction * (current - derivative / slope)
        derivative += slope * gap
        current = knots[k]
        slope += jumps[k]
    # At the last knot the derivative must be nonnegative (up to roundoff).
    return direction * current


@njit
def _objective(r, beta, sigma, rho, a, penalty, intercept=False):
    n = len(r)
    value = a * sigma
    for j in range(len(beta)):
        if not (intercept and j == 0):
            value += penalty * abs(beta[j])
    if sigma == 0:
        for i in range(n):
            value += rho * abs(r[i]) / n
    else:
        delta = rho * sigma
        for i in range(n):
            t = abs(r[i])
            value += (0.5 * t * t / sigma if t <= delta
                      else rho * t - 0.5 * rho * rho * sigma) / n
    return value


@njit
def _diagnostics(X, r, beta, sigma, rho, a, penalty, intercept):
    if sigma == 0:
        return np.inf, np.inf
    n, p = X.shape
    violation = 0.0
    squared = 0.0
    for i in range(n):
        z = min(rho, max(-rho, r[i] / sigma))
        squared += z * z / n
    for j in range(p):
        gradient = 0.0
        for i in range(n):
            gradient -= X[i, j] * min(rho, max(-rho, r[i] / sigma)) / n
        lam = 0.0 if intercept and j == 0 else penalty
        if beta[j] > 0:
            v = abs(gradient + lam)
        elif beta[j] < 0:
            v = abs(gradient - lam)
        else:
            v = max(0.0, abs(gradient) - lam)
        violation = max(violation, v)
    return violation, abs(a - 0.5 * squared)


@njit
def _solve(X, y, penalty, rho, a, sigma, estimate_scale, tol, max_iter, intercept, scale_mm_steps):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()
    history = np.empty(max_iter + 1)
    history[0] = _objective(r, beta, sigma, rho, a, penalty, intercept)
    active = np.ones(p, dtype=np.bool_)
    sweeps = 0
    for iteration in range(max_iter):
        if estimate_scale:
            for scale_step in range(scale_mm_steps):
                squared = 0.0
                delta = rho * sigma
                for i in range(n):
                    squared += min(r[i] ** 2, delta ** 2) / n
                new_sigma = np.sqrt(squared / (2 * a))
                sigma = new_sigma
                if sigma == 0:
                    break
                scale_gradient = a
                for i in range(n):
                    scale_gradient -= 0.5 * min((r[i] / sigma) ** 2, rho ** 2) / n
                if abs(scale_gradient) <= tol * 0.5:
                    break
        if sigma == 0:
            # Closed boundary is reported explicitly; no hidden scale floor.
            value = _objective(r, beta, sigma, rho, a, penalty, intercept)
            history[iteration + 1] = value
            return beta, sigma, iteration + 1, history[:iteration + 2], False, np.inf, np.inf, sweeps
        delta = rho * sigma
        # Full check every sweep: safe active-set filtering is refreshed at the new scale.
        for j in range(p):
            if beta[j] == 0 and not (intercept and j == 0):
                gradient = 0.0
                for i in range(n):
                    gradient -= X[i, j] * min(delta, max(-delta, r[i])) / n
                active[j] = abs(gradient) > sigma * penalty
            else:
                active[j] = True
        for j in range(p):
            if not active[j]:
                continue
            u = r + X[:, j] * beta[j]
            lam = 0.0 if intercept and j == 0 else sigma * penalty
            new = _coordinate(u, X[:, j], delta, lam)
            for i in range(n):
                r[i] -= X[i, j] * (new - beta[j])
            beta[j] = new
        sweeps += 1
        history[iteration + 1] = _objective(r, beta, sigma, rho, a, penalty, intercept)
        vb, vs = _diagnostics(X, r, beta, sigma, rho, a, penalty, intercept)
        if vb <= tol and (not estimate_scale or vs <= tol):
            return beta, sigma, iteration + 1, history[:iteration + 2], True, vb, vs, sweeps
    return beta, sigma, max_iter, history, False, vb, vs, sweeps


@dataclass
class HuberCD:
    """Dense lasso-Huber CD; optional concomitant MM scale estimation.

    Objective: mean sigma*h_rho((y-intercept-X beta)/sigma)
    + scale_penalty*sigma + penalty*||beta||_1.
    For fixed scale the scale term is constant. No standardization is applied.
    A zero-scale boundary is returned with converged_=False pending a boundary
    dual certificate; the present implementation certifies positive scales only.
    """
    penalty: float = 0.05
    rho: float = 1.345
    scale_penalty: float = 0.5
    estimate_scale: bool = True
    initial_scale: float | None = None
    fit_intercept: bool = True
    tol: float = 1e-7
    max_iter: int = 2000
    scale_mm_steps: int = 8

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or not len(y):
            raise ValueError('Expected finite X with shape (n,p) and y with shape (n,)')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('X and y must be finite')
        values = (self.penalty, self.rho, self.scale_penalty, self.tol)
        if not all(np.isfinite(v) for v in values) or self.penalty < 0 or min(values[1:]) <= 0:
            raise ValueError('penalty must be nonnegative; rho, scale_penalty and tol positive')
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        if not isinstance(self.scale_mm_steps, (int, np.integer)) or self.scale_mm_steps < 1:
            raise ValueError('scale_mm_steps must be a positive integer')
        self.n_features_in_ = X.shape[1]
        design = np.column_stack((np.ones(len(y)), X)) if self.fit_intercept else X
        design = np.asfortranarray(design)
        sigma = self.initial_scale
        if sigma is None:
            sigma = max(float(np.median(np.abs(y - np.median(y))) / 0.67448975),
                        float(np.std(y)) * 1e-3, np.finfo(float).eps)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('initial_scale must be finite and strictly positive')
        result = _solve(design, y, self.penalty, self.rho, self.scale_penalty,
                        sigma, self.estimate_scale, self.tol, self.max_iter, self.fit_intercept, self.scale_mm_steps)
        b, self.scale_, self.n_iter_, self.objective_history_, self.converged_, self.coef_kkt_, self.scale_kkt_, self.n_sweeps_ = result
        self.intercept_ = float(b[0]) if self.fit_intercept else 0.0
        self.coef_ = b[1:].copy() if self.fit_intercept else b.copy()
        self.objective_ = float(self.objective_history_[-1])
        self.status_ = 'converged' if self.converged_ else ('boundary_uncertified' if self.scale_ == 0 else 'max_iter')
        if not self.converged_:
            warnings.warn(f'HuberCD ended with status {self.status_}', RuntimeWarning)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Prediction X has incorrect shape')
        return self.intercept_ + X @ self.coef_
