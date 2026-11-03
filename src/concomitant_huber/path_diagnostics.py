"""Opt-in warm path traces; never used for timing."""
import ast
import inspect
import textwrap
import numpy as np
from numba import njit
from . import solver


def _instrument_warm():
    tree = ast.parse(textwrap.dedent(inspect.getsource(solver._solve.py_func)))
    function = tree.body[0]
    function.name = '_traced_solve'
    function.decorator_list = []
    function.args.args.append(ast.arg(arg='beta_start'))
    class WarmInitialize(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'beta':
                    node.value = ast.parse('beta_start.copy()', mode='eval').body
                elif node.targets[0].id == 'r':
                    node.value = ast.parse('y-X@beta', mode='eval').body
            return node
    tree = WarmInitialize().visit(tree)
    function.args.args.extend([ast.arg(arg='beta_trace'), ast.arg(arg='scale_trace')])

    class Recorder(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0]
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'history'):
                index = ast.unparse(target.slice)
                extra = ast.parse(f'beta_trace[{index}, :] = beta\nscale_trace[{index}] = sigma').body
                return [node, *extra]
            return node
    tree = ast.fix_missing_locations(Recorder().visit(tree))
    namespace = dict(vars(solver))
    exec(compile(tree, '<production-kernel-trace>', 'exec'), namespace)
    return njit(namespace['_traced_solve'])


_warm_trace_kernel = _instrument_warm()


def trace_warm_point(X, y, beta, sigma, penalty, *, rho=1.345,
                     scale_penalty=.5, estimate_scale=True, tol=1e-7,
                     max_iter=2000, scale_mm_steps=8):
    """Diagnostic warm continuation, verified against the production path kernel.

    No intercept or preprocessing. Inputs follow huber_path conventions.
    This performs two solves and is deliberately unsuitable for runtime comparisons.
    """
    from .path import _solve_warm
    X = np.asfortranarray(X, dtype=float)
    y, beta = np.asarray(y, dtype=float), np.asarray(beta, dtype=float)
    if (X.ndim != 2 or y.shape != (len(X),) or beta.shape != (X.shape[1],)
            or len(y) == 0 or not all(np.isfinite(v).all() for v in (X, y, beta))):
        raise ValueError('Invalid warm state')
    if (not all(np.isfinite(v) and v > 0 for v in (sigma, rho, scale_penalty, tol))
            or not np.isfinite(penalty) or penalty < 0
            or not isinstance(max_iter, int) or max_iter < 1
            or not isinstance(scale_mm_steps, int) or scale_mm_steps < 1):
        raise ValueError('Invalid constants or budgets')
    args = (X, y, penalty, rho, scale_penalty, sigma, estimate_scale,
            tol, max_iter, False, scale_mm_steps, beta)
    expected = _solve_warm(*args)
    beta_trace = np.empty((max_iter+1, X.shape[1]))
    scale_trace = np.empty(max_iter+1)
    actual = _warm_trace_kernel(*args, beta_trace, scale_trace)
    for j in (0, 1, 3, 5, 6):
        np.testing.assert_allclose(actual[j], expected[j], rtol=0, atol=1e-13)
    assert actual[2] == expected[2] and actual[4] == expected[4]
    length = actual[2]+1
    return {'beta': beta_trace[:length].copy(), 'sigma': scale_trace[:length].copy(),
            'objective': actual[3].copy(), 'converged': bool(actual[4]),
            'coefficient_kkt': float(actual[5]), 'scale_kkt': float(actual[6])}
