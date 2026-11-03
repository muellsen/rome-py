"""Opt-in sweep traces generated from the production kernel's syntax tree.

Instrumentation adds state copies only; it is excluded from all timing studies.
"""
import ast
import inspect
import textwrap

import numpy as np
from numba import njit
from . import solver


def _instrument():
    tree = ast.parse(textwrap.dedent(inspect.getsource(solver._solve.py_func)))
    function = tree.body[0]
    function.name = '_traced_solve'
    function.decorator_list = []
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


_trace_kernel = _instrument()


def fit_trace(model, X, y):
    """Fit a HuberCD model and return sweep states, including initialization.

    The ordinary fit validates arguments and supplies the public result. The
    instrumented rerun must reproduce that result before states are returned.
    This diagnostic performs two fits and must never be used for timing.
    """
    model.fit(X, y)
    X, y = np.asarray(X, float), np.asarray(y, float)
    design = np.column_stack((np.ones(len(y)), X)) if model.fit_intercept else X
    design = np.asfortranarray(design)
    sigma = model.initial_scale
    if sigma is None:
        sigma = max(float(np.median(np.abs(y-np.median(y)))/0.67448975),
                    float(np.std(y))*1e-3, np.finfo(float).eps)
    beta = np.empty((model.max_iter+1, design.shape[1]))
    scale = np.empty(model.max_iter+1)
    result = _trace_kernel(design, y, model.penalty, model.rho, model.scale_penalty,
                           sigma, model.estimate_scale, model.tol, model.max_iter,
                           model.fit_intercept, model.scale_mm_steps, beta, scale)
    expected = np.r_[model.intercept_, model.coef_] if model.fit_intercept else model.coef_
    np.testing.assert_allclose(result[0], expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result[1], model.scale_, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result[3], model.objective_history_, rtol=0, atol=1e-13)
    length = result[2]+1
    return {'beta': beta[:length].copy(), 'sigma': scale[:length].copy(),
            'objective': result[3].copy()}
