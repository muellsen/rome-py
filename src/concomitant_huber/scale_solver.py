"""Use one production coefficient loop for all three scale-update options.

Only the scale block is replaced in the parsed production kernel. Existing
MM timing artifacts continue to use the unchanged optimized.solve_warm.
"""
import ast,inspect,textwrap
from numba import njit
from . import optimized
from .scale_updates import _scale_update


def _strategy_kernel():
    tree=ast.parse(textwrap.dedent(inspect.getsource(optimized.solve_warm.py_func)))
    f=tree.body[0];f.name='solve_scale_strategy';f.decorator_list=[]
    for name,default in [('scale_mode',0),('scale_inner_tol',1e-8),('scale_max_iter',2000),('prox_gamma',1.)]:
        f.args.args.append(ast.arg(arg=name));f.args.defaults.append(ast.Constant(default))
    outer=next(node for node in f.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='iteration')
    assert isinstance(outer.body[0],ast.If) and ast.unparse(outer.body[0].test)=='estimate_scale'
    outer.body[0].body=ast.parse('sigma, inner_iterations, inner_kkt, inner_gap, inner_ok = _scale_update(r, sigma, rho, a, scale_mode, scale_inner_tol, scale_max_iter, scale_mm_steps, prox_gamma)').body
    location=f.body.index(outer)
    f.body[location:location]=ast.parse('scale_iterations = 0\ninexact_scale_updates = 0').body
    outer.body[0].body.extend(ast.parse('scale_iterations += inner_iterations\ninexact_scale_updates += int(not inner_ok)').body)
    for node in ast.walk(f):
        if isinstance(node,ast.Return):
            assert isinstance(node.value,ast.Tuple)
            node.value.elts.extend([ast.Name(id='scale_iterations',ctx=ast.Load()),ast.Name(id='inexact_scale_updates',ctx=ast.Load())])
    tree=ast.fix_missing_locations(tree)
    namespace=dict(vars(optimized));namespace['_scale_update']=_scale_update
    exec(compile(tree,'<shared-CD-three-scale-options>','exec'),namespace)
    return njit(namespace['solve_scale_strategy'])

solve_scale_strategy=_strategy_kernel()
