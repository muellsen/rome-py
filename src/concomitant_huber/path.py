"""Warm-start paths using the production exact-CD/MM update equations.

No sequential strong rule is assumed valid when scale changes.
"""
import ast
import inspect
import textwrap
from dataclasses import dataclass
import numpy as np
from numba import njit
from . import solver


def _warm_kernel():
    tree=ast.parse(textwrap.dedent(inspect.getsource(solver._solve.py_func)))
    f=tree.body[0];f.name='_solve_warm';f.decorator_list=[]
    f.args.args.append(ast.arg(arg='beta_start'))
    class Initialize(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id=='beta':node.value=ast.parse('beta_start.copy()',mode='eval').body
                if node.targets[0].id=='r':node.value=ast.parse('y-X@beta',mode='eval').body
            return node
    tree=ast.fix_missing_locations(Initialize().visit(tree))
    ns=dict(vars(solver));exec(compile(tree,'<production-warm-path>','exec'),ns)
    return njit(ns['_solve_warm'])

_solve_warm=_warm_kernel()


@dataclass
class PathResult:
    coefficients: np.ndarray
    scales: np.ndarray
    iterations: np.ndarray
    status: list
    coefficient_kkt: np.ndarray
    scale_kkt: np.ndarray
    objectives: np.ndarray


def huber_path(X,y,penalties,*,rho=1.345,scale_penalty=.5,initial_scale=1.,
               estimate_scale=True,tol=1e-7,max_iter=3000,scale_mm_steps=8):
    """Descending dense lasso path; return every attempted point and explicit tail status.

    Penalties are absolute coefficients of the averaged perspective objective.
    A failed/boundary point stops the path; unattempted values are NaN, never
    silently filled by a previous solution. No intercept or preprocessing.
    """
    X=np.asfortranarray(X,dtype=float);y=np.asarray(y,dtype=float)
    penalties=np.asarray(penalties,dtype=float)
    if X.ndim!=2 or y.ndim!=1 or len(y)!=len(X) or len(y)==0:raise ValueError('Invalid design')
    if penalties.ndim!=1 or not len(penalties) or not np.isfinite(penalties).all() or min(penalties)<0 or np.any(np.diff(penalties)>0):raise ValueError('Penalties must be finite, nonnegative and descending')
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite data')
    if not all(np.isfinite(v) and v>0 for v in (rho,scale_penalty,initial_scale,tol)):raise ValueError('Positive finite constants required')
    if not isinstance(max_iter,int) or max_iter<1 or not isinstance(scale_mm_steps,int) or scale_mm_steps<1:raise ValueError('Positive iteration budgets required')
    L,p=len(penalties),X.shape[1]
    coefficients=np.full((L,p),np.nan);scales=np.full(L,np.nan);iterations=np.zeros(L,dtype=int)
    vb=np.full(L,np.nan);vs=vb.copy();objectives=vb.copy();status=['not_attempted']*L
    beta=np.zeros(p);sigma=initial_scale
    for k,lam in enumerate(penalties):
        result=_solve_warm(X,y,lam,rho,scale_penalty,sigma,estimate_scale,tol,max_iter,False,scale_mm_steps,beta)
        beta,sigma,iteration,history,ok,cb,cs,_=result
        coefficients[k]=beta;scales[k]=sigma;iterations[k]=iteration;vb[k]=cb;vs[k]=cs;objectives[k]=history[-1]
        status[k]='converged' if ok else ('boundary_uncertified' if sigma==0 else 'max_iter')
        if not ok:break
    return PathResult(coefficients,scales,iterations,status,vb,vs,objectives)
