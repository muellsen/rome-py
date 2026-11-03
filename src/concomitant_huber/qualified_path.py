"""Full warm paths with explicit certified and approximate returns.

The optimized kernels are unchanged. A boundary LAD-lasso candidate is a
candidate for the original closed perspective, never automatically certified.
"""
import time
import numpy as np
from .optimized import solve_warm
from .scale_solver import solve_scale_strategy
from .boundary import boundary_solution
from .solver import _objective,_diagnostics


def accuracy(X,y,beta,sigma,penalty,*,rho,a,estimate_scale,intercept=False):
    """Original primal and a feasible dual lower bound, independent of the loop."""
    residual=y-X@beta;n=len(y)
    if sigma>0:
        v=np.sign(residual)*np.minimum(abs(residual)/sigma,rho)
        cb,cs=_diagnostics(X,residual,beta,sigma,rho,a,penalty,intercept)
    else:
        v=rho*np.sign(residual);cb=cs=np.inf
    if intercept:v-=v.mean()
    weights=np.full(X.shape[1],penalty)
    if intercept:weights[0]=0.
    shrink=1.
    box=float(max(abs(v)))
    if box>0:shrink=min(shrink,rho/box)
    if estimate_scale:
        square=float(np.mean(v*v))
        if square>0:shrink=min(shrink,np.sqrt(2*a/square))
    corr=abs(X.T@v/n)
    for j in range(int(intercept),len(weights)):
        if corr[j]>0:shrink=min(shrink,weights[j]/corr[j])
    v*=shrink
    primal=float(_objective(residual,beta,sigma,rho,a,penalty,intercept))
    dual=float(y@v/n)
    if not estimate_scale:dual+=a*sigma-.5*sigma*np.mean(v*v)
    return dict(objective=primal,dual_lower_bound=dual,
        normalized_gap=max(0.,primal-dual)/max(1.,abs(primal)),
        coefficient_kkt=float(cb),scale_kkt=float(cs),dual_vector=v)


def qualified_path(X,y,penalties,*,rho,a,estimate_scale=True,initial_scale=1.,
                   tol=1e-6,practical_gap_tol=1e-4,max_iter=4000):
    """Visit every penalty, returning finite candidates and explicit accuracy.

    Strict positive-scale certification uses the existing joint KKT rule;
    boundary certification uses the LP perspective dual. Uncertified candidates
    remain available and warm-start the next penalty. No iteration cap is
    relabeled convergence. Accuracy auditing after the timed solve is excluded
    from prepared CPU, while candidate selection and LP calls are included.
    """
    X=np.asfortranarray(X,dtype=float);y=np.asarray(y,dtype=float)
    penalties=np.asarray(penalties,dtype=float)
    if (X.ndim!=2 or y.shape!=(len(X),) or not len(y) or
        not np.isfinite(X).all() or not np.isfinite(y).all() or
        penalties.ndim!=1 or not len(penalties) or not np.isfinite(penalties).all() or
        np.any(penalties<=0) or min(rho,a,initial_scale,tol,practical_gap_tol)<=0 or max_iter<1):
        raise ValueError('Finite data, nonempty positive penalties and positive solver settings required')
    L=len(penalties);n,p=X.shape;beta=np.zeros(p);sigma=float(initial_scale)
    coefficients=np.empty((L,p));scales=np.empty(L);sweeps=np.zeros(L,dtype=int)
    lp_duals=np.full((L,n),np.nan);strict=np.zeros(L,dtype=bool)
    statuses=[];lp_calls=0;lp_cpu=0.;lp_wall=0.
    cpu=time.process_time();wall=time.perf_counter()
    for k,lam in enumerate(penalties):
        cert=None;it=0;ok=False
        if estimate_scale and sigma==0:
            lc=time.process_time();lw=time.perf_counter();lp_calls+=1
            cert=boundary_solution(X,y,lam,rho=rho,a=a,intercept=False,tol=tol)
            lp_cpu+=time.process_time()-lc;lp_wall+=time.perf_counter()-lw
            if cert['certified']:
                beta=cert['coefficients'];sigma=0.;ok=True
        if not ok:
            # An uncertified zero-scale warm start does not permanently force
            # all later penalties onto the boundary.
            seed=sigma if sigma>0 else float(initial_scale)
            if estimate_scale:
                rr=solve_scale_strategy(X,y,lam,rho,a,seed,True,tol,max_iter,False,8,beta,
                                        5,5,0,tol*.5,2000,1.)
            else:
                rr=solve_warm(X,y,lam,rho,a,initial_scale,False,tol,max_iter,False,8,beta,5,5)
            beta,sigma,it=rr[:3];ok=bool(rr[4])
        if not ok and estimate_scale:
            if cert is None:
                lc=time.process_time();lw=time.perf_counter();lp_calls+=1
                cert=boundary_solution(X,y,lam,rho=rho,a=a,intercept=False,tol=tol)
                lp_cpu+=time.process_time()-lc;lp_wall+=time.perf_counter()-lw
            if 'coefficients' in cert:
                current=_objective(y-X@beta,beta,sigma,rho,a,lam,False)
                if cert['objective']<=current:
                    beta=cert['coefficients'];sigma=0.;ok=bool(cert['certified'])
        if sigma==0 and cert is not None and 'dual_vector' in cert:
            lp_duals[k]=cert['dual_vector']
        strict[k]=ok;coefficients[k]=beta;scales[k]=sigma;sweeps[k]=it
        statuses.append(('boundary_certified' if sigma==0 else 'converged') if ok else
                        ('approximate_lad' if sigma==0 else 'approximate_iterate'))
    info=dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
        returned_points=L,strict_points=int(strict.sum()),strict_complete=bool(strict.all()),
        statuses=statuses,total_sweeps=int(sweeps.sum()),boundary_lp_calls=lp_calls,
        boundary_lp_cpu_seconds=lp_cpu,boundary_lp_wall_seconds=lp_wall)
    checks=[];duals=np.empty((L,n))
    for k,lam in enumerate(penalties):
        q=accuracy(X,y,coefficients[k],scales[k],lam,rho=rho,a=a,estimate_scale=estimate_scale)
        if scales[k]==0 and np.isfinite(lp_duals[k]).all():
            q['dual_vector']=lp_duals[k]
            q['dual_lower_bound']=float(y@lp_duals[k]/n)
            q['normalized_gap']=max(0.,q['objective']-q['dual_lower_bound'])/max(1.,abs(q['objective']))
        duals[k]=q.pop('dual_vector');checks.append(q)
    gaps=np.array([q['normalized_gap'] for q in checks]);practical=gaps<=practical_gap_tol
    info.update(practical_gap_points=int(practical.sum()),practical_gap_complete=bool(practical.all()),
        maximum_normalized_gap=float(max(gaps)),approximate_points=int((~strict).sum()))
    arrays=dict(coefficients=coefficients,scales=scales,iterations=sweeps,penalties=penalties,
        strict=strict,practical_gap=practical,dual_vectors=duals,
        **{name:np.array([q[name] for q in checks]) for name in checks[0]})
    return arrays,info
