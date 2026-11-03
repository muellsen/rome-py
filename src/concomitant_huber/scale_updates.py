"""Three descent-controlled updates for the same closed perspective scale block.

The prox option is Douglas--Rachford on observation-wise residual/scale pairs
with fixed-residual and common-scale projection; it is not one prox/projection.
All inner computations use r/max(abs(r)); no scale floor changes the problem.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

METHODS={'mm':0,'exact':1,'prox':2}


@njit
def _block_value(q,t,rho,a):
    value=a*t
    for i in range(len(q)):
        w=abs(q[i])
        if t==0:value+=rho*w/len(q)
        elif w<=rho*t:value+=.5*w*(w/t)/len(q)
        else:value+=(rho*w-.5*rho*rho*t)/len(q)
    return value


@njit
def _block_certificate(q,t,rho,a):
    sq=0.;product=0.
    for i in range(len(q)):
        w=abs(q[i]);z=0.
        if w>0:
            z=rho if t==0 or w>=rho*t else w/t
        sq+=z*z/len(q);product+=w*z/len(q)
    derivative=a-.5*sq
    residual=abs(derivative) if t>0 else max(-derivative,0.)
    shrink=min(1.,np.sqrt(2*a/sq)) if sq>0 else 1.
    primal=_block_value(q,t,rho,a)
    gap=max(0.,primal-shrink*product)/max(1.,abs(primal))
    return residual,gap,derivative


@njit
def _perspective_prox(u,s,gamma,rho,a):
    """prox_gamma [s*h_rho(u/s)+a*s], closed at s=0 (u first)."""
    x=u/gamma;t=s/gamma
    bounded=min(abs(x),rho)
    violation=t+.5*bounded*bounded-a
    if violation<=0:return u-gamma*np.sign(x)*bounded,0.
    lo=0.;hi=violation
    for _ in range(80):
        mu=(lo+hi)*.5
        v=min(abs(x)/(1+mu),rho)
        f=t-mu+.5*v*v-a
        if f>0:lo=mu
        else:hi=mu
        if hi-lo<=1e-14*max(1.,mu):break
    mu=(lo+hi)*.5
    v=np.sign(x)*min(abs(x)/(1+mu),rho)
    return u-gamma*v,gamma*mu


@njit
def _scale_update(r,sigma,rho,a,method,tol,max_iter,mm_steps,gamma):
    """Return scale, iterations, block residual, normalized dual gap, certified.

    Exact/prox use the same block certificate. Finite MM batches need only
    decrease the objective; outer convergence always needs a joint certificate.
    """
    magnitude=0.
    for i in range(len(r)):magnitude=max(magnitude,abs(r[i]))
    if magnitude==0:return 0.,0,0.,0.,True
    q=r/magnitude;t=sigma/magnitude;n=len(q)
    old=_block_value(q,t,rho,a)
    boundary_kkt,boundary_gap,_=_block_certificate(q,0.,rho,a)
    if boundary_kkt==0:return 0.,0,0.,boundary_gap,True
    residual,gap,_=_block_certificate(q,t,rho,a)
    if residual<=tol and gap<=tol:return sigma,0,residual,gap,True
    candidate=t;used=0
    if method==0:
        for k in range(mm_steps):
            squared=0.;limit=rho*candidate
            for i in range(n):squared+=min(q[i]*q[i],limit*limit)/n
            candidate=np.sqrt(squared/(2*a));used=k+1
            residual,gap,_=_block_certificate(q,candidate,rho,a)
            if residual<=tol and gap<=tol:break
    elif method==1:
        squared=0.
        for i in range(n):squared+=q[i]*q[i]
        lo=0.;hi=np.sqrt(squared/(2*n*a))
        # The previous scale tightens the bracket without changing the root.
        _,_,derivative=_block_certificate(q,t,rho,a)
        if derivative<0:lo=t
        else:hi=min(hi,t)
        for k in range(max_iter):
            candidate=(lo+hi)*.5;used=k+1
            residual,gap,derivative=_block_certificate(q,candidate,rho,a)
            if residual<=tol and gap<=tol:break
            if derivative<0:lo=candidate
            else:hi=candidate
            if hi==lo:break
    else:
        # Multiply the averaged block objective by n: each local term is
        # s_i*h_rho(u_i/s_i)+a*s_i. Its prox uses gamma (not gamma/n).
        zu=q.copy();zs=np.full(n,t);pu=np.empty(n);ps=np.empty(n)
        best=t;best_value=old
        for k in range(max_iter):
            mean_scale=0.;mean_reflection=0.
            for i in range(n):
                pu[i],ps[i]=_perspective_prox(zu[i],zs[i],gamma,rho,a)
                mean_scale+=ps[i]/n
                mean_reflection+=(2*ps[i]-zs[i])/n
            value=_block_value(q,mean_scale,rho,a)
            if value<best_value:best=mean_scale;best_value=value
            residual,gap,_=_block_certificate(q,mean_scale,rho,a)
            used=k+1
            if residual<=tol and gap<=tol and value<=old:
                best=mean_scale;break
            for i in range(n):
                zu[i]+=q[i]-pu[i]
                zs[i]+=mean_reflection-ps[i]
        candidate=best
    # A truncated inner solve is allowed only as a descent step. No inner
    # convergence flag substitutes for the full outer joint certificate.
    if _block_value(q,candidate,rho,a)>old:candidate=t
    residual,gap,_=_block_certificate(q,candidate,rho,a)
    return magnitude*candidate,used,residual,gap,residual<=tol and gap<=tol


@dataclass(frozen=True)
class ScaleUpdateResult:
    scale: float
    method: str
    iterations: int
    block_kkt: float
    relative_gap: float
    converged: bool


def update_scale(residuals,scale,*,method='mm',rho=1.345,a=.355082274,
                 tol=1e-8,max_iter=2000,mm_steps=8,prox_gamma=1.):
    """Apply MM, exact scalar root, or coordinate-prox consensus to fixed r.

    prox_gamma is the DR step size after dividing (r,scale) by max(abs(r))
    and multiplying the averaged block objective by n. The returned gap is
    certified for that normalized block. Fixed-residual scale certification
    does not certify the regression coefficients or a joint boundary solution.
    """
    r=np.asarray(residuals,float)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all():raise ValueError('Finite nonempty residual vector required')
    if method not in METHODS:raise ValueError('scale update must be mm, exact or prox')
    if not all(np.isfinite(v) for v in (scale,rho,a,tol,prox_gamma)) or scale<0 or min(rho,a,tol,prox_gamma)<=0:
        raise ValueError('Nonnegative scale and positive finite parameters required')
    for v in (max_iter,mm_steps):
        if not isinstance(v,(int,np.integer)) or v<1:raise ValueError('Positive integer inner budgets required')
    result=_scale_update(r,scale,rho,a,METHODS[method],tol,max_iter,mm_steps,prox_gamma)
    return ScaleUpdateResult(float(result[0]),method,int(result[1]),float(result[2]),float(result[3]),bool(result[4]))
