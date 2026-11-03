"""Exact Huber CD/MM with local affine roots and reusable working buffers.

The reference solver is preserved in solver.py. No changed objective, scale
floor, approximate coordinate update, or permanent screening is introduced.
"""
import numpy as np
from numba import njit
from .solver import _coordinate, _objective


@njit
def _local_coordinate(r, x, old, delta, penalty, scratch):
    """Exact affine root when it stays in the current Huber cell; else kink sweep."""
    n=len(r);g=0.;curvature=0.
    for i in range(n):
        g-=x[i]*np.sign(r[i])*min(abs(r[i]),delta)/n
        if abs(r[i])<delta:curvature+=x[i]*x[i]/n
    if curvature>0:
        raw=old-g/curvature
        candidate=np.sign(raw)*max(abs(raw)-penalty/curvature,0.)
        valid=True
        for i in range(n):
            new=r[i]-x[i]*(candidate-old)
            if abs(r[i])<delta:
                if abs(new)>delta:valid=False;break
            elif r[i]>=delta:
                if new<delta:valid=False;break
            elif new>-delta:valid=False;break
        if valid:return candidate,True
    for i in range(n):scratch[i]=r[i]+x[i]*old
    return _coordinate(scratch,x,delta,penalty),False


@njit
def _certificate(X,r,beta,sigma,rho,a,penalty,intercept,score,active,tol):
    """One score evaluation shared by every coefficient and the scale check."""
    if sigma==0:return np.inf,np.inf
    n,p=X.shape;squared=0.
    for i in range(n):
        score[i]=np.sign(r[i])*min(abs(r[i])/sigma,rho)
        squared+=score[i]*score[i]/n
    violation=0.
    for j in range(p):
        g=0.
        for i in range(n):g-=X[i,j]*score[i]/n
        lam=0. if intercept and j==0 else penalty
        if beta[j]!=0:v=abs(g+lam*np.sign(beta[j]))
        else:v=max(abs(g)-lam,0.)
        if v>tol:active[j]=True
        violation=max(violation,v)
    return violation,abs(a-.5*squared)


@njit
def solve_warm(X,y,penalty,rho,a,sigma,estimate_scale,tol,max_iter,intercept,scale_mm_steps,beta_start,check_every=5,newton_every=0):
    n,p=X.shape;beta=beta_start.copy();r=y-X@beta
    history=np.empty(max_iter+1);history[0]=_objective(r,beta,sigma,rho,a,penalty,intercept)
    active=np.ones(p,dtype=np.bool_);scratch=np.empty(n);score=np.empty(n)
    vb=np.inf;vs=np.inf;fast=0;fallback=0
    for iteration in range(max_iter):
        if estimate_scale:
            for step in range(scale_mm_steps):
                squared=0.;delta=rho*sigma
                for i in range(n):squared+=min(r[i]*r[i],delta*delta)/n
                sigma=np.sqrt(squared/(2*a))
                if sigma==0:break
                g=a
                for i in range(n):g-=.5*min((r[i]/sigma)**2,rho*rho)/n
                if abs(g)<=tol*.5:break
        if sigma==0:
            history[iteration+1]=_objective(r,beta,sigma,rho,a,penalty,intercept)
            return beta,sigma,iteration+1,history[:iteration+2],False,np.inf,np.inf,iteration,fast,fallback
        if estimate_scale and sigma<1e-10*np.std(y):
            history[iteration+1]=_objective(r,beta,sigma,rho,a,penalty,intercept)
            return beta,sigma,iteration+1,history[:iteration+2],False,np.inf,np.inf,iteration+1,fast,fallback
        delta=rho*sigma;change=0.
        for j in range(p):
            if not active[j]:continue
            lam=0. if intercept and j==0 else sigma*penalty
            new,local=_local_coordinate(r,X[:,j],beta[j],delta,lam,scratch)
            if local:fast+=1
            else:fallback+=1
            difference=new-beta[j];change=max(change,abs(difference))
            if difference!=0:
                for i in range(n):r[i]-=X[i,j]*difference
            beta[j]=new
            active[j]=(new!=0) or (intercept and j==0)
        if newton_every>0 and (iteration+1)%newton_every==0:
            sigma,accepted=_polish(X,y,r,beta,sigma,rho,a,penalty,intercept,estimate_scale)
        history[iteration+1]=_objective(r,beta,sigma,rho,a,penalty,intercept)
        if iteration%check_every==0 or change<=tol or iteration==max_iter-1:
            vb,vs=_certificate(X,r,beta,sigma,rho,a,penalty,intercept,score,active,tol)
            if vb<=tol and (not estimate_scale or vs<=tol):
                return beta,sigma,iteration+1,history[:iteration+2],True,vb,vs,iteration+1,fast,fallback
    return beta,sigma,max_iter,history,False,vb,vs,max_iter,fast,fallback


@njit
def _polish(X,y,r,beta,sigma,rho,a,penalty,intercept,estimate_scale):
    """Safeguarded Newton step on the current nonzero support (optional).

    Acceptance uses the original nonsmooth perspective objective. The ridge
    regularizes only the search direction, never the statistical objective.
    A full KKT certificate is still required after acceptance.
    """
    n,p=X.shape
    support=np.empty(p,dtype=np.int64);k=0
    for j in range(p):
        if beta[j]!=0 or (intercept and j==0):support[k]=j;k+=1
    if k==0 or k>128:return sigma,False
    dim=k+int(estimate_scale);H=np.zeros((dim,dim));g=np.zeros(dim)
    for i in range(n):
        score=np.sign(r[i])*min(abs(r[i])/sigma,rho)
        for j in range(k):g[j]-=X[i,support[j]]*score/n
        if abs(r[i])<rho*sigma:
            for j in range(k):
                for l in range(k):H[j,l]+=X[i,support[j]]*X[i,support[l]]/(n*sigma)
                if estimate_scale:
                    H[j,k]+=X[i,support[j]]*r[i]/(n*sigma*sigma)
                    H[k,j]=H[j,k]
            if estimate_scale:H[k,k]+=r[i]*r[i]/(n*sigma**3)
        if estimate_scale:g[k]-=.5*score*score/n
    for j in range(k):
        if not(intercept and support[j]==0):g[j]+=penalty*np.sign(beta[support[j]])
    if estimate_scale:g[k]+=a
    largest=0.
    for j in range(dim):largest=max(largest,H[j,j])
    if largest==0:return sigma,False
    for j in range(dim):H[j,j]+=largest*1e-10
    direction=np.linalg.solve(H,-g)
    if not np.isfinite(direction).all() or np.dot(g,direction)>=0:return sigma,False
    old=_objective(r,beta,sigma,rho,a,penalty,intercept)
    trial=beta.copy();rr=np.empty(n);step=1.
    for attempt in range(24):
        ss=sigma+step*direction[k] if estimate_scale else sigma
        if ss>0:
            for j in range(k):trial[support[j]]=beta[support[j]]+step*direction[j]
            for i in range(n):
                rr[i]=r[i]
                for j in range(k):rr[i]-=step*X[i,support[j]]*direction[j]
            new=_objective(rr,trial,ss,rho,a,penalty,intercept)
            if new<old and new<=old+1e-4*step*np.dot(g,direction):
                beta[:]=trial;r[:]=rr
                return ss,True
        step*=.5
    return sigma,False
