"""Certified zero-scale solutions of the closed Huber perspective."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix,hstack,eye


def boundary_solution(X,y,penalty,*,rho,a,intercept=True,tol=1e-6):
    """Solve the boundary LAD-lasso LP and certify the perspective dual ball.

    X includes an intercept column when intercept=True. The LP dual must
    additionally satisfy ||v||_2² <= 2*n*a to certify the joint scale problem.
    A failed certificate is returned explicitly; it is never a scale floor.
    """
    X=np.asarray(X,float);y=np.asarray(y,float);n,p=X.shape
    if y.shape!=(n,) or not np.isfinite(X).all() or not np.isfinite(y).all() or min(penalty,rho,a,tol)<=0:
        raise ValueError('Finite matching data and positive parameters required')
    weights=np.full(p,penalty)
    if intercept:weights[0]=0.
    identity=eye(n,format='csc');design=csc_matrix(X)
    equality=hstack([design,-design,identity,-identity],format='csc')
    objective=np.r_[weights,weights,np.full(2*n,rho/n)]
    lp=linprog(objective,A_eq=equality,b_eq=y,bounds=(0,None),method='highs',
               options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not lp.success:return dict(certified=False,status='boundary_lp_failed',message=lp.message)
    beta=lp.x[:p]-lp.x[p:2*p];r=y-X@beta;v=n*lp.eqlin.marginals
    # Construct a feasible perspective dual point before evaluating the gap.
    if intercept:v-=v.mean()
    shrink=1.
    if max(abs(v))>0:shrink=min(shrink,rho/max(abs(v)))
    if np.linalg.norm(v)>0:shrink=min(shrink,np.sqrt(2*n*a)/np.linalg.norm(v))
    correlations=abs(X.T@v/n)
    for j in range(p):
        if weights[j]>0 and correlations[j]>0:shrink=min(shrink,weights[j]/correlations[j])
    v*=shrink
    primal=float(rho*np.mean(abs(r))+weights@abs(beta));dual=float(y@v/n)
    slope=np.max(np.maximum(abs(X.T@v/n)-weights,0.))
    box=max(0.,float(max(abs(v))-rho));ball=max(0.,float(np.mean(v*v)-2*a))
    relative_gap=abs(primal-dual)/max(1.,abs(primal))
    certified=bool(max(slope,box,ball,relative_gap)<=tol)
    return dict(certified=certified,status='boundary_certified' if certified else 'boundary_gap_failed',
                coefficients=beta,scale=0.,objective=primal,dual=dual,dual_vector=v,
                coefficient_kkt=float(slope),scale_kkt=float(.5*ball),relative_gap=relative_gap,
                box_violation=box,dual_ball_violation=ball)
