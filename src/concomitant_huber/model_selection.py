"""Training-only preprocessing and warm paths for reproducible cross-validation."""
from dataclasses import dataclass
import numpy as np
from .path import _solve_warm


def low_correlation_columns(X, keep=500):
    """Greedy minimum maximum absolute correlation; variance/index tie breaks."""
    X=np.asarray(X,float)
    if X.ndim!=2 or len(X)<2 or not np.isfinite(X).all() or keep<1:
        raise ValueError('Finite matrix with at least two rows and positive keep required')
    n,p=X.shape
    if keep>=p:return np.arange(p)
    centered=X-X.mean(0)
    norm=np.linalg.norm(centered,axis=0);norm[norm==0]=1
    unit=centered/norm
    corr=abs(unit.T@unit);np.fill_diagonal(corr,0.)
    variance=X.var(0,ddof=1)
    chosen=[int(np.lexsort((np.arange(p),-variance))[0])]
    maximum=corr[:,chosen[0]].copy();maximum[chosen[0]]=np.inf
    for _ in range(1,keep):
        j=int(np.lexsort((np.arange(p),-variance,maximum))[0])
        chosen.append(j);maximum=np.maximum(maximum,corr[:,j]);maximum[chosen]=np.inf
    return np.sort(chosen)


@dataclass
class TrainingTransform:
    columns: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def fit(cls,X,y,keep=None):
        X,y=np.asarray(X,float),np.asarray(y,float)
        if X.ndim!=2 or y.shape!=(len(X),) or len(y)<2 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Finite training design/response required')
        columns=np.arange(X.shape[1]) if keep is None else low_correlation_columns(X,keep)
        mean=X[:,columns].mean(0);scale=X[:,columns].std(0);scale[scale==0]=1
        ys=float(y.std())
        if ys==0:raise ValueError('Constant response')
        return cls(columns,mean,scale,float(y.mean()),ys)

    def design(self,X):
        return np.asfortranarray((np.asarray(X,float)[:,self.columns]-self.x_mean)/self.x_scale)

    def response(self,y):
        return (np.asarray(y,float)-self.y_mean)/self.y_scale

    def predict(self,X,coefficients):
        return self.y_mean+self.y_scale*(coefficients[0]+self.design(X)@coefficients[1:])


def standardized_path(X,y,penalties,*,rho=1.345,a=.355082274,estimate_scale=True,
                      initial_scale=1.,tol=1e-6,max_iter=4000,backend="reference"):
    """Warm-start lasso path with an unpenalized intercept; no preprocessing."""
    X=np.asarray(X,float);y=np.asarray(y,float);penalties=np.asarray(penalties,float)
    if (X.ndim!=2 or y.shape!=(len(X),) or not len(y) or
        not np.isfinite(X).all() or not np.isfinite(y).all() or
        penalties.ndim!=1 or not len(penalties) or not np.isfinite(penalties).all() or
        np.any(penalties<=0) or np.any(np.diff(penalties)>0)):
        raise ValueError('Finite design and descending positive penalties required')
    if min(rho,a,initial_scale,tol)<=0 or not all(np.isfinite(v) for v in (rho,a,initial_scale,tol)):
        raise ValueError('Positive finite constants required')
    if backend not in ("reference","optimized","polished"):raise ValueError("Unknown solver backend")
    from .optimized import solve_warm as optimized_solve
    kernel=_solve_warm if backend=="reference" else optimized_solve
    design=np.asfortranarray(np.column_stack([np.ones(len(y)),X]))
    beta=np.zeros(design.shape[1]);sigma=initial_scale;L=len(penalties)
    coefficients=np.full((L,len(beta)),np.nan);scales=np.full(L,np.nan)
    boundary=np.zeros(L,dtype=bool);boundary_gaps=np.full(L,np.nan)
    violation=np.full((L,2),np.nan);status=['not_attempted']*L
    for k,lam in enumerate(penalties):
        args=(design,y,lam,rho,a,sigma,estimate_scale,tol,max_iter,True,8,beta)
        if sigma==0 and backend=="polished":
            ok=False;cb=np.inf;cs=np.inf
        else:
            result=kernel(*args,newton_every=5) if backend=="polished" else kernel(*args)
            beta,sigma,it,history,ok,cb,cs,_=result[:8]
        if not ok and estimate_scale and backend=="polished":
            from .boundary import boundary_solution
            certificate=boundary_solution(design,y,lam,rho=rho,a=a,intercept=True,tol=tol)
            if certificate['certified']:
                beta=certificate['coefficients'];sigma=0.;ok=True
                cb=certificate['coefficient_kkt'];cs=certificate['scale_kkt']
                boundary[k]=True;boundary_gaps[k]=certificate['relative_gap']
        coefficients[k]=beta;scales[k]=sigma;violation[k]=[cb,cs]
        status[k]='converged' if ok else ('boundary_uncertified' if sigma==0 else 'max_iter')
        if not ok and backend!="polished":break
    return dict(coefficients=coefficients,scales=scales,status=status,kkt=violation,boundary=boundary,boundary_gaps=boundary_gaps)


def null_penalty_anchor(X,y,*,rho=1.345,a=.355082274,estimate_scale=True,fixed_scale=1.):
    """Training-fold null intercept/scale and exact positive-scale KKT anchor.

    The canonical joint penalty is held fixed during fitting. Boundary null
    optima raise an explicit error; no positive floor is silently imposed.
    """
    from scipy.optimize import brentq
    X=np.asarray(X,float);y=np.asarray(y,float)
    if (X.ndim!=2 or y.shape!=(len(X),) or len(y)<2 or
        not np.isfinite(X).all() or not np.isfinite(y).all() or
        not all(np.isfinite(v) and v>0 for v in (rho,a,fixed_scale)) or np.ptp(y)==0):
        raise ValueError('Finite nonconstant training response and positive constants required')
    def location(sigma):
        return brentq(lambda b:np.mean(np.sign(y-b)*np.minimum(abs(y-b)/sigma,rho)),
                      float(y.min()),float(y.max()),xtol=1e-13)
    if estimate_scale:
        low=float(y.std())*1e-12
        def scale_gradient(sigma):
            b=location(sigma);score=np.sign(y-b)*np.minimum(abs(y-b)/sigma,rho)
            return a-.5*np.mean(score**2)
        if scale_gradient(low)>=0:
            raise ValueError('Null optimum is at or numerically near zero scale; boundary certificate required')
        high=max(float(y.std()),fixed_scale)
        while scale_gradient(high)<0:high*=2
        sigma=brentq(scale_gradient,low,high,xtol=1e-12)
    else:sigma=fixed_scale
    b=location(sigma);score=np.sign(y-b)*np.minimum(abs(y-b)/sigma,rho)
    return dict(intercept=float(b),scale=float(sigma),
                lambda_max=float(np.max(abs(X.T@score))/len(y)),
                intercept_kkt=float(abs(score.mean())),scale_kkt=float(abs(a-.5*np.mean(score**2))))
