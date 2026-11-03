"""Public optimized estimator; the original HuberCD remains the reference."""
from dataclasses import dataclass
import warnings
import numpy as np
from .solver import HuberCD
from .scale_solver import solve_scale_strategy
from .scale_updates import METHODS
from .boundary import boundary_solution


@dataclass
class FastHuberCD(HuberCD):
    """Exact coefficient CD with mm/exact/prox scale updates and optional polishing.

    newton_every=0 disables support Newton polishing. A zero-scale result is
    accepted only with a feasible perspective dual certificate. The reference
    HuberCD objective and all parameter normalizations are unchanged.
    """
    check_every: int = 5
    newton_every: int = 5
    certify_boundary: bool = True
    scale_update: str = 'mm'
    scale_inner_tol: float | None = None
    scale_max_iter: int = 2000
    prox_gamma: float = 1.

    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        if (X.ndim!=2 or y.shape!=(len(X),) or not len(y) or
            not np.isfinite(X).all() or not np.isfinite(y).all()):
            raise ValueError('Expected finite matching design and response')
        if (not all(np.isfinite(v) for v in (self.penalty,self.rho,self.scale_penalty,self.tol)) or
            self.penalty<0 or min(self.rho,self.scale_penalty,self.tol)<=0):
            raise ValueError('Nonnegative penalty and positive finite constants required')
        for name in ('max_iter','scale_mm_steps','check_every','newton_every','scale_max_iter'):
            value=getattr(self,name)
            if not isinstance(value,(int,np.integer)) or value<(0 if name=='newton_every' else 1):
                raise ValueError('Invalid iteration/check budget: '+name)
        if self.scale_update not in METHODS:raise ValueError('scale_update must be mm, exact or prox')
        inner_tol=self.tol*.5 if self.scale_inner_tol is None else self.scale_inner_tol
        if not all(np.isfinite(v) and v>0 for v in (inner_tol,self.prox_gamma)):
            raise ValueError('Positive finite inner tolerance and prox_gamma required')
        self.n_features_in_=X.shape[1]
        D=np.asfortranarray(np.column_stack([np.ones(len(y)),X]) if self.fit_intercept else X)
        sigma=self.initial_scale
        if sigma is None:
            sigma=max(float(np.median(abs(y-np.median(y)))/.67448975),float(y.std())*1e-3,np.finfo(float).eps)
        if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive finite initial_scale required')
        result=solve_scale_strategy(D,y,self.penalty,self.rho,self.scale_penalty,sigma,self.estimate_scale,
                          self.tol,self.max_iter,self.fit_intercept,self.scale_mm_steps,
                          np.zeros(D.shape[1]),self.check_every,self.newton_every,
                          METHODS[self.scale_update],inner_tol,self.scale_max_iter,self.prox_gamma)
        b,self.scale_,self.n_iter_,history,ok,cb,cs,self.n_sweeps_,self.local_roots_,self.kink_sweeps_=result[:10]
        self.scale_inner_iterations_,self.inexact_scale_updates_=result[10:12]
        self.boundary_certificate_=None
        if not ok and self.estimate_scale and self.certify_boundary and self.penalty>0:
            certificate=boundary_solution(D,y,self.penalty,rho=self.rho,a=self.scale_penalty,
                                          intercept=self.fit_intercept,tol=self.tol)
            self.boundary_certificate_=certificate
            if certificate['certified']:
                b=certificate['coefficients'];self.scale_=0.;ok=True
                cb=certificate['coefficient_kkt'];cs=certificate['scale_kkt']
                history=np.r_[history,certificate['objective']]
        self.converged_=ok;self.coef_kkt_=cb;self.scale_kkt_=cs
        self.intercept_=float(b[0]) if self.fit_intercept else 0.
        self.coef_=b[1:].copy() if self.fit_intercept else b.copy()
        self.objective_history_=history;self.objective_=float(history[-1])
        self.status_=('boundary_certified' if self.scale_==0 else 'converged') if ok else ('boundary_uncertified' if self.scale_==0 else 'max_iter')
        if not ok:warnings.warn('FastHuberCD ended without an optimality certificate',RuntimeWarning)
        return self
