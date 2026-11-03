import sys
import unittest
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import huber

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from concomitant_huber import HuberCD
from concomitant_huber.solver import _coordinate


class SolverTests(unittest.TestCase):
    def test_exact_coordinate_against_independent_scalar_optimizer(self):
        rng = np.random.default_rng(71)
        for n in (3, 20, 100):
            for penalty in (0., 0.03, 1.):
                x, u = rng.normal(size=(2, n))
                x[0] = 0
                delta = .7
                fun = lambda b: np.mean(huber(delta, u-x*b)) + penalty*abs(b)
                b = _coordinate(u, x, delta, penalty)
                ref = minimize_scalar(fun, method='brent', options={'xtol': 1e-12})
                self.assertLessEqual(fun(b), ref.fun + 1e-10)

    def test_mm_surrogate_is_global_and_tangent(self):
        rng = np.random.default_rng(7)
        r = rng.standard_t(3, 100)
        old, rho, a = 1.7, 1.345, .4
        o = np.sign(r)*np.maximum(abs(r)-rho*old, 0)
        q = lambda s: np.mean((r-o)**2)/(2*s) + a*s + rho*np.mean(abs(o))
        f = lambda s: np.mean(s*huber(rho, r/s)) + a*s
        self.assertAlmostEqual(q(old), f(old), places=12)
        for s in np.geomspace(.01, 100, 40):
            self.assertGreaterEqual(q(s)+1e-12, f(s))
        new = np.sqrt(np.mean((r-o)**2)/(2*a))
        self.assertLessEqual(f(new), f(old)+1e-12)

    def test_joint_solution_against_independent_bound_optimizer(self):
        rng = np.random.default_rng(95)
        X = rng.normal(size=(70, 5))
        y = 1.2+X @ np.array([2., -1., 0, 0, 0]) + rng.standard_t(3,70)
        lam, rho, a = .08, 1.345, .3
        model = HuberCD(penalty=lam, scale_penalty=a, tol=1e-8).fit(X,y)
        A = np.column_stack((np.ones(len(y)),X)); p=A.shape[1]
        weights = np.r_[0., np.full(p-1,lam)]
        def fun(z):
            b = z[:p]-z[p:2*p]; s=z[-1]; r=y-A@b; psi=np.clip(r/s,-rho,rho)
            value = np.mean(s*huber(rho,r/s))+a*s+weights@(z[:p]+z[p:2*p])
            grad = -A.T@psi/len(y)
            return value, np.r_[grad+weights,-grad+weights,a-.5*np.mean(psi**2)]
        z = np.r_[np.zeros(2*p),np.std(y)]
        ref = minimize(fun,z,jac=True,bounds=[(0,None)]*(2*p)+[(1e-9,None)],method='L-BFGS-B', options={'ftol':1e-14,'gtol':1e-10,'maxiter':5000})
        self.assertTrue(ref.success,ref.message)
        self.assertTrue(model.converged_)
        self.assertAlmostEqual(model.objective_,ref.fun,places=8)
        self.assertLess(np.max(np.diff(model.objective_history_)),1e-10)
        self.assertLess(model.coef_kkt_,1e-8)
        self.assertLess(model.scale_kkt_,1e-8)
        mm1 = HuberCD(penalty=lam,scale_penalty=a,tol=1e-8,scale_mm_steps=1).fit(X,y)
        self.assertTrue(mm1.converged_)
        self.assertAlmostEqual(mm1.objective_,ref.fun,places=8)
        fixed = HuberCD(penalty=lam,scale_penalty=a,estimate_scale=False, initial_scale=model.scale_,tol=1e-8).fit(X,y)
        np.testing.assert_allclose(model.predict(X),fixed.predict(X),atol=3e-7)

    def test_degenerate_columns_and_uncertified_boundary(self):
        X=np.zeros((10,2)); y=np.zeros(10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=HuberCD().fit(X,y)
        self.assertEqual(model.scale_,0)
        self.assertEqual(model.status_,'boundary_uncertified')
        self.assertFalse(model.converged_)
        self.assertTrue(np.isfinite(model.objective_))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            HuberCD(initial_scale=0).fit(np.ones((5,2)),np.ones(5))
        with self.assertRaises(ValueError):
            HuberCD().fit(np.ones((5,2)),np.ones(4))
