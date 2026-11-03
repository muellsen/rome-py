import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from concomitant_huber import HuberCD
from concomitant_huber.diagnostics import fit_trace

class TraceTests(unittest.TestCase):
    def test_trace_preserves_kernel_and_records_every_sweep(self):
        rng = np.random.default_rng(71)
        X = rng.normal(size=(40, 12)); y = X[:, 0]+rng.normal(size=40)
        for intercept in (True, False):
            for estimate in (True, False):
                m = HuberCD(penalty=.2, fit_intercept=intercept, estimate_scale=estimate)
                trace = fit_trace(m, X, y)
                self.assertEqual(len(trace['sigma']), m.n_iter_+1)
                self.assertTrue(np.all(np.diff(trace['objective']) <= 1e-12))
                self.assertTrue(np.all(trace['beta'][0] == 0))
                self.assertEqual(trace['sigma'][-1], m.scale_)

    def test_response_units_and_amp_objective_conversion(self):
        from concomitant_huber.solver import _objective
        from scipy.special import huber
        rng = np.random.default_rng(82)
        n,p=19,7
        X=rng.normal(size=(n,p)); y=rng.normal(size=n); b=rng.normal(size=p)
        sigma,rho,a,lam=.8,1.345,.355,.12
        original=_objective(y-X@b,b,sigma,rho,a,lam)
        # AMP uses column-L2 design and a sum, with rescaled coefficients.
        design=X/np.sqrt(n); B=np.sqrt(n)*b; penalty=np.sqrt(n)*lam
        summed=np.sum(sigma*huber(rho,(y-design@B)/sigma))+n*a*sigma+penalty*np.sum(abs(B))
        self.assertAlmostEqual(summed/n,original,places=12)
        self.assertAlmostEqual(_objective(3*(y-X@b),3*b,3*sigma,rho,a,lam),3*original,places=12)

    def test_warm_trace_preserves_nonzero_initial_state_and_objective(self):
        from concomitant_huber.path_diagnostics import trace_warm_point
        from concomitant_huber.solver import _objective
        rng = np.random.default_rng(118)
        X = rng.normal(size=(35, 9))
        y = X[:, 0]+rng.normal(size=35)
        beta = rng.normal(size=9)*.1
        for estimate in (True, False):
            trace = trace_warm_point(X, y, beta, 1.2, .15,
                                     estimate_scale=estimate, max_iter=300)
            np.testing.assert_array_equal(trace['beta'][0], beta)
            self.assertAlmostEqual(trace['objective'][0],
                                   _objective(y-X@beta, beta, 1.2, 1.345, .5, .15))
            self.assertTrue(np.all(np.diff(trace['objective'])<=1e-12))
            self.assertTrue(trace['converged'])
        np.testing.assert_array_equal(trace['sigma'], np.full(len(trace['sigma']), 1.2))
