import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber import HuberCD
from concomitant_huber.path import huber_path

class PathTests(unittest.TestCase):
    def test_warm_path_matches_independent_cold_solutions(self):
        rng=np.random.default_rng(301)
        X=rng.normal(size=(70,12));y=X[:,0]+rng.normal(size=70)
        penalties=np.array([.5,.3,.2])
        for estimate in (True,False):
            path=huber_path(X,y,penalties,estimate_scale=estimate,tol=1e-9)
            self.assertEqual(path.status,['converged']*3)
            for k,lam in enumerate(penalties):
                cold=HuberCD(penalty=lam,estimate_scale=estimate,initial_scale=1.,fit_intercept=False,tol=1e-9).fit(X,y)
                np.testing.assert_allclose(X@path.coefficients[k],cold.predict(X),atol=2e-7)
                self.assertAlmostEqual(path.scales[k],cold.scale_,places=7)
    def test_failed_path_does_not_impute_its_tail(self):
        rng=np.random.default_rng(3)
        path=huber_path(rng.normal(size=(30,50)),rng.normal(size=30),[.1,.05,.01],max_iter=1)
        self.assertEqual(path.status[1:],['not_attempted']*2)
        self.assertTrue(np.isnan(path.coefficients[1:]).all())
