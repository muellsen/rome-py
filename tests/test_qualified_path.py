"""Qualified paths retain useful returns without manufacturing convergence."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber.qualified_path import qualified_path,accuracy


class QualifiedPathTests(unittest.TestCase):
    def test_tiny_budget_still_visits_every_penalty_and_marks_approximation(self):
        rng=np.random.default_rng(82);X=rng.normal(size=(35,7));y=X[:,0]+rng.normal(size=35)
        data,info=qualified_path(X,y,[.4,.2,.1],rho=1.345,a=.355,max_iter=1)
        self.assertEqual(info['returned_points'],3)
        self.assertTrue(np.isfinite(data['coefficients']).all())
        self.assertGreater(info['approximate_points'],0)
        self.assertEqual(sum(data['strict']),info['strict_points'])
        self.assertTrue(np.all(data['dual_lower_bound']<=data['objective']+1e-10))

    def test_zero_boundary_and_fixed_scale_have_different_duals(self):
        X=np.eye(6);y=np.arange(6.)
        joint,ji=qualified_path(X,y,[.01,.001],rho=1.345,a=.355,max_iter=30)
        self.assertTrue(ji['strict_complete']);np.testing.assert_array_equal(joint['scales'],0.)
        fixed,fi=qualified_path(X,y,[.01,.001],rho=1.345,a=.355,estimate_scale=False,max_iter=30)
        self.assertTrue(fi['strict_complete']);np.testing.assert_array_equal(fixed['scales'],1.)
        self.assertLess(max(fixed['normalized_gap']),1e-8)

    def test_uncertified_lad_is_candidate_not_automatic_replacement(self):
        X=np.eye(4);y=np.arange(4.)
        # A deliberately inferior LAD candidate must never replace a better
        # original-objective iterate merely because scale is small.
        bad=dict(certified=False,coefficients=np.full(4,100.),objective=1e6,dual_vector=np.zeros(4))
        with patch('concomitant_huber.qualified_path.boundary_solution',return_value=bad):
            data,info=qualified_path(X,y,[.1],rho=1.345,a=.355,max_iter=1)
        self.assertFalse(np.all(data['coefficients'][0]==100.))
        self.assertFalse(info['strict_complete'])

    def test_feasible_joint_dual_bound_without_stationarity(self):
        rng=np.random.default_rng(91);X=rng.normal(size=(20,5));y=rng.normal(size=20)
        q=accuracy(X,y,np.zeros(5),.3,.1,rho=1.345,a=.355,estimate_scale=True)
        v=q['dual_vector']
        self.assertLessEqual(max(abs(v)),1.345+1e-12)
        self.assertLessEqual(np.mean(v*v),.71+1e-12)
        self.assertLessEqual(max(abs(X.T@v/20)),.1+1e-12)
        self.assertLessEqual(q['dual_lower_bound'],q['objective']+1e-12)
