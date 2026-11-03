import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.model_selection import standardized_path

class BoundaryTests(unittest.TestCase):
    def test_interpolating_boundary_has_feasible_dual_and_zero_gap(self):
        X=np.asfortranarray(np.column_stack([np.ones(8),np.eye(8)]));y=np.arange(8.)
        z=boundary_solution(X,y,.001,rho=1.345,a=.355)
        self.assertTrue(z['certified']);self.assertLess(z['relative_gap'],1e-9)
        self.assertLess(max(abs(y-X@z['coefficients'])),1e-9)
        self.assertLessEqual(np.mean(z['dual_vector']**2),.71+1e-12)
        self.assertLess(abs(z['dual_vector'].mean()),1e-12)
    def test_boundary_is_rejected_when_interior_has_lower_objective(self):
        X=np.ones((8,1));y=np.arange(8.)
        z=boundary_solution(X,y,.1,rho=1.345,a=.355)
        self.assertFalse(z['certified']);self.assertGreater(z['relative_gap'],1e-4)
    def test_path_does_not_truncate_after_an_uncertified_point(self):
        X=np.eye(8);y=np.arange(8.)
        z=standardized_path(X,y,[.3,.001],a=.355,backend='polished',max_iter=1)
        self.assertNotEqual(z['status'][0],'converged')
        self.assertEqual(z['status'][1],'converged');self.assertTrue(z['boundary'][1])

    def test_path_continues_after_certified_zero_scale(self):
        X=np.eye(8);y=np.arange(8.)
        z=standardized_path(X,y,[.01,.001],a=.355,backend='polished',max_iter=1000)
        self.assertEqual(z['status'],['converged','converged'])
        self.assertTrue(z['boundary'].all());np.testing.assert_array_equal(z['scales'],0.)
