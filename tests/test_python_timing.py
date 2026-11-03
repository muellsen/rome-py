"""A timing return must pass independent accuracy checks before comparison."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from python_scale_timing import check_path

class PythonTimingTests(unittest.TestCase):
    def test_fixed_scale_and_joint_have_distinct_required_certificates(self):
        X=np.asfortranarray(np.eye(4));y=np.zeros(4)
        def trial(s):
            return dict(penalties=np.array([.1]),coefficients=np.zeros((1,4)),
                        scales=np.array([s]),boundary=np.array([False]),dual_vectors=np.full((1,4),np.nan))
        fixed=dict(statuses=['converged'])
        check_path(X,y,trial(1.),fixed,estimate_scale=False,rho=1.345,a=.355,tol=1e-6)
        self.assertTrue(fixed['qualified_complete_path'])
        wrong=dict(statuses=['converged'])
        check_path(X,y,trial(2.),wrong,estimate_scale=False,rho=1.345,a=.355,tol=1e-6)
        self.assertFalse(wrong['qualified_complete_path'])
        joint=dict(statuses=['converged'])
        check_path(X,y,trial(1.),joint,estimate_scale=True,rho=1.345,a=.355,tol=1e-6)
        self.assertFalse(joint['qualified_complete_path'])

    def test_zero_scale_requires_a_dual_gap_certificate(self):
        X=np.asfortranarray(np.eye(4));y=np.arange(4.);lam=.01
        def trial(v):
            return dict(penalties=np.array([lam]),coefficients=y[None,:],scales=np.array([0.]),
                        boundary=np.array([True]),dual_vectors=v[None,:])
        good=dict(statuses=['converged'])
        check_path(X,y,trial(4*lam*np.sign(y)),good,estimate_scale=True,rho=1.345,a=.355,tol=1e-6)
        self.assertTrue(good['qualified_complete_path'])
        bad=dict(statuses=['converged'])
        check_path(X,y,trial(np.zeros(4)),bad,estimate_scale=True,rho=1.345,a=.355,tol=1e-6)
        self.assertFalse(bad['qualified_complete_path'])
