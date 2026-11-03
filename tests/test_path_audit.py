"""Independent checks of the objective bounds used in the diagnostic report."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from audit_incomplete_paths import dual_bound, boundary_lp

class AuditBoundsTests(unittest.TestCase):
    def test_boundary_and_joint_duals_match_known_one_observation_solution(self):
        # With lambda < min(rho, sqrt(2a)), beta=y and sigma=0 is optimal.
        X = np.ones((1, 1))
        y = np.array([2.])
        lam, rho, a = .2, .5, .1
        v, report = dual_bound(X, y, lam, rho, a)
        self.assertLessEqual(abs(v[0]), rho)
        self.assertLessEqual(abs((X.T@v)[0]), lam)
        self.assertLessEqual(v@v/2, a)
        self.assertAlmostEqual(report['value'], .4, places=9)
        beta, lower, primal = boundary_lp(X, y, lam, rho)
        self.assertAlmostEqual(beta[0], 2.)
        self.assertLessEqual(lower, primal)
        self.assertAlmostEqual(lower, .4, places=9)
