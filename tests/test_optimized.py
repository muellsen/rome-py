import sys,unittest
from pathlib import Path
import numpy as np
from scipy.special import huber
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber.optimized import _local_coordinate,solve_warm
from concomitant_huber.solver import _solve
from concomitant_huber.model_selection import null_penalty_anchor

class OptimizedTests(unittest.TestCase):
    def test_local_and_kink_roots_against_independent_minimizer(self):
        rng=np.random.default_rng(91);used=[False,False]
        for k in range(100):
            n=30;x=rng.normal(size=n);u=rng.normal(size=n);old=rng.normal()
            delta=.5 if k%2 else 10.;lam=.01
            b,local=_local_coordinate(u-x*old,x,old,delta,lam,np.empty(n));used[int(local)]=True
            fun=lambda v:np.mean(huber(delta,u-x*v))+lam*abs(v)
            ref=minimize_scalar(fun,method='brent')
            self.assertLessEqual(fun(b),ref.fun+1e-10)
        self.assertTrue(all(used))
    def test_joint_and_fixed_match_reference_and_descend(self):
        rng=np.random.default_rng(82);X=np.asfortranarray(np.column_stack([np.ones(80),rng.normal(size=(80,12))]));y=1+2*X[:,1]+rng.standard_t(3,80)
        for joint in [False,True]:
            args=(X,y,.12,1.345,.355,1.,joint,1e-8,4000,True,8)
            ref=_solve(*args);fit=solve_warm(*args,np.zeros(X.shape[1]))
            self.assertTrue(ref[4] and fit[4]);np.testing.assert_allclose(fit[0],ref[0],atol=1e-6)
            self.assertAlmostEqual(fit[1],ref[1],places=6)
            self.assertLess(np.max(np.diff(fit[3])),1e-10)
    def test_safeguarded_polish_matches_independent_reference(self):
        rng=np.random.default_rng(205)
        X=rng.normal(size=(100,10));X[:,1]=X[:,0]+.03*rng.normal(size=100)
        X=np.asfortranarray(np.column_stack([np.ones(100),X]));y=1+2*X[:,1]+rng.standard_t(4,100)
        args=(X,y,.15,1.345,.355,1.,True,1e-8,10000,True,8)
        ref=_solve(*args);fit=solve_warm(*args,np.zeros(X.shape[1]),newton_every=5)
        self.assertTrue(ref[4] and fit[4])
        self.assertAlmostEqual(fit[3][-1],ref[3][-1],places=9)
        np.testing.assert_allclose(fit[0],ref[0],atol=2e-5)
        self.assertLess(np.max(np.diff(fit[3])),1e-10)

    def test_public_fast_estimator_and_zero_scale(self):
        from concomitant_huber import FastHuberCD
        X=np.eye(8);y=np.arange(8.)
        fit=FastHuberCD(penalty=.001,max_iter=1000).fit(X,y)
        self.assertTrue(fit.converged_);self.assertEqual(fit.status_,'boundary_certified')
        self.assertTrue(fit.boundary_certificate_['certified'])
        np.testing.assert_allclose(fit.predict(X),y,atol=1e-7)

    def test_model_specific_anchor_certifies_null(self):
        rng=np.random.default_rng(10);X=np.asfortranarray(rng.normal(size=(70,8)));y=2+rng.standard_t(3,70)
        for joint in [False,True]:
            z=null_penalty_anchor(X,y,estimate_scale=joint)
            self.assertLess(z['intercept_kkt'],1e-9)
            if joint:self.assertLess(z['scale_kkt'],1e-9)
            D=np.asfortranarray(np.column_stack([np.ones(len(y)),X]));b=np.r_[z['intercept'],np.zeros(X.shape[1])]
            fit=solve_warm(D,y,z['lambda_max']*1.0001,1.345,.355082274,z['scale'],joint,1e-8,3000,True,8,b)
            self.assertTrue(fit[4]);self.assertLess(max(abs(fit[0][1:])),1e-8)
