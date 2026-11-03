import sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import huber
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber.scale_updates import update_scale,_perspective_prox

class ScaleUpdateTests(unittest.TestCase):
    def test_exact_and_prox_match_independent_scalar_minimum(self):
        r=np.array([0.,-.2,.4,1.,-2.,7.]);rho=1.345;a=.355
        objective=lambda t:np.mean(t*huber(rho,r/t))+a*t if t>0 else rho*np.mean(abs(r))
        ref=minimize_scalar(objective,bounds=(1e-10,20.),method='bounded',options={'xatol':1e-12})
        for method in ['exact','prox']:
            fit=update_scale(r,2.,method=method,rho=rho,a=a,tol=1e-9,max_iter=3000)
            self.assertTrue(fit.converged,(method,fit))
            self.assertLessEqual(objective(fit.scale),ref.fun+1e-9)
            self.assertLess(abs(fit.scale-ref.x),1e-5)
    def test_finite_mm_batch_is_descent_but_not_claimed_exact(self):
        r=np.array([1.,2.,3.]);rho=1.345;a=.355
        z=update_scale(r,.001,method='mm',rho=rho,a=a,mm_steps=1,tol=1e-12)
        value=lambda t:np.mean(t*huber(rho,r/t))+a*t
        self.assertLessEqual(value(z.scale),value(.001)+1e-12)
        self.assertFalse(z.converged)
    def test_boundary_and_homogeneity(self):
        for method in ['mm','exact','prox']:
            self.assertEqual(update_scale(np.zeros(5),1.,method=method).scale,0.)
            z=update_scale([0.,0.,2.],1.,method=method,rho=1.,a=.3)
            self.assertTrue(z.converged);self.assertEqual(z.scale,0.)
        r=np.array([-.3,1.,2.,4.])
        for method in ['exact','prox']:
            base=update_scale(r,2.,method=method,tol=1e-9,max_iter=3000)
            scaled=update_scale(r*1e-100,2e-100,method=method,tol=1e-9,max_iter=3000)
            self.assertTrue(base.converged and scaled.converged)
            self.assertAlmostEqual(scaled.scale/1e-100,base.scale,places=8)
    def test_joint_prox_against_independent_constrained_projection(self):
        for u,s in [(0.,2.),(2.,-1.),(.4,1.),(3.,2.),(-2.,.1)]:
            gamma=.7;rho=1.345;a=.355
            z=np.array([u/gamma,s/gamma])
            ref=minimize(lambda v:.5*np.sum((v-z)**2),np.array([0.,min(s/gamma,a)]),
                jac=lambda v:v-z,bounds=[(-rho,rho),(None,None)],
                constraints=[{'type':'ineq','fun':lambda v:a-v[1]-.5*v[0]**2,
                              'jac':lambda v:np.array([-v[0],-1.])}],
                method='SLSQP',options={'ftol':1e-10,'maxiter':1000})
            self.assertTrue(ref.success)
            actual=np.array(_perspective_prox(u,s,gamma,rho,a))
            np.testing.assert_allclose(actual,np.array([u,s])-gamma*ref.x,atol=2e-6)
    def test_truncated_prox_reports_inexactness_and_retains_descent(self):
        r=np.array([-.1,1.,3.,8.]);old=5.
        z=update_scale(r,old,method='prox',max_iter=1,tol=1e-12)
        value=lambda s:np.mean(s*huber(1.345,r/s))+.355082274*s if s>0 else 1.345*np.mean(abs(r))
        self.assertLessEqual(value(z.scale),value(old)+1e-12)
        self.assertFalse(z.converged)


class ScaleStrategyEstimatorTests(unittest.TestCase):
    def test_three_methods_match_same_joint_solution(self):
        from concomitant_huber import FastHuberCD
        rng=np.random.default_rng(7);X=rng.normal(size=(35,5));y=1+X[:,0]+rng.normal(size=35)
        fits=[FastHuberCD(penalty=.2,scale_update=m,newton_every=0,tol=1e-8,max_iter=500).fit(X,y)
              for m in ['mm','exact','prox']]
        for fit in fits:
            self.assertTrue(fit.converged_)
            self.assertLess(max(fit.coef_kkt_,fit.scale_kkt_),1e-8)
            self.assertLessEqual(np.max(np.diff(fit.objective_history_)),1e-10)
            np.testing.assert_allclose(fit.coef_,fits[0].coef_,atol=2e-6)
            self.assertAlmostEqual(fit.scale_,fits[0].scale_,places=5)
            self.assertAlmostEqual(fit.objective_,fits[0].objective_,places=9)
    def test_fixed_scale_ignores_scale_strategy(self):
        from concomitant_huber import FastHuberCD
        rng=np.random.default_rng(81);X=rng.normal(size=(25,4));y=X[:,0]+rng.normal(size=25)
        fits=[FastHuberCD(penalty=.1,scale_update=m,estimate_scale=False,initial_scale=1.7).fit(X,y)
              for m in ['mm','exact','prox']]
        for fit in fits:
            self.assertTrue(fit.converged_);self.assertEqual(fit.scale_,1.7)
            self.assertEqual(fit.scale_inner_iterations_,0)
            np.testing.assert_array_equal(fit.coef_,fits[0].coef_)
