import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concomitant_huber.model_selection import TrainingTransform,low_correlation_columns,standardized_path
from concomitant_huber import HuberCD

class ModelSelectionTests(unittest.TestCase):
    def test_training_transform_and_predictions_ignore_validation_distribution(self):
        X=np.array([[1.,3.],[2.,3.],[3.,3.],[4.,3.]])
        y=np.array([1.,2.,4.,5.])
        transform=TrainingTransform.fit(X,y)
        np.testing.assert_allclose(transform.design(X).mean(0),0.,atol=1e-15)
        # Transforming extreme new observations must not refit training statistics.
        before=transform.x_mean.copy()
        transform.design(np.full((3,2),1e6))
        np.testing.assert_array_equal(before,transform.x_mean)
        b=np.array([.3,.5,0.])
        expected=transform.y_mean+transform.y_scale*(b[0]+transform.design(X)@b[1:])
        np.testing.assert_allclose(transform.predict(X,b),expected)

    def test_intercept_path_matches_estimator_and_warm_initialization(self):
        rng=np.random.default_rng(112)
        X=rng.normal(size=(45,7));y=2+1.3*X[:,0]+rng.normal(size=45)
        for scale in (False,True):
            path=standardized_path(X,y,[.3,.2],a=.5,estimate_scale=scale,tol=1e-8,max_iter=3000)
            model=HuberCD(penalty=.2,rho=1.345,scale_penalty=.5,estimate_scale=scale,
                          initial_scale=1.,fit_intercept=True,tol=1e-8,max_iter=3000).fit(X,y)
            self.assertEqual(path['status'][-1],'converged')
            np.testing.assert_allclose(path['coefficients'][-1],np.r_[model.intercept_,model.coef_],atol=1e-6)
            self.assertAlmostEqual(path['scales'][-1],model.scale_,places=6)

    def test_filter_has_deterministic_variance_and_index_ties(self):
        # Distinct orthogonal centered columns have zero correlation; largest variance first.
        base=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
        X=base*np.array([1.,3.,2.])
        np.testing.assert_array_equal(low_correlation_columns(X,2),[1,2])
        np.testing.assert_array_equal(low_correlation_columns(X,3),[0,1,2])
