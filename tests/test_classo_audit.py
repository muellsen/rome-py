"""Verify off-clock setup leaves stock c-lasso arithmetic unchanged."""
import sys
import unittest
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
try:
    import r4_benchmark as bench
except ImportError:
    bench=None


@unittest.skipIf(bench is None,'optional c-lasso benchmark dependency unavailable')
class ClassoAuditTest(unittest.TestCase):
    def test_snapshots_match_imported_source(self):
        import classo
        registry=Path(__file__).resolve().parents[1]/'sources/classo-1.0.11-audit/registry.json'
        for entry in json.loads(registry.read_text())['files']:
            file=Path(classo.__file__).parent/Path(entry['path']).name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),entry['sha256'])

    def test_prepared_calls_match_stock(self):
        import numpy as np
        rng=np.random.default_rng(51)
        X=rng.normal(size=(20,5));y=X[:,0]*2+rng.normal(size=20)
        penalty=.2
        for method in ('DR','Path-Alg'):
            pb=bench.r4.problem_R4((X,np.zeros((1,5)),y),method,1.345)
            pb.tol=1e-5
            expected=bench.r4.Classo_R4(pb,2*len(y)*penalty/pb.lambdamax)
            actual=bench.prepare(method,X,y,penalty,1.,dr_tol=1e-5)()
            np.testing.assert_allclose(actual[0],expected[0],atol=1e-12,rtol=1e-12)
            self.assertAlmostEqual(actual[1],expected[1],places=12)

    def test_identity_variant_matches_stock_dr(self):
        import numpy as np
        rng=np.random.default_rng(3)
        X=rng.normal(size=(20,5));y=X[:,0]*2+rng.normal(size=20)
        ordinary=bench.prepare('DR',X,y,.2,1.)()
        identity=bench.prepare('DR-identity',X,y,.2,1.)()
        np.testing.assert_allclose(ordinary[0],identity[0],atol=1e-10)
        self.assertAlmostEqual(ordinary[1],identity[1],places=10)

    def test_balanced_dr_meets_common_gate(self):
        import numpy as np
        rng=np.random.default_rng(9)
        X=rng.normal(size=(25,6));y=X[:,0]*2+rng.normal(size=25)
        b,s=bench.prepare('Path-Alg-eps0',X,y,.2,1.)()
        objective=bench.diagnostics(X,y,b,s,.2)['objective']
        b,s=bench.prepare('DR-balanced',X,y,.2,1.,reference=objective)()
        self.assertTrue(bench.qualified(bench.diagnostics(X,y,b,s,.2),objective))
