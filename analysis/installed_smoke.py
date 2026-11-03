"""Exercise installed distribution from outside the checkout; no sys.path edits."""
from pathlib import Path
import importlib.metadata
import numpy as np
import rome_py
import concomitant_huber
from rome_py import HuberCD, FastHuberCD
from rome_py.path import huber_path

root=Path(__file__).resolve().parents[1]
module=Path(rome_py.__file__).resolve()
assert HuberCD is concomitant_huber.HuberCD
assert FastHuberCD is concomitant_huber.FastHuberCD
if module.is_relative_to(root/'src'):
    raise RuntimeError('Smoke check imported the source checkout instead of the installed wheel')
rng=np.random.default_rng(71)
X=rng.normal(size=(80,12));y=2*X[:,0]+rng.normal(size=80)
for estimate in (False,True):
    model=HuberCD(penalty=.25,estimate_scale=estimate,initial_scale=1.).fit(X,y)
    if not model.converged_ or not np.isfinite(model.predict(X)).all():
        raise RuntimeError('Installed estimator smoke check failed')
for method in ('mm','exact','prox'):
    fast=FastHuberCD(penalty=.25,initial_scale=1.,scale_update=method).fit(X,y)
    if not fast.converged_:raise RuntimeError('Installed scale strategy failed: '+method)
path=huber_path(X,y,[.5,.35,.25])
if path.status!=['converged']*3:raise RuntimeError('Installed path smoke check failed')
print('Installed wheel passed:',importlib.metadata.version('rome-py'),module)
