"""Check the AISTATS riboflavin preprocessing and support-calibration protocol."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
import time
import csv
import numpy as np
from scipy.io import loadmat
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber import HuberCD
from concomitant_huber.solver import _objective, _solve
from pipeline import OUT,capture,save
spec=importlib.util.spec_from_file_location('source_protocol_ramp',ROOT/'sources/amp-aistats2027/perspective_huber_ramp_solver.py')
ramp=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ramp;spec.loader.exec_module(ramp)
tree=ast.parse((ROOT/'sources/amp-aistats2027/riboflavin.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_greedy_decorrelated_genes')
namespace={'np':np,'RiboflavinData':SimpleNamespace}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<reference-filter>','exec'),namespace)
mat=loadmat(ROOT/'sources/data/riboflavinData.mat')['riboData']
data=SimpleNamespace(x=mat[1:].T,y=mat[0],gene_names=np.arange(mat.shape[0]-1))
data,selected=namespace['select_greedy_decorrelated_genes'](data,keep_k=500)
# Match source workflow: unsupervised filter precedes split, seed and fraction as supplied.
perm=np.random.default_rng(20261006).permutation(len(data.y))
train=np.sort(perm[:round(.8*len(data.y))]);test=np.sort(perm[round(.8*len(data.y)):])
X=data.x[train]-data.x[train].mean(0);X/=np.sqrt(np.mean(X**2,axis=0))
y=(data.y[train]-data.y[train].mean())/data.y[train].std(ddof=1)
n,p=X.shape;rho=1.345
a=.5*(1-2*rho*np.exp(-rho**2/2)/np.sqrt(2*np.pi)+2*(rho**2-1)*(1-ndtr(rho)))
rows=[];arrays=dict(selected=selected,train=train,test=test,X=X,y=y)
# Include the draft's chosen lambda=2.5 and support=14; grid is an accuracy audit, not CV.
for lamp in (2.5,3.,4.,5.):
 lam=lamp/np.sqrt(n)
 with warnings.catch_warnings(record=True):
  ref=HuberCD(penalty=lam,rho=rho,scale_penalty=a,fit_intercept=False,tol=1e-9,max_iter=10000).fit(X,y)
 arrays[f'cd_{lamp}']=ref.coef_
 for count in range(1,31):
  row=dict(lambda_AMP=lamp,lambda0=lam,target_support=count,omega=count/p,reference_status=ref.status_,reference_sigma=ref.scale_,reference_objective=ref.objective_)
  try:
   b,h=ramp.run_perspective_huber_ramp(X/np.sqrt(n),y,lam=lamp,omega=count/p,iterations=1000,a=a,
       params=ramp.HuberSpec(c=rho),convergence=ramp.SolverConvergence(max_iterations=1000))
   b/=np.sqrt(n);sigma=h[-1]['sigma'];r=y-X@b;psi=np.clip(r/sigma,-rho,rho)
   grad=-X.T@psi/n
   vb=float(max(abs(b-np.sign(b-grad)*np.maximum(abs(b-grad)-lam,0))))
   vs=float(abs(a-.5*np.mean(psi**2)))
   gap=float((_objective(r,b,sigma,rho,a,lam)-ref.objective_)/max(1,abs(ref.objective_)))
   qualified=ref.converged_ and vb<=1e-5 and vs<=1e-5 and abs(gap)<=1e-8
   row.update(status='qualified' if qualified else 'accuracy_failed',qualified=bool(qualified),
       coefficient_residual=vb,scale_residual=vs,relative_objective_gap=gap,
       q=h[-1]['q'],q_minus_omega=h[-1]['q']-count/p,sigma=sigma,iterations=len(h),
       coefficient_distance=float(np.linalg.norm(b-ref.coef_)),prediction_distance=float(np.linalg.norm(X@(b-ref.coef_))),
       native_converged=bool(h[-1]['converged']),support=int(np.sum(abs(b)>1e-8)),history=h)
   arrays[f'amp_{lamp}_{count}']=b
  except (RuntimeError,FloatingPointError,ValueError) as e:
   row.update(status='numerical_failure',qualified=False,error=str(e))
  rows.append(row)
 print(lamp,{k:sum(r['status']==k for r in rows if r['lambda_AMP']==lamp) for k in ('qualified','accuracy_failed','numerical_failure')},flush=True)
save(dict(protocol='AISTATS source preprocessing: global unsupervised greedy filter k500, 80% training, seed20261006; no CV; four lambdas, target support1..30',n=n,p=p,a=a,results=rows),OUT/'amp-source-protocol-summary.json')
np.savez_compressed(OUT/'amp-source-protocol-data.npz',**arrays)
lines=['| AMP penalty | Qualified | Accuracy failed | Numerical failure |','|--:|--:|--:|--:|']
for lam in (2.5,3.,4.,5.):
 rr=[r for r in rows if r['lambda_AMP']==lam]
 lines.append(f"| {lam} | {sum(r['status']=='qualified' for r in rr)} | {sum(r['status']=='accuracy_failed' for r in rr)} | {sum(r['status']=='numerical_failure' for r in rr)} |")
(OUT/'amp-source-protocol-results.qmd').write_text('\n'.join(lines)+'\n')

# Time only settings already shown to solve the same objective, with fresh state.
# This is a conditional implementation comparison, not a calibration-search timer.
timings=[]
design=np.asfortranarray(X);amp_design=X/np.sqrt(n)
initial=max(float(np.median(abs(y-np.median(y)))/.67448975),float(np.std(y))*1e-3,np.finfo(float).eps)
for lamp,count in ((2.5,14),(3.,6)):
 lam=lamp/np.sqrt(n)
 for repeat in range(5):
  for method in (('MM-CD','RAMP') if repeat%2==0 else ('RAMP','MM-CD')):
   start=time.perf_counter()
   if method=='MM-CD':
    result=_solve(design,y,lam,rho,a,initial,True,1e-9,10000,False,8)
    b,sigma,iterations=result[:3]
   else:
    b,h=ramp.run_perspective_huber_ramp(amp_design,y,lam=lamp,omega=count/p,iterations=1000,a=a,
      params=ramp.HuberSpec(c=rho),convergence=ramp.SolverConvergence(max_iterations=1000))
    sigma=h[-1]['sigma'];iterations=len(h)
   elapsed=time.perf_counter()-start
   if method=='RAMP':b=b/np.sqrt(n)
   residual=y-X@b;psi=np.clip(residual/sigma,-rho,rho);gradient=-X.T@psi/n
   vb=float(max(abs(b-np.sign(b-gradient)*np.maximum(abs(b-gradient)-lam,0))))
   vs=float(abs(a-.5*np.mean(psi**2)))
   reference=next(r for r in rows if r['lambda_AMP']==lamp and r['target_support']==count)
   gap=float((_objective(residual,b,sigma,rho,a,lam)-reference['reference_objective'])/max(1.,abs(reference['reference_objective'])))
   qualified=vb<=1e-5 and vs<=1e-5 and abs(gap)<=1e-8
   if not qualified:raise RuntimeError('Timed return failed common accuracy gate')
   timings.append(dict(method=method,lambda_AMP=lamp,target_support=count,repeat=repeat,
       seconds=elapsed,iterations=iterations,coefficient_residual=vb,scale_residual=vs,relative_objective_gap=gap,qualified=qualified))
with (OUT/'amp-matched-timings.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
lines=['| AMP penalty / support | MM-CD ms | RAMP ms |','|:--|--:|--:|']
for lamp,count in ((2.5,14),(3.,6)):
 medians=[1000*np.median([r['seconds'] for r in timings if r['method']==method and r['lambda_AMP']==lamp]) for method in ('MM-CD','RAMP')]
 lines.append(f'| {lamp} / {count} | {medians[0]:.2f} | {medians[1]:.2f} |')
(OUT/'amp-matched-timings.qmd').write_text('\n'.join(lines)+'\n')
outputs=[OUT/n for n in ('amp-source-protocol-summary.json','amp-source-protocol-data.npz','amp-source-protocol-results.qmd','amp-matched-timings.csv','amp-matched-timings.qmd')]
save(capture(outputs,inputs=[ROOT/'sources/data/riboflavinData.mat',ROOT/'sources/amp-aistats2027/riboflavin.py',ROOT/'sources/amp-aistats2027/perspective_huber_ramp_solver.py'],
 configuration=dict(global_predictor_filter=500,seed=20261006,training_fraction=.8,target_support=list(range(1,31)),lambda_AMP=[2.5,3.,4.,5.],max_iterations=1000,timing_repeats=5,timing_settings=[[2.5,14],[3.,6]],timing_excludes=['imports','JIT','filtering','standardization','matrix preparation','calibration search']),seed=20261006),OUT/'amp-source-protocol.provenance.json')
