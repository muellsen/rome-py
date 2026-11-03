"""Fresh complete paths: native ROME versus optimized concomitant fitting.

A declared four-case validation, not a replacement for the 360-path baseline.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,shutil
import numpy as np
from pipeline import ROOT,OUT,capture,save,verify
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.optimized import solve_warm
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.solver import _diagnostics
base=json.loads((OUT/'revision-rome/summary.json').read_text());a=base['configuration']['a']
cases=['100_100_Block-AR_71','500_100_Block-AR_71','100_500_AR-t2_71','100_100_Contaminated-AR_71']
CONFIG=dict(cases=cases,tol=1e-6,max_sweeps=4000,path_wall_budget=120,rho=.5,a=a,check_every=5,newton_every=5,maximum_Newton_support=128,replications=1,scope='four declared validation paths; separate from 360-path reference study',timer='prepared process CPU and wall; JIT, serialization and process startup excluded')
DEST=OUT/'optimized-timing';DEST.mkdir(exist_ok=True)
small=(np.asfortranarray(np.eye(6)),np.arange(6.),.1,.5,a,1.,True,1e-6,6,False,8,np.zeros(6))
solve_warm(*small,newton_every=5);_diagnostics(small[0],small[1],small[-1],1.,.5,a,.1,False)
# Initialize LP outside timers, including its backend runtime.
boundary_solution(np.eye(3),np.arange(3.),.01,rho=.5,a=a,intercept=False)
rows=[];inputs=[]
for tag in cases:
 d=DEST/tag;d.mkdir(exist_ok=True);source=OUT/'revision-rome/cases'/tag
 for name in ['X.tsv','y.tsv','lambda.tsv']:shutil.copy2(source/name,d/name);inputs.append(source/name)
 X=np.asfortranarray(np.loadtxt(d/'X.tsv'));y=np.loadtxt(d/'y.tsv');penalties=np.loadtxt(d/'lambda.tsv')
 record=dict(case=tag,methods={});arrays={}
 try:
  r=subprocess.run(['Rscript','analysis/revision_rome_reference.R',str(d)],cwd=ROOT,capture_output=True,text=True,timeout=120)
  (d/'R-console.txt').write_text(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(r.stderr)
  b=np.atleast_2d(np.loadtxt(d/'rome-beta.tsv'));tt=np.loadtxt(d/'rome-time.tsv')
  kval=[_diagnostics(X,y-X@bb,bb,1.,.5,a,lam,False)[0] for bb,lam in zip(b,penalties)]
  record['methods']['ROME']=dict(complete=len(b)==100,cpu_seconds=float(tt[1]),wall_seconds=float(tt[0]),maximum_coefficient_kkt=float(max(kval)))
 except (RuntimeError,subprocess.TimeoutExpired) as e:record['methods']['ROME']=dict(complete=False,error=str(e))
 beta=np.zeros(X.shape[1]);sigma=1.;statuses=[];coeff=[];scales=[];violations=[];iterations=[];boundary=[]
 cpu=time.process_time();wall=time.perf_counter()
 for lam in penalties:
  if time.perf_counter()-wall>120:statuses.append('wall_budget');break
  if sigma>0:
   rr=solve_warm(X,y,lam,.5,a,sigma,True,1e-6,4000,False,8,beta,newton_every=5)
   beta,sigma,it,history,ok,cb,cs=rr[:7]
  else:ok=False;it=0;cb=np.inf;cs=np.inf
  isboundary=False
  if not ok:
   cert=boundary_solution(X,y,lam,rho=.5,a=a,intercept=False,tol=1e-6)
   if cert['certified']:
    beta=cert['coefficients'];sigma=0.;ok=True;cb=cert['coefficient_kkt'];cs=cert['scale_kkt'];isboundary=True
  statuses.append('converged' if ok else 'uncertified');coeff.append(beta.copy());scales.append(sigma);violations.append([cb,cs]);iterations.append(it);boundary.append(isboundary)
  if not ok:break
 record['methods']['Optimized']=dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,complete=len(statuses)==100 and all(s=='converged' for s in statuses),qualified_points=statuses.count('converged'),statuses=statuses,boundary_points=sum(boundary))
 np.savez_compressed(d/'solutions.npz',beta=coeff,sigma=scales,kkt=violations,iterations=iterations,boundary=boundary)
 save(record,d/'result.json');rows.append(record);print(record,flush=True)
save(dict(configuration=CONFIG,results=rows),DEST/'summary.json')
lines=['| Validation design $(n,p)$ | ROME CPU s | Optimized CPU s | Certified joint points |','|:--|--:|--:|--:|']
for r in rows:
 m=r['methods'];old=m['ROME'].get('cpu_seconds',float('nan'));new=m['Optimized']['cpu_seconds']
 lines.append(f"| {r['case'].replace('_',' / ')} | {old:.4f} | {new:.4f} | {m['Optimized']['qualified_points']}/100 |")
(DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
save(capture([DEST/'summary.json',DEST/'results.qmd',*DEST.rglob('result.json'),*DEST.rglob('solutions.npz'),*DEST.rglob('*.tsv'),*DEST.rglob('R-console.txt')],inputs=inputs,configuration=CONFIG),DEST/'provenance.json')
