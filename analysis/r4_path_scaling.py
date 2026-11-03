"""Size scaling of unconstrained R4: Path-Alg first, fixed accuracy/budgets."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r4_benchmark import prepare,diagnostics,qualified,_solve
from pipeline import OUT,capture,save

CONFIG=dict(dimensions=[[50,100],[100,100],[100,200],[100,400],[100,800],[200,400],[400,800]],
 seeds=[71,72],repeats=3,methods=['MM-CD','Path-Alg-eps0','Path-Alg'],rho=1.345,
 a=.5,penalty_fraction=.6,stationarity_tolerance=1e-5,relative_objective_tolerance=1e-8,
 primary='Path-Alg-eps0 vs MM-CD; stock eps1e-3 retained as accuracy diagnostic',
 scaling_axes=['n increases at p=2n','p increases at n=100'],
 timers='fresh prepared solves; setup and setup+solve also recorded',
 constraints='one zero row',intercept=False,noise='Gaussian',design='independent Gaussian, centered columns RMS1')
OUT.mkdir(exist_ok=True)
rows=[];arrays={}
_solve(np.asfortranarray(np.eye(4)),np.arange(4,dtype=float),.5,1.345,.5,1.,True,1e-5,100,False,8)
for n,p in CONFIG['dimensions']:
 for seed in CONFIG['seeds']:
  rng=np.random.default_rng(seed)
  X=rng.normal(size=(n,p));X-=X.mean(0);X/=np.sqrt(np.mean(X**2,0))
  truth=np.zeros(p);truth[:10]=rng.choice([-1.,1.],10)*rng.uniform(1,2,10)
  y=X@truth+rng.normal(size=n)
  initial=max(float(np.median(abs(y-np.median(y)))/.67448975),1e-6)
  lam=.6*max(abs(X.T@np.clip(y/initial,-1.345,1.345)))/n
  ref=_solve(np.asfortranarray(X),y,lam,1.345,.5,initial,True,1e-9,10000,False,8)
  if not ref[4]:
   rows.append(dict(n=n,p=p,seed=seed,repeat=-1,method='reference',penalty=lam,status='reference_uncertified',qualified=False,scale=float(ref[1])))
   print('Reference uncertified',n,p,seed,flush=True)
   continue
  reference=diagnostics(X,y,ref[0],ref[1],lam)['objective']
  key=f'{n}_{p}_{seed}';arrays[key+'_X']=X;arrays[key+'_y']=y;arrays[key+'_reference_beta']=ref[0]
  for repeat in range(CONFIG['repeats']):
   methods=CONFIG['methods'][::1 if repeat%2==0 else -1]
   for method in methods:
    row=dict(n=n,p=p,seed=seed,repeat=repeat,method=method,penalty=lam,reference=reference)
    try:
     start=time.perf_counter();call=prepare(method,X,y,lam,initial,reference=reference)
     setup=time.perf_counter()-start
     start=time.perf_counter();b,sigma=call();solve=time.perf_counter()-start
     d=diagnostics(X,y,b,sigma,lam)
     row.update(setup_seconds=setup,solve_seconds=solve,total_seconds=setup+solve,scale=sigma,
                qualified=bool(qualified(d,reference)),status='returned',**d)
    except (RuntimeError,ValueError,np.linalg.LinAlgError) as e:
     row.update(status='failure',qualified=False,error=str(e))
    rows.append(row)
  print(n,p,seed,[(m,sum(r['qualified'] for r in rows if r['n']==n and r['p']==p and r['seed']==seed and r['method']==m)) for m in CONFIG['methods']],flush=True)
raw=OUT/'r4-path-scaling.csv'
keys=sorted(set().union(*(r.keys() for r in rows)))
with raw.open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
np.savez_compressed(OUT/'r4-path-scaling-data.npz',**arrays)
summary=[]
lines=['| n | p | MM-CD ms | Path0 ms | Path0 setup ms | Stock path passes |','|--:|--:|--:|--:|--:|:--|']
for n,p in CONFIG['dimensions']:
 med={};counts={}
 for m in CONFIG['methods']:
  rr=[r for r in rows if r['n']==n and r['p']==p and r['method']==m]
  passed=[r for r in rr if r['qualified']]
  counts[m]=[len(passed),len(rr)]
  med[m]={k:float(np.median([r[k] for r in passed])) if passed else None for k in ('solve_seconds','setup_seconds','total_seconds')}
 summary.append(dict(n=n,p=p,median_seconds=med,passes=counts))
 fmt=lambda v:'—' if v is None else f'{1000*v:.2f}'
 lines.append(f"| {n} | {p} | {fmt(med['MM-CD']['solve_seconds'])} | {fmt(med['Path-Alg-eps0']['solve_seconds'])} | {fmt(med['Path-Alg-eps0']['setup_seconds'])} | {counts['Path-Alg'][0]}/{counts['Path-Alg'][1]} |")
(OUT/'r4-path-scaling-results.qmd').write_text('\n'.join(lines)+'\n')
save(dict(configuration=CONFIG,results=summary),OUT/'r4-path-scaling-summary.json')
fig,axes=plt.subplots(1,3,figsize=(10,3.2),constrained_layout=True)
for ax,axis,title,kind in ((axes[0],'n','p = 2n','solve_seconds'),(axes[1],'p','n = 100','solve_seconds'),(axes[2],'p','n = 100, including setup','total_seconds')):
 points=[r for r in summary if (r['p']==2*r['n'] if axis=='n' else r['n']==100)]
 points=sorted(points,key=lambda r:r[axis])
 for method,label in (('MM-CD','MM-CD'),('Path-Alg-eps0','Path0')):
  values=[r['median_seconds'][method][kind] for r in points]
  ax.loglog([r[axis] for r in points],values,'o-',label=label)
 ax.set(xlabel=axis,ylabel='Median seconds',title=title);ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'r4-path-scaling.pdf');fig.savefig(OUT/'r4-path-scaling.png',dpi=180);plt.close(fig)
outputs=[OUT/n for n in ('r4-path-scaling.csv','r4-path-scaling-data.npz','r4-path-scaling-results.qmd','r4-path-scaling-summary.json','r4-path-scaling.pdf','r4-path-scaling.png')]
save(capture(outputs,inputs=[Path(__file__).resolve().parents[1]/'sources/classo-1.0.11-audit/registry.json'],configuration=CONFIG,seed=CONFIG['seeds']),OUT/'r4-path-scaling.provenance.json')
