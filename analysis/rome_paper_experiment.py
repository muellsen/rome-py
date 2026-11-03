"""Mirror ROME Section4.1, comparing only original rome with its joint extension."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json,subprocess,time,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.path import huber_path
from pipeline import OUT,capture,save
CONFIG=json.loads((ROOT/'analysis/configs/rome.json').read_text())


def ar(rng,n,p,rho):
    X=rng.normal(size=(n,p))
    for j in range(1,p):X[:,j]=rho*X[:,j-1]+np.sqrt(1-rho*rho)*X[:,j]
    return X


def generate(n,p,design,seed):
    rng=np.random.default_rng(seed)
    if design=='Compound-N':X=np.sqrt(.8)*rng.normal(size=(n,1))+np.sqrt(.2)*rng.normal(size=(n,p))
    elif design=='AR-t2':X=ar(rng,n,p,.8)/np.sqrt(rng.chisquare(2,n)[:,None]/2)
    elif design=='Contaminated-AR':X=np.column_stack((ar(rng,n,p-1,.8),rng.standard_t(1,n)))
    elif design=='Block-AR':X=np.column_stack((ar(rng,n,p//2,.2)/np.sqrt(rng.chisquare(1,n)[:,None]),ar(rng,n,p-p//2,.8)))
    else:raise ValueError(design)
    beta=np.zeros(p);beta[:16]=CONFIG['beta_first16']
    y=X@beta+rng.normal(size=n)
    maximum=max(abs(X.T@np.clip(y,-.5,.5)))/n
    penalties=maximum*np.exp(np.log(.05)*np.arange(1,101)/100)
    return X,y,beta,penalties


def kkt(X,y,b,sigma,lam,rho=.5):
    if sigma<=0:return float('inf')
    g=-X.T@np.clip((y-X@b)/sigma,-rho,rho)/len(y)
    return float(max(abs(b-np.sign(b-g)*np.maximum(abs(b-g)-lam,0))))


def run(full=False):
    dimensions=CONFIG['dimensions_full'] if full else CONFIG['validation_dimensions']
    seeds=list(range(71,171)) if full else CONFIG['validation_seeds']
    mode='full' if full else 'validation'
    directory=OUT/f'rome-paper-{mode}';directory.mkdir(exist_ok=True)
    rows=[];arrays={};a=.5*(1-2*.5*np.exp(-.5**2/2)/np.sqrt(2*np.pi)+2*(.5**2-1)*(1-ndtr(.5)))
    # JIT outside every timer.
    huber_path(np.eye(5),np.arange(5,dtype=float),[.5],rho=.5,scale_penalty=a,tol=1e-5)
    for n,p in dimensions:
      for design in CONFIG['designs']:
       for seed in seeds:
        X,y,truth,penalties=generate(n,p,design,seed)
        tag=f'{n}_{p}_{design}_{seed}';case=directory/tag;case.mkdir(exist_ok=True)
        for name,value in (('X',X),('y',y),('lambda',penalties)):np.savetxt(case/(name+'.tsv'),value,delimiter='\t',fmt='%.17g')
        # R process startup and serialization are outside the native R timer.
        result=subprocess.run(['Rscript','analysis/rome_fixed_path.R',str(case)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        (case/'R-console.txt').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('ROME reference failed: '+result.stderr)
        original=np.atleast_2d(np.loadtxt(case/'rome-beta.tsv'))
        original_lambda=np.atleast_1d(np.loadtxt(case/'rome-lambda.tsv'))
        original_iterations=np.atleast_1d(np.loadtxt(case/'rome-iter.tsv'))
        runtime=(case/'rome-runtime.txt').read_text().splitlines();r_seconds=float(runtime[0])
        if len(original_lambda)!=100 or not np.allclose(original_lambda,penalties):raise RuntimeError('Original rome path was truncated or reordered')
        start=time.perf_counter()
        joint=huber_path(X,y,penalties,rho=.5,scale_penalty=a,initial_scale=1.,
                         tol=1e-6,max_iter=2000,estimate_scale=True)
        joint_seconds=time.perf_counter()-start
        for method,coefficients,scales,iterations,status,seconds in (
            ('rome-fixed',original,np.ones(100),original_iterations,['returned']*100,r_seconds),
            ('concomitant',joint.coefficients,joint.scales,joint.iterations,joint.status,joint_seconds)):
          attempted=sum(s!='not_attempted' for s in status)
          passed=0
          for k in range(100):
            b=coefficients[k]
            valid=np.isfinite(b).all()
            cb=kkt(X,y,b,scales[k],penalties[k]) if valid else None
            vs=float(abs(a-.5*np.mean(np.clip((y-X@b)/scales[k],-.5,.5)**2))) if valid and scales[k]>0 else None
            ok=bool(valid and cb<=1e-5 and (method=='rome-fixed' or (vs is not None and vs<=1e-5 and status[k]=='converged')))
            passed+=ok
            rows.append(dict(n=n,p=p,design=design,seed=seed,method=method,path_index=k+1,
              penalty=float(penalties[k]),scale=float(scales[k]) if np.isfinite(scales[k]) else None,
              status=status[k],qualified=ok,coefficient_residual=cb,scale_residual=vs,
              iterations=int(iterations[k]),relative_beta_error=float(np.linalg.norm(b-truth)/np.linalg.norm(truth)) if valid else None,
              full_path_seconds=seconds,attempted_points=attempted))
          print(tag,method,'qualified',passed,'attempted',attempted,'seconds',round(seconds,3),flush=True)
        arrays[tag+'_truth']=truth;arrays[tag+'_original_beta']=original;arrays[tag+'_joint_beta']=joint.coefficients
        arrays[tag+'_joint_sigma']=joint.scales;arrays[tag+'_penalties']=penalties
        # Checkpoint every pair; partial completion is explicit.
        save(dict(configuration=CONFIG,mode=mode,completed_pairs=len(arrays)//5,results=rows),directory/'progress.json')
    raw=directory/'results.csv'
    with raw.open('w') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(directory/'paths.npz',**arrays)
    save(dict(configuration=CONFIG,mode=mode,completed_pairs=len(arrays)//5,a=a,results=rows),directory/'summary.json')
    lines=['| n | p | Design | ROME path s | Joint path s | ROME / joint certified |','|--:|--:|:--|--:|--:|:--|']
    for n,p in dimensions:
      for design in CONFIG['designs']:
        rr=[r for r in rows if r['n']==n and r['p']==p and r['design']==design]
        fixed=[r for r in rr if r['method']=='rome-fixed'];joint=[r for r in rr if r['method']=='concomitant']
        lines.append(f"| {n} | {p} | {design} | {np.median([r['full_path_seconds'] for r in fixed]):.3f} | {np.median([r['full_path_seconds'] for r in joint]):.3f} | {sum(r['qualified'] for r in fixed)}/{len(fixed)}; {sum(r['qualified'] for r in joint)}/{len(joint)} |")
    (directory/'results.qmd').write_text('\n'.join(lines)+'\n')
    # One path panel per original design, first declared dimension only.
    fig,axes=plt.subplots(2,2,figsize=(8,5.5),constrained_layout=True)
    n,p=dimensions[0]
    for ax,design in zip(axes.flat,CONFIG['designs']):
      for method,label in (('rome-fixed','ROME fixed'),('concomitant','Concomitant')):
        rr=[r for r in rows if r['n']==n and r['p']==p and r['design']==design and r['method']==method and r['seed']==seeds[0] and r['qualified']]
        ax.semilogy([r['path_index'] for r in rr],[max(r['relative_beta_error'],1e-16) for r in rr],label=label)
      ax.set(title=design,xlabel='Penalty index',ylabel='Relative coefficient error');ax.legend(fontsize=8)
    fig.savefig(directory/'recovery.pdf');fig.savefig(directory/'recovery.png',dpi=180);plt.close(fig)
    inputs=[ROOT/'analysis/configs/rome.json',ROOT/'sources/rome-99d6f07.tar.gz']
    inputs.extend(p for p in directory.rglob('*.tsv'));inputs.extend(directory.rglob('rome-runtime.txt'));inputs.extend(directory.rglob('R-console.txt'))
    outputs=[directory/n for n in ('results.csv','summary.json','paths.npz','results.qmd','recovery.pdf','recovery.png')]
    save(capture(outputs,inputs=inputs,configuration=dict(protocol=CONFIG,mode=mode,dimensions=dimensions,seeds=seeds,a=a,R_version=runtime[1],rome_version=runtime[2]),seed=seeds),directory/'provenance.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
    run(args.full)
