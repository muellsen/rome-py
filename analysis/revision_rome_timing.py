"""ROME Section 4.1, 10 paired replications; resumable per-case checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
import argparse,json,sys,time,subprocess,csv
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from rome_paper_experiment import generate,CONFIG as ORIGINAL
from pipeline import ROOT,OUT,capture,save
from concomitant_huber.path import _solve_warm
from concomitant_huber.solver import _diagnostics

DEST=OUT/'revision-rome'
CONFIG=dict(replications=10,dimensions=ORIGINAL['dimensions_full'],designs=ORIGINAL['designs'],
            seeds=list(range(71,81)),rho=.5,a=float(.5*(1-2*.5*np.exp(-.5**2/2)/np.sqrt(2*np.pi)+2*(.5**2-1)*(1-ndtr(.5)))),
            length=100,minimum_ratio=.05,tol=1e-6,max_sweeps=4000,scale_batch=8,
            joint_path_wall_budget=120,R_wall_budget=120,
            R_native_eps=1e-4,comparison='original R/C ROME vs concomitant Python MM-CD',
            timer='CPU and wall times on loaded path inputs; JIT, serialization and R process startup excluded',
            accuracy='full path separately from joint certificate; never rank an incomplete path',
            scope='10 rather than original 100 replications per setting; original script distributions and coefficient vector')


def case(n,p,design,seed):
    tag=f'{n}_{p}_{design}_{seed}'
    directory=DEST/'cases'/tag;directory.mkdir(parents=True,exist_ok=True)
    metadata=directory/'result.json'
    if metadata.exists():
        stored=json.loads(metadata.read_text())
        if stored.get('configuration')!=CONFIG:raise ValueError('Existing run configuration differs')
        return stored
    X,y,truth,penalties=generate(n,p,design,seed);X=np.asfortranarray(X)
    for name,value in [('X',X),('y',y),('lambda',penalties)]:
        np.savetxt(directory/(name+'.tsv'),value,fmt='%.17g',delimiter='\t')
    data=dict(case=tag,n=n,p=p,design=design,seed=seed,configuration=CONFIG)
    try:
        run=subprocess.run(['Rscript','analysis/revision_rome_reference.R',str(directory)],
                           cwd=ROOT,capture_output=True,text=True,timeout=CONFIG['R_wall_budget'])
        (directory/'R-console.txt').write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(run.stderr)
        rb=np.atleast_2d(np.loadtxt(directory/'rome-beta.tsv'))
        rl=np.loadtxt(directory/'rome-lambda.tsv')
        if rb.shape!=(100,p) or not np.allclose(rl,penalties):raise RuntimeError('Truncated R path')
        rt=np.loadtxt(directory/'rome-time.tsv')
        rv=np.array([_diagnostics(X,y-X@b,b,1.,.5,CONFIG['a'],lam,False)[0] for b,lam in zip(rb,penalties)])
        data['rome']=dict(status='returned',complete=True,wall_seconds=float(rt[0]),cpu_seconds=float(rt[1]),
                          qualified_points=int(sum(rv<=1e-5)),maximum_coefficient_kkt=float(max(rv)))
    except (subprocess.TimeoutExpired,RuntimeError) as e:
        rb=np.full((100,p),np.nan)
        data['rome']=dict(status='failure',complete=False,error=str(e))
    betas=np.full((100,p),np.nan);scales=np.full(100,np.nan);vb=np.full(100,np.nan);vs=vb.copy()
    iterations=np.zeros(100,dtype=int);beta=np.zeros(p);sigma=1.;statuses=['not_attempted']*100
    start=time.perf_counter();cpu=time.process_time()
    for k,lam in enumerate(penalties):
        if time.perf_counter()-start>CONFIG['joint_path_wall_budget']:
            statuses[k]='path_wall_budget';break
        result=_solve_warm(X,y,lam,.5,CONFIG['a'],sigma,True,1e-6,4000,False,8,beta)
        beta,sigma,it,history,ok,cb,cs,_=result
        betas[k]=beta;scales[k]=sigma;iterations[k]=it;vb[k]=cb;vs[k]=cs
        statuses[k]='converged' if ok else ('boundary_uncertified' if sigma==0 else 'max_iter')
        if not ok:break
    joint_wall=time.perf_counter()-start;joint_cpu=time.process_time()-cpu
    complete=bool(all(s=='converged' for s in statuses))
    data['joint']=dict(status='converged' if complete else next(s for s in statuses if s!='converged'),
                       complete=complete,wall_seconds=joint_wall,cpu_seconds=joint_cpu,
                       qualified_points=int(sum(s=='converged' for s in statuses)),
                       attempted_points=int(np.isfinite(scales).sum()),statuses=statuses)
    np.savez_compressed(directory/'solutions.npz',truth=truth,penalties=penalties,
                        rome_beta=rb,joint_beta=betas,joint_scale=scales,
                        coefficient_kkt=vb,scale_kkt=vs,iterations=iterations)
    save(data,metadata)
    print(tag,'R',data['rome']['status'],'joint',data['joint']['status'],
          data['joint']['qualified_points'],'CPU',round(joint_cpu,2),flush=True)
    return data


def assemble(results):
    save(dict(configuration=CONFIG,completed_cases=len(results),planned_cases=360,results=results),DEST/'summary.json')
    lines=['| Design | n | p | ROME CPU s | MM-CD CPU s | Complete joint paths |',
           '|:--|--:|--:|--:|--:|:--|']
    for design in CONFIG['designs']:
      for n,p in CONFIG['dimensions']:
        rr=[r for r in results if r['design']==design and r['n']==n and r['p']==p]
        paired=[r for r in rr if r['rome']['complete'] and r['joint']['complete']]
        fmt=lambda method:f"{np.mean([r[method]['cpu_seconds'] for r in paired]):.3f}" if paired else '—'
        lines.append(f"| {design} | {n} | {p} | {fmt('rome')} | {fmt('joint')} | {sum(r['joint']['complete'] for r in rr)}/{len(rr)} |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')


def main(limit=None):
    DEST.mkdir(exist_ok=True)
    # Both compiled warm kernel and BLAS initialization are outside every timer.
    _solve_warm(np.asfortranarray(np.eye(5)),np.arange(5,dtype=float),.1,.5,CONFIG['a'],1.,True,1e-6,10,False,8,np.zeros(5))
    results=[]
    # Replications interleaved across all settings, preserving useful checkpoints.
    for seed in CONFIG['seeds']:
      for n,p in CONFIG['dimensions']:
       for design in CONFIG['designs']:
        results.append(case(n,p,design,seed));assemble(results)
        if limit and len(results)>=limit:break
       if limit and len(results)>=limit:break
      if limit and len(results)>=limit:break
    inputs=[ROOT/'sources/rome-99d6f07.tar.gz',ROOT/'analysis/configs/rome.json']
    inputs.extend((DEST/'cases').rglob('*.tsv'));inputs.extend((DEST/'cases').rglob('R-console.txt'))
    outputs=[DEST/'summary.json',DEST/'results.qmd',*(DEST/'cases').rglob('result.json'),*(DEST/'cases').rglob('solutions.npz')]
    save(capture(outputs,inputs=inputs,configuration=CONFIG,seed=CONFIG['seeds']),DEST/'provenance.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args();main(args.limit)
