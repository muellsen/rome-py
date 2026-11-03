"""Declared validation of three scale updates; no winner or full-study claim."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys,time,warnings
import numpy as np
from pipeline import ROOT,OUT,capture,save
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber import FastHuberCD,HuberCD
from concomitant_huber.scale_solver import solve_scale_strategy
from concomitant_huber.solver import _diagnostics

DEST=OUT/'scale-strategies'
CONFIG=dict(seed=20261102,methods=['mm','exact','prox'],fixtures=['Gaussian','Correlated','Contaminated'],
            n=45,p=8,penalty=.15,rho=1.345,a=.3550822741345242,initial_scale=1.,intercept=True,
            tol=1e-7,scale_inner_tol=5e-8,max_sweeps=500,scale_max_iter=2000,mm_steps=8,
            prox_gamma=1.,newton_every=0,check_every=5,timing_repeats=3,
            order='cyclic rotation of method order across repeats',
            timer='process CPU and wall around estimator fit; JIT/reference fitting/serialization/post-fit validation excluded',
            scope='small correctness and work comparison; not a method-selection or full timing study',winner=None)


def main():
    DEST.mkdir(exist_ok=True);rng=np.random.default_rng(CONFIG['seed'])
    tiny=np.asfortranarray(np.column_stack([np.ones(6),np.eye(6)]));yy=np.arange(6.)
    for code in range(3):
        solve_scale_strategy(tiny,yy,.15,1.345,CONFIG['a'],1.,True,1e-7,3,True,8,np.zeros(7),5,0,code,5e-8,2000,1.)
    rows=[];outputs=[]
    for index,name in enumerate(CONFIG['fixtures']):
        X=rng.normal(size=(45,8))
        if name=='Correlated':X[:,1]=.95*X[:,0]+.1*rng.normal(size=45)
        y=.7+X@np.array([1.5,-.8,.6,0.,0.,0.,0.,0.])+rng.normal(size=45)
        if name=='Contaminated':y[::7]+=6.
        np.savez_compressed(DEST/(name+'-data.npz'),X=X,y=y);outputs.append(DEST/(name+'-data.npz'))
        reference=HuberCD(penalty=.15,rho=1.345,scale_penalty=CONFIG['a'],initial_scale=1.,tol=1e-9,max_iter=4000).fit(X,y)
        if not reference.converged_:raise RuntimeError('Reference did not certify')
        D=np.asfortranarray(np.column_stack([np.ones(len(y)),X]))
        for repeat in range(3):
            methods=CONFIG['methods'][repeat:]+CONFIG['methods'][:repeat]
            for method in methods:
                start=time.perf_counter();cpu=time.process_time()
                model=FastHuberCD(penalty=.15,rho=1.345,scale_penalty=CONFIG['a'],initial_scale=1.,
                    scale_update=method,tol=1e-7,scale_inner_tol=5e-8,max_iter=500,scale_max_iter=2000,
                    scale_mm_steps=8,prox_gamma=1.,newton_every=0,check_every=5).fit(X,y)
                seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
                b=np.r_[model.intercept_,model.coef_]
                cb,cs=_diagnostics(D,y-D@b,b,model.scale_,1.345,CONFIG['a'],.15,True)
                gap=abs(model.objective_-reference.objective_)/max(1.,abs(reference.objective_))
                qualified=bool(model.converged_ and max(cb,cs)<=1e-7 and gap<=1e-8)
                item=dict(fixture=name,repeat=repeat,method=method,qualified=qualified,
                    cpu_seconds=cpu_seconds,wall_seconds=seconds,coefficient_kkt=float(cb),scale_kkt=float(cs),
                    objective=float(model.objective_),reference_relative_objective_gap=float(gap),
                    maximum_coefficient_difference=float(max(abs(model.coef_-reference.coef_))),
                    scale_difference=float(abs(model.scale_-reference.scale_)),scale=float(model.scale_),
                    sweeps=int(model.n_sweeps_),scale_inner_iterations=int(model.scale_inner_iterations_),
                    inexact_scale_updates=int(model.inexact_scale_updates_),
                    maximum_objective_increase=float(max(np.diff(model.objective_history_))))
                np.savez_compressed(DEST/f'{name}-{repeat}-{method}.npz',coefficients=b,scale=model.scale_,objective=model.objective_history_)
                rows.append(item);print(name,repeat,method,qualified,item['sweeps'],item['scale_inner_iterations'],round(cpu_seconds*1000,3),flush=True)
    outputs.extend(DEST.glob('*-mm.npz'));outputs.extend(DEST.glob('*-exact.npz'));outputs.extend(DEST.glob('*-prox.npz'))
    save(dict(configuration=CONFIG,results=rows),DEST/'summary.json')
    lines=['| Fixture | Update | CPU ms | Sweeps | Inner iterations | Qualified |',
           '|:--|:--|--:|--:|--:|:--|']
    for fixture in CONFIG['fixtures']:
        for method in CONFIG['methods']:
            rr=[r for r in rows if r['fixture']==fixture and r['method']==method]
            cpu_ms=1000*np.median([r['cpu_seconds'] for r in rr])
            sweeps=int(np.median([r['sweeps'] for r in rr]))
            inner=int(np.median([r['scale_inner_iterations'] for r in rr]))
            qualified=sum(r['qualified'] for r in rr)
            lines.append(f"| {fixture} | {method} | {cpu_ms:.2f} | {sweeps} | {inner} | {qualified}/3 |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    count=sum(r['qualified'] for r in rows)
    beta=max(r['maximum_coefficient_difference'] for r in rows);scale=max(r['scale_difference'] for r in rows)
    text=(f'{count} of {len(rows)} validation fits satisfy the common joint residual and objective requirements. '
          f'Maximum coefficient and scale differences from the independently run reference solver are {beta:.3g} and {scale:.3g}. '
          'Inner work counts have different meanings: scalar MM steps, scalar bisection steps, or product-space DR iterations, each containing local joint proxs. '
          'These three small fixtures validate the implementations; they do not select a winning method. Historical timing and CV results in the following sections use the recorded MM backend.\n')
    # Mathematical scientific notation is rendered consistently in both formats.
    import re
    text=re.sub(r'([0-9.]+)e-([0-9]+)',lambda m:'$'+m[1]+r'\times10^{-'+str(int(m[2]))+'}$',text)
    (DEST/'findings.qmd').write_text(text)
    outputs.extend(DEST/n for n in ['summary.json','results.qmd','findings.qmd'])
    save(capture(outputs,configuration=CONFIG,seed=CONFIG['seed']),DEST/'provenance.json')

if __name__=='__main__':main()
