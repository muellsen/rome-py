"""Audited single-lambda R4 comparison, separating preparation and solving.

Parameter construction is moved outside the prepared timer. DR uses the
common external stopping gate; its updates and the path updates are unchanged.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-mpl')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import csv
import hashlib
import importlib.metadata
import json
import inspect
import time
from contextlib import contextmanager
import numpy as np
from scipy.special import huber
import classo.path_alg as path
import classo.solve_R4 as r4
from classo.solve_R3 import problem_R3, Classo_R3
from concomitant_huber.solver import _solve
from pipeline import OUT, capture, save, verify

CONFIG={'dimensions':[[80,120],[150,600],[300,1000]],'noise':['gaussian','t3'],
        'seeds':[71,72], 'repeats':3, 'rho':1.345, 'scale_penalty':.5,
        'penalty_fraction':.4, 'stationarity_target':1e-5,'relative_objective_gap_target':1e-8,
        'reference_tolerance':1e-9, 'methods':['MM-CD','DR-stock','DR-balanced','Path-Alg','Path-Alg-eps0'],
        'constraint':'zero row C=0; no coefficient restrictions', 'intercept':False,
        'timing':'prepared numerical solve; setup separate; end-to-end is setup+solve',
        'dr_max_iter':5000, 'dr_stock_iteration_budget':1000, 'dr_balanced_metric_c':1.0, 'dr_stopping':'same objective/stationarity gate as other methods, checked every 10 iterations; native iterate-change stop disabled', 'dr_tolerance_candidates':[1e-5,1e-6,1e-7],
        'path_eps':{'Path-Alg':1e-3,'Path-Alg-eps0':0.0},'threads':1}


def diagnostics(X,y,b,sigma,penalty):
    if sigma<=0 or not np.isfinite(sigma):
        return {'objective':None,'coef_residual':None,'scale_residual':None}
    r=y-X@b
    psi=np.clip(r/sigma,-CONFIG['rho'],CONFIG['rho'])
    g=-X.T@psi/len(y)
    z=b-g
    prox=np.sign(z)*np.maximum(abs(z)-penalty,0)
    return {'objective':float(np.mean(sigma*huber(CONFIG['rho'],r/sigma))+.5*sigma+penalty*np.sum(abs(b))),
            'coef_residual':float(np.max(abs(b-prox))),
            'scale_residual':float(abs(.5-.5*np.mean(psi**2)))}


class Identity:
    def dot(self,x):
        return x


@contextmanager
def cached_path_parameter(parameter):
    original=path.parameters_for_update
    calls=[]
    def prepared(*args,**kwargs):
        calls.append(1)
        if len(calls)!=1:
            raise RuntimeError('Unexpected additional path initialization')
        return parameter
    path.parameters_for_update=prepared
    try:
        yield calls
    finally:
        path.parameters_for_update=original


def common_stop_dr(pb,normalized_lambda,gate):
    source=inspect.getsource(r4.Classo_R4)
    old='if i > 0 and LA.norm(b - nv_b) * Anorm + LA.norm(s - nv_s) < 2 * tol:'
    if source.count(old)!=1 or source.count('s, b = nv_s, nv_b')!=1:
        raise RuntimeError('Audited c-lasso source changed')
    source=source.replace(old,'if False:')
    source=source.replace('s, b = nv_s, nv_b',
        's, b = nv_s, nv_b\n            if i % 10 == 0 and _benchmark_gate(b, sum(s) / len(s), i + 1):\n                return b, sum(s) / len(s)')
    namespace=dict(r4.__dict__);namespace['_benchmark_gate']=gate
    exec(compile(source,'<audited-r4-common-stop>','exec'),namespace)
    function=namespace['Classo_R4']
    return lambda:function(pb,normalized_lambda)


def prepare(method,X,y,penalty,initial,dr_tol=1e-5,reference=None):
    """All parameter/model construction and JIT compilation lie outside solve timing."""
    n,p=X.shape
    if method=='MM-CD':
        design=np.asfortranarray(X)
        def run():
            result=_solve(design,y,penalty,1.345,.5,initial,True,1e-5,3000,False,8)
            if not result[4]: raise RuntimeError('MM-CD did not converge')
            return result[0],result[1]
        return run
    C=np.zeros((1,p))
    absolute_lambda=2*n*penalty
    if method.startswith('DR'):
        pb=r4.problem_R4((X,C,y),'DR',1.345,intercept=False)
        pb.tol=dr_tol; pb.N=CONFIG['dr_stock_iteration_budget'] if method=='DR-stock' else CONFIG['dr_max_iter']
        if method=='DR-balanced':pb.c=CONFIG['dr_balanced_metric_c']
        normalized_lambda=absolute_lambda/pb.lambdamax
        if normalized_lambda<1e-5: raise RuntimeError('Would trigger silent path dispatch')
        pb.compute_param()
        if method=='DR-identity':pb.Proj=Identity()
        # Keep regpath=False and its ordinary output; only skip already completed setup.
        pb.compute_param=lambda:None
        if reference is None:
            return lambda:r4.Classo_R4(pb,normalized_lambda)
        def gate(b,s,iteration):
            return qualified(diagnostics(X,y,b,s,penalty),reference)
        return common_stop_dr(pb,normalized_lambda,gate)
    # Algebraically identical to Classo_R4's mean-shift path branch.
    # Do not time the unused R4 constructor's spectral norm for this method.
    A=np.sqrt(2)*np.column_stack((X,absolute_lambda/(2*1.345)*np.eye(n)))
    C_aug=np.zeros((1,p+n)); yy=np.sqrt(2)*y
    pb=problem_R3((A,C_aug,yy),'Path-Alg')
    ratio=absolute_lambda/pb.lambdamax
    if ratio<1e-5: raise RuntimeError('Would trigger unpenalized fallback')
    parameter=path.parameters_for_update((A,C_aug,yy),0,0,'R3',eps_L2=CONFIG['path_eps'][method])
    def run():
        with cached_path_parameter(parameter):
            b,s=Classo_R3(pb,ratio)
        return b[:-n],s/2
    return run


def qualified(d,reference):
    if d['objective'] is None:return False
    gap=abs(d['objective']-reference)/max(1,abs(reference))
    return max(d['coef_residual'],d['scale_residual'])<=CONFIG['stationarity_target'] and gap<=CONFIG['relative_objective_gap_target']


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter()
    _solve(np.asfortranarray(np.eye(4)),np.arange(4,dtype=float),.5,1.345,.5,1.,True,1e-5,100,False,8)
    compile_seconds=time.perf_counter()-t
    rows=[];calibration=[];arrays={}
    for n,p in CONFIG['dimensions']:
        for noise in CONFIG['noise']:
            for seed in CONFIG['seeds']:
                rng=np.random.default_rng(seed)
                X=rng.normal(size=(n,p));X-=X.mean(0);X/=np.sqrt(np.mean(X**2,0))
                truth=np.zeros(p);truth[:min(10,p)]=rng.choice([-1.,1.],min(10,p))*rng.uniform(1,2,min(10,p))
                eps=rng.normal(size=n) if noise=='gaussian' else rng.standard_t(3,n)/np.sqrt(3)
                y=X@truth+eps
                initial=max(float(np.median(abs(y-np.median(y)))/.67448975),1e-6)
                penalty=.4*float(max(abs(X.T@np.clip(y/initial,-1.345,1.345)))/n)
                ref=_solve(np.asfortranarray(X),y,penalty,1.345,.5,initial,True,1e-9,10000,False,8)
                if not ref[4]:raise RuntimeError('Uncertified reference')
                reference=diagnostics(X,y,ref[0],ref[1],penalty)['objective']
                prefix=f'{n}_{p}_{noise}_{seed}'
                arrays[prefix+'_X']=X;arrays[prefix+'_y']=y;arrays[prefix+'_reference_beta']=ref[0]
                print(f'Start {n}x{p} {noise} seed={seed}',flush=True)
                choices={}
                returned={}
                for method in CONFIG['methods']:
                    candidates=[None]
                    chosen=None
                    returned[method]=False
                    for tol in candidates:
                        setup_ms=None;solver_start=None
                        try:
                            start=time.perf_counter_ns()
                            f=prepare(method,X,y,penalty,initial,dr_tol=tol or 1e-5,reference=reference)
                            setup_ms=(time.perf_counter_ns()-start)/1e6
                            solver_start=time.perf_counter_ns()
                            b,s=f();elapsed=(time.perf_counter_ns()-solver_start)/1e6
                            returned[method]=True
                            d=diagnostics(X,y,b,s,penalty)
                            ok=qualified(d,reference)
                            calibration.append(dict(n=n,p=p,noise=noise,seed=seed,method=method,tolerance=tol,qualified=ok,setup_ms=setup_ms,solver_ms=elapsed,**d))
                            if ok:chosen=tol;break
                        except Exception as e:
                            elapsed=(time.perf_counter_ns()-solver_start)/1e6 if solver_start is not None else None
                            calibration.append(dict(n=n,p=p,noise=noise,seed=seed,method=method,tolerance=tol,qualified=False,error=str(e),setup_ms=setup_ms,solver_ms=elapsed))
                            rows.append(dict(n=n,p=p,noise=noise,seed=seed,repeat=-1,method=method,penalty=penalty,qualified=False,error=str(e),setup_ms=setup_ms,solver_ms=elapsed,total_ms=setup_ms+elapsed if setup_ms is not None and elapsed is not None else None))
                    print(f'  {method}: returned={returned[method]}',flush=True)
                    choices[method]=chosen
                print(f'  timing repeated successful returns',flush=True)
                for repeat in range(CONFIG['repeats']):
                    order=CONFIG['methods'].copy()
                    if repeat%2:order.reverse()
                    for method in order:
                        if not returned[method]:continue
                        row=dict(n=n,p=p,noise=noise,seed=seed,repeat=repeat,method=method,penalty=penalty,
                                 dr_tolerance=choices[method] if method.startswith('DR') else None)
                        try:
                            t=time.perf_counter_ns()
                            f=prepare(method,X,y,penalty,initial,dr_tol=choices[method] or 1e-5,reference=reference)
                            setup_ms=(time.perf_counter_ns()-t)/1e6
                            t=time.perf_counter_ns();b,s=f();solver_ms=(time.perf_counter_ns()-t)/1e6
                            d=diagnostics(X,y,b,s,penalty)
                            row.update(setup_ms=setup_ms,solver_ms=solver_ms,total_ms=setup_ms+solver_ms,
                                       scale=float(s),qualified=qualified(d,reference),
                                       relative_objective_gap=abs(d['objective']-reference)/max(1,abs(reference)),
                                       prediction_max_error=float(max(abs(X@(b-ref[0])))),**d)
                            arrays[prefix+'_'+method+'_beta']=b
                        except Exception as e:
                            row.update(qualified=False,error=str(e))
                        rows.append(row)
                # Flush progress in case a subsequent algorithm fails.
                save({'config':CONFIG,'calibration':calibration,'timings':rows},OUT/'r4-progress.json')
    raw=OUT/'r4-timings.csv'
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with raw.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    data=OUT/'r4-data.npz';np.savez_compressed(data,**arrays)
    summary=[]
    for n,p in CONFIG['dimensions']:
        for noise in CONFIG['noise']:
            for method in CONFIG['methods']:
                rr=[r for r in rows if r['n']==n and r['noise']==noise and r['method']==method]
                successes=[r for r in rr if r.get('qualified')]
                summary.append(dict(n=n,p=p,noise=noise,method=method,qualified=len(successes),total=len(rr),
                                    median_solver_ms=float(np.median([r['solver_ms'] for r in rr if 'solver_ms' in r])),
                                    median_setup_ms=float(np.median([r['setup_ms'] for r in rr if 'setup_ms' in r])),
                                    median_total_ms=float(np.median([r['total_ms'] for r in rr if 'total_ms' in r])),
                                    max_residual=max([max(r['coef_residual'],r['scale_residual']) for r in rr if r.get('objective') is not None],default=None)))
    report={'config':CONFIG,'package_versions':{k:importlib.metadata.version(k) for k in ('c-lasso','numpy','numba','scipy')},
            'jit_first_call_seconds':compile_seconds,'calibration':calibration,'summary':summary}
    dest=OUT/'r4-summary.json';save(report,dest)
    manifest=capture([raw,data,dest],configuration=CONFIG,seed=CONFIG['seeds'],inputs=[Path(__file__).resolve().parents[1]/'sources/classo-1.0.11-audit/registry.json'])
    save(manifest,OUT/'r4.provenance.json');verify(manifest)
    for r in summary:print(r,flush=True)

if __name__=='__main__':run()
