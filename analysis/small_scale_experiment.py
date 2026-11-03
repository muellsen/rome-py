"""Paired timing of joint MM/CD and fixed-scale exact CD; not a speed guarantee."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import csv
import json
import time
import platform
import numpy as np
from concomitant_huber import HuberCD
from pipeline import OUT, capture, save, verify

CONFIG = {'dimensions': [[150,600],[300,1000]], 'noise': ['gaussian','t3','contaminated'],
          'seeds': [71,72,73], 'repeats': 7, 'rho': 1.345, 'scale_penalty': .5,
          'penalty_fraction': .4, 'tol': 1e-7, 'max_iter': 3000,
          'intercept': True, 'standardization': 'centered, unit RMS columns',
          'fixed_baseline': 'scale fixed at converged joint estimate; same coefficient optimum',
          'initialization': 'zero coefficients; same initial scale for both joint methods and initial-fixed control',
          'timing': 'fit wall time, warmed Numba, alternating method order; compilation reported separately',
          'scale_mm_steps': [1,8], 'threads': 1, 'protocol_revision': 'Initial fraction .15 reached near-zero scale and failed KKT; retained in pilot-initial-failure.json. Fraction .4 selected for positive-scale pilot, not on timing.'}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    warm_x=np.arange(40,dtype=float).reshape(10,4)/10
    warm_y=np.sin(np.arange(10))
    t=time.perf_counter()
    for joint in (True,False):
        HuberCD(penalty=.3,estimate_scale=joint,tol=1e-6).fit(warm_x,warm_y)
    compile_s=time.perf_counter()-t
    rows=[]
    arrays={}
    for n,p in CONFIG['dimensions']:
        for noise in CONFIG['noise']:
            for seed in CONFIG['seeds']:
                rng=np.random.default_rng(seed)
                X=rng.normal(size=(n,p)); X-=X.mean(axis=0); X/=np.sqrt(np.mean(X**2,axis=0))
                beta=np.zeros(p); beta[:10]=rng.choice([-1.,1.],10)*rng.uniform(1,2,10)
                eps=rng.normal(size=n) if noise!='t3' else rng.standard_t(3,n)/np.sqrt(3)
                if noise=='contaminated':
                    idx=rng.choice(n,max(1,n//10),replace=False); eps[idx]+=rng.normal(0,10,len(idx))
                y=1+X@beta+eps
                initial=max(float(np.median(abs(y-np.median(y)))/.67448975),1e-6)
                psi=np.clip((y-np.median(y))/initial,-CONFIG['rho'],CONFIG['rho'])
                penalty=CONFIG['penalty_fraction']*float(np.max(abs(X.T@psi))/n)
                args=dict(penalty=penalty,rho=CONFIG['rho'],scale_penalty=CONFIG['scale_penalty'],tol=CONFIG['tol'],max_iter=CONFIG['max_iter'])
                joint=HuberCD(**args,initial_scale=initial).fit(X,y)
                if not joint.converged_: raise RuntimeError(f'Joint pilot failed: {n}, {p}, {noise}, {seed}')
                print(f'Pilot {n}x{p} {noise} seed={seed}: {joint.n_iter_} sweeps, scale={joint.scale_:.4g}', flush=True)
                def fit(method):
                    return HuberCD(**args,initial_scale=initial if method!='fixed_at_joint_scale' else joint.scale_,estimate_scale=method.startswith('joint'),scale_mm_steps=1 if method=='joint_mm1' else 8).fit(X,y)
                for repeat in range(CONFIG['repeats']):
                    methods=['joint_mm1','joint_mm8','fixed_at_joint_scale','fixed_at_initial_scale']
                    if repeat%2: methods.reverse()
                    for method in methods:
                        start=time.perf_counter_ns(); model=fit(method); elapsed=(time.perf_counter_ns()-start)/1e6
                        if not model.converged_: raise RuntimeError(f'{method} failed')
                        diff=float(np.max(abs(model.predict(X)-joint.predict(X))))
                        if method=='fixed_at_joint_scale' and diff>1e-4: raise RuntimeError('Coefficient solutions disagree')
                        if np.max(np.diff(model.objective_history_))>1e-9: raise RuntimeError('Descent failure')
                        rows.append(dict(n=n,p=p,noise=noise,seed=seed,repeat=repeat,method=method,milliseconds=elapsed,
                                         sweeps=model.n_iter_,scale=float(model.scale_),penalty=penalty,coef_kkt=float(model.coef_kkt_),
                                         scale_kkt=float(model.scale_kkt_) if method.startswith('joint') else None,
                                         objective=model.objective_,prediction_difference_from_joint=diff))
                key=f'{n}_{p}_{noise}_{seed}'
                arrays[key+'_X']=X; arrays[key+'_y']=y; arrays[key+'_truth']=beta
                arrays[key+'_joint_beta']=joint.coef_; arrays[key+'_history']=joint.objective_history_
    raw=OUT/'small-scale-timings.csv'
    with raw.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    data=OUT/'small-scale-data.npz';np.savez_compressed(data,**arrays)
    summaries=[]
    for n,p in CONFIG['dimensions']:
        for noise in CONFIG['noise']:
            medians={}
            ratios=[]
            for method in ('joint_mm1','joint_mm8','fixed_at_joint_scale','fixed_at_initial_scale'):
                medians[method]=float(np.median([r['milliseconds'] for r in rows if r['n']==n and r['noise']==noise and r['method']==method]))
            for seed in CONFIG['seeds']:
                med={m:float(np.median([r['milliseconds'] for r in rows if r['n']==n and r['noise']==noise and r['seed']==seed and r['method']==m])) for m in medians}
                ratios.append(med['joint_mm8']/med['fixed_at_joint_scale'])
            summaries.append(dict(n=n,p=p,noise=noise,median_ms=medians,median_paired_ratio=float(np.median(ratios)),seed_ratios=ratios))
    summary=OUT/'small-scale-summary.json'
    summary.write_text(json.dumps({'config':CONFIG,'compile_and_first_calls_seconds':compile_s,'hardware':platform.processor(),'results':summaries},indent=2)+'\n')
    lines=['| n | p | Noise | Fixed scale (ms) | Joint MM1 (ms) | Joint MM8 (ms) | MM8/fixed paired ratio |','|---:|---:|:---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['n']} | {r['p']} | {r['noise']} | {r['median_ms']['fixed_at_joint_scale']:.2f} | {r['median_ms']['joint_mm1']:.2f} | {r['median_ms']['joint_mm8']:.2f} | {r['median_paired_ratio']:.2f} |")
    report=OUT/'small-scale-results.qmd';report.write_text('\n'.join(lines)+'\n\nTimes exclude JIT compilation. Three data seeds and seven repetitions per seed;\nratios are medians of within-seed median ratios. The fixed-scale baseline uses\nthe estimated scale and reaches the same coefficient optimum. The fixed-at-initial-scale\ncontrol solves a different problem and is recorded separately in the raw table.\nThese are pilot results, not evidence of universal speed parity.\n')
    manifest=capture([raw,data,summary,report,OUT/'pilot-initial-failure.json'],configuration=CONFIG,seed=CONFIG['seeds'])
    save(manifest,OUT/'small-scale.provenance.json');verify(manifest)
    print(report.read_text());print(f'Compilation/first calls: {compile_s:.2f} seconds')

if __name__=='__main__': run()
