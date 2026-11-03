"""Repeated held-out evaluation with inner five-fold CV; no test-set preprocessing."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
from pathlib import Path
import json,subprocess,time,sys
import numpy as np
from scipy.io import loadmat
from scipy.special import ndtr
from pipeline import ROOT,OUT,capture,save
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.model_selection import TrainingTransform,standardized_path
RHO=1.345
A=float(.5*(1-2*RHO*np.exp(-RHO**2/2)/np.sqrt(2*np.pi)+2*(RHO**2-1)*(1-ndtr(RHO))))
FRACTIONS=np.exp(np.linspace(0,np.log(.001),50))
DEST=OUT/'revision-real'
CONFIG=dict(outer_repeats=10,folds=5,seed=20261030,rho=RHO,a=A,fractions=FRACTIONS.tolist(),
            train_size={'Glass':120,'B-subtilis':50},B_subtilis_filter=500,
            selection='minimum pooled inner-fold MSE; only points certified in every joint fold',
            preprocessing='fit X/y centering and scaling and B.subtilis predictor filter independently inside each inner fold',
            intercept='unpenalized, optimized',penalty='lasso alpha=1 for both methods',
            real_data_change='lasso adaptation; original glass application used elastic-net alpha=.3',
            solver_tol=1e-6,max_sweeps=4000,reference='fixed-scale exact CD in the same Python kernel; original ROME objective; no implementation-language confounding')


def fitted_paths(X,y,columns,directory):
    directory.mkdir(parents=True,exist_ok=True)
    anchor=float(max(abs(X.T@(np.sign(y)*np.minimum(abs(y),RHO))))/len(y))
    penalties=anchor*FRACTIONS
    for name,value in [('X',X),('y',y),('lambda',penalties)]:
        np.savetxt(directory/(name+'.tsv'),value,fmt='%.17g',delimiter='\t')
    np.save(directory/'columns.npy',columns)
    metadata=directory/'fits.json'
    if metadata.exists():
        info=json.loads(metadata.read_text())
        if info['configuration']!=CONFIG:raise ValueError('CV checkpoint configuration mismatch')
        return np.load(directory/'fits.npz'),info
    start=time.perf_counter()
    joint=standardized_path(X,y,penalties,rho=RHO,a=A,tol=1e-6,max_iter=4000)
    joint_time=time.perf_counter()-start
    start=time.perf_counter()
    fixed=standardized_path(X,y,penalties,rho=RHO,a=A,tol=1e-6,max_iter=4000,estimate_scale=False)
    full=fixed['coefficients'];rtime=time.perf_counter()-start;error=None
    # Record independent fixed-scale stationarity; never hide native accuracy.
    D=np.column_stack([np.ones(len(y)),X])
    r_kkt=np.full(len(penalties),np.nan)
    for k,b in enumerate(full):
        if not np.isfinite(b).all():continue
        z=y-D@b;g=-D.T@(np.sign(z)*np.minimum(abs(z),RHO))/len(y)
        lam=np.r_[0.,np.full(len(b)-1,penalties[k])]
        prox=np.sign(b-g)*np.maximum(abs(b-g)-lam,0.)
        r_kkt[k]=max(abs(b-prox))
    np.savez_compressed(directory/'fits.npz',rome=full,joint=joint['coefficients'],
                        sigma=joint['scales'],joint_kkt=joint['kkt'],rome_kkt=r_kkt,penalties=penalties)
    info=dict(configuration=CONFIG,fixed_status=fixed['status'],joint_status=joint['status'],joint_wall_seconds=joint_time,
              rome_wall_seconds=rtime,rome_error=error)
    save(info,metadata)
    return np.load(directory/'fits.npz'),info


def run_case(name,X,y,outer):
    directory=DEST/name/f'split-{outer:02d}';directory.mkdir(parents=True,exist_ok=True)
    final=directory/'result.json'
    if final.exists():return json.loads(final.read_text())
    rng=np.random.default_rng(CONFIG['seed']+outer)
    permutation=rng.permutation(len(y))
    train=permutation[:CONFIG['train_size'][name]];test=permutation[len(train):]
    fold_order=np.random.default_rng(CONFIG['seed']+1000+outer).permutation(len(train))
    folds=np.array_split(fold_order,5)
    keep=500 if name=='B-subtilis' else None
    losses={m:np.full((5,len(FRACTIONS)),np.nan) for m in ('rome','joint')}
    eligibility={m:np.zeros((5,len(FRACTIONS)),bool) for m in losses}
    state=dict(train=train,test=test,fold_assignment=np.array([next(k for k,f in enumerate(folds) if i in f) for i in range(len(train))]))
    for fold,validation in enumerate(folds):
        subtrain=np.setdiff1d(np.arange(len(train)),validation)
        ii,jj=train[subtrain],train[validation]
        transform=TrainingTransform.fit(X[ii],y[ii],keep=keep)
        fits,info=fitted_paths(transform.design(X[ii]),transform.response(y[ii]),transform.columns,directory/f'fold-{fold}')
        for method in losses:
            beta=fits[method]
            prediction=transform.y_mean+transform.y_scale*(beta[:,0,None]+beta[:,1:]@transform.design(X[jj]).T)
            losses[method][fold]=np.mean((prediction-y[jj])**2,axis=1)
            eligibility[method][fold]=np.isfinite(beta).all(1)
            eligibility[method][fold]&=np.array(info['joint_status'] if method=='joint' else info['fixed_status'])=='converged'
        for key in ('columns','x_mean','x_scale'):
            state[f'fold{fold}_{key}']=getattr(transform,key)
    # Joint fractions require convergence in all inner fits. ROME fractions require native returns.
    # No asymmetric removal of low-error folds and no validation/test rows in transforms.
    selected={}
    for method in losses:
        eligible=eligibility[method].all(0)
        pooled=np.average(losses[method],axis=0,weights=[len(v) for v in folds])
        pooled[~eligible]=np.inf
        selected[method]=int(np.argmin(pooled)) if eligible.any() else None
        state[method+'_cv_loss']=losses[method];state[method+'_eligible']=eligibility[method]
    transform=TrainingTransform.fit(X[train],y[train],keep=keep)
    fits,info=fitted_paths(transform.design(X[train]),transform.response(y[train]),transform.columns,directory/'refit')
    output=dict(dataset=name,split=outer,configuration=CONFIG,methods={})
    for method,k in selected.items():
        if k is None:
            output['methods'][method]=dict(status='no_eligible_cv_point');continue
        beta=fits[method][k]
        status='returned' if np.isfinite(beta).all() else 'refit_failed'
        if (info['joint_status'] if method=='joint' else info['fixed_status'])[k]!='converged':status='refit_failed'
        record=dict(status=status,selected_index=k,fraction=float(FRACTIONS[k]),
                    penalty=float(fits['penalties'][k]),eligible_cv_points=int(eligibility[method].all(0).sum()))
        if status=='returned':
            prediction=transform.predict(X[test],beta)
            state[method+'_prediction']=prediction;state[method+'_beta']=beta
            record.update(mse=float(np.mean((prediction-y[test])**2)),
                normalized_mse=float(np.mean((prediction-y[test])**2)/transform.y_scale**2),
                mae=float(np.mean(abs(prediction-y[test]))),
                nonzero=int(sum(abs(beta[1:])>1e-8)),
                scale=float(fits['sigma'][k]*transform.y_scale) if method=='joint' else transform.y_scale,
                coefficient_kkt=float(fits['joint_kkt'][k,0] if method=='joint' else fits['rome_kkt'][k]))
        output['methods'][method]=record
    state.update(test_y=y[test],columns=transform.columns,x_mean=transform.x_mean,
                 x_scale=transform.x_scale,y_mean=np.array(transform.y_mean),y_scale=np.array(transform.y_scale))
    np.savez_compressed(directory/'evaluation.npz',**state);save(output,final)
    print(name,outer,output['methods'],flush=True)
    return output


def main():
    DEST.mkdir(exist_ok=True)
    # Warm compilation outside all recorded fit durations.
    standardized_path(np.eye(6),np.arange(6.),[.5],rho=RHO,a=A,max_iter=5)
    mat=loadmat(ROOT/'sources/data/riboflavinData.mat')['riboData']
    data={'Glass':(np.loadtxt(ROOT/'sources/data/glass-X.tsv'),np.loadtxt(ROOT/'sources/data/glass-y.tsv')),
          'B-subtilis':(mat[1:].T,mat[0])}
    rows=[]
    for outer in range(10):
      for name,(X,y) in data.items():
        rows.append(run_case(name,X,y,outer))
        save(dict(configuration=CONFIG,results=rows),DEST/'summary.json')
    lines=['| Data | Method | Successful splits | Test normalized MSE | Nonzeros |',
           '|:--|:--|--:|--:|--:|']
    for name in data:
      for method,label in [('rome','Fixed-scale CD'),('joint','Concomitant CD')]:
        rr=[r['methods'][method] for r in rows if r['dataset']==name and r['methods'][method]['status']=='returned']
        fmt=lambda key:f"{np.mean([r[key] for r in rr]):.3f} ({np.std([r[key] for r in rr],ddof=1):.3f})" if len(rr)>1 else '—'
        lines.append(f"| {name} | {label} | {len(rr)}/10 | {fmt('normalized_mse')} | {fmt('nonzero')} |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    inputs=[ROOT/'sources/data'/n for n in ('glass-X.tsv','glass-y.tsv','riboflavinData.mat')]
    outputs=[DEST/'summary.json',DEST/'results.qmd',*DEST.rglob('result.json'),*DEST.rglob('evaluation.npz'),*DEST.rglob('fits.npz'),*DEST.rglob('fits.json')]
    save(capture(outputs,inputs=inputs,configuration=CONFIG,seed=CONFIG['seed']),DEST/'provenance.json')

if __name__=='__main__':main()
