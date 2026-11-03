"""Sweep diagnostics and finite-sample AMP audit; no solver timing claims."""
import ast
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from concomitant_huber import HuberCD
from concomitant_huber.diagnostics import fit_trace
from concomitant_huber.solver import _objective
from pipeline import OUT, capture, save

spec = importlib.util.spec_from_file_location('pinned_ramp', ROOT/'sources/amp-aistats2027/perspective_huber_ramp_solver.py')
ramp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ramp
spec.loader.exec_module(ramp)
# Execute the exact snapshotted greedy filter without importing unrelated CV modules.
filter_tree=ast.parse((ROOT/'sources/amp-aistats2027/riboflavin.py').read_text())
filter_function=next(n for n in filter_tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_greedy_decorrelated_genes')
filter_namespace={'np':np,'RiboflavinData':SimpleNamespace}
exec(compile(ast.Module(body=[filter_function],type_ignores=[]),'<pinned-correlation-filter>','exec'),filter_namespace)
correlation_filter=filter_namespace['select_greedy_decorrelated_genes']
RHO = 1.345
# Analytic Gaussian consistency, shared by CD and AMP; unlike timing pilots a != 1/2.
A = .5*(1-2*RHO*np.exp(-RHO**2/2)/np.sqrt(2*np.pi)
          +2*(RHO**2-1)*(1-ndtr(RHO)))
SEED = 20261020
OUT.mkdir(exist_ok=True)
arrays, rows, traces = {}, [], []


def diagnostics(X, y, beta, sigma, penalty):
    if sigma <= 0 or not np.isfinite(sigma) or not np.isfinite(beta).all():
        return dict(objective=None, coefficient_residual=None, scale_residual=None)
    r = y-X@beta
    psi = np.clip(r/sigma, -RHO, RHO)
    gradient = -X.T@psi/len(y)
    prox = beta-np.sign(beta-gradient)*np.maximum(abs(beta-gradient)-penalty, 0)
    return dict(objective=float(_objective(r,beta,sigma,RHO,A,penalty)),
                coefficient_residual=float(np.max(abs(prox))),
                scale_residual=float(abs(A-.5*np.mean(psi**2))))


def cd(X, y, lam, *, fixed=False, scale=1., trace=False, tol=1e-7):
    m = HuberCD(penalty=lam, rho=RHO, scale_penalty=A, fit_intercept=False,
                estimate_scale=not fixed, initial_scale=scale, tol=tol, max_iter=10000)
    with warnings.catch_warnings(record=True):
        t = fit_trace(m,X,y) if trace else None
        if not trace: m.fit(X,y)
    return m,t


def amp_case(name,X,y,lam,reference,omega,label,test=None):
    record = dict(case=name,method='Perspective-RAMP',penalty=lam,omega=omega,
                  omega_rule=label,n=len(y),p=X.shape[1],a=A)
    try:
        b,h = ramp.run_perspective_huber_ramp(X/np.sqrt(len(y)), y,
            lam=np.sqrt(len(y))*lam, omega=omega, iterations=1000, a=A,
            params=ramp.HuberSpec(c=RHO,gamma=1.),
            convergence=ramp.SolverConvergence(max_iterations=1000))
        b = b/np.sqrt(len(y)); sigma = h[-1]['sigma']
        d = diagnostics(X,y,b,sigma,lam)
        gap = ((d['objective']-reference.objective_)/max(1.,abs(reference.objective_)))
        record.update(d,scale=sigma,iterations=len(h),native_converged=bool(h[-1]['converged']),
            q=h[-1]['q'],q_minus_omega=h[-1]['q']-omega,relative_objective_gap=gap,
            qualified=bool(d['coefficient_residual']<=1e-5 and d['scale_residual']<=1e-5 and abs(gap)<=1e-8))
        record['status'] = 'qualified' if record['qualified'] else 'accuracy_failed'
        arrays[f'{name}_{len(rows)}_amp_beta']=b
        # Save all iterate diagnostics, including failed qualification.
        record['history']=h
        if test is not None: record['test_mse']=float(np.mean((test[1]-test[0]@b)**2))
    except (RuntimeError,FloatingPointError,ValueError) as e:
        record.update(status='numerical_failure',qualified=False,error=str(e))
    rows.append(record)
    print(name,label,record['status'],record.get('coefficient_residual'),flush=True)


def prepare(X,y,train,test):
    mean=X[train].mean(0); scale=X[train].std(0); scale[scale==0]=1
    ym=y[train].mean(); ys=y[train].std()
    return ((X[train]-mean)/scale,(y[train]-ym)/ys,
            (X[test]-mean)/scale,(y[test]-ym)/ys,
            dict(x_mean=mean,x_scale=scale,y_mean=np.array(ym),y_scale=np.array(ys)))


def reference_record(name,X,y,lam,m,test=None,method='MM-CD'):
    row=dict(case=name,method=method,n=len(y),p=X.shape[1],penalty=lam,a=A,
             scale=m.scale_,iterations=m.n_iter_,status=m.status_,qualified=bool(m.converged_),
             **diagnostics(X,y,m.coef_,m.scale_,lam))
    if test is not None:row['test_mse']=float(np.mean((test[1]-test[0]@m.coef_)**2))
    rows.append(row)


rng=np.random.default_rng(SEED)
for design in ('Gaussian','AR08-t3'):
    n,p=120,240
    X=rng.normal(size=(n,p))
    if design=='AR08-t3':
        for j in range(1,p):X[:,j]=.8*X[:,j-1]+.6*X[:,j]
    X-=X.mean(0);X/=np.sqrt(np.mean(X**2,axis=0))
    truth=np.zeros(p);truth[:5]=[2.,1.5,-1.,.8,-.5]
    noise=rng.normal(size=n) if design=='Gaussian' else rng.standard_t(3,n)/np.sqrt(3)
    y=X@truth+noise;y-=y.mean()
    sigma0=float(np.median(abs(y-np.median(y)))/.67448975)
    lam=.4*max(abs(X.T@np.clip(y/sigma0,-RHO,RHO)/n))
    m,t=cd(X,y,lam,scale=sigma0,trace=True,tol=1e-9)
    reference_record(design,X,y,lam,m)
    arrays.update({f'{design}_X':X,f'{design}_y':y,f'{design}_truth':truth})
    for key,value in t.items():arrays[f'{design}_{key}']=value
    traces.append((design,X,y,lam,m,t))
    # Reference-informed omega is an explicit diagnostic, not a usable tuning rule.
    k=max(1,int(np.sum(abs(m.coef_)>1e-8)))
    for count,label in ((3,'prespecified_3'),(10,'prespecified_10'),(k,'reference_support_diagnostic')):
        amp_case(design,X,y,lam,m,count/p,label)

mat=loadmat(ROOT/'sources/data/riboflavinData.mat')['riboData']
datasets=(('Glass',np.loadtxt(ROOT/'sources/data/glass-X.tsv'),
           np.loadtxt(ROOT/'sources/data/glass-y.tsv'),120),
          ('B-subtilis',mat[1:].T,mat[0],50))
for name,Xraw,yraw,ntrain in datasets:
    perm=np.random.default_rng(SEED).permutation(len(yraw)); train,test=perm[:ntrain],perm[ntrain:]
    if name=='B-subtilis':
        # Feature selection sees training X only; the same columns go to every solver.
        filtered,selected=correlation_filter(SimpleNamespace(x=Xraw[train],y=yraw[train],
                                            gene_names=np.arange(Xraw.shape[1])),keep_k=500)
        arrays[name+'_selected_original_columns']=selected
        Xraw=Xraw[:,selected]
    X,y,Xtest,ytest,pre=prepare(Xraw,yraw,train,test)
    arrays.update({f'{name}_train':train,f'{name}_test':test,f'{name}_X':X,f'{name}_y':y,
                   f'{name}_Xtest':Xtest,f'{name}_ytest':ytest})
    for k,v in pre.items():arrays[f'{name}_{k}']=v
    sigma0=float(np.median(abs(y-np.median(y)))/.67448975)
    maximum=max(abs(X.T@np.clip(y/sigma0,-RHO,RHO)/len(y)))
    for fraction in (.4,.6,.8):
        lam=fraction*maximum; tag=f'{name}-{fraction}'
        m,t=cd(X,y,lam,scale=sigma0,trace=fraction==.6,tol=1e-8)
        reference_record(tag,X,y,lam,m,(Xtest,ytest))
        fixed,_=cd(X,y,lam,fixed=True,scale=1.)
        reference_record(tag,X,y,lam,fixed,(Xtest,ytest),method='Fixed-CD-sigma1')
        arrays[tag+'_beta']=m.coef_
        if t is not None:
            traces.append((tag,X,y,lam,m,t))
            for key,value in t.items():arrays[f'{tag}_{key}']=value
        for count in (3,10,20):
            amp_case(tag,X,y,lam,m,count/X.shape[1],f'prespecified_{count}',(Xtest,ytest))

# Plot the full parameter trajectory and objective convergence, never elapsed time.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(len(traces),3,figsize=(9,2.15*len(traces)),constrained_layout=True)
trace_summary=[]
for ax,(name,X,y,lam,m,t) in zip(axes,traces):
    steps=np.arange(len(t['sigma'])); final=m.coef_
    selected=np.argsort(abs(final))[-min(5,len(final)):]
    ax[0].plot(steps,t['sigma']);ax[0].set_ylabel(name+'\nscale');ax[0].set_xlabel('Full coefficient sweep')
    for j in selected:ax[1].plot(steps,t['beta'][:,j],label=str(j+1))
    ax[1].set_ylabel('Largest final coefficients');ax[1].set_xlabel('Full coefficient sweep')
    relative=np.linalg.norm(t['beta']-final,axis=1)/max(1.,np.linalg.norm(final))
    gap=np.maximum((t['objective']-m.objective_)/max(1.,abs(m.objective_)),1e-16)
    ax[2].semilogy(steps,np.maximum(relative,1e-16),label='Coefficient distance')
    ax[2].semilogy(steps,gap,label='Objective gap')
    ax[2].set_xlabel('Full coefficient sweep');ax[2].legend(fontsize=7)
    trace_summary.append(dict(case=name,sweeps=m.n_iter_,scale=m.scale_,status=m.status_,
        maximum_objective_increase=float(np.max(np.diff(t['objective']))),
        coefficient_kkt=m.coef_kkt_,scale_kkt=m.scale_kkt_))
fig.savefig(OUT/'convergence-traces.pdf');fig.savefig(OUT/'convergence-traces.png',dpi=180);plt.close(fig)
np.savez_compressed(OUT/'convergence-amp-data.npz',**arrays)
summary=dict(seed=SEED,a=A,rho=RHO,trace_summary=trace_summary,results=rows,
    protocol='Single prespecified holdout; three unselected penalty fractions; no CV; AMP accuracy audit, not timing benchmark')
save(summary,OUT/'convergence-amp-summary.json')
# Compact manuscript tables, with failures retained in the full JSON.
lines=['| Case | Sweeps | Final scale | Coefficient KKT | Scale KKT |','|:--|--:|--:|--:|--:|']
for t in trace_summary:lines.append(f"| {t['case']} | {t['sweeps']} | {t['scale']:.4f} | {t['coefficient_kkt']:.1e} | {t['scale_kkt']:.1e} |")
(OUT/'convergence-results.qmd').write_text('\n'.join(lines)+'\n')
lines=['| Design | AMP returns | Qualified | Numerical failures |','|:--|--:|--:|--:|']
for name in ('Gaussian','AR08-t3','Glass','B-subtilis'):
    rr=[r for r in rows if r['method']=='Perspective-RAMP' and r['case'].startswith(name)]
    lines.append(f"| {name} | {len(rr)} | {sum(r['qualified'] for r in rr)} | {sum(r['status']=='numerical_failure' for r in rr)} |")
(OUT/'amp-results.qmd').write_text('\n'.join(lines)+'\n')
lines=['| Data | Penalty fraction | MM-CD scale | MM-CD test MSE | Fixed-scale test MSE |','|:--|--:|--:|--:|--:|']
for name in ('Glass','B-subtilis'):
 for f in (.4,.6,.8):
    tag=f'{name}-{f}'
    a=next(r for r in rows if r['case']==tag and r['method']=='MM-CD')
    b=next(r for r in rows if r['case']==tag and r['method']=='Fixed-CD-sigma1')
    lines.append(f"| {name} | {f} | {a['scale']:.4f} | {a['test_mse']:.4f} | {b['test_mse']:.4f} |")
(OUT/'real-data-results.qmd').write_text('\n'.join(lines)+'\n')
outputs=[OUT/n for n in ('convergence-traces.pdf','convergence-traces.png','convergence-amp-data.npz',
                         'convergence-amp-summary.json','convergence-results.qmd','amp-results.qmd','real-data-results.qmd')]
inputs=[p for directory in ('sources/data','sources/amp-aistats2027') for p in (ROOT/directory).glob('*') if p.is_file()]
save(capture(outputs,inputs=inputs,configuration=dict(seed=SEED,a=A,rho=RHO,max_amp_iterations=1000,
    coefficient_tolerance=1e-5,scale_tolerance=1e-5,objective_tolerance=1e-8,
    train_sizes={'Glass':120,'B-subtilis':50},B_subtilis_correlation_filter=dict(keep_k=500,fit='training predictors only',method='pinned greedy maximum absolute correlation'),penalty_fractions=[.4,.6,.8],
    amp_omega_counts=[3,10,20],preprocessing='training only; predictor RMS1; response std1'),seed=SEED),
    OUT/'convergence-amp.provenance.json')
print(json.dumps(trace_summary,indent=2),flush=True)
