"""A declared ROME-design comparison with native and objective-matched adaHuber.

The native adaptive routine is unchanged. The matched R adapter alternates
an exact scalar scale solve with adaHuber's compiled coefficient steps and
updates eta=sigma*lambda0. All numerical audits are outside the timers.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/rome-py-mpl')
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT,OUT,capture,save,verify
from rome_paper_experiment import generate
sys.path.insert(0,str(ROOT/'src'))
from rome_py import FastHuberCD
from rome_py.qualified_path import accuracy
from concomitant_huber.diagnostics import fit_trace
from rome_py import HuberCD
DEST=OUT/'adahuber'
CONFIG=dict(design='Compound-N',n=100,p=100,seeds=[71],fractions=[.6,.4,.2],
    rho=1.345,scale_calibration='a=rho**2*log(n*p)/(2*n), matching the intended adaptive-lasso equation',
    repeats=3,timing_batch=1,tolerance=1e-5,inner_tolerance=1e-6,
    reference_tolerance=1e-11,reference_max_iter=20000,max_outer=1000,max_inner=20000,coefficient_batch=100,
    root_tolerance=1e-11,root_max=1000,bracket_expansion=2.,phi0=.01,gamma=1.2,inner_check_every=5,
    standardization_tolerance=1e-10,wrapper_match_tolerance=1e-7,
    fit_intercept=True,coefficient_and_scale_tolerance=1e-5,objective_tolerance=1e-8,
    initial_scale='joint intercept-only null optimum',penalty='fraction of model-specific null maximum',
    native_penalty='initial_scale*lambda0, held fixed by unchanged adaHuber',
    timing='prepared numerical fit CPU and wall; native public fit measured separately; startup, compilation, input and audit excluded',
    cd_newton_every=0,scale_mm_steps=8,
    timing_order='alternate R and Python timing blocks by penalty index')


def diagnostics(X,y,beta,sigma,lam,a,reference):
    Z=np.column_stack((np.ones(len(y)),X))
    q=accuracy(Z,y,beta,sigma,lam,rho=CONFIG['rho'],a=a,estimate_scale=True,intercept=True)
    q.pop('dual_vector')
    score=np.sign(y-Z@beta)*np.minimum(abs(y-Z@beta)/sigma,CONFIG['rho'])
    gradient=-Z.T@score/len(y);weights=np.r_[0.,np.full(X.shape[1],lam)]
    shifted=beta-gradient
    continuous=float(max(abs(beta-np.sign(shifted)*np.maximum(abs(shifted)-weights,0))))
    q['continuous_coefficient_residual']=continuous
    q['joint_residual']=max(continuous,q['scale_kkt'])
    q['relative_objective_difference']=abs(q['objective']-reference.objective_)/max(1.,abs(reference.objective_))
    q['beta_difference']=float(max(abs(beta-np.r_[reference.intercept_,reference.coef_])))
    q['scale_difference']=abs(sigma-reference.scale_)
    q['prediction_rmse_difference']=float(np.sqrt(np.mean((Z@(beta-np.r_[reference.intercept_,reference.coef_]))**2)))
    q['qualified']=bool(q['joint_residual']<=CONFIG['coefficient_and_scale_tolerance'] and q['relative_objective_difference']<=CONFIG['objective_tolerance'])
    return q


def case(seed,fraction):
    from rome_py.model_selection import null_penalty_anchor
    folder=DEST/f'seed-{seed}-fraction-{fraction:g}';folder.mkdir(parents=True,exist_ok=True)
    X,y,truth,_=generate(CONFIG['n'],CONFIG['p'],CONFIG['design'],seed)
    originalX=X.copy();originaly=y.copy()
    X=np.asfortranarray((X-X.mean(0))/X.std(0,ddof=1));y=y-y.mean()
    rho=CONFIG['rho'];a=rho*rho*np.log(len(y)*X.shape[1])/(2*len(y))
    anchor=null_penalty_anchor(X,y,rho=rho,a=a)
    lam=fraction*anchor['lambda_max'];initial=anchor['scale']
    ref=FastHuberCD(penalty=lam,rho=rho,scale_penalty=a,initial_scale=initial,
        tol=CONFIG['reference_tolerance'],max_iter=CONFIG['reference_max_iter'],scale_update='exact').fit(X,y)
    if not ref.converged_ or ref.scale_<=0:raise RuntimeError('Positive-scale reference not certified')
    rq=diagnostics(X,y,np.r_[ref.intercept_,ref.coef_],ref.scale_,lam,a,ref)
    if rq['normalized_gap']>CONFIG['objective_tolerance']:raise RuntimeError('Reference dual gap too large')
    for name,value in [('X',X),('y',y)]:np.savetxt(folder/(name+'.tsv'),value,delimiter='\t',fmt='%.17g')
    controls={**{k:CONFIG[k] for k in ('rho','tolerance','inner_tolerance','max_outer','max_inner','coefficient_batch','root_tolerance',
        'root_max','bracket_expansion','phi0','gamma','repeats','timing_batch','inner_check_every','standardization_tolerance','wrapper_match_tolerance')},
        'a':a,'lambda0':lam,'sigma0':initial}
    (folder/'controls.tsv').write_text('key\tvalue\n'+''.join(f'{k}\t{v}\n' for k,v in controls.items()))
    def cd():
        return FastHuberCD(penalty=lam,rho=rho,scale_penalty=a,initial_scale=initial,
            tol=CONFIG['tolerance'],max_iter=CONFIG['reference_max_iter'],newton_every=CONFIG['cd_newton_every'],
            scale_mm_steps=CONFIG['scale_mm_steps']).fit(X,y)
    cd() # JIT/BLAS outside every timer.
    def measure_cd():
        cd_rows=[]
        for repeat in range(1,CONFIG['repeats']+1):
            start=time.perf_counter();cpu=time.process_time()
            for _ in range(CONFIG['timing_batch']):model=cd()
            elapsed=(time.perf_counter()-start)/CONFIG['timing_batch'];elapsed_cpu=(time.process_time()-cpu)/CONFIG['timing_batch']
            b=np.r_[model.intercept_,model.coef_];q=diagnostics(X,y,b,model.scale_,lam,a,ref)
            cd_rows.append(dict(method='cd',repeat=repeat,cpu_seconds=elapsed_cpu,wall_seconds=elapsed,sigma=model.scale_,**q))
            cd_solution=np.r_[b,model.scale_]
        return cd_rows,cd_solution
    cd_timing=measure_cd() if CONFIG['fractions'].index(fraction)%2==0 else None
    command=['Rscript','analysis/adahuber_reference.R',str(folder)]
    print('Running R comparison:',seed,fraction,flush=True)
    started=time.perf_counter();r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
    (folder/'r-console.txt').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    rrows=list(csv.DictReader((folder/'r-timings.tsv').open(),delimiter='\t'))
    coefficient_data=np.genfromtxt(folder/'r-coefficients.tsv',names=True,delimiter='\t',deletechars='')
    rows=[];solutions={};timed=[]
    for row in rrows:
        if row['method']=='native_public':continue
        method=row['method'];repeat=int(row['repeat_id']);b=coefficient_data[f'{method}-{repeat}']
        sigma=float(row['sigma']);q=diagnostics(X,y,b,sigma,lam,a,ref)
        rows.append(dict(method=method,repeat=repeat,cpu_seconds=float(row['cpu_seconds']),wall_seconds=float(row['wall_seconds']),
            sigma=sigma,**q))
        solutions[method]=np.r_[b,sigma]
    public_cpu=float(np.median([float(row['cpu_seconds']) for row in rrows if row['method']=='native_public']))
    if cd_timing is None:cd_timing=measure_cd()
    rows.extend(cd_timing[0]);solutions['cd']=cd_timing[1]
    native=next(row for row in rows if row['method']=='native')
    matched=next(row for row in rows if row['method']=='adapted')
    result=dict(configuration=CONFIG,seed=seed,fraction=fraction,a=a,lambda0=lam,initial_scale=initial,
        reference_scale=ref.scale_,reference_quality=rq,native_public_cpu_seconds=public_cpu,results=rows,
        process_wall_seconds=time.perf_counter()-started)
    save(result,folder/'result.json')
    np.savez_compressed(folder/'solutions.npz',X=X,y=y,original_X=originalX,original_y=originaly,truth=truth,
        reference=np.r_[ref.intercept_,ref.coef_,ref.scale_],**solutions)
    if seed==CONFIG['seeds'][0] and fraction==CONFIG['fractions'][1]:
        trace_model=HuberCD(penalty=lam,rho=rho,scale_penalty=a,initial_scale=initial,
            tol=CONFIG['tolerance'],max_iter=CONFIG['reference_max_iter'])
        tr=fit_trace(trace_model,X,y)
        np.savez_compressed(folder/'cd-trace.npz',**tr)
    print(seed,fraction,'native',native['qualified'],'adapter',matched['qualified'],
          'beta',f"{matched['beta_difference']:.2g}",'sigma',f"{matched['scale_difference']:.2g}",flush=True)
    return result


def assemble(results):
    save(dict(configuration=CONFIG,cases=results),DEST/'summary.json')
    labels={'cd':'MM coordinate descent','native':'adaHuber, native','adapted':'adaHuber, matched adapter'}
    scientific=lambda value: '$'+('{:.2e}'.format(value).split('e')[0])+r'\times10^{'+str(int('{:.2e}'.format(value).split('e')[1]))+'}$'
    lines=['| Method | CPU ms | $\\lVert\\Delta\\beta\\rVert_\\infty$ | $\\lvert\\Delta\\sigma\\rvert$ | Joint residual | Passed |',
           '|:--|--:|--:|--:|--:|:--|']
    details=['| Seed | Fraction | Method | CPU ms | Beta difference | Scale difference | Objective difference | Joint residual |',
             '|--:|--:|:--|--:|--:|--:|--:|--:|']
    for method in ('cd','native','adapted'):
        rows=[row for result in results for row in result['results'] if row['method']==method]
        med=float(np.median([row['cpu_seconds'] for row in rows]))*1000
        maximum=lambda key:max(row[key] for row in rows)
        passed=sum(all(row['qualified'] for row in result['results'] if row['method']==method) for result in results)
        lines.append(f"| {labels[method]} | {med:.3f} | {scientific(maximum('beta_difference'))} | {scientific(maximum('scale_difference'))} | {scientific(maximum('joint_residual'))} | {passed}/{len(results)} |")
        for result in results:
            rr=[row for row in result['results'] if row['method']==method];row=rr[0]
            details.append(f"| {result['seed']} | {result['fraction']} | {labels[method]} | {np.median([r['cpu_seconds'] for r in rr])*1000:.3f} | {row['beta_difference']:.2e} | {row['scale_difference']:.2e} | {row['relative_objective_difference']:.2e} | {row['joint_residual']:.2e} |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n');(DEST/'details.qmd').write_text('\n'.join(details)+'\n')
    agreement=['| Penalty fraction | CD scale | Adapter scale | Maximum coefficient difference | Relative objective difference |',
               '|--:|--:|--:|--:|--:|']
    pair_beta=[];pair_scale=[];pair_objective=[]
    for result in results:
        folder=DEST/f"seed-{result['seed']}-fraction-{result['fraction']:g}"
        values=np.load(folder/'solutions.npz')
        difference=max(abs(values['cd'][:-1]-values['adapted'][:-1]))
        sd=abs(values['cd'][-1]-values['adapted'][-1])
        cdq=next(r for r in result['results'] if r['method']=='cd')
        aq=next(r for r in result['results'] if r['method']=='adapted')
        od=abs(cdq['objective']-aq['objective'])/max(1.,abs(aq['objective']))
        pair_beta.append(difference);pair_scale.append(sd);pair_objective.append(od)
        agreement.append(f"| {result['fraction']} | {values['cd'][-1]:.7f} | {values['adapted'][-1]:.7f} | {scientific(difference)} | {scientific(od)} |")
    (DEST/'solutions.qmd').write_text('\n'.join(agreement)+'\n')
    passing=lambda method:sum(all(r['qualified'] for r in result['results'] if r['method']==method) for result in results)
    findings=(f"CD passes the common accuracy checks at {passing('cd')}/{len(results)} penalties and the matched adapter at {passing('adapted')}/{len(results)}. "
        f"Their largest coefficient and scale differences are {scientific(max(pair_beta))} and {scientific(max(pair_scale))}. "
        f"The largest relative objective difference is {scientific(max(pair_objective))}. "
        f"Native adaHuber passes at {passing('native')}/{len(results)} penalties. Its runtime describes its own calibration procedure, rather than a solve of the matched objective.\n")
    (DEST/'findings.qmd').write_text(findings)
    fig,axes=plt.subplots(1,3,figsize=(8,2.8),constrained_layout=True)
    for method,marker,label in [('native','x','Native adaHuber'),('adapted','o','Matched adapter')]:
        xx=[];yy=[];sx=[];sy=[]
        for result in results:
            d=np.load(DEST/f"seed-{result['seed']}-fraction-{result['fraction']:g}"/'solutions.npz')
            xx.extend(d['reference'][:-1]);yy.extend(d[method][:-1]);sx.append(d['reference'][-1]);sy.append(d[method][-1])
        axes[0].scatter(xx,yy,s=8,alpha=.5,marker=marker,label=label)
        axes[1].scatter(sx,sy,s=22,marker=marker,label=label)
    for ax in axes[:2]:
        low=min(ax.get_xlim()[0],ax.get_ylim()[0]);high=max(ax.get_xlim()[1],ax.get_ylim()[1])
        ax.plot([low,high],[low,high],':',color='0.4',lw=1);ax.set(xlim=(low,high),ylim=(low,high))
    axes[0].set(xlabel='Reference coefficient',ylabel='R coefficient');axes[1].set(xlabel='Reference scale',ylabel='R scale')
    for method,label in labels.items():
        values=[row['cpu_seconds']*1000 for result in results for row in result['results'] if row['method']==method]
        axes[2].scatter([label]*len(values),values,s=7,alpha=.45)
    axes[2].set(ylabel='Prepared CPU ms',yscale='log');axes[2].tick_params(axis='x',rotation=35)
    axes[0].legend(fontsize=7)
    for ax in axes:ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
    fig.savefig(DEST/'agreement.pdf');fig.savefig(DEST/'agreement.png',dpi=180);plt.close(fig)
    if len(results)>1:
        target=DEST/f"seed-{CONFIG['seeds'][0]}-fraction-{CONFIG['fractions'][1]:g}"
        if (target/'cd-trace.npz').exists():
            data=np.load(target/'solutions.npz');cd=np.load(target/'cd-trace.npz');rt=np.loadtxt(target/'r-trace.tsv',skiprows=1)
            ref=data['reference'];fig,axes=plt.subplots(1,2,figsize=(6.7,2.7),constrained_layout=True)
            for label,B,S in [('MM-CD',cd['beta'],cd['sigma']),('Matched adapter',rt[:,1:],rt[:,0])]:
                steps=np.arange(len(S));err=np.max(abs(B-ref[:-1]),axis=1)
                axes[0].semilogy(steps,np.maximum(err,1e-15),label=label)
                axes[1].semilogy(steps,np.maximum(abs(S-ref[-1]),1e-15),label=label)
            for ax in axes:ax.set(xlabel='Method outer iteration');ax.legend(fontsize=8);ax.tick_params(labelsize=8)
            axes[0].set(ylabel='Maximum coefficient difference');axes[1].set(ylabel='Absolute scale difference')
            fig.savefig(DEST/'trajectory.pdf');fig.savefig(DEST/'trajectory.png',dpi=180);plt.close(fig)


def main(limit=None):
    DEST.mkdir(exist_ok=True)
    install=OUT/'adahuber-install.provenance.json'
    verify(json.loads(install.read_text()),source_paths=['analysis/install_adahuber_reference.py'])
    results=[]
    for seed in CONFIG['seeds']:
        for fraction in CONFIG['fractions']:
            result=case(seed,fraction);results.append(result);assemble(results)
            if limit and len(results)>=limit:return
    inputs=[ROOT/'sources/adahuber.json',ROOT/'sources/adaHuber_1.1.tar.gz',install,
            ROOT/'sources/private/adahuber-1.1/src/adaHuber.cpp',ROOT/'sources/private/adahuber-1.1/R/adaHuber.R']
    inputs.extend(DEST.glob('seed-*/*.tsv'));inputs.extend(DEST.glob('seed-*/r-session.txt'))
    outputs=[p for p in DEST.rglob('*') if p.is_file() and p not in inputs and p.name!='provenance.json']
    save(capture(outputs,inputs=inputs,configuration=CONFIG,seed=CONFIG['seeds']),DEST/'provenance.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    args=parser.parse_args();main(args.limit)
