"""Moderate matched-objective R4 scaling and same-instance CD/DR trajectories."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
import sys,time,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r4_benchmark import prepare,diagnostics,qualified,common_stop_dr,r4
from pipeline import ROOT,OUT,capture,save
from concomitant_huber.solver import _solve
from concomitant_huber.diagnostics import fit_trace
from concomitant_huber import HuberCD
DEST=OUT/'revision-r4'
CONFIG=dict(dimensions=[[50,100],[100,100],[100,200],[100,400],[200,400]],
            seeds=[71,72],repeats=3,rho=1.345,a=.5,fraction=.6,
            methods=['MM-CD','Path-Alg-eps0','DR-balanced'],
            DR_metric=1.,DR_max_iterations=5000,DR_gate_every=10,
            coefficient_and_scale_gate=1e-5,relative_objective_gate=1e-8,
            threads=1,constraints='one zero row; no restrictions',
            path='audited mean-shift R4 branch; eps_L2=0; unchanged knot updates',
            timings='prepared numerical solve, setup and total separately; imports/JIT excluded')


def generate(n,p,seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(n,p));X-=X.mean(0);X/=np.sqrt(np.mean(X**2,0))
    truth=np.zeros(p);truth[:10]=rng.choice([-1.,1.],10)*rng.uniform(1,2,10)
    y=X@truth+rng.normal(size=n)
    initial=max(float(np.median(abs(y-np.median(y)))/.67448975),1e-6)
    score=np.sign(y/initial)*np.minimum(abs(y/initial),1.345)
    lam=.6*max(abs(X.T@score))/n
    return X,y,initial,lam


def scaling():
    records=[];arrays={}
    for n,p in CONFIG['dimensions']:
     for seed in CONFIG['seeds']:
      checkpoint=DEST/f'{n}-{p}-{seed}.json'
      X,y,initial,lam=generate(n,p,seed)
      arrays[f'{n}-{p}-{seed}-X']=X;arrays[f'{n}-{p}-{seed}-y']=y
      if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved['configuration']!=CONFIG:raise ValueError('R4 configuration mismatch')
        records+=saved['results'];continue
      ref=_solve(np.asfortranarray(X),y,lam,1.345,.5,initial,True,1e-9,10000,False,8)
      if not ref[4]:raise RuntimeError('Reference uncertified')
      target=diagnostics(X,y,ref[0],ref[1],lam)['objective']
      current=[]
      for rep in range(3):
       for method in CONFIG['methods'][::1 if rep%2==0 else -1]:
        row=dict(n=n,p=p,seed=seed,repeat=rep,method=method,penalty=lam,reference=target)
        begin=time.perf_counter();cpu=time.process_time()
        try:
            call=prepare(method,X,y,lam,initial,reference=target)
            setup=time.perf_counter()-begin
            start=time.perf_counter();start_cpu=time.process_time();b,sigma=call()
            elapsed=time.perf_counter()-start;elapsed_cpu=time.process_time()-start_cpu
            d=diagnostics(X,y,b,sigma,lam)
            row.update(status='returned',qualified=bool(qualified(d,target)),setup_seconds=setup,
                       solve_seconds=elapsed,solve_cpu_seconds=elapsed_cpu,
                       total_seconds=setup+elapsed,scale=float(sigma),**d)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            row.update(status='failure',qualified=False,error=str(error),
                       elapsed_wall_seconds=time.perf_counter()-begin,
                       elapsed_cpu_seconds=time.process_time()-cpu)
        current.append(row)
      records+=current;save(dict(configuration=CONFIG,results=current),checkpoint)
      print(n,p,seed,[(m,sum(r['qualified'] for r in current if r['method']==m)) for m in CONFIG['methods']],flush=True)
    np.savez_compressed(DEST/'scaling-data.npz',**arrays)
    save(dict(configuration=CONFIG,results=records),DEST/'scaling.json')
    return records


def trajectory():
    X,y,initial,lam=generate(80,40,91)
    b,s=prepare('Path-Alg-eps0',X,y,lam,initial)()
    d=diagnostics(X,y,b,s,lam);target=d['objective']
    if not qualified(d,target):raise RuntimeError('Path reference not stationary')
    model=HuberCD(penalty=lam,initial_scale=initial,fit_intercept=False,
                  rho=1.345,scale_penalty=.5,tol=1e-8,max_iter=10000)
    cd=fit_trace(model,X,y)
    if not model.converged_:raise RuntimeError('CD trace uncertified')
    pb=r4.problem_R4((X,np.zeros((1,X.shape[1])),y),'DR',1.345,intercept=False)
    pb.c=1.;pb.N=5000;pb.compute_param();pb.compute_param=lambda:None
    dr_beta=[];dr_sigma=[];dr_iteration=[]
    def gate(beta,sigma,iteration):
        dr_beta.append(beta.copy());dr_sigma.append(float(sigma));dr_iteration.append(iteration)
        return qualified(diagnostics(X,y,beta,sigma,lam),target)
    call=common_stop_dr(pb,2*len(y)*lam/pb.lambdamax,gate)
    try:
        db,ds=call();dr_status='qualified'
    except (ValueError,RuntimeError) as error:
        db=np.asarray(dr_beta[-1]);ds=dr_sigma[-1];dr_status=str(error)
    B=np.asarray(dr_beta);S=np.asarray(dr_sigma)
    arrays=dict(X=X,y=y,penalty=np.array(lam),path_beta=b,path_sigma=np.array(s),
                cd_beta=cd['beta'],cd_sigma=cd['sigma'],cd_objective=cd['objective'],
                dr_beta=B,dr_sigma=S,dr_iteration=np.array(dr_iteration))
    np.savez_compressed(DEST/'trajectory.npz',**arrays)
    result=dict(n=80,p=40,seed=91,penalty=lam,path=d,path_scale=float(s),dr_status=dr_status,
                cd=diagnostics(X,y,model.coef_,model.scale_,lam),
                dr=diagnostics(X,y,db,ds,lam),
                cd_beta_max_difference=float(max(abs(model.coef_-b))),
                dr_beta_max_difference=float(max(abs(db-b))),
                cd_scale_difference=float(abs(model.scale_-s)),
                dr_scale_difference=float(abs(ds-s)))
    save(result,DEST/'trajectory.json')
    selected=np.argsort(abs(b))[-3:]
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.9),constrained_layout=True)
    for color,j in zip(['C0','C1','C2'],selected):
        axes[0].plot(np.arange(len(cd['beta'])),cd['beta'][:,j],color=color,label=f'j={j+1}')
        axes[0].plot(dr_iteration,B[:,j],ls='--',color=color)
        axes[0].axhline(b[j],color=color,lw=.5,ls=':',alpha=.5)
    axes[0].set(ylabel='Coefficient',xlabel='Method iteration')
    axes[0].legend(fontsize=8)
    for values,steps,label in [(cd['sigma'],np.arange(len(cd['sigma'])),'CD'),(S,dr_iteration,'DR')]:
        axes[1].plot(steps,values,label=label,ls='--' if label=='DR' else '-')
    axes[1].axhline(s,color='k',ls=':',label='Path-Alg solution')
    axes[1].set(ylabel='Scale',xlabel='Method iteration');axes[1].legend(fontsize=8)
    for betas,scales,steps,label in [(cd['beta'],cd['sigma'],np.arange(len(cd['sigma'])),'CD'),(B,S,dr_iteration,'DR')]:
        error=np.maximum(np.linalg.norm(betas-b,axis=1)+abs(scales-s),1e-16)
        axes[2].semilogy(steps,error,label=label,ls='--' if label=='DR' else '-')
    axes[2].set(ylabel='Distance to Path-Alg solution',xlabel='Method iteration');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale('symlog',linthresh=10)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
    fig.savefig(DEST/'trajectory.pdf');fig.savefig(DEST/'trajectory.png',dpi=180);plt.close(fig)


def assemble(rows):
    lines=['| $n$ | $p$ | CD ms | Path-Alg ms | DR ms | Qualified (of 6) |',
           '|--:|--:|--:|--:|--:|:--|']
    fig,axes=plt.subplots(1,2,figsize=(7.5,3),constrained_layout=True)
    for n,p in CONFIG['dimensions']:
        med={};counts=[]
        for method in CONFIG['methods']:
            rr=[r for r in rows if r['n']==n and r['p']==p and r['method']==method]
            counts.append(str(sum(r['qualified'] for r in rr)))
            common=set.intersection(*[
                {(r['seed'],r['repeat']) for r in rows if r['n']==n and r['p']==p and r['method']==m and r['qualified']}
                for m in CONFIG['methods']])
            valid=[r for r in rr if r['qualified'] and (r['seed'],r['repeat']) in common]
            med[method]=float(np.median([r['solve_seconds'] for r in valid])) if valid else None
        fmt=lambda x:'—' if x is None else f'{1000*x:.2f}'
        lines.append(f"| {n} | {p} | {fmt(med['MM-CD'])} | {fmt(med['Path-Alg-eps0'])} | {fmt(med['DR-balanced'])} | {', '.join(counts)} |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    for ax,axis in zip(axes,['n','p']):
        dims=sorted([v for v in CONFIG['dimensions'] if (v[1]==2*v[0] if axis=='n' else v[0]==100)])
        for method,label in zip(CONFIG['methods'],['CD','Path-Alg','DR']):
            xx=[];yy=[]
            for n,p in dims:
                common=set.intersection(*[
                    {(r['seed'],r['repeat']) for r in rows if r['n']==n and r['p']==p and r['method']==m and r['qualified']}
                    for m in CONFIG['methods']])
                valid=[r['solve_seconds'] for r in rows if r['n']==n and r['p']==p and r['method']==method and r['qualified'] and (r['seed'],r['repeat']) in common]
                if valid:xx.append(n if axis=='n' else p);yy.append(np.median(valid))
            ax.loglog(xx,yy,'o-',label=label)
        ax.set(xlabel=axis,ylabel='Prepared solve seconds',title='p=2n' if axis=='n' else 'n=100');ax.legend(fontsize=8)
    fig.savefig(DEST/'scaling.pdf');fig.savefig(DEST/'scaling.png',dpi=180);plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    _solve(np.asfortranarray(np.eye(5)),np.arange(5.),.2,1.345,.5,1.,True,1e-5,10,False,8)
    rows=scaling();trajectory();assemble(rows)
    outputs=[DEST/n for n in ('scaling-data.npz','scaling.json','trajectory.npz','trajectory.json','results.qmd','scaling.pdf','scaling.png','trajectory.pdf','trajectory.png')]
    save(capture(outputs,inputs=[ROOT/'sources/classo-1.0.11-audit/registry.json'],configuration=CONFIG,seed=CONFIG['seeds']),DEST/'provenance.json')

if __name__=='__main__':main()
