"""Paired Python fixed-scale and concomitant timing on saved validation paths.

Five timing repeats per dataset are distinct from simulation replications.
The same optimized package kernel, buffers, tolerances and polishing settings
are used in both modes; compilation and independent post-fit checks are untimed.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,sys,time
import numpy as np
from pipeline import ROOT,OUT,capture,save,verify
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.optimized import solve_warm
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.solver import _diagnostics,_objective

DEST=OUT/'python-scale-timing'
CASES=['100_100_Block-AR_71','500_100_Block-AR_71',
       '100_500_AR-t2_71','100_100_Contaminated-AR_71']


def solve_path(X,y,penalties,*,estimate_scale,rho,a,tol,max_iter,wall_budget):
    """Time fresh path fits only; common warm starts within each penalty path."""
    L=len(penalties);n,p=X.shape
    beta=np.zeros(p);sigma=1.
    coefficients=np.full((L,p),np.nan);scales=np.full(L,np.nan)
    iterations=np.zeros(L,dtype=int);boundary=np.zeros(L,dtype=bool)
    duals=np.full((L,n),np.nan);statuses=['not_attempted']*L
    local_roots=0;kink_sweeps=0;lp_calls=0;lp_cpu_seconds=0.;lp_wall_seconds=0.
    start=time.perf_counter();cpu=time.process_time()
    for k,lam in enumerate(penalties):
        if time.perf_counter()-start>wall_budget:
            statuses[k]='wall_budget';break
        if sigma>0:
            result=solve_warm(X,y,lam,rho,a,sigma,estimate_scale,tol,max_iter,False,8,beta,
                              check_every=5,newton_every=5)
            beta,sigma,it,history,ok,cb,cs=result[:7]
            local_roots+=int(result[8]);kink_sweeps+=int(result[9])
        else:ok=False;it=0
        if not ok and estimate_scale:
            lp_calls+=1
            lp_cpu=time.process_time();lp_wall=time.perf_counter()
            cert=boundary_solution(X,y,lam,rho=rho,a=a,intercept=False,tol=tol)
            lp_cpu_seconds+=time.process_time()-lp_cpu;lp_wall_seconds+=time.perf_counter()-lp_wall
            if cert['certified']:
                beta=cert['coefficients'];sigma=0.;ok=True
                boundary[k]=True;duals[k]=cert['dual_vector']
        coefficients[k]=beta;scales[k]=sigma;iterations[k]=it
        statuses[k]='converged' if ok else 'uncertified'
        if not ok:break
    cpu_seconds=time.process_time()-cpu;wall_seconds=time.perf_counter()-start
    arrays=dict(coefficients=coefficients,scales=scales,iterations=iterations,
                boundary=boundary,dual_vectors=duals,penalties=penalties)
    info=dict(cpu_seconds=cpu_seconds,wall_seconds=wall_seconds,statuses=statuses,
              complete=all(s=='converged' for s in statuses),
              points_returned=statuses.count('converged'),total_sweeps=int(iterations.sum()),
              local_roots=local_roots,kink_sweeps=kink_sweeps,boundary_lp_calls=lp_calls,
              boundary_points=int(boundary.sum()),boundary_lp_cpu_seconds=lp_cpu_seconds,
              boundary_lp_wall_seconds=lp_wall_seconds)
    return arrays,info


def check_path(X,y,arrays,info,*,estimate_scale,rho,a,tol):
    """Independent positive-scale stationarity or zero-scale dual feasibility/gap."""
    L=len(arrays['penalties']);checks=np.full((L,4),np.nan)
    qualified=np.zeros(L,dtype=bool)
    for k,lam in enumerate(arrays['penalties']):
        b=arrays['coefficients'][k];s=arrays['scales'][k]
        if not np.isfinite(b).all() or not np.isfinite(s):continue
        r=y-X@b
        if s>0:
            cb,cs=_diagnostics(X,r,b,s,rho,a,lam,False)
            checks[k]=[cb,cs,0.,0.]
            qualified[k]=cb<=tol and (not estimate_scale or cs<=tol)
            if not estimate_scale:qualified[k]&=s==1.
        elif arrays['boundary'][k]:
            v=arrays['dual_vectors'][k]
            feasibility=max(0.,float(max(abs(v))-rho),float(np.mean(v*v)-2*a),
                            float(max(abs(X.T@v)/len(y))-lam))
            primal=_objective(r,b,0.,rho,a,lam,False)
            gap=abs(primal-y@v/len(y))/max(1.,abs(primal))
            checks[k]=[0.,0.,feasibility,gap]
            qualified[k]=feasibility<=tol and gap<=tol
    qualified&=np.array(info['statuses'])=='converged'
    arrays['external_checks']=checks;arrays['qualified']=qualified
    info['externally_qualified_points']=int(qualified.sum())
    info['qualified_complete_path']=bool(qualified.all())
    return info


def assemble(rows,repeats):
    lines=['| Design $(n,p)$ | Fixed CPU ms | Joint CPU ms | Joint/fixed | Qualified pairs |',
           '|:--|--:|--:|--:|:--|'];details=[]
    for tag in CASES:
        rr=[r for r in rows if r['case']==tag]
        valid=[r for r in rr if all(r['methods'][m]['qualified_complete_path'] for m in ('fixed','joint'))]
        fixed=np.median([r['methods']['fixed']['cpu_seconds'] for r in valid]) if valid else np.nan
        joint=np.median([r['methods']['joint']['cpu_seconds'] for r in valid]) if valid else np.nan
        ratios=[r['methods']['joint']['cpu_seconds']/r['methods']['fixed']['cpu_seconds'] for r in valid]
        ratio=float(np.median(ratios)) if valid else np.nan
        n,p,design,seed=tag.split('_')
        lines.append(f'| {design} ({n}, {p}) | {1000*fixed:.2f} | {1000*joint:.2f} | {ratio:.2f} | {len(valid)}/{repeats} |')
        details.append(dict(case=tag,qualified_pairs=len(valid),fixed_median_cpu_seconds=float(fixed),
                            joint_median_cpu_seconds=float(joint),median_paired_cpu_ratio=ratio))
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    complete=sum(all(r['methods'][m]['qualified_complete_path'] for m in ('fixed','joint')) for r in rows)
    text=(f'All {complete} of {len(rows)} paired timing repeats qualify over the complete 100-penalty path. '
          if complete==len(rows) else f'{complete} of {len(rows)} paired repeats qualify over the complete path; failed returns remain in the raw records. ')
    text+=('The ratio is the median of within-repeat joint/fixed CPU ratios, not the ratio of separately summarized medians. '
           'These measurements remove implementation-language differences, but include the changed coefficient supports, curvature and boundary solves caused by estimating scale. '
           'They do not measure only the scalar MM update or establish uniform speed parity.\n')
    affected=[r['methods']['joint'] for r in rows if r['methods']['joint']['boundary_lp_calls']>0]
    if affected:
        share=100*np.median([r['boundary_lp_cpu_seconds']/r['cpu_seconds'] for r in affected])
        text+=f'In the AR-t2 validation dataset, each joint path uses 54 boundary LP solves, accounting for a median {share:.1f} percent of its CPU time.\n'
    (DEST/'findings.qmd').write_text(text)
    return details


def main(repeats=5):
    if repeats<1:raise ValueError('Positive timing repeat count required')
    baseline=json.loads((OUT/'optimized-timing/provenance.json').read_text())
    verify(baseline,source_paths=['src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py',
                                'analysis/optimized_timing_validation.py'])
    a=json.loads((OUT/'optimized-timing/summary.json').read_text())['configuration']['a']
    config=dict(cases=CASES,timing_repeats=repeats,path_length=100,seed=71,rho=.5,a=a,initial_scale=1.,
                fixed_scale=1.,tol=1e-6,max_sweeps=4000,check_every=5,newton_every=5,scale_mm_steps=8,
                maximum_Newton_support=128,path_wall_budget=120,intercept=False,standardization=False,
                order='alternate fixed/joint order by repeat and dataset',
                timer='prepared path process CPU and wall; input loading, JIT, startup, serialization and independent post-fit validation excluded; internal certificates and boundary LPs included',
                scope='five timing repeats of each of four declared validation datasets; original 360-case ROME study retained separately')
    DEST.mkdir(exist_ok=True)
    tiny=np.asfortranarray(np.eye(6));yy=np.arange(6.)
    for joint in (False,True):
        solve_warm(tiny,yy,.1,.5,a,1.,joint,1e-6,6,False,8,np.zeros(6),check_every=5,newton_every=5)
    _diagnostics(tiny,yy,np.zeros(6),1.,.5,a,.1,False)
    _objective(yy,np.zeros(6),0.,.5,a,.1,False)
    boundary_solution(np.eye(3),np.arange(3.),.01,rho=.5,a=a,intercept=False)
    rows=[];inputs=[OUT/'optimized-timing/provenance.json'];outputs=[]
    for index,tag in enumerate(CASES):
        d=OUT/'optimized-timing'/tag
        X=np.asfortranarray(np.loadtxt(d/'X.tsv'));y=np.loadtxt(d/'y.tsv');penalties=np.loadtxt(d/'lambda.tsv')
        inputs.extend(d/n for n in ['X.tsv','y.tsv','lambda.tsv'])
        for repeat in range(repeats):
            row=dict(case=tag,repeat=repeat,methods={});data={}
            order=['fixed','joint'] if (index+repeat)%2==0 else ['joint','fixed']
            for method in order:
                arrays,info=solve_path(X,y,penalties,estimate_scale=method=='joint',rho=.5,a=a,
                                       tol=1e-6,max_iter=4000,wall_budget=120)
                check_path(X,y,arrays,info,estimate_scale=method=='joint',rho=.5,a=a,tol=1e-6)
                row['methods'][method]=info
                data.update({method+'_'+k:v for k,v in arrays.items()})
            destination=DEST/tag;destination.mkdir(exist_ok=True)
            archive=destination/f'repeat-{repeat:02d}.npz';np.savez_compressed(archive,**data);outputs.append(archive)
            rows.append(row)
            print(tag,repeat,{m:(round(v['cpu_seconds'],4),v['externally_qualified_points'],v['total_sweeps']) for m,v in row['methods'].items()},flush=True)
    details=assemble(rows,repeats)
    save(dict(configuration=config,results=rows,paired_summary=details),DEST/'summary.json')
    outputs.extend(DEST/n for n in ['summary.json','results.qmd','findings.qmd'])
    save(capture(outputs,inputs=inputs,configuration=config,seed=71),DEST/'provenance.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=5)
    main(parser.parse_args().repeats)
