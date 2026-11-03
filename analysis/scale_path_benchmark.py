"""Declared 36-pair MM/root study; checkpoints retain failures and partial paths.

Primary timings call the unmodified production coefficient kernel. Separate
diagnostic reruns record scale-block inputs and prefix trajectories, untimed
with respect to the primary comparison. No historical result is overwritten.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
import argparse,ast,inspect,json,sys,time,textwrap
import numpy as np
from scipy.special import ndtr
from numba import njit
from pipeline import ROOT,OUT,capture,save,digest
sys.path.insert(0,str(ROOT/'src'))
from rome_paper_experiment import generate
from concomitant_huber import optimized
from concomitant_huber.scale_solver import solve_scale_strategy
from concomitant_huber.scale_updates import _scale_update
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.solver import _diagnostics,_objective
from python_scale_timing import check_path

DEST=OUT/'scale-path-benchmark'
DESIGNS=['Compound-N','AR-t2','Contaminated-AR','Block-AR']
RHO=.5
A=float(.5*(1-2*RHO*np.exp(-RHO**2/2)/np.sqrt(2*np.pi)+2*(RHO**2-1)*(1-ndtr(RHO))))
CONFIG=dict(designs=DESIGNS,dimensions=[[100,100],[100,500],[500,500]],seeds=[71,72,73],
    replications=3,methods=['mm','exact'],path_length=30,minimum_ratio=.05,
    grid='lambda_max(fixed scale 1) * .05**(k/30), k=1,...,30; identical absolute grid for both methods',
    rho=RHO,a=A,initial_scale=1.,initial_beta=0.,intercept=False,standardization=False,
    tol=1e-6,scale_inner_tol=5e-7,max_sweeps=4000,scale_mm_steps=8,scale_max_iter=2000,
    check_every=5,newton_every=5,maximum_Newton_support=128,path_wall_budget=120,
    budget='soft path budget checked between penalties; a solve already running is allowed to finish',
    order='alternate MM/root order across consecutive design/size/seed cases; no concurrent timing workers',
    timer='prepared path CPU/wall includes internal checks and boundary LPs; excludes JIT, data generation, serialization, and independent post-fit checks',
    qualification='all 30 points pass independent original-objective KKT (positive scale) or feasible-dual/gap (zero scale), tolerance 1e-6',
    agreement='paired objectives and fitted values; coefficients/scales descriptive under possible nonuniqueness',
    diagnostic_cases='seed 71, n=p=100, all four designs; cold start at path index 10 (zero based)',
    diagnostic_prefixes=[1,2,5,10,20,50,100,200,500,1000,2000,4000],
    diagnostic_timing='prefix restarts from identical cold state, not cumulative sums; 3 repetitions; excludes JIT and external checks',
    scale_profile='separate recorded-input rerun at diagnostic penalty; batched standalone scale-block replay, not subtraction from primary path CPU',
    scope='moderate method-selection pilot; not full original-paper replication')


def kernel(X,y,lam,beta,sigma,mode,max_iter=4000):
    return solve_scale_strategy(X,y,lam,RHO,A,sigma,True,1e-6,max_iter,False,8,beta,
                               5,5,mode,5e-7,2000,1.)


def solve_path(X,y,penalties,mode):
    L=len(penalties);n,p=X.shape;beta=np.zeros(p);sigma=1.
    arrays=dict(coefficients=np.full((L,p),np.nan),scales=np.full(L,np.nan),
        iterations=np.zeros(L,dtype=int),boundary=np.zeros(L,dtype=bool),
        dual_vectors=np.full((L,n),np.nan),penalties=penalties,
        objectives=np.full(L,np.nan),scale_iterations=np.zeros(L,dtype=int),
        inexact_scale_updates=np.zeros(L,dtype=int))
    statuses=['not_attempted']*L;lp_calls=0;lp_cpu=0.;lp_wall=0.
    cpu=time.process_time();wall=time.perf_counter()
    for k,lam in enumerate(penalties):
        if time.perf_counter()-wall>CONFIG['path_wall_budget']:
            statuses[k]='wall_budget';break
        it=0;ok=False
        if sigma>0:
            rr=kernel(X,y,lam,beta,sigma,mode)
            beta,sigma,it,history,ok=rr[:5]
            arrays['scale_iterations'][k]=rr[10]
            arrays['inexact_scale_updates'][k]=rr[11]
        if not ok:
            lp_calls+=1;lc=time.process_time();lw=time.perf_counter()
            cert=boundary_solution(X,y,lam,rho=RHO,a=A,intercept=False,tol=1e-6)
            lp_cpu+=time.process_time()-lc;lp_wall+=time.perf_counter()-lw
            if cert['certified']:
                beta=cert['coefficients'];sigma=0.;ok=True
                arrays['boundary'][k]=True;arrays['dual_vectors'][k]=cert['dual_vector']
        arrays['coefficients'][k]=beta;arrays['scales'][k]=sigma
        arrays['iterations'][k]=it
        arrays['objectives'][k]=_objective(y-X@beta,beta,sigma,RHO,A,lam,False)
        statuses[k]='converged' if ok else 'uncertified'
        if not ok:break
    info=dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
        statuses=statuses,complete=all(s=='converged' for s in statuses),
        total_sweeps=int(arrays['iterations'].sum()),scale_inner_iterations=int(arrays['scale_iterations'].sum()),
        inexact_scale_updates=int(arrays['inexact_scale_updates'].sum()),
        boundary_lp_calls=lp_calls,boundary_points=int(arrays['boundary'].sum()),
        boundary_lp_cpu_seconds=lp_cpu,boundary_lp_wall_seconds=lp_wall)
    check_path(X,y,arrays,info,estimate_scale=True,rho=RHO,a=A,tol=1e-6)
    return arrays,info


def agreement(X,mm,exact):
    common=mm['qualified']&exact['qualified']
    if not common.any():return dict(common_qualified_points=0)
    bm=mm['coefficients'][common];be=exact['coefficients'][common]
    fm=bm@X.T;fe=be@X.T
    objm=mm['objectives'][common];obje=exact['objectives'][common]
    return dict(common_qualified_points=int(common.sum()),
        maximum_relative_objective_difference=float(np.max(abs(objm-obje)/np.maximum(1.,abs(objm)))),
        maximum_coefficient_difference=float(np.max(abs(bm-be))),
        maximum_scale_difference=float(np.max(abs(mm['scales'][common]-exact['scales'][common]))),
        maximum_prediction_rmse=float(np.max(np.sqrt(np.mean((fm-fe)**2,axis=1)))))


def dependency_hashes():
    names=['analysis/scale_path_benchmark.py','analysis/python_scale_timing.py','analysis/rome_paper_experiment.py',
        'analysis/configs/rome.json',*['src/concomitant_huber/'+s+'.py' for s in
            ['scale_solver','scale_updates','optimized','solver','boundary']]]
    return {n:digest(ROOT/n) for n in names}


def case(n,p,design,seed,index):
    tag=f'{n}_{p}_{design}_{seed}';d=DEST/tag;d.mkdir(exist_ok=True)
    checkpoint=d/'result.json';hashes=dependency_hashes()
    if checkpoint.exists():
        row=json.loads(checkpoint.read_text())
        if row['configuration']!=CONFIG or row['source_hashes']!=hashes:raise ValueError('Stale checkpoint '+tag)
        return row
    X,y,truth,_=generate(n,p,design,seed);X=np.asfortranarray(X)
    score=np.sign(y)*np.minimum(abs(y),RHO)
    maximum=float(max(abs(X.T@score))/n)
    penalties=maximum*np.exp(np.log(.05)*np.arange(1,31)/30)
    np.savez_compressed(d/'inputs.npz',X=X,y=y,truth=truth,penalties=penalties)
    row=dict(case=tag,n=n,p=p,design=design,seed=seed,configuration=CONFIG,source_hashes=hashes,methods={})
    data={};order=['mm','exact'] if index%2==0 else ['exact','mm']
    row['execution_order']=order
    for method in order:
        arrays,info=solve_path(X,y,penalties,int(method=='exact'))
        row['methods'][method]=info;data[method]=arrays
        print(tag,method,info['externally_qualified_points'],'/30',round(info['cpu_seconds'],3),
              's',info['total_sweeps'],'sweeps',info['boundary_lp_calls'],'LPs',flush=True)
    row['agreement']=agreement(X,data['mm'],data['exact'])
    np.savez_compressed(d/'solutions.npz',**{m+'_'+k:v for m,aa in data.items() for k,v in aa.items()})
    save(row,checkpoint)
    return row


def recording_kernel():
    """Analysis-only production clone recording the exact scale-block inputs."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(optimized.solve_warm.py_func)))
    f=tree.body[0];f.name='record_scale_inputs';f.decorator_list=[]
    f.args.args.append(ast.arg(arg='scale_mode'));f.args.defaults.append(ast.Constant(0))
    outer=next(t for t in f.body if isinstance(t,ast.For) and getattr(t.target,'id',None)=='iteration')
    position=f.body.index(outer)
    f.body[position:position]=ast.parse('record_r=np.empty((max_iter,n))\nrecord_s=np.empty(max_iter)').body
    outer.body[0].body=ast.parse('record_r[iteration,:]=r\nrecord_s[iteration]=sigma\nsigma, ni, ck, cg, co = _scale_update(r,sigma,rho,a,scale_mode,5e-7,2000,scale_mm_steps,1.)').body
    for node in ast.walk(f):
        if isinstance(node,ast.Return):
            node.value.elts.extend(ast.parse('record_r[:iteration+1],record_s[:iteration+1]',mode='eval').body.elts)
    ns=dict(vars(optimized));ns['_scale_update']=_scale_update
    exec(compile(ast.fix_missing_locations(tree),'<analysis-scale-recording>','exec'),ns)
    return njit(ns['record_scale_inputs'])


@njit
def replay_scale_inputs(residuals,scales,mode,batches):
    checksum=0.
    for _ in range(batches):
        for j in range(len(scales)):
            result=_scale_update(residuals[j],scales[j],RHO,A,mode,5e-7,2000,8,1.)
            checksum+=result[0]
    return checksum


def diagnostics(rows):
    record=recording_kernel();summary=[];prefixes=[]
    tiny=np.asfortranarray(np.eye(6));yy=np.arange(6.)
    record(tiny,yy,.1,RHO,A,1.,True,1e-6,3,False,8,np.zeros(6),5,5,0)
    replay_scale_inputs(np.ones((2,6)),np.ones(2),0,1)
    for row in rows:
        if row['n']!=100 or row['p']!=100 or row['seed']!=71:continue
        d=DEST/row['case'];data=np.load(d/'inputs.npz')
        X=np.asfortranarray(data['X']);y=data['y'];lam=float(data['penalties'][10])
        for method in ('mm','exact'):
            mode=int(method=='exact')
            rr=record(X,y,lam,RHO,A,1.,True,1e-6,4000,False,8,np.zeros(100),5,5,mode)
            unrecorded=kernel(X,y,lam,np.zeros(100),1.,mode)
            if not (np.array_equal(rr[0],unrecorded[0]) and rr[1]==unrecorded[1] and rr[2]==unrecorded[2]):
                raise RuntimeError('Recording changed deterministic trajectory')
            residuals=np.ascontiguousarray(rr[-2]);scales=rr[-1].copy()
            # Enough repetitions for stable isolated block timing, without
            # representing replay CPU as an additive part of primary path CPU.
            batches=max(1,min(1000,10000//len(scales)))
            replay_scale_inputs(residuals,scales,mode,1)
            timings=[]
            for repeat in range(3):
                cpu=time.process_time();checksum=replay_scale_inputs(residuals,scales,mode,batches)
                timings.append((time.process_time()-cpu)/batches)
            summary.append(dict(case=row['case'],method=method,penalty=lam,sweeps=int(rr[2]),
                converged=bool(rr[4]),block_calls=len(scales),batches=batches,
                replay_cpu_seconds=float(np.median(timings)),checksum=float(checksum)))
            np.savez_compressed(d/f'diagnostic-{method}.npz',residuals=residuals,scales_before=scales)
            for budget in CONFIG['diagnostic_prefixes']:
                timings=[];walls=[]
                for repeat in range(3):
                    cpu=time.process_time();wall=time.perf_counter()
                    fit=kernel(X,y,lam,np.zeros(100),1.,mode,max_iter=budget)
                    timings.append(time.process_time()-cpu);walls.append(time.perf_counter()-wall)
                b,s=fit[:2];cb,cs=_diagnostics(X,y-X@b,b,s,RHO,A,lam,False) if s>0 else (np.inf,np.inf)
                prefixes.append(dict(case=row['case'],method=method,penalty=lam,budget=budget,
                    actual_sweeps=int(fit[2]),cpu_seconds=float(np.median(timings)),
                    wall_seconds=float(np.median(walls)),coefficient_kkt=float(cb),scale_kkt=float(cs),
                    scale=float(s),objective=float(_objective(y-X@b,b,s,RHO,A,lam,False)),
                    qualified=bool(max(cb,cs)<=1e-6)))
                np.savez_compressed(d/f'prefix-{method}-{budget}.npz',beta=b,scale=s)
                if fit[4] or s<=0:break
            print('Diagnostics',row['case'],method,'done',flush=True)
    save(dict(configuration=CONFIG,scale_replay=summary,prefixes=prefixes),DEST/'diagnostics.json')


def assemble(rows):
    lines=['| Design | $(n,p)$ | MM paths | Root paths | MM CPU s | Root CPU s | Root/MM |',
        '|:--|:--|--:|--:|--:|--:|--:|'];details=[]
    for design in DESIGNS:
        for n,p in CONFIG['dimensions']:
            selected=[r for r in rows if r['design']==design and r['n']==n and r['p']==p]
            paired=[r for r in selected if all(r['methods'][m]['qualified_complete_path'] for m in ('mm','exact'))]
            counts={m:sum(r['methods'][m]['qualified_complete_path'] for r in selected) for m in ('mm','exact')}
            med={m:float(np.median([r['methods'][m]['cpu_seconds'] for r in paired])) if paired else None for m in ('mm','exact')}
            ratio=float(np.median([r['methods']['exact']['cpu_seconds']/r['methods']['mm']['cpu_seconds'] for r in paired])) if paired else None
            fmt=lambda v:'—' if v is None else f'{v:.3f}'
            lines.append(f"| {design} | ({n}, {p}) | {counts['mm']}/{len(selected)} | {counts['exact']}/{len(selected)} | {fmt(med['mm'])} | {fmt(med['exact'])} | {fmt(ratio)} |")
            details.append(dict(design=design,n=n,p=p,cases=len(selected),qualified=counts,paired=len(paired),median_cpu_seconds=med,median_paired_root_mm_ratio=ratio))
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    save(dict(configuration=CONFIG,completed_cases=len(rows),planned_cases=36,results=rows,paired_summary=details),DEST/'summary.json')


def main(limit=None,skip_diagnostics=False):
    DEST.mkdir(exist_ok=True)
    tiny=np.asfortranarray(np.eye(6));y=np.arange(6.)
    for mode in (0,1):kernel(tiny,y,.1,np.zeros(6),1.,mode,max_iter=6)
    _diagnostics(tiny,y,np.zeros(6),1.,RHO,A,.1,False);_objective(y,np.zeros(6),0.,RHO,A,.1,False)
    boundary_solution(np.eye(3),np.arange(3.),.01,rho=RHO,a=A,intercept=False)
    rows=[]
    for seed in CONFIG['seeds']:
        for n,p in CONFIG['dimensions']:
            for design in DESIGNS:
                rows.append(case(n,p,design,seed,len(rows)));assemble(rows)
                if limit and len(rows)>=limit:break
            if limit and len(rows)>=limit:break
        if limit and len(rows)>=limit:break
    if not skip_diagnostics:diagnostics(rows)
    outputs=[p for p in DEST.rglob('*') if p.is_file() and p.name!='provenance.json']
    save(capture(outputs,configuration=CONFIG,seed=CONFIG['seeds']),DEST/'provenance.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    parser.add_argument('--skip-diagnostics',action='store_true');args=parser.parse_args()
    main(args.limit,args.skip_diagnostics)
