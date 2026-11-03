"""Fresh full Figure1: native ROME, fixed Python, qualified concomitant Python.

Historical data and measurements remain immutable. Each case is resumable.
All100 penalties are visited, including after an approximate return.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,subprocess,time,sys
import numpy as np
from pipeline import ROOT,OUT,capture,save,digest
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.qualified_path import qualified_path
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.solver import _diagnostics,_objective
from concomitant_huber.scale_solver import solve_scale_strategy
from concomitant_huber.optimized import solve_warm

DEST=OUT/'figure1-complete'
BASE=json.loads((OUT/'revision-rome/summary.json').read_text())
CONFIG=dict(dimensions=BASE['configuration']['dimensions'],designs=BASE['configuration']['designs'],
    seeds=list(range(71,81)),replications=10,path_length=100,rho=.5,a=BASE['configuration']['a'],
    tol=1e-6,practical_normalized_gap_tol=1e-4,max_sweeps=4000,initial_scale=1.,
    intercept=False,standardization=False,check_every=5,newton_every=5,scale_mm_steps=8,
    policy='visit every penalty; choose lower-original-objective iterate/LAD candidate when uncertified; continue with qualification',
    zero_restart='try LAD first; if uncertified restart positive-scale CD at initial scale with previous coefficients',
    native_ROME='fresh R/C run; native eps1e-4; original path reference script; separate external stationarity check',
    timer='serial single-threaded prepared path CPU/wall; includes internal checks, selection and LPs; excludes compilation, startup, serialization and external audit',
    execution_order='Python fixed/joint order alternates by case; fresh native ROME run follows',
    accuracy='strict KKT/boundary certificate and feasible-dual normalized gap reported separately; neither approximate nor native returns imply certification')


def dependencies():
    return {n:digest(ROOT/n) for n in ['analysis/figure1_complete.py','analysis/revision_rome_reference.R',
        *['src/concomitant_huber/'+s+'.py' for s in ['qualified_path','scale_solver','scale_updates','optimized','boundary','solver']]]}


def case(row,index):
    tag=row['case'];d=DEST/tag;d.mkdir(exist_ok=True);checkpoint=d/'result.json'
    hashes=dependencies()
    if checkpoint.exists():
        result=json.loads(checkpoint.read_text())
        if result['configuration']!=CONFIG or result['sources']!=hashes:raise ValueError('Stale checkpoint '+tag)
        return result
    source=OUT/'revision-rome/cases'/tag
    X=np.asfortranarray(np.loadtxt(source/'X.tsv'));y=np.loadtxt(source/'y.tsv');penalties=np.loadtxt(source/'lambda.tsv')
    result={k:row[k] for k in ('case','n','p','design','seed')}
    result.update(configuration=CONFIG,sources=hashes,methods={})
    methods=['fixed','joint'] if index%2==0 else ['joint','fixed']
    for method in methods:
        arrays,info=qualified_path(X,y,penalties,rho=.5,a=CONFIG['a'],estimate_scale=method=='joint',
            tol=1e-6,practical_gap_tol=1e-4,max_iter=4000)
        np.savez_compressed(d/(method+'.npz'),**arrays);result['methods'][method]=info
    for name in ('X.tsv','y.tsv','lambda.tsv'):
        path=d/name
        if not path.exists():path.symlink_to(source/name)
    run=subprocess.run(['Rscript','analysis/revision_rome_reference.R',str(d)],cwd=ROOT,capture_output=True,text=True,timeout=120)
    (d/'R-console.txt').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stderr)
    b=np.atleast_2d(np.loadtxt(d/'rome-beta.tsv'));tt=np.loadtxt(d/'rome-time.tsv')
    if b.shape!=(100,X.shape[1]) or not np.allclose(np.loadtxt(d/'rome-lambda.tsv'),penalties):raise RuntimeError('Incomplete native ROME return')
    kval=np.array([_diagnostics(X,y-X@bb,bb,1.,.5,CONFIG['a'],lam,False)[0] for bb,lam in zip(b,penalties)])
    result['methods']['rome']=dict(cpu_seconds=float(tt[1]),wall_seconds=float(tt[0]),returned_points=100,
        strict_points=int(sum(kval<=1e-6)),strict_complete=bool((kval<=1e-6).all()),
        maximum_coefficient_kkt=float(max(kval)))
    np.savez_compressed(d/'rome-checks.npz',coefficient_kkt=kval)
    save(result,checkpoint)
    print(tag,{m:(round(v['cpu_seconds'],3),v['strict_points'],v.get('practical_gap_points')) for m,v in result['methods'].items()},flush=True)
    return result


def main(limit=None):
    DEST.mkdir(exist_ok=True)
    X=np.asfortranarray(np.eye(6));y=np.arange(6.);a=CONFIG['a']
    for joint in (False,True):
        qualified_path(X,y,np.array([.1]),rho=.5,a=a,estimate_scale=joint,max_iter=6)
    boundary_solution(np.eye(3),np.arange(3.),.01,rho=.5,a=a,intercept=False)
    rows=[]
    for index,row in enumerate(BASE['results']):
        rows.append(case(row,index))
        save(dict(configuration=CONFIG,completed_cases=len(rows),planned_cases=360,results=rows),DEST/'summary.json')
        if limit and len(rows)>=limit:break
    inputs=[OUT/'revision-rome/summary.json']
    outputs=[p for p in DEST.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='provenance.json']
    inputs.extend(p for p in DEST.rglob('*.tsv') if p.is_symlink())
    save(capture(outputs,inputs=inputs,configuration=CONFIG,seed=CONFIG['seeds']),DEST/'provenance.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);main(p.parse_args().limit)
