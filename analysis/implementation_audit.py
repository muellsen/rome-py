"""Audit pinned ROME C and Python kernels, without changing Figure1 solvers.

The C coordinate primitive is compiled verbatim with standard-library headers
instead of unused R headers. It is checked against independent scalar
optimization, not timed concurrently with the Figure1 experiment.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,ctypes,hashlib,json,platform,subprocess,sys,tarfile
import numpy as np
from scipy.optimize import minimize_scalar
from pipeline import ROOT,OUT,capture,save,verify
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.solver import _coordinate

DEST=OUT/'implementation-audit'


def coordinate_checks():
    archive=ROOT/'sources/rome-99d6f07.tar.gz'
    with tarfile.open(archive) as tar:
        raw=tar.extractfile('yk748-rome-99d6f07/src/rome_fit.c').read().decode()
    body='\n'.join(line for line in raw.splitlines() if not line.startswith('#include'))
    src=DEST/'rome-coordinate-verbatim.c'
    src.write_text('#include <math.h>\n#include <stdlib.h>\n#include <string.h>\n'+body)
    library=DEST/'rome-coordinate.dylib'
    subprocess.run(['cc','-O2','-shared','-fPIC',str(src),'-o',str(library)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(library));fun=lib.huber_fit;pointer=ctypes.POINTER(ctypes.c_double)
    fun.argtypes=[pointer,pointer,pointer,pointer,pointer,ctypes.c_double,ctypes.c_double,
                  ctypes.c_double,ctypes.c_int,ctypes.c_double]
    fun.restype=None
    rho=.5;rng=np.random.default_rng(731);fixtures=[]
    fixtures.append(('zero-optimum',np.array([-1.,.2,.5]),np.ones(3),.1))
    fixtures.append(('zero-optimum-n100',np.resize(np.array([-1.,.2,.5]),100),np.ones(100),.1))
    for k in range(20):
        x=rng.uniform(.2,2.,size=20)*rng.choice([-1.,1.],20)
        u=rng.normal(size=20)
        fixtures.append((f'random-{k:02d}',u,x,.1))
    results=[]
    for name,u,x,lam in fixtures:
        rj=np.ascontiguousarray(u/x);first=np.ascontiguousarray(x)
        second=np.ascontiguousarray(x*x);threshold=np.ascontiguousarray(rho/abs(x))
        answer=ctypes.c_double()
        fun(ctypes.byref(answer),rj.ctypes.data_as(pointer),first.ctypes.data_as(pointer),
            second.ctypes.data_as(pointer),threshold.ctypes.data_as(pointer),lam,1.,rho,len(x),float(len(x)))
        def objective(b):
            r=abs(u-x*b)
            return float(np.mean(np.where(r<=rho,.5*r*r,rho*r-.5*rho*rho))+lam*abs(b))
        bound=float(max(abs(u/x))+max(rho/abs(x))+1.)
        candidates=[0.,minimize_scalar(objective,bounds=(-bound,0.),method='bounded',options={'xatol':1e-13}).x,
                    minimize_scalar(objective,bounds=(0.,bound),method='bounded',options={'xatol':1e-13}).x]
        reference=min(candidates,key=objective);python=float(_coordinate(u,x,rho,lam))
        results.append(dict(fixture=name,n=len(x),penalty=lam,C_coefficient=answer.value,
            Python_coefficient=python,reference_coefficient=float(reference),
            C_objective_excess=objective(answer.value)-objective(reference),
            Python_objective_excess=objective(python)-objective(reference)))
    save(dict(scope='22 coordinate fixtures, not a whole-package correctness rate',results=results,
        original_C_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        compiler='cc -O2 -shared -fPIC; primitive unchanged; standard headers replace R headers'),DEST/'coordinate-checks.json')
    subprocess.run(['Rscript','analysis/rome_coordinate_check.R',str(DEST/'installed-R-check.csv')],cwd=ROOT,check=True,capture_output=True)
    return results,src,library


def main():
    DEST.mkdir(exist_ok=True)
    study=json.loads((OUT/'figure1-complete/summary.json').read_text());rows=study['results']
    if len(rows)!=360:raise RuntimeError('Run diagnostic compilation only after primary timing is complete')
    coordinates,src,library=coordinate_checks()
    compiler=dict(python=sys.version,Python_architecture=platform.machine(),
        R_version=subprocess.check_output(['Rscript','--version'],text=True,stderr=subprocess.STDOUT).strip(),
        R_configured_C_compiler=subprocess.check_output(['R','CMD','config','CC'],text=True).strip(),
        R_configured_C_flags=subprocess.check_output(['R','CMD','config','CFLAGS'],text=True).strip(),
        scope='R configuration records current defaults, not a reconstruction of all historical build flags')
    save(compiler,DEST/'compiler-environment.json')
    reuse=[]
    for row in rows:
        if row['methods']['joint']['boundary_lp_calls']==0:continue
        z=np.load(OUT/'figure1-complete'/row['case']/'joint.npz')
        possible=0
        for k in range(1,100):
            if z['scales'][k-1]!=0 or not z['strict'][k-1]:continue
            # Scaling a feasible dual down preserves every joint constraint.
            ratio=min(1.,z['penalties'][k]/z['penalties'][k-1])
            b=z['coefficients'][k-1]
            primal=z['objective'][k-1]+(z['penalties'][k]-z['penalties'][k-1])*sum(abs(b))
            dual=z['dual_lower_bound'][k-1]*ratio
            if max(0.,primal-dual)/max(1.,abs(primal))<=1e-6:possible+=1
        reuse.append(dict(case=row['case'],boundary_LP_calls=row['methods']['joint']['boundary_lp_calls'],
                         subsequent_penalties_with_reusable_certified_boundary_candidate=possible))
    save(dict(scope='read-only certificate reuse opportunities; no claimed speedup or changed timing',results=reuse),DEST/'boundary-reuse.json')
    possible=sum(r['subsequent_penalties_with_reusable_certified_boundary_candidate'] for r in reuse)
    calls=sum(r['boundary_LP_calls'] for r in reuse)
    (DEST/'reuse-findings.qmd').write_text(
        f'The unchanged boundary candidate passes this reuse check at {possible} of the {calls} recorded boundary LP calls. '
        'Thus this simple reuse opportunity covers only a small fraction of the measured LP workload. '
        'Broader improvements would require LP warm starts or avoiding an eager fallback when an existing iterate already has sufficient objective accuracy; those improvements are not timed here.\n')
    examples=['100_100_Contaminated-AR_71','100_500_AR-t2_71','1000_1000_Block-AR_71']
    lines=['| Case | ROME CPU s | Fixed Python CPU s | Joint Python CPU s | Joint LP share |',
           '|:--|--:|--:|--:|--:|'];details=[]
    work=['| Case | ROME sweeps | Fixed Python sweeps | Joint Python sweeps |',
          '|:--|--:|--:|--:|'];iteration_outputs=[]
    for tag in examples:
        row=next(r for r in rows if r['case']==tag);m=row['methods'];j=m['joint']
        share=j['boundary_lp_cpu_seconds']/j['cpu_seconds']
        lines.append(f"| {tag} | {m['rome']['cpu_seconds']:.3f} | {m['fixed']['cpu_seconds']:.3f} | {j['cpu_seconds']:.3f} | {100*share:.1f}% |")
        details.append(dict(case=tag,methods=m))
        iteration_file=DEST/(tag+'-R-iterations.csv')
        subprocess.run(['Rscript','analysis/rome_iteration_check.R',str(OUT/'figure1-complete'/tag),str(iteration_file)],
                       cwd=ROOT,check=True,capture_output=True)
        with iteration_file.open() as stream:iterations=list(csv.DictReader(stream))
        count=sum(int(r['iterations']) for r in iterations)
        work.append(f"| {tag} | {count} | {m['fixed']['total_sweeps']} | {j['total_sweeps']} |")
        iteration_outputs.append(iteration_file)
    (DEST/'timings.qmd').write_text('\n'.join(lines)+'\n')
    (DEST/'work.qmd').write_text('\n'.join(work)+'\n')
    save(dict(examples=details,coordinate_checks=coordinates),DEST/'summary.json')
    outputs=[DEST/'coordinate-checks.json',DEST/'installed-R-check.csv',DEST/'compiler-environment.json',DEST/'summary.json',DEST/'timings.qmd',DEST/'work.qmd',DEST/'boundary-reuse.json',DEST/'reuse-findings.qmd',src,library,*iteration_outputs]
    inputs=[ROOT/'sources/rome-99d6f07.tar.gz',OUT/'figure1-complete/provenance.json']
    inputs.extend(p for p in (OUT/'r-library/rome/libs').glob('*') if p.is_file())
    save(capture(outputs,inputs=inputs),DEST/'provenance.json')
    print('C objective discrepancies above 1e-10:',sum(r['C_objective_excess']>1e-10 for r in coordinates))
    print('Python objective discrepancies above 1e-10:',sum(r['Python_objective_excess']>1e-10 for r in coordinates))


if __name__=='__main__':
    if '--record-html' in sys.argv:
        manifest=DEST/'provenance.json'
        verify(json.loads(manifest.read_text()),source_paths=['analysis/implementation_audit.py','analysis/rome_coordinate_check.R','analysis/rome_iteration_check.R'])
        document=ROOT/'analysis/implementation-audit.html'
        if not document.exists():raise FileNotFoundError(document)
        save(capture([document],inputs=[manifest,ROOT/'analysis/implementation-audit.qmd']),DEST/'html.provenance.json')
    else:main()
