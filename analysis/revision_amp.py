"""Fresh paired timings for the single matched AMP riboflavin problem."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import importlib.util,json,sys,time
import numpy as np
from pipeline import ROOT,OUT,capture,save,verify
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.solver import _solve,_objective
DEST=OUT/'revision-amp'

def main():
    DEST.mkdir(exist_ok=True)
    manifest=OUT/'amp-source-protocol.provenance.json'
    verify(json.loads(manifest.read_text()),source_paths=['src/concomitant_huber/solver.py','analysis/amp_source_protocol.py'])
    arrays=np.load(OUT/'amp-source-protocol-data.npz')
    X,y=arrays['X'],arrays['y']
    n,p=X.shape;assert (n,p)==(57,500)
    source=ROOT/'sources/amp-aistats2027/perspective_huber_ramp_solver.py'
    spec=importlib.util.spec_from_file_location('matched_ramp',source)
    ramp=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ramp;spec.loader.exec_module(ramp)
    original=json.loads((OUT/'amp-source-protocol-summary.json').read_text())
    a=original['a'];rho=1.345;lamp=2.5;lam=lamp/np.sqrt(n);omega=14/p
    initial=max(float(np.median(abs(y-np.median(y)))/.67448975),float(np.std(y))*1e-3,np.finfo(float).eps)
    design=np.asfortranarray(X);amp_design=X/np.sqrt(n)
    reference=_solve(design,y,lam,rho,a,initial,True,1e-9,10000,False,8)
    assert reference[4]
    target=reference[3][-1];rows=[];coefficients={}
    for repeat in range(5):
      for method in (['CD','AMP'] if repeat%2==0 else ['AMP','CD']):
        start=time.perf_counter();start_cpu=time.process_time()
        if method=='CD':
            result=_solve(design,y,lam,rho,a,initial,True,1e-9,10000,False,8)
            beta,sigma,iterations=result[:3];native=bool(result[4])
        else:
            beta,history=ramp.run_perspective_huber_ramp(amp_design,y,lam=lamp,omega=omega,
              iterations=1000,a=a,params=ramp.HuberSpec(c=rho),
              convergence=ramp.SolverConvergence(max_iterations=1000))
            sigma=history[-1]['sigma'];iterations=len(history);native=bool(history[-1]['converged'])
        elapsed=time.perf_counter()-start;cpu=time.process_time()-start_cpu
        if method=='AMP':beta=beta/np.sqrt(n)
        residual=y-X@beta;score=np.sign(residual/sigma)*np.minimum(abs(residual/sigma),rho)
        gradient=-X.T@score/n
        mapping=beta-np.sign(beta-gradient)*np.maximum(abs(beta-gradient)-lam,0)
        vb=float(max(abs(mapping)));vs=float(abs(a-.5*np.mean(score**2)))
        gap=float(abs(_objective(residual,beta,sigma,rho,a,lam)-target)/max(1.,abs(target)))
        passed=bool(vb<=1e-5 and vs<=1e-5 and gap<=1e-8)
        if not passed:raise RuntimeError('Matched timing return failed common accuracy gate')
        coefficients[method]=beta
        rows.append(dict(method=method,repeat=repeat,seconds=elapsed,cpu_seconds=cpu,
                         iterations=int(iterations),native_converged=native,qualified=passed,
                         sigma=float(sigma),coefficient_kkt=vb,scale_kkt=vs,relative_objective_gap=gap))
    difference=float(max(abs(coefficients['CD']-coefficients['AMP'])))
    summary=dict(configuration=dict(n=n,p=p,rho=rho,a=a,lambda_AMP=lamp,lambda0=lam,
         omega=omega,source_seed=20261006,correlation_filter=500,repeats=5,
         scope='one user-requested same-solution instance; no support-calibration search',
         timing_excludes=['preprocessing','imports','JIT','calibration search','matrix preparation']),
         maximum_coefficient_difference=difference,results=rows)
    save(summary,DEST/'summary.json')
    np.savez_compressed(DEST/'solutions.npz',CD=coefficients['CD'],AMP=coefficients['AMP'])
    save(capture([DEST/'summary.json',DEST/'solutions.npz'],inputs=[manifest,
         OUT/'amp-source-protocol-data.npz',ROOT/'sources/data/riboflavinData.mat',source],
         configuration=summary['configuration'],seed=20261006),DEST/'provenance.json')
    print('Matched AMP comparison:',difference,'maximum coefficient difference; all 10 calls qualified.',flush=True)

if __name__=='__main__':main()
