"""Selected sweep trajectories using original ROME cases plus real-data pilots."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber import HuberCD
from concomitant_huber.diagnostics import fit_trace
from pipeline import OUT,capture,save,verify
directory=OUT/'rome-paper-validation'
verify(json.loads((directory/'provenance.json').read_text()),source_paths=[
 'src/concomitant_huber/solver.py','src/concomitant_huber/path.py','analysis/rome_paper_experiment.py','analysis/rome_fixed_path.R','analysis/configs/rome.json'])
verify(json.loads((OUT/'convergence-amp.provenance.json').read_text()),source_paths=[
 'src/concomitant_huber/solver.py','src/concomitant_huber/diagnostics.py','analysis/convergence_amp_experiment.py'])
rho=.5;a=.5*(1-2*rho*np.exp(-rho*rho/2)/np.sqrt(2*np.pi)+2*(rho*rho-1)*(1-ndtr(rho)))
traces=[];arrays={};summary=[];inputs=[directory/'provenance.json',OUT/'convergence-amp.provenance.json',OUT/'convergence-amp-data.npz']
for design in ('Compound-N','AR-t2'):
 case=directory/f'500_100_{design}_71'
 X=np.loadtxt(case/'X.tsv');y=np.loadtxt(case/'y.tsv');penalties=np.loadtxt(case/'lambda.tsv')
 model=HuberCD(penalty=float(penalties[49]),rho=rho,scale_penalty=a,initial_scale=1.,
               fit_intercept=False,tol=1e-8,max_iter=10000)
 trace=fit_trace(model,X,y)
 if not model.converged_:raise RuntimeError('Selected ROME trace not certified')
 traces.append((design,trace,model.coef_))
 summary.append(dict(case=design,n=500,p=100,path_index=50,rho=rho,a=a,scale=model.scale_,
     sweeps=model.n_iter_,coefficient_kkt=model.coef_kkt_,scale_kkt=model.scale_kkt_))
 for key,value in trace.items():arrays[design+'_'+key]=value
 inputs.extend(case/n for n in ('X.tsv','y.tsv','lambda.tsv'))
 print(design,model.n_iter_,model.scale_,flush=True)
z=np.load(OUT/'convergence-amp-data.npz')
previous=json.loads((OUT/'convergence-amp-summary.json').read_text())
inputs.append(OUT/'convergence-amp-summary.json')
for name in ('Glass-0.6','B-subtilis-0.6'):
 trace={key:z[name+'_'+key] for key in ('beta','sigma','objective')}
 traces.append((name,trace,trace['beta'][-1]))
 record=next(r for r in previous['trace_summary'] if r['case']==name)
 summary.append(dict(record,n=120 if name.startswith('Glass') else 50,p=486 if name.startswith('Glass') else 500,rho=previous['rho'],a=previous['a']))
 for key,value in trace.items():arrays[name+'_'+key]=value
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(4,3,figsize=(7,8.6),constrained_layout=True)
for ax,(name,t,final) in zip(axes,traces):
 steps=np.arange(len(t['sigma']));selected=np.argsort(abs(final))[-5:]
 ax[0].plot(steps,t['sigma']);ax[0].set(xlabel='Full coefficient sweep',ylabel=name+'\nscale')
 for j in selected:ax[1].plot(steps,t['beta'][:,j])
 ax[1].set(xlabel='Full coefficient sweep',ylabel='Largest final coefficients')
 error=np.linalg.norm(t['beta']-final,axis=1)/max(1.,np.linalg.norm(final))
 gap=(t['objective']-t['objective'][-1])/max(1.,abs(t['objective'][-1]))
 ax[2].semilogy(steps,np.maximum(error,1e-16),label='Coefficient distance')
 ax[2].semilogy(steps,np.maximum(gap,1e-16),label='Objective gap')
 ax[2].set_xlabel('Full coefficient sweep');ax[2].legend(fontsize=8)
fig.savefig(OUT/'selected-convergence.pdf');fig.savefig(OUT/'selected-convergence.png',dpi=180);plt.close(fig)
np.savez_compressed(OUT/'selected-convergence-data.npz',**arrays)
save(dict(results=summary),OUT/'selected-convergence-summary.json')
lines=['| Case | Sweeps | Final scale | Coefficient KKT | Scale KKT |','|:--|--:|--:|--:|--:|']
for r in summary:lines.append(f"| {r['case']} | {r['sweeps']} | {r['scale']:.4f} | {r['coefficient_kkt']:.1e} | {r['scale_kkt']:.1e} |")
(OUT/'selected-convergence-results.qmd').write_text('\n'.join(lines)+'\n')
outputs=[OUT/n for n in ('selected-convergence.pdf','selected-convergence.png','selected-convergence-data.npz','selected-convergence-summary.json','selected-convergence-results.qmd')]
save(capture(outputs,inputs=inputs,configuration=dict(ROME_cases=['Compound-N','AR-t2'],n=500,p=100,seed=71,path_index=50,tolerance=1e-8,real_data_penalty_fraction=.6),seed=71),OUT/'selected-convergence.provenance.json')
