"""Assemble a provenance-checked decision report from the paired path study."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pipeline import ROOT,capture,save,verify
from scale_path_benchmark import DEST,DESIGNS,RHO,A
from concomitant_huber.boundary import boundary_solution
from concomitant_huber.solver import _diagnostics


def main():
    source=DEST/'provenance.json'
    verify(json.loads(source.read_text()),source_paths=[
        'analysis/scale_path_benchmark.py','analysis/python_scale_timing.py','analysis/rome_paper_experiment.py',
        'src/concomitant_huber/scale_solver.py','src/concomitant_huber/scale_updates.py',
        'src/concomitant_huber/optimized.py','src/concomitant_huber/solver.py','src/concomitant_huber/boundary.py'])
    study=json.loads((DEST/'summary.json').read_text());rows=study['results']
    diag=json.loads((DEST/'diagnostics.json').read_text())
    paired=[r for r in rows if all(r['methods'][m]['qualified_complete_path'] for m in ('mm','exact'))]
    ratios=np.array([r['methods']['exact']['cpu_seconds']/r['methods']['mm']['cpu_seconds'] for r in paired])
    metrics={k:max(r['agreement'].get(k,0) for r in rows) for k in
        ['maximum_relative_objective_difference','maximum_coefficient_difference','maximum_scale_difference','maximum_prediction_rmse']}
    findings=dict(cases=len(rows),paired_complete=len(paired),
        qualified_complete={m:sum(r['methods'][m]['qualified_complete_path'] for r in rows) for m in ('mm','exact')},
        median_root_mm_cpu_ratio=float(np.median(ratios)),ratio_quartiles=np.quantile(ratios,[.25,.75]).tolist(),
        ratio_range=[float(min(ratios)),float(max(ratios))],agreement=metrics)
    outputs=[]
    fails=[]
    for row in rows:
        d=DEST/row['case'];data=np.load(d/'inputs.npz');solutions=np.load(d/'solutions.npz')
        for method in ('mm','exact'):
            info=row['methods'][method]
            if info['qualified_complete_path']:continue
            k=next(k for k,status in enumerate(info['statuses']) if status!='converged')
            beta=solutions[method+'_coefficients'][k];sigma=float(solutions[method+'_scales'][k])
            if not np.isfinite(beta).all():
                fails.append(dict(case=row['case'],method=method,index=k,status=info['statuses'][k]));continue
            X=data['X'];y=data['y'];lam=float(data['penalties'][k])
            cb,cs=_diagnostics(X,y-X@beta,beta,sigma,RHO,A,lam,False) if sigma>0 else (np.inf,np.inf)
            cert=boundary_solution(X,y,lam,rho=RHO,a=A,intercept=False,tol=1e-6)
            guard=bool(sigma<1e-10*np.std(y))
            fails.append(dict(case=row['case'],method=method,index=k,penalty=lam,scale=sigma,
                sweeps=int(solutions[method+'_iterations'][k]),coefficient_kkt=float(cb),scale_kkt=float(cs),
                small_scale_handoff=guard,
                stopping_reason='small-scale handoff without a certified boundary solution' if guard else '4000-sweep limit without a certified boundary solution',
                boundary_status=cert['status'],boundary_relative_gap=float(cert.get('relative_gap',np.inf))))
    save(dict(findings=findings,failed_returns=fails),DEST/'audit.json');outputs.append(DEST/'audit.json')
    text=(f"The declared study ran all {len(rows)} paired 30-penalty paths. MM completed and independently certified "
          f"{findings['qualified_complete']['mm']}/{len(rows)} paths; direct scalar minimization completed "
          f"{findings['qualified_complete']['exact']}/{len(rows)}. Across the {len(paired)} pairs completed by both, "
          f"the median root/MM CPU ratio was {np.median(ratios):.3f} "
          f"(interquartile range {np.quantile(ratios,.25):.3f}–{np.quantile(ratios,.75):.3f}; "
          f"range {min(ratios):.3f}–{max(ratios):.3f}). A ratio above one means that the root option took longer. "
          "Failed and partial paths do not enter timing ratios.\n\n"
          f"Over points independently qualified by both methods, the largest relative objective difference was "
          f"{metrics['maximum_relative_objective_difference']:.3g}; the largest prediction RMSE difference was "
          f"{metrics['maximum_prediction_rmse']:.3g}. Maximum coefficient and scale differences were "
          f"{metrics['maximum_coefficient_difference']:.3g} and {metrics['maximum_scale_difference']:.3g}, respectively. "
          "These are empirical agreement measures, not coefficient-error bounds under nonuniqueness.\n\n"
          "**Interpretation.** The scalar root and MM blocks give closely agreeing qualified solutions and similar "
          "typical total path times in this pilot. Direct scalar minimization does not show a general speed advantage. "
          "The current MM option completed two more paths, so these results support retaining it as the default "
          "while keeping the root option available. This is not a final method-selection result. All incomplete "
          "paths are AR-t2 cases; the near-zero-scale handoff and joint convergence need attention before the full study.\n")
    (DEST/'findings.qmd').write_text(text);outputs.append(DEST/'findings.qmd')
    fail_lines=['| Case | Update | First failed point | Scale | Sweeps | Reason |',
        '|:--|:--|--:|--:|--:|:--|']
    for f in fails:
        fail_lines.append(f"| {f['case']} | {f['method']} | {f['index']+1} | {f.get('scale',np.nan):.3g} | {f.get('sweeps',0)} | {f.get('stopping_reason',f.get('status'))} |")
    (DEST/'failures.qmd').write_text('\n'.join(fail_lines)+'\n');outputs.append(DEST/'failures.qmd')
    # Every complete pair is a point; failures remain visible in a separate
    # completion panel rather than appearing as spuriously fast observations.
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for j,design in enumerate(DESIGNS):
        rr=[r for r in paired if r['design']==design]
        axes[0].scatter([r['methods']['mm']['cpu_seconds'] for r in rr],
                        [r['methods']['exact']['cpu_seconds'] for r in rr],label=design,s=36,color=colors[j],alpha=.85)
    lo=min(r['methods'][m]['cpu_seconds'] for r in paired for m in ('mm','exact'))*.75
    hi=max(r['methods'][m]['cpu_seconds'] for r in paired for m in ('mm','exact'))*1.3
    axes[0].plot([lo,hi],[lo,hi],':',color='black',lw=1)
    axes[0].set(xscale='log',yscale='log',xlabel='MM path CPU seconds',ylabel='Scalar-root path CPU seconds',
                title=f'{len(paired)} complete, qualified pairs',xlim=(lo,hi),ylim=(lo,hi))
    axes[0].legend(fontsize=8)
    positions=np.arange(4)
    for j,method in enumerate(('mm','exact')):
        counts=[sum(r['methods'][method]['qualified_complete_path'] for r in rows if r['design']==design) for design in DESIGNS]
        bars=axes[1].bar(positions+(j-.5)*.32,counts,width=.32,label='MM' if method=='mm' else 'Scalar root',color=colors[j])
        axes[1].bar_label(bars,labels=[str(n)+'/9' for n in counts],fontsize=9)
    axes[1].set(xticks=positions,xticklabels=DESIGNS,ylim=(0,10.8),ylabel='Certified complete paths',title='Completion includes every attempted case')
    axes[1].tick_params(axis='x',labelrotation=15);axes[1].legend(fontsize=9,loc='lower left')
    for ext in ('png','pdf'):
        path=DEST/('paired-timing.'+ext);fig.savefig(path,dpi=180);outputs.append(path)
    plt.close(fig)
    # Prefix CPU is separately measured from a fresh cold start for each cap.
    fig,axes=plt.subplots(3,4,figsize=(12,8.3),layout='constrained')
    replay_lines=['| Design | Update | Sweeps | Isolated scale-block CPU ms |', '|:--|:--|--:|--:|']
    for j,design in enumerate(DESIGNS):
        tag=f'100_100_{design}_71';d=DEST/tag
        mm=[r for r in diag['prefixes'] if r['case']==tag and r['method']=='mm']
        reference=np.load(d/f"prefix-mm-{mm[-1]['budget']}.npz")['beta']
        for index,method in enumerate(('mm','exact')):
            selected=[r for r in diag['prefixes'] if r['case']==tag and r['method']==method]
            x=np.array([r['cpu_seconds'] for r in selected])*1000
            axes[0,j].semilogy(x,[max(1e-14,r['coefficient_kkt'],r['scale_kkt']) for r in selected],
                            'o-',ms=3,label='MM' if method=='mm' else 'Scalar root',color=colors[index])
            axes[1,j].plot(x,[r['scale'] for r in selected],'o-',ms=3,color=colors[index])
            distance=[]
            for r in selected:
                b=np.load(d/f"prefix-{method}-{r['budget']}.npz")['beta']
                distance.append(max(1e-10,float(np.linalg.norm(b-reference)/max(1.,np.linalg.norm(reference)))))
            axes[2,j].semilogy(x,distance,'o-',ms=3,color=colors[index])
            replay=next(r for r in diag['scale_replay'] if r['case']==tag and r['method']==method)
            replay_lines.append(f"| {design} | {method} | {replay['sweeps']} | {1000*replay['replay_cpu_seconds']:.3f} |")
        axes[0,j].axhline(1e-6,color='gray',ls=':',lw=1);axes[0,j].set_title(design)
        axes[2,j].set_xlabel('Standalone prefix CPU ms')
        for i in range(3):
            axes[i,j].grid(alpha=.15)
            axes[i,j].xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[0,0].set_ylabel('Joint KKT residual');axes[0,0].legend(fontsize=8)
    axes[1,0].set_ylabel(r'$\sigma$');axes[2,0].set_ylabel(r'Relative $\beta$ distance')
    for ext in ('png','pdf'):
        path=DEST/('prefix-convergence.'+ext);fig.savefig(path,dpi=180);outputs.append(path)
    plt.close(fig)
    (DEST/'scale-replay.qmd').write_text('\n'.join(replay_lines)+'\n');outputs.append(DEST/'scale-replay.qmd')
    save(capture(outputs,inputs=[source,DEST/'summary.json',DEST/'diagnostics.json'],configuration=study['configuration']),
         DEST/'report.provenance.json')
    print(json.dumps(findings,indent=2))


if __name__=='__main__':
    if '--record-html' in sys.argv:
        manifest=DEST/'report.provenance.json'
        verify(json.loads(manifest.read_text()),source_paths=['analysis/scale_path_report.py'])
        html=ROOT/'analysis/scale-path-report.html'
        if not html.exists():raise FileNotFoundError(html)
        save(capture([html],inputs=[manifest,ROOT/'analysis/scale-path-report.qmd'],
            configuration={'scope':'standalone rendered paired-path pilot report'}),DEST/'html.provenance.json')
    else:main()
