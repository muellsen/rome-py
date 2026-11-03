"""Assemble the focused note from saved numerical outputs and verified inputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/concomitant-huber-mpl')
import json,sys,argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT,OUT,capture,save,verify
DEST=OUT/'optimized-paper'

def read(name):
    return json.loads((OUT/name).read_text())

def historical_timing(draft=False):
    source=OUT/'revision-rome/summary.json'
    data=json.loads(source.read_text());rows=data['results']
    if not draft and len(rows)!=360:raise RuntimeError(f"Timing experiment incomplete: {len(rows)}/360")
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.6),constrained_layout=True)
    dims=data['configuration']['dimensions']
    complete=sum(r['joint']['complete'] for r in rows)
    no_joint_cells=0
    for ax,design in zip(axes.flat,data['configuration']['designs']):
        groups=[[r for r in rows if r['design']==design and r['n']==n and r['p']==p] for n,p in dims]
        counts=[sum(r['joint']['complete'] for r in group) for group in groups]
        for method,label,offset,color in [('rome','ROME: native returns',-.08,'C0'),('joint','Earlier MM: certified paths',.08,'C1')]:
            yy=np.full(len(dims),np.nan);sd=np.full(len(dims),np.nan)
            for position,group in enumerate(groups):
                eligible=[r for r in group if r[method]['complete']]
                if eligible:
                    v=[r[method]['cpu_seconds'] for r in eligible]
                    yy[position]=np.mean(v);sd[position]=np.std(v,ddof=1) if len(v)>1 else 0.
            # NaNs leave a visible gap instead of connecting across failed cells.
            ax.errorbar(np.arange(len(dims))+offset,yy,yerr=sd,marker='o',ms=3,
                        label=label,capsize=2,color=color)
        failed=[j for j,c in enumerate(counts) if c==0]
        no_joint_cells+=len(failed)
        if failed:
            ax.scatter(np.array(failed)+.08,np.full(len(failed),.045),transform=ax.get_xaxis_transform(),
                       marker='x',color='crimson',s=25,label='No certified joint path',zorder=5)
        labels=[f'{n},{p}\n{c}/{len(group)}' for (n,p),c,group in zip(dims,counts,groups)]
        ax.set(yscale='log',title=design,ylabel='Mean CPU seconds',
               xlabel='(n,p); joint paths completed / attempted',xticks=range(len(dims)))
        ax.set_xticklabels(labels,rotation=55,ha='right',fontsize=8)
        for tick,count in zip(ax.get_xticklabels(),counts):
            if count==0:tick.set_color('crimson')
        ax.legend(fontsize=7,loc='best')
    fig.savefig(DEST/'rome-timing.pdf');fig.savefig(DEST/'rome-timing.png',dpi=180);plt.close(fig)
    statuses={}
    for r in rows:
        label=r['joint']['status'];statuses[label]=statuses.get(label,0)+1
    pending='**Interim run:** ' if len(rows)!=360 else ''
    native=sum(r['rome']['complete'] for r in rows)
    text=(f"{pending}**Earlier MM baseline, not the current three-option comparison.** "
          f"All {len(rows)} planned paths were attempted. Native ROME returned all 100 penalties "
          f"in {native} paths; the earlier concomitant MM implementation certified all 100 in "
          f"{complete}. The other returns comprise "
          +', '.join(f"{count} "+{'max_iter':'paths reaching the per-penalty sweep limit','path_wall_budget':'paths reaching the path time limit'}.get(label,label) for label,count in statuses.items() if label!='converged')
          +f". In {no_joint_cells} of the 36 design/size cells, no joint path completed. "
          "The revised figure retains all available native ROME timings and marks those failed joint cells with red crosses. "
          "A cross denotes a failed full path, not a CPU-time observation; each tick reports the joint completion count. "
          "Joint times summarize complete paths only, whereas native times use all returned paths. "
          "These are descriptive completion-conditioned curves, not paired speed-ratio estimates. "
          "The records do not establish uniform speed parity or describe the performance of the newer scale options.\n")
    (DEST/'timing-findings.qmd').write_text(text)
    lines=['| Design | Native ROME returns | Joint certified paths | Sweep limit | Time limit |',
           '|:--|--:|--:|--:|--:|']
    for design in data['configuration']['designs']:
        rr=[r for r in rows if r['design']==design]
        lines.append(f"| {design} | {sum(r['rome']['complete'] for r in rr)}/{len(rr)} | "
                     f"{sum(r['joint']['complete'] for r in rr)}/{len(rr)} | "
                     f"{sum(r['joint']['status']=='max_iter' for r in rr)} | "
                     f"{sum(r['joint']['status']=='path_wall_budget' for r in rr)} |")
    (DEST/'rome-completion.qmd').write_text('\n'.join(lines)+'\n')
    return source


def timing(draft=False):
    source=OUT/'figure1-complete/summary.json'
    if not source.exists():return historical_timing(draft)
    data=json.loads(source.read_text());rows=data['results']
    if len(rows)!=360:raise RuntimeError(f"Figure1 rerun incomplete: {len(rows)}/360")
    verify(read('figure1-complete/accuracy.provenance.json'),source_paths=['analysis/figure1_accuracy.py'])
    quality={r['case']:r['methods'] for r in read('figure1-complete/accuracy.json')['results']}
    verify(read('figure1-complete/provenance.json'),source_paths=[
        'analysis/figure1_complete.py','analysis/revision_rome_reference.R',
        *['src/concomitant_huber/'+s+'.py' for s in ['qualified_path','scale_solver','scale_updates','optimized','boundary','solver']]])
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.5),constrained_layout=True)
    dims=data['configuration']['dimensions']
    for ax,design in zip(axes.flat,data['configuration']['designs']):
        groups=[[r for r in rows if r['design']==design and r['n']==n and r['p']==p] for n,p in dims]
        for method,label,offset,color,marker in [
            ('rome','ROME (R/C), fixed scale',-.12,'C0','o'),
            ('fixed','Python CD, fixed scale',0.,'C2','s'),
            ('joint','Python CD, concomitant',.12,'C1','D')]:
            means=[];sd=[]
            for group in groups:
                values=[r['methods'][method]['cpu_seconds'] for r in group]
                if len(values)!=10 or any(r['methods'][method]['returned_points']!=100 for r in group):
                    raise RuntimeError('Cannot plot incomplete path as full-path timing')
                means.append(np.mean(values));sd.append(np.std(values,ddof=1))
            positions=np.arange(len(dims))+offset
            ax.errorbar(positions,means,yerr=sd,marker=marker,ms=3,label=label,capsize=2,color=color)
            if method!='rome':
                approx=[j for j,group in enumerate(groups) if any(not quality[r['case']][method]['strict_complete'] for r in group)]
                if approx:
                    ax.scatter(positions[approx],np.array(means)[approx],marker=marker,s=24,
                               facecolors='white',edgecolors=color,zorder=5)
        ax.set(yscale='log',title=design,ylabel='Mean CPU seconds',xlabel='(n,p)',xticks=range(len(dims)))
        ax.set_xticklabels([f'{n},{p}' for n,p in dims],rotation=55,ha='right',fontsize=9)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.005),
               ncol=3,fontsize=9,frameon=False)
    for ext in ('pdf','png'):fig.savefig(DEST/('rome-timing.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    counts={m:sum(quality[r['case']][m]['strict_complete'] for r in rows) for m in ('fixed','joint')}
    gap_count=sum(quality[r['case']]['joint']['practical_gap_complete'] for r in rows)
    total_approx=sum(quality[r['case']]['joint']['approximate_points'] for r in rows)
    gaps=[quality[r['case']]['joint']['maximum_normalized_gap'] for r in rows]
    text=(f"All three implementations return all 100 penalties on all 360 datasets. "
          f"The Python fixed-scale and concomitant solvers certify every point in {counts['fixed']}/360 and "
          f"{counts['joint']}/360 paths, respectively. The concomitant solver returns {total_approx} "
          "points with an approximate-solution status; these points and their continuation costs remain in the timings. "
          f"A feasible-dual normalized objective gap at most $10^{{-4}}$ holds throughout {gap_count}/360 joint paths. "
          f"The largest recorded joint path gap is {max(gaps):.3g}. "
          "Hollow Python markers identify cells containing at least one approximate path. "
          "Native ROME retains its original iterate-change stopping rule and is not presented as jointly accuracy-matched. "
          "Historical truncated MM paths remain archived separately.\n")
    (DEST/'timing-findings.qmd').write_text(text)
    lines=['| Design | ROME paths | Fixed Python certified points | Joint Python certified points | Joint gap $\\le10^{-4}$ points |',
           '|:--|--:|--:|--:|--:|']
    for design in data['configuration']['designs']:
        rr=[r for r in rows if r['design']==design]
        lines.append(f"| {design} | {len(rr)}/{len(rr)} | "
            f"{sum(quality[r['case']]['fixed']['strict_points'] for r in rr)}/{100*len(rr)} | "
            f"{sum(quality[r['case']]['joint']['strict_points'] for r in rr)}/{100*len(rr)} | "
            f"{sum(quality[r['case']]['joint']['practical_gap_points'] for r in rr)}/{100*len(rr)} |")
    (DEST/'rome-completion.qmd').write_text('\n'.join(lines)+'\n')
    return source


def equivalence(draft=False):
    data=read('revision-r4/trajectory.json')
    math=lambda v:'$'+f'{v:.2g}'+'$'
    text=(f"In this example, the maximum coefficient differences from Path-Alg are "
          f"{math(data['cd_beta_max_difference'])} for CD and "
          f"{math(data['dr_beta_max_difference'])} for DR. "
          f"The corresponding scale differences are "
          f"{math(data['cd_scale_difference'])} and {math(data['dr_scale_difference'])}. "
          "All three returned solutions satisfy the common accuracy requirements.\n")
    import re
    text=re.sub(r'([0-9.]+)e([+-])([0-9]+)',lambda m:m[1]+r'\times10^{'+('-' if m[2]=='-' else '')+str(int(m[3]))+'}',text)
    (DEST/'equivalence-findings.qmd').write_text(text)
    if not draft:verify(read('revision-r4/provenance.json'),source_paths=[
        'src/concomitant_huber/solver.py','analysis/r4_benchmark.py','analysis/revision_r4.py',
        'src/concomitant_huber/diagnostics.py'])


def real_figure(draft=False):
    if not draft:verify(read('optimized-real/provenance.json'),source_paths=[
        'src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
        'src/concomitant_huber/model_selection.py','src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','analysis/optimized_real_data.py','analysis/optimized_real_parallel.py'])
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.3),constrained_layout=True)
    for i,name in enumerate(['Glass','B-subtilis']):
        directory=OUT/'optimized-real'/name/'split-00'
        arrays=np.load(directory/'evaluation.npz')
        info=json.loads((directory/'result.json').read_text())
        fractions=np.array(info['configuration']['fractions'])
        for method,label in [('rome','Fixed scale'),('joint','Concomitant')]:
            losses=arrays[method+'_cv_loss']
            valid=arrays[method+'_eligible'].all(0)
            mean=np.mean(losses[:,valid],axis=0)/float(arrays['y_scale'])**2
            sem=np.std(losses[:,valid],axis=0,ddof=1)/np.sqrt(5)/float(arrays['y_scale'])**2
            line=axes[i,0].errorbar(fractions[valid],mean,yerr=sem,marker='.',ms=3,lw=.8,label=label)
            axes[i,0].axvline(info['methods'][method]['fraction'],
                              color=line.lines[0].get_color(),ls=':',lw=.8)
            prediction=arrays[method+'_prediction']
            axes[i,1].scatter(arrays['test_y'],prediction,s=15,alpha=.7,label=label)
        values=arrays['test_y'];low,high=min(values),max(values)
        axes[i,1].plot([low,high],[low,high],color='k',lw=.7,ls=':')
        axes[i,0].set(xscale='log',xlabel=r'$\lambda_0/\lambda_{\max}$ (model-specific)',ylabel='CV normalized MSE',title=name)
        axes[i,1].set(xlabel='Observed response',ylabel='Held-out prediction',title=name)
        for ax in axes[i]:
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=9)
            ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
    fig.savefig(DEST/'real-data.pdf');fig.savefig(DEST/'real-data.png',dpi=180);plt.close(fig)


def amp_cached_preflight():
    verify(read('amp-source-protocol.provenance.json'),
           source_paths=['src/concomitant_huber/solver.py','analysis/amp_source_protocol.py'])
    data=read('amp-source-protocol-summary.json')
    matched=[r for r in data['results'] if r['lambda_AMP']==2.5 and r['target_support']==14]
    if len(matched)!=1 or not matched[0]['qualified']:raise RuntimeError('Matched AMP point not qualified')
    with (OUT/'amp-matched-timings.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if float(r['lambda_AMP'])==2.5]
    lines=['| Matched riboflavin instance | CD | AMP |','|:--|--:|--:|']
    med={method:1000*np.median([float(r['seconds']) for r in rows if r['method']==method]) for method in ['MM-CD','RAMP']}
    lines.append(f"| Prepared solve ms, median of five runs | {med['MM-CD']:.2f} | {med['RAMP']:.2f} |")
    lines.append(f"| Scale | {matched[0]['reference_sigma']:.8f} | {matched[0]['sigma']:.8f} |")
    (DEST/'amp-matched.qmd').write_text('\n'.join(lines)+'\n')


def amp(draft=False):
    current=OUT/'revision-amp/provenance.json'
    if not current.exists():
        if not draft:raise RuntimeError('Run the matched AMP instance before final assembly')
        amp_cached_preflight()
        item=next(r for r in read('amp-source-protocol-summary.json')['results']
                  if r['lambda_AMP']==2.5 and r['target_support']==14)
        sigma_cd=item['reference_sigma'];sigma_amp=item['sigma']
        difference=item['coefficient_distance']
    else:
        verify(json.loads(current.read_text()),source_paths=[
            'src/concomitant_huber/solver.py','analysis/revision_amp.py'])
        data=read('revision-amp/summary.json');rows=data['results']
        if len(rows)!=10 or not all(r['qualified'] for r in rows):raise RuntimeError('Matched AMP checks failed')
        med={m:1000*np.median([r['seconds'] for r in rows if r['method']==m]) for m in ['CD','AMP']}
        sigma_cd=next(r['sigma'] for r in rows if r['method']=='CD')
        sigma_amp=next(r['sigma'] for r in rows if r['method']=='AMP')
        difference=data['maximum_coefficient_difference']
        lines=['| Matched riboflavin instance | CD | AMP |','|:--|--:|--:|',
               f"| Prepared solve ms, median of five runs | {med['CD']:.2f} | {med['AMP']:.2f} |",
               f"| Scale | {sigma_cd:.8f} | {sigma_amp:.8f} |"]
        (DEST/'amp-matched.qmd').write_text('\n'.join(lines)+'\n')
    import re
    error=re.sub(r'([0-9.]+)e-([0-9]+)',lambda m:m[1]+r'\times10^{-'+str(int(m[2]))+'}',f'{difference:.2g}')
    text=(r'At $\lambda_{\rm AMP}=2.5$ and $\omega=14/500$, both methods select 14 coefficients. '
          +f"The fitted scales are {sigma_cd:.5f} and {sigma_amp:.5f}; "
          +r'the maximum coefficient difference is $'+error+'$. '
          +r'Every timed return satisfies the common $10^{-5}$ joint residual '
          +r'and $10^{-8}$ relative objective requirements.'+'\n')
    (DEST/'amp-findings.qmd').write_text(text)

def findings():
    rows=read('revision-r4/scaling.json')['results']
    counts={m:sum(r['qualified'] for r in rows if r['method']==m)
            for m in ['MM-CD','Path-Alg-eps0','DR-balanced']}
    large=[r for r in rows if r['n']==200 and r['p']==400 and r['qualified']]
    med={m:np.median([r['solve_seconds'] for r in large if r['method']==m]) for m in counts}
    text=(f"CD and Path-Alg qualify in {counts['MM-CD']} and {counts['Path-Alg-eps0']} of 30 calls, "
          f"respectively; DR qualifies in {counts['DR-balanced']} of 30. "
          f"At the largest setting, prepared CD takes {1000*med['MM-CD']:.2f} ms, "
          f"compared with {1000*med['Path-Alg-eps0']:.2f} ms for Path-Alg and "
          f"{1000*med['DR-balanced']:.2f} ms for DR. "
          "The comparison concerns these implementations, including compiled CD versus "
          "Python-level c-lasso updates; it does not identify an intrinsic algorithmic complexity ratio.\n")
    (DEST/'r4-findings.qmd').write_text(text)
    rows=read('optimized-real/summary.json')['results']
    parts=[]
    for name in ['Glass','B-subtilis']:
        fixed=[r['methods']['rome'] for r in rows if r['dataset']==name and r['methods']['rome']['status']=='returned']
        joint=[r['methods']['joint'] for r in rows if r['dataset']==name and r['methods']['joint']['status']=='returned']
        if not fixed or not joint:continue
        parts.append(f"For {name}, mean normalized test MSE is "
                     f"{np.mean([r['normalized_mse'] for r in fixed]):.3f} for fixed-scale CD and "
                     f"{np.mean([r['normalized_mse'] for r in joint]):.3f} for concomitant CD. "
                     f"The median number of CV-eligible penalties is "
                     f"{np.median([r['eligible_cv_points'] for r in fixed]):.0f} and "
                     f"{np.median([r['eligible_cv_points'] for r in joint]):.0f}, respectively.")
    all_complete=all(r['methods'][m].get('eligible_cv_points')==50 for r in rows for m in ('rome','joint'))
    if all_complete:parts.append('All 50 penalties are eligible in both models in every outer split, including certified zero-scale solutions.')
    else:
        incomplete=[r['methods']['joint'].get('eligible_cv_points',0) for r in rows if r['methods']['joint'].get('eligible_cv_points',0)<50]
        parts.append(f'All fixed-scale grids are complete; {len(incomplete)} concomitant split(s) exclude uncertified inner-fold points, with eligible grid sizes '+', '.join(map(str,incomplete))+'. Later grid points are still attempted and certified individually.')
    boundaries=sum(r['methods']['joint'].get('scale',-1)==0 for r in rows)
    parts.append(f'CV selects a zero-scale concomitant refit in {boundaries} of {len(rows)} splits.')
    (DEST/'real-findings.qmd').write_text(' '.join(parts)+'\n')

def main(draft=False):
    DEST.mkdir(exist_ok=True)
    source=timing(draft);equivalence(draft);real_figure(draft);amp(draft);findings()
    if not draft:
        verify(read('scale-strategies/provenance.json'),source_paths=['src/concomitant_huber/solver.py','src/concomitant_huber/optimized.py','src/concomitant_huber/scale_updates.py','src/concomitant_huber/scale_solver.py','src/concomitant_huber/fast_estimator.py','analysis/scale_strategy_comparison.py'])
        verify(read('python-scale-timing/provenance.json'),source_paths=['src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','src/concomitant_huber/solver.py','analysis/python_scale_timing.py'])
        verify(read('optimized-timing/provenance.json'),source_paths=['src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','analysis/optimized_timing_validation.py'])
    if draft and not (OUT/'optimized-real/provenance.json').exists():
        rows=read('optimized-real/summary.json')['results']
        lines=['| Data | Method | Completed evaluations | Test normalized MSE |',
               '|:--|:--|--:|--:|']
        for name in ['Glass','B-subtilis']:
          for method,label in [('rome','Fixed-scale CD'),('joint','Concomitant CD')]:
            rr=[r['methods'][method] for r in rows if r['dataset']==name and r['methods'][method]['status']=='returned']
            value=f"{np.mean([r['normalized_mse'] for r in rr]):.3f}" if rr else '—'
            lines.append(f"| {name} | {label} | {len(rr)}/10 (interim) | {value} |")
        (OUT/'optimized-real/results.qmd').write_text('\n'.join(lines)+'\n')

    if not draft:
        verify(read('revision-rome/provenance.json'),
               source_paths=['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
                             'analysis/revision_rome_timing.py','analysis/revision_rome_parallel.py',
                             'analysis/revision_rome_reference.R','analysis/rome_paper_experiment.py'])
    files=[DEST/n for n in ['rome-timing.pdf','rome-timing.png','timing-findings.qmd','rome-completion.qmd',
                            'equivalence-findings.qmd','real-data.pdf','real-data.png','amp-matched.qmd','amp-findings.qmd','r4-findings.qmd','real-findings.qmd']]
    inputs=[source,OUT/'revision-r4/provenance.json',OUT/'amp-source-protocol.provenance.json']
    if not draft:inputs.extend([OUT/'optimized-real/provenance.json',OUT/'revision-amp/provenance.json',OUT/'optimized-timing/provenance.json',OUT/'python-scale-timing/provenance.json',OUT/'scale-strategies/provenance.json'])
    else:
        inputs.append(OUT/'optimized-real/summary.json')
        inputs.extend((OUT/'optimized-real').rglob('result.json'))
    if not draft:
        inputs.append(OUT/'revision-rome/provenance.json')
        if (OUT/'figure1-complete/provenance.json').exists():
            inputs.extend([OUT/'figure1-complete/provenance.json',OUT/'figure1-complete/accuracy.provenance.json'])
    else:inputs.extend((OUT/'revision-rome/cases').rglob('result.json'))
    save(capture(files,inputs=inputs,configuration={'focused_rewrite':True,'draft':draft}),
         DEST/'provenance.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args();main(args.draft)
