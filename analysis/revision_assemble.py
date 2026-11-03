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
DEST=OUT/'revision-paper'

def read(name):
    return json.loads((OUT/name).read_text())

def timing(draft=False):
    source=OUT/'revision-rome/summary.json'
    data=json.loads(source.read_text());rows=data['results']
    if not draft and len(rows)!=360:raise RuntimeError(f"Timing experiment incomplete: {len(rows)}/360")
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.2),constrained_layout=True)
    dims=data['configuration']['dimensions']
    complete=sum(r['joint']['complete'] for r in rows)
    for ax,design in zip(axes.flat,data['configuration']['designs']):
        for method,label,offset in [('rome','ROME',-.08),('joint','Concomitant CD',.08)]:
            xx=[];yy=[];sd=[]
            for position,(n,p) in enumerate(dims):
                rr=[r for r in rows if r['design']==design and r['n']==n and r['p']==p]
                paired=[r for r in rr if r['joint']['complete'] and r['rome']['complete']]
                if paired:
                    xx.append(position+offset);v=[r[method]['cpu_seconds'] for r in paired]
                    yy.append(np.mean(v));sd.append(np.std(v,ddof=1) if len(v)>1 else 0)
            ax.errorbar(xx,yy,yerr=sd,marker='o',ms=3,label=label,capsize=2)
        for j,(n,p) in enumerate(dims):
            rr=[r for r in rows if r['design']==design and r['n']==n and r['p']==p]
            ax.text(j,.02,f"{sum(r['joint']['complete'] for r in rr)}/{len(rr)}",
                    transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set(yscale='log',title=design,ylabel='Mean CPU seconds',xlabel='(n,p)',xticks=range(len(dims)))
        ax.set_xticklabels([f'{n},{p}' for n,p in dims],rotation=60,fontsize=8)
        ax.legend(fontsize=7)
    fig.savefig(DEST/'rome-timing.pdf');fig.savefig(DEST/'rome-timing.png',dpi=180);plt.close(fig)
    statuses={}
    for r in rows:
        label=r['joint']['status'];statuses[label]=statuses.get(label,0)+1
    pending='**Interim run:** ' if len(rows)!=360 else ''
    text=(f"{pending}The experiment records {len(rows)} of 360 paired paths; "
          f"{complete} concomitant paths complete all 100 penalties with the joint certificate. "
          f"The remaining recorded returns comprise "
          +', '.join(f"{count} {label.replace('_',' ')}" for label,count in statuses.items() if label!='converged')
          +". Full-path CPU summaries use only paired complete paths; the complete pointwise results "
          "and native ROME stationarity checks are retained in the reproducibility materials. "
          "These data do not support uniform speed parity between fixed-scale and concomitant fitting.\n")
    (DEST/'timing-findings.qmd').write_text(text)
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
    if not draft:verify(read('revision-real/provenance.json'),source_paths=[
        'src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
        'src/concomitant_huber/model_selection.py','analysis/revision_real_data.py'])
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.3),constrained_layout=True)
    for i,name in enumerate(['Glass','B-subtilis']):
        directory=OUT/'revision-real'/name/'split-00'
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
        axes[i,0].set(xscale='log',xlabel='Penalty fraction',ylabel='CV normalized MSE',title=name)
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
    rows=read('revision-real/summary.json')['results']
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
    parts.append("The prediction comparison is conditional on certified tuning points; "
                 "it does not establish performance over the uncomputed or uncertified part of a joint path.")
    (DEST/'real-findings.qmd').write_text(' '.join(parts)+'\n')

def main(draft=False):
    DEST.mkdir(exist_ok=True)
    source=timing(draft);equivalence(draft);real_figure(draft);amp(draft);findings()
    if draft:
        rows=read('revision-real/summary.json')['results']
        lines=['| Data | Method | Completed evaluations | Test normalized MSE |',
               '|:--|:--|--:|--:|']
        for name in ['Glass','B-subtilis']:
          for method,label in [('rome','Fixed-scale CD'),('joint','Concomitant CD')]:
            rr=[r['methods'][method] for r in rows if r['dataset']==name and r['methods'][method]['status']=='returned']
            value=f"{np.mean([r['normalized_mse'] for r in rr]):.3f}" if rr else '—'
            lines.append(f"| {name} | {label} | {len(rr)}/10 (interim) | {value} |")
        (OUT/'revision-real/results.qmd').write_text('\n'.join(lines)+'\n')

    if not draft:
        verify(read('revision-rome/provenance.json'),
               source_paths=['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
                             'analysis/revision_rome_timing.py','analysis/revision_rome_parallel.py',
                             'analysis/revision_rome_reference.R','analysis/rome_paper_experiment.py'])
    files=[DEST/n for n in ['rome-timing.pdf','rome-timing.png','timing-findings.qmd',
                            'equivalence-findings.qmd','real-data.pdf','real-data.png','amp-matched.qmd','amp-findings.qmd','r4-findings.qmd','real-findings.qmd']]
    inputs=[source,OUT/'revision-r4/provenance.json',OUT/'amp-source-protocol.provenance.json']
    if not draft:inputs.extend([OUT/'revision-real/provenance.json',OUT/'revision-amp/provenance.json'])
    else:
        inputs.append(OUT/'revision-real/summary.json')
        inputs.extend((OUT/'revision-real').rglob('result.json'))
    if not draft:inputs.append(OUT/'revision-rome/provenance.json')
    else:inputs.extend((OUT/'revision-rome/cases').rglob('result.json'))
    save(capture(files,inputs=inputs,configuration={'focused_rewrite':True,'draft':draft}),
         DEST/'provenance.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args();main(args.draft)
