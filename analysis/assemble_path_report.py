"""Render saved ROME path returns, keeping certification separate from recovery."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT,OUT,capture,save,verify
directory=OUT/'rome-paper-validation'
manifest=directory/'provenance.json'
verify(json.loads(manifest.read_text()),source_paths=['src/concomitant_huber/solver.py',
 'src/concomitant_huber/path.py','analysis/rome_paper_experiment.py','analysis/rome_fixed_path.R','analysis/configs/rome.json'])
summary=json.loads((directory/'summary.json').read_text());rows=summary['results']
fig,axes=plt.subplots(2,2,figsize=(8,5.3),constrained_layout=True)
for ax,design in zip(axes.flat,summary['configuration']['designs']):
 for method,label in (('rome-fixed','ROME native returns'),('concomitant','Joint attempted returns')):
  rr=[r for r in rows if r['n']==100 and r['p']==100 and r['design']==design and r['method']==method and r['relative_beta_error'] is not None]
  ax.semilogy([r['path_index'] for r in rr],[max(r['relative_beta_error'],1e-16) for r in rr],label=label)
  if method=='concomitant':
   failed=[r for r in rr if r['status']!='converged']
   if failed:ax.scatter([r['path_index'] for r in failed],[r['relative_beta_error'] for r in failed],marker='x',c='red',s=45,label='Joint uncertified')
 ax.set(title=design,xlabel='Penalty index',ylabel='Relative coefficient error');ax.legend(fontsize=7)
fig.savefig(directory/'returned-paths.pdf');fig.savefig(directory/'returned-paths.png',dpi=180);plt.close(fig)
save(capture([directory/'returned-paths.pdf',directory/'returned-paths.png'],
 inputs=[manifest,directory/'summary.json'],configuration={'plot':'all returned vectors, joint failure marked; no imputed tails; original native tolerance'}),directory/'figure.provenance.json')
