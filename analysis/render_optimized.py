"""Verify the focused paper's numerical inputs and render or record its output."""
import argparse,json,subprocess
from pathlib import Path
from pipeline import ROOT,OUT,capture,save,verify
from fetch_algorithm_template import ensure

CHECKS={
 'figure1-complete/accuracy.provenance.json':['analysis/figure1_accuracy.py'],
 'figure1-complete/provenance.json':['analysis/figure1_complete.py','analysis/revision_rome_reference.R',
   *['src/concomitant_huber/'+s+'.py' for s in ['qualified_path','scale_solver','scale_updates','optimized','boundary','solver']]],
 'revision-rome/provenance.json':['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
   'analysis/revision_rome_timing.py','analysis/revision_rome_parallel.py','analysis/revision_rome_reference.R','analysis/rome_paper_experiment.py'],
 'revision-r4/provenance.json':['src/concomitant_huber/solver.py','analysis/r4_benchmark.py',
   'analysis/revision_r4.py','src/concomitant_huber/diagnostics.py'],
 'optimized-real/provenance.json':['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
   'src/concomitant_huber/model_selection.py','src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','analysis/optimized_real_data.py','analysis/optimized_real_parallel.py'],
 'revision-amp/provenance.json':['src/concomitant_huber/solver.py','analysis/revision_amp.py'],
 'amp-source-protocol.provenance.json':['src/concomitant_huber/solver.py','analysis/amp_source_protocol.py'],
 'optimized-timing/provenance.json':['src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','analysis/optimized_timing_validation.py'],
 'optimization-pilot/provenance.json':['src/concomitant_huber/solver.py','analysis/optimization_pilot.py'],
 'python-scale-timing/provenance.json':['src/concomitant_huber/optimized.py','src/concomitant_huber/boundary.py','src/concomitant_huber/solver.py','analysis/python_scale_timing.py'],
 'scale-strategies/provenance.json':['src/concomitant_huber/solver.py','src/concomitant_huber/optimized.py','src/concomitant_huber/scale_updates.py','src/concomitant_huber/scale_solver.py','src/concomitant_huber/fast_estimator.py','analysis/scale_strategy_comparison.py'],
 'scale-path-benchmark/provenance.json':['analysis/scale_path_benchmark.py',
   *['src/concomitant_huber/'+n+'.py' for n in ('solver','optimized','scale_solver','scale_updates','boundary')]],
 'scale-path-benchmark/report.provenance.json':['analysis/scale_path_report.py'],
 'adahuber/provenance.json':['analysis/adahuber_benchmark.py','analysis/adahuber_reference.R',
   'analysis/rome_paper_experiment.py',
   *['src/concomitant_huber/'+n+'.py' for n in ('solver','optimized','scale_solver','scale_updates','fast_estimator','qualified_path','diagnostics','model_selection','path','boundary')],
   *['src/rome_py/'+n+'.py' for n in ('__init__','qualified_path','model_selection')]],
 'adahuber-install.provenance.json':['analysis/install_adahuber_reference.py'],
 'optimized-paper/provenance.json':['analysis/optimized_assemble.py']}

def run(fmt,record=False,draft=False):
    ensure()
    for name,dependencies in CHECKS.items():
        if draft and (name.startswith('revision-rome/') or not (OUT/name).exists()):continue
        verify(json.loads((OUT/name).read_text()),source_paths=dependencies)
    assembled=json.loads((OUT/'optimized-paper/provenance.json').read_text())
    if assembled['configuration']['draft'] and not draft:raise ValueError('Refusing a final render from interim results')
    command=['quarto','render','paper/manuscript.qmd','--to',fmt]
    if not record:subprocess.run(command,cwd=ROOT,check=True)
    document=ROOT/'_build/paper'/('manuscript.'+fmt)
    if not document.exists():raise RuntimeError('Rendered manuscript missing')
    sources=[ROOT/'paper'/n for n in ('manuscript.qmd','algorithm.qmd','algorithm-html.qmd','scale-options.qmd','supplement.qmd','numerical-controls.qmd','support-step.qmd','jcgs-preamble.tex','references.bib','blue-additions.lua')]
    sources.append(ROOT/'_quarto.yml')
    if document.stat().st_mtime<max(p.stat().st_mtime for p in sources):raise ValueError('Manuscript output predates current sources')
    inputs=[OUT/n for n in CHECKS if not(draft and n.startswith('revision-rome/'))]
    inputs.extend(OUT/'optimized-paper'/n for n in ('timing-findings.qmd','rome-completion.qmd','equivalence-findings.qmd','amp-matched.qmd','amp-findings.qmd','rome-timing.pdf','real-data.pdf','r4-findings.qmd','real-findings.qmd'))
    inputs.extend([OUT/'revision-r4/results.qmd',OUT/'revision-r4/scaling.pdf',OUT/'revision-r4/trajectory.pdf',OUT/'optimized-real/results.qmd',OUT/'optimized-timing/results.qmd',OUT/'python-scale-timing/results.qmd',OUT/'python-scale-timing/findings.qmd',OUT/'scale-strategies/results.qmd',OUT/'scale-strategies/findings.qmd',OUT/'scale-path-benchmark/results.qmd'])
    inputs.extend(OUT/'adahuber'/n for n in ('results.qmd','solutions.qmd','findings.qmd','agreement.pdf','trajectory.pdf'))
    save(capture([document],inputs=inputs,configuration={'format':fmt,'focused_rewrite':True,'draft':draft},
                 command=command),OUT/f'manuscript-optimized-{fmt}.provenance.json')
    print('Verified and recorded:',document)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('format',choices=['pdf','html'])
    p.add_argument('--record-existing',action='store_true');p.add_argument('--draft',action='store_true')
    args=p.parse_args();run(args.format,args.record_existing,args.draft)
