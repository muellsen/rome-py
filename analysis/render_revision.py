"""Verify the focused paper's numerical inputs and render or record its output."""
import argparse,json,subprocess
from pathlib import Path
from pipeline import ROOT,OUT,capture,save,verify
from fetch_algorithm_template import ensure

CHECKS={
 'revision-rome/provenance.json':['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
   'analysis/revision_rome_timing.py','analysis/revision_rome_parallel.py','analysis/revision_rome_reference.R','analysis/rome_paper_experiment.py'],
 'revision-r4/provenance.json':['src/concomitant_huber/solver.py','analysis/r4_benchmark.py',
   'analysis/revision_r4.py','src/concomitant_huber/diagnostics.py'],
 'revision-real/provenance.json':['src/concomitant_huber/solver.py','src/concomitant_huber/path.py',
   'src/concomitant_huber/model_selection.py','analysis/revision_real_data.py','analysis/revision_real_parallel.py'],
 'revision-amp/provenance.json':['src/concomitant_huber/solver.py','analysis/revision_amp.py'],
 'amp-source-protocol.provenance.json':['src/concomitant_huber/solver.py','analysis/amp_source_protocol.py'],
 'revision-paper/provenance.json':['analysis/revision_assemble.py']}

def run(fmt,record=False,draft=False):
    ensure()
    for name,dependencies in CHECKS.items():
        if draft and (name.startswith('revision-rome/') or not (OUT/name).exists()):continue
        verify(json.loads((OUT/name).read_text()),source_paths=dependencies)
    assembled=json.loads((OUT/'revision-paper/provenance.json').read_text())
    if assembled['configuration']['draft'] and not draft:raise ValueError('Refusing a final render from interim results')
    command=['quarto','render','paper/manuscript.qmd','--to',fmt]
    if not record:subprocess.run(command,cwd=ROOT,check=True)
    document=ROOT/'_build/paper'/('manuscript.'+fmt)
    if not document.exists():raise RuntimeError('Rendered manuscript missing')
    sources=[ROOT/'paper'/n for n in ('manuscript.qmd','algorithm.qmd','algorithm-html.qmd','jcgs-preamble.tex','references.bib','blue-additions.lua')]
    sources.append(ROOT/'_quarto.yml')
    if document.stat().st_mtime<max(p.stat().st_mtime for p in sources):raise ValueError('Manuscript output predates current sources')
    inputs=[OUT/n for n in CHECKS if not(draft and n.startswith('revision-rome/'))]
    inputs.extend(OUT/'revision-paper'/n for n in ('timing-findings.qmd','equivalence-findings.qmd','amp-matched.qmd','amp-findings.qmd','rome-timing.pdf','real-data.pdf','r4-findings.qmd','real-findings.qmd'))
    inputs.extend([OUT/'revision-r4/results.qmd',OUT/'revision-r4/scaling.pdf',OUT/'revision-r4/trajectory.pdf',OUT/'revision-real/results.qmd'])
    save(capture([document],inputs=inputs,configuration={'format':fmt,'focused_rewrite':True,'draft':draft},
                 command=command),OUT/f'manuscript-{fmt}.provenance.json')
    print('Verified and recorded:',document)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('format',choices=['pdf','html'])
    p.add_argument('--record-existing',action='store_true');p.add_argument('--draft',action='store_true')
    args=p.parse_args();run(args.format,args.record_existing,args.draft)
