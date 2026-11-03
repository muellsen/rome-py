"""Verify the primary report's data dependencies and record its rendered artifact."""
import json,subprocess
from pathlib import Path
from pipeline import ROOT,OUT,capture,save,verify
CHECKS={
 'rome-paper-validation/provenance.json':['src/concomitant_huber/solver.py',
     'src/concomitant_huber/path.py','analysis/rome_paper_experiment.py','analysis/rome_fixed_path.R','analysis/configs/rome.json'],
 'rome-paper-validation/figure.provenance.json':['analysis/assemble_path_report.py'],
 'r4-path-scaling.provenance.json':['src/concomitant_huber/solver.py','analysis/r4_benchmark.py','analysis/r4_path_scaling.py']}
for name,deps in CHECKS.items():verify(json.loads((OUT/name).read_text()),source_paths=deps)
command=['quarto','render','analysis/report.qmd','--profile','public','--to','html']
if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--verify-only']:
        print('Primary report dependencies verified.')
    else:
        if sys.argv[1:]!=['--record-existing']:subprocess.run(command,cwd=ROOT,check=True)
        output=ROOT/'_build/analysis/report.html'
        if not output.exists():raise SystemExit('Missing rendered primary report')
        if output.stat().st_mtime<(ROOT/'analysis/report.qmd').stat().st_mtime:
            raise SystemExit('Report predates its source')
        save(capture([output],inputs=[*[OUT/n for n in CHECKS],ROOT/'_quarto-public.yml'],
            configuration={'scope':'public primary experiments','optional_AMP_required':False},
            command=command),OUT/'primary-report.provenance.json')
        print('Primary report recorded:',output)
