"""Assemble both equally weighted comparisons from verified numerical artifacts."""
from pathlib import Path
import json
from pipeline import OUT,ROOT,capture,save,verify


def table(headers,rows,caption,label):
    # Quarto/Pandoc native tables, narrow enough to remain legible in PDF.
    body=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    body+=['| '+' | '.join(map(str,row))+' |' for row in rows]
    body+=['',f': {caption} {{#{label}}}','']
    return '\n'.join(body)


def run():
    pilot_manifest=OUT/'small-scale.provenance.json'
    r4_manifest=OUT/'r4.provenance.json'
    verify(json.loads(pilot_manifest.read_text()),source_paths=['src/concomitant_huber/solver.py','analysis/small_scale_experiment.py'])
    verify(json.loads(r4_manifest.read_text()),source_paths=['src/concomitant_huber/solver.py','analysis/r4_benchmark.py','sources/classo-1.0.11-audit/registry.json'])
    pilot=json.loads((OUT/'small-scale-summary.json').read_text())
    rows=[]
    noise={'gaussian':'Gaussian','t3':'$t_3$','contaminated':'Contam.'}
    for r in pilot['results']:
        rows.append([f"{r['n']} x {r['p']}",noise[r['noise']],f"{r['median_ms']['fixed_at_joint_scale']:.1f}",f"{r['median_ms']['joint_mm8']:.1f}",f"{r['median_paired_ratio']:.2f}"])
    fixed=OUT/'fixed-scale-results.qmd'
    fixed.write_text(table(['Design','Noise','Fixed (ms)','Joint (ms)','Ratio'],rows,
        'Fixed-scale exact CD and concomitant MM8-CD. Times are warmed estimator-fit medians; ratios are medians of within-seed ratios. Three seeds and seven repeats per seed.','tbl-fixed'))
    data=json.loads((OUT/'r4-summary.json').read_text())
    results=data['summary']
    rows=[];stock=[]
    for n,p in data['config']['dimensions']:
        for ns in data['config']['noise']:
            group={r['method']:r for r in results if r['n']==n and r['noise']==ns}
            def cell(method):
                r=group[method]
                text=f"{r['median_solver_ms']:.1f}"
                return text+('†' if r['qualified']<r['total'] else '')
            rows.append([f'{n} x {p}',noise[ns],cell('MM-CD'),cell('DR-balanced'),cell('Path-Alg'),cell('Path-Alg-eps0')])
            r=group['DR-stock']
            stock.append([f'{n} x {p}',noise[ns],f"{r['median_solver_ms']:.1f}",f"{r['qualified']}/{r['total']}"])
    r4=OUT/'r4-results.qmd'
    r4.write_text(table(['Design','Noise','MM-CD','DR*','Path','Path0*'],rows,
        'Prepared numerical solve time in milliseconds for unconstrained R4. Two seeds and three repeats per seed. DR* uses metric c=1; Path0* sets the path diagonal correction to zero. † marks a cell containing accuracy-gate failures and excludes it from speed rankings.','tbl-r4'))
    setup_rows=[]
    for method in data['config']['methods']:
        rr=[r for r in results if r['method']==method]
        import statistics
        setup_rows.append([method,f"{statistics.median(r['median_setup_ms'] for r in rr):.2f}",sum(r['qualified'] for r in rr),sum(r['total'] for r in rr)])
    setup=OUT/'r4-setup.qmd'
    setup.write_text(table(['Method','Setup (ms)','Qualified','Attempts'],setup_rows,
        'Median setup cost across the six setting-level medians, and accuracy qualification counts. The stock DR metric has a 1000-update diagnostic budget; successful returns are repeated three times.','tbl-r4-setup')+'\n'+table(['Design','Noise','Budget time (ms)','Qualified'],stock,
        'Stock-metric DR diagnostics. Times are costs of the budgeted attempt, not time to an accurate solution; no speed ranking uses unsuccessful attempts.','tbl-r4-stock'))
    outputs=[fixed,r4,setup]
    manifest=capture(outputs,inputs=[pilot_manifest,r4_manifest,OUT/'small-scale-summary.json',OUT/'r4-summary.json',OUT/'r4-native-stop-attempt.json'],configuration={'role':'two equally weighted main comparisons; separate setup and stock diagnostic tables'})
    save(manifest,OUT/'paper-tables.provenance.json');verify(manifest)
    print('Assembled both main comparison tables and setup diagnostics.')

if __name__=='__main__':run()
