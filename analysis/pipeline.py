"""Small provenance and manuscript workflow; standard library only."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/outputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def sources():
    candidates = [ROOT / n for n in ('pyproject.toml', 'Makefile', '_quarto.yml', 'AGENTS.md', 'README.md',
                                                  'LICENSE', 'CITATION.cff', 'THIRD_PARTY.md',
                                                  'CONTRIBUTING.md', 'MANIFEST.in')]
    candidates.extend(ROOT.glob('_quarto-*.yml'))
    for directory in ('src', 'analysis', 'paper', 'sources', 'skills', 'tests', '.github'):
        candidates.extend((ROOT / directory).rglob('*'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(candidates))
            if p.is_file() and p != ROOT / 'paper/manuscript.tex' and not any(s in p.parts for s in
                ('__pycache__', 'outputs', 'private', '.quarto'))
            and not any(part.endswith(('.egg-info', '_files')) for part in p.parts)
            and p.suffix not in ('.aux', '.log', '.toc', '.out', '.bcf', '.blg', '.bbl',
                                 '.fls', '.fdb_latexmk')
            and not p.name.endswith(('.run.xml', '.synctex.gz'))
            and (p.suffix not in ('.html', '.pdf', '.pyc', '.zip')
                 or p.is_relative_to(ROOT / 'sources')
                 or p.is_relative_to(ROOT / 'paper/vendor'))}


def capture(outputs, *, inputs=(), configuration=None, seed=None, command=None):
    return {
        'schema_version': 1,
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'command': command or [sys.executable, *sys.argv],
        'cwd': str(ROOT),
        'configuration': configuration or {}, 'seed': seed,
        'git': {'commit': git('rev-parse', 'HEAD'),
                'status': git('status', '--porcelain', '--untracked-files=all'),
                'diff_sha256': hashlib.sha256((git('diff', 'HEAD') or '').encode()).hexdigest()},
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(),
                        'packages': sorted(f'{d.metadata["Name"]}=={d.version}'
                                           for d in importlib.metadata.distributions()),
                        'threads': {k: os.environ.get(k) for k in
                                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'sources': sources(),
        'inputs': {str(Path(p).relative_to(ROOT)): digest(p) for p in inputs},
        'outputs': {str(Path(p).relative_to(ROOT)): digest(p) for p in outputs},
    }


def save(manifest, path):
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


def verify(manifest, source_paths=None):
    for group in ('sources', 'inputs', 'outputs'):
        entries = manifest[group]
        if group == 'sources' and source_paths is not None:
            missing = set(source_paths) - set(entries)
            if missing:
                raise ValueError(f'Missing source dependency in manifest: {missing}')
            entries = {name: entries[name] for name in source_paths}
        for name, expected in entries.items():
            path = ROOT / name
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'Stale or missing {group} artifact: {name}')
    if source_paths is None and manifest['sources'] != sources():
        raise ValueError('Source inventory changed; regenerate outputs')


def smoke():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'status.qmd'
    output.write_text('Provenance smoke check completed; see the separate pilot report for solver results.\n')
    manifest = capture([output], inputs=[ROOT / 'analysis/configs/rome.json',
                                        ROOT / 'analysis/configs/r4.json'],
                       configuration={'stage': 'workflow-smoke', 'scientific_results': False},
                       seed=None)
    save(manifest, OUT / 'status.provenance.json')
    verify(manifest)
    print('Provenance smoke check passed.')


def render(fmt, *, already_rendered=False):
    verify(json.loads((OUT / 'status.provenance.json').read_text()))
    pilot_manifest = OUT / 'small-scale.provenance.json'
    if not pilot_manifest.exists():
        raise RuntimeError('Run make pilot before rendering the manuscript')
    verify(json.loads(pilot_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'src/concomitant_huber/__init__.py',
        'analysis/small_scale_experiment.py'])
    r4_manifest = OUT / 'r4.provenance.json'
    verify(json.loads(r4_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'analysis/r4_benchmark.py',
        'sources/classo-1.0.11-audit/registry.json'])
    table_manifest = OUT / 'paper-tables.provenance.json'
    verify(json.loads(table_manifest.read_text()), source_paths=['analysis/assemble_tables.py'])
    diagnostic_manifest = OUT / 'convergence-amp.provenance.json'
    verify(json.loads(diagnostic_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'src/concomitant_huber/diagnostics.py',
        'analysis/convergence_amp_experiment.py', 'analysis/export_glass.R'])
    amp_manifest = OUT / 'amp-source-protocol.provenance.json'
    verify(json.loads(amp_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'analysis/amp_source_protocol.py'])
    scaling_manifest = OUT / 'r4-path-scaling.provenance.json'
    verify(json.loads(scaling_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'analysis/r4_benchmark.py', 'analysis/r4_path_scaling.py'])
    rome_manifest = OUT / 'rome-paper-validation/provenance.json'
    verify(json.loads(rome_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'src/concomitant_huber/path.py',
        'analysis/rome_paper_experiment.py', 'analysis/rome_fixed_path.R', 'analysis/configs/rome.json'])
    rome_figure_manifest = OUT / 'rome-paper-validation/figure.provenance.json'
    verify(json.loads(rome_figure_manifest.read_text()), source_paths=['analysis/assemble_path_report.py'])
    selected_manifest = OUT / 'selected-convergence.provenance.json'
    verify(json.loads(selected_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'src/concomitant_huber/diagnostics.py', 'analysis/rome_convergence.py'])
    incomplete_manifest = OUT / 'incomplete-path-audit/provenance.json'
    verify(json.loads(incomplete_manifest.read_text()), source_paths=[
        'src/concomitant_huber/solver.py', 'src/concomitant_huber/path.py',
        'src/concomitant_huber/path_diagnostics.py', 'analysis/audit_incomplete_paths.py'])
    command = ['quarto', 'render', 'paper/manuscript.qmd', '--to', fmt]
    version = subprocess.check_output(['quarto', '--version'], text=True).strip()
    if not already_rendered:
        subprocess.run(command, cwd=ROOT, check=True)
    outputs = sorted((ROOT / '_build').rglob('manuscript.' + fmt))
    tex = ROOT / 'paper/manuscript.tex'
    if fmt == 'pdf' and tex.exists():
        outputs.append(tex)
    if not outputs:
        raise RuntimeError('Quarto returned without the expected document')
    if already_rendered:
        render_sources = [ROOT / p for p in ('paper/manuscript.qmd', 'paper/algorithm.qmd',
                           'paper/blue-additions.lua', 'paper/references.bib', '_quarto.yml')]
        document = next(p for p in outputs if p.suffix == '.' + fmt)
        if document.stat().st_mtime < max(p.stat().st_mtime for p in render_sources):
            raise ValueError('External render predates manuscript sources')
    save(capture(outputs, inputs=[OUT / 'status.qmd', OUT / 'status.provenance.json',
                                 OUT / 'fixed-scale-results.qmd', OUT / 'r4-results.qmd', OUT / 'r4-setup.qmd', pilot_manifest, r4_manifest, table_manifest, diagnostic_manifest, OUT / 'convergence-results.qmd',
                                 OUT / 'amp-results.qmd', OUT / 'real-data-results.qmd',
                                 OUT / 'convergence-traces.pdf', amp_manifest,
                                 OUT / 'amp-source-protocol-results.qmd', OUT / 'amp-matched-timings.qmd',
                                 scaling_manifest, OUT / 'r4-path-scaling-results.qmd', OUT / 'r4-path-scaling.pdf',
                                 rome_manifest, OUT / 'rome-paper-validation/results.qmd',
                                 OUT / 'rome-paper-validation/returned-paths.pdf', rome_figure_manifest,
                                 incomplete_manifest, OUT / 'incomplete-path-audit/summary.json',
                                 selected_manifest, OUT / 'selected-convergence-results.qmd', OUT / 'selected-convergence.pdf'],
                 configuration={'format': fmt, 'quarto_version': version, 'renderer_run_externally': already_rendered}, command=command),
         OUT / f'manuscript-{fmt}.provenance.json')
    print('Rendered:', ', '.join(str(p.relative_to(ROOT)) for p in outputs))


if __name__ == '__main__':
    if sys.argv[1:] == ['smoke']:
        smoke()
    elif len(sys.argv) == 3 and sys.argv[1] == 'render' and sys.argv[2] in ('html', 'pdf'):
        render(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'record' and sys.argv[2] in ('html', 'pdf'):
        render(sys.argv[2], already_rendered=True)
    else:
        raise SystemExit('Usage: python analysis/pipeline.py smoke | {render,record} {html,pdf}')
