"""Create a reviewable GitHub source archive from nonignored Git candidates."""
from pathlib import Path
import subprocess,zipfile
from pipeline import ROOT,OUT,capture,save
subprocess.run(['python3','analysis/submission_check.py'],cwd=ROOT,check=True)
result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                      cwd=ROOT,check=True,capture_output=True)
files=sorted(set(ROOT/p.decode() for p in result.stdout.split(b'\0') if p))
destination=ROOT/'dist/rome-py-github.zip';destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_file():archive.write(path,'rome-py/'+str(path.relative_to(ROOT)))
artifacts=[destination,*sorted((ROOT/'dist').glob('*.whl')),*sorted((ROOT/'dist').glob('*.tar.gz'))]
save(capture(artifacts,inputs=[p for p in files if p.is_file()],
     configuration={'scope':'nonignored Git source candidates; no remote publication'}),
     OUT/'github-submission.provenance.json')
print('GitHub source archive:',destination)
