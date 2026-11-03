"""Audit Git submission candidates and distribution contents without staging files."""
from pathlib import Path
import json,re,subprocess,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                      cwd=ROOT,check=True,capture_output=True)
paths=sorted(set(p.decode() for p in result.stdout.split(b'\0') if p))
bad=[];total=0
for name in paths:
    path=ROOT/name
    if not path.is_file():continue
    if name=='analysis/outputs/.gitkeep':continue
    total+=path.stat().st_size
    if (name.startswith(('analysis/outputs/','sources/private/','sources/amp-aistats2027/','_build/','dist/'))
        or path.suffix in ('.pdf','.RData','.mat','.zip','.gz','.pyc','.html')
        or any(part.endswith('_files') for part in path.parts)):
        bad.append(name)
    if path.suffix in ('.py','.json','.qmd','.md','.toml','.yml','.yaml'):
        if re.search(r"/Users/[A-Za-z][A-Za-z0-9_.-]*/",path.read_text(errors='replace')):
            bad.append(name+' (local absolute user path)')
if bad:raise SystemExit('Excluded/private material in Git candidates:\n'+'\n'.join(bad))
distributions=[]
for path in (ROOT/'dist').glob('*'):
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as archive:names=archive.namelist()
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path) as archive:names=archive.getnames()
    else:continue
    if any(any(part in ('analysis','paper','sources','skills','tests','.github') for part in Path(n).parts) for n in names):
        raise SystemExit('Research/private tree leaked into distribution: '+path.name)
    distributions.append(path.name)
print(json.dumps({'git_submission_files':len(paths),'git_submission_bytes':total,
                  'distributions_checked':distributions,'private_or_binary_candidates':0},indent=2))
