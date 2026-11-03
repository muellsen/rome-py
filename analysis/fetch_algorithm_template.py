"""Fetch pinned CTAN packages used for the ROME algorithm environment."""
import hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def ensure():
    registry=json.loads((ROOT/'sources/algorithm-template.json').read_text())
    target=ROOT/'paper/vendor/algorithm2e';target.mkdir(parents=True,exist_ok=True)
    if all((target/name).exists() for name in ('algorithm2e.sty','ifoddpage.sty','relsize.sty','ydocstrip.tex')):
        return
    for name,record in registry.items():
        with tempfile.TemporaryDirectory() as work:
            archive=Path(work)/(name+'.zip')
            subprocess.run(['curl','-L','--fail','--max-time','60',record['url'],'-o',str(archive)],check=True)
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('CTAN archive differs from recorded source')
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    if member.endswith(('.sty','.dtx','.ins')):
                        (target/Path(member).name).write_bytes(z.read(member))
    extraction=(r'\input docstrip.tex'+'\n'+r'\keepsilent\askforoverwritefalse'+'\n'
                +r'\generate{\file{ydocstrip.tex}{\from{ydoc.dtx}{ydocstrip.tex}}}'+'\n'
                +r'\endbatchfile'+'\n')
    (target/'build-ydoc.ins').write_text(extraction)
    for installer in ('build-ydoc.ins','ifoddpage.ins'):
        subprocess.run(['latex','-interaction=nonstopmode',installer],cwd=target,check=True,capture_output=True)

if __name__=='__main__':ensure()
