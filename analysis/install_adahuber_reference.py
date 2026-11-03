"""Install checksum-pinned adaHuber without changing its numerical sources."""
from pathlib import Path
import json,os,subprocess,tarfile,sys
from fetch_sources import fetch
from pipeline import ROOT,OUT,capture,save


def main():
    registry=ROOT/'sources/adahuber.json'
    entry=json.loads(registry.read_text());fetch(entry)
    target=ROOT/'sources/private/adahuber-1.1';target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(ROOT/entry['path']) as archive:
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).suffix in ('.o','.so','.dll'):continue
            destination=target.joinpath(*Path(member.name).parts[1:]).resolve()
            if not destination.is_relative_to(target.resolve()):raise ValueError('Unsafe source path')
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
    library=OUT/'r-library';library.mkdir(exist_ok=True)
    env=os.environ.copy()
    if sys.platform=='darwin':
        sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
        compiler=subprocess.check_output(['xcrun','--find','clang++'],text=True).strip()
        rhome=Path(subprocess.check_output(['R','RHOME'],text=True).strip())
        flags=OUT/'adahuber-Makevars'
        lines=[f'CXX11 = {compiler} -isysroot {sdk} -isystem {sdk}/usr/include/c++/v1',
               'CXX11FLAGS = -O2 -g']
        fortran=rhome/'lib/libgfortran.5.dylib';quad=rhome/'lib/libquadmath.0.dylib'
        if fortran.exists() and quad.exists():lines.append(f'FLIBS = {fortran} {quad} -lm')
        flags.write_text('\n'.join(lines)+'\n');env['R_MAKEVARS_USER']=str(flags)
    command=['R','CMD','INSTALL','--preclean','--library='+str(library),str(target)]
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    log=OUT/'adahuber-install.log';log.write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('adaHuber installation failed. See '+str(log))
    binary=next((library/'adaHuber/libs').glob('*.so'))
    inputs=[registry,ROOT/entry['path'],target/'src/adaHuber.cpp',target/'R/adaHuber.R']
    if env.get('R_MAKEVARS_USER'):inputs.append(Path(env['R_MAKEVARS_USER']))
    save(capture([binary,log],inputs=inputs,configuration={'version':'1.1','numerical_source_modified':False},command=command),OUT/'adahuber-install.provenance.json')
    print('Installed unchanged adaHuber 1.1:',binary)

if __name__=='__main__':main()
