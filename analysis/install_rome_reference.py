"""Build the pinned original rome package in the ignored project-local R library."""
from pathlib import Path
import subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'sources/private/rome-build';target.mkdir(parents=True,exist_ok=True)
with tarfile.open(ROOT/'sources/rome-99d6f07.tar.gz') as archive:
    for member in archive.getmembers():
        if not member.isfile() or Path(member.name).suffix in ('.o','.dll','.so'):continue
        destination=target.joinpath(*Path(member.name).parts[1:]).resolve()
        if not destination.is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(archive.extractfile(member).read())
library=ROOT/'analysis/outputs/r-library';library.mkdir(exist_ok=True)
subprocess.run(['R','CMD','INSTALL','--preclean','--library='+str(library),str(target)],check=True,cwd=ROOT)
