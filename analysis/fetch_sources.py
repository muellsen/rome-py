"""Retrieve public inputs by recorded SHA-256; never overwrite changed inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(entry, offline=False):
    destination=ROOT/entry['path']
    if destination.exists():
        if sha(destination)!=entry['sha256']:
            raise ValueError(f'Changed cached input: {entry["path"]}')
        return
    if offline:raise FileNotFoundError(f'Missing cached input: {entry["path"]}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    request=urllib.request.Request(entry['url'],headers={'User-Agent':'concomitant-huber-reproducibility/0.0.1'})
    with urllib.request.urlopen(request,timeout=90) as response:
        contents=response.read()
    if hashlib.sha256(contents).hexdigest()!=entry['sha256']:
        raise ValueError(f'Checksum mismatch: {entry["url"]}; input was not installed')
    temporary=destination.with_name(destination.name+'.download')
    temporary.write_bytes(contents)
    temporary.replace(destination)


def main(scope='primary',offline=False):
    registry=json.loads((ROOT/'sources/registry.json').read_text())
    for entry in registry:
        if scope in ('literature','all-public') or entry['path'].endswith('rome-99d6f07.tar.gz'):
            fetch(entry,offline)
    # Portable source audit registry; do not replace historical local provenance.
    audit=ROOT/'sources/classo-1.0.11-audit'
    registry_path=audit/'registry.json'
    if not registry_path.exists():
        registry_path.write_bytes((audit/'registry.public.json').read_bytes())
    portable=json.loads((audit/'registry.public.json').read_text())
    for entry in portable['files']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Changed c-lasso audit snapshot')
    if scope in ('data','all-public'):
        data=ROOT/'sources/data';data.mkdir(exist_ok=True)
        with tarfile.open(ROOT/'sources/rome-99d6f07.tar.gz') as archive:
            for basename in ('glass.RData','Section4_5.R'):
                destination=data/basename
                member=next(m for m in archive.getmembers() if m.name.endswith('/'+basename))
                contents=archive.extractfile(member).read()
                if destination.exists() and destination.read_bytes()!=contents:raise ValueError('Changed glass input')
                destination.write_bytes(contents)
        recorded=json.loads((data/'registry.json').read_text())
        fetch({'path':'sources/data/riboflavinData.mat','url':recorded['sources']['riboflavin'],
               'sha256':recorded['sha256']['sources/data/riboflavinData.mat']},offline)
        for name in ('glass.RData','Section4_5.R'):
            path='sources/data/'+name
            if sha(ROOT/path)!=recorded['sha256'][path]:raise ValueError('Glass checksum mismatch')
    print('Public source inputs verified:',scope)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=('primary','data','literature','all-public'),default='primary')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();main(args.scope,args.offline)
