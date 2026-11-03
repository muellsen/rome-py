"""Validate user-supplied unpublished AMP inputs without distributing them."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
registry=json.loads((ROOT/'sources/amp-reference.json').read_text())
for entry in registry['files']:
    path=ROOT/entry['path']
    if not path.exists():
        raise SystemExit('Optional AMP reference missing: '+entry['path']+
                         '. See sources/README.md. Package and primary experiments do not require it.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise SystemExit('Optional AMP reference checksum mismatch: '+entry['path'])
print('Optional unpublished AMP reference files verified.')
