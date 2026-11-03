"""Wait for checked computations, run final R4 timings without other project jobs, assemble."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,sys,time,subprocess
from pathlib import Path
from pipeline import ROOT,OUT,save

def ready():
    for name,count in [('revision-rome',360),('revision-real',20)]:
        manifest=OUT/name/'provenance.json';summary=OUT/name/'summary.json'
        if not manifest.exists() or not summary.exists():return False
        data=json.loads(summary.read_text())
        if len(data['results'])!=count:return False
    return True

def main():
    while not ready():
        time.sleep(15)
    print('Both primary computations finished. Starting final R4 timing run.',flush=True)
    current=OUT/'revision-r4';preflight=OUT/'revision-r4-preflight'
    if current.exists() and not preflight.exists():current.rename(preflight)
    subprocess.run([sys.executable,'analysis/revision_r4.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'analysis/revision_amp.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'analysis/revision_assemble.py'],cwd=ROOT,check=True)
    save({'stage':'all numerical computations and focused-paper assembly complete',
          'ready_for_render':True},OUT/'revision-ready.json')
    print('Numerical workflow complete; canonical manuscript ready for final render.',flush=True)

if __name__=='__main__':main()
