"""Run independent ROME replications on four workers, reporting process CPU time."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
from revision_rome_timing import CONFIG,DEST,case,assemble,ROOT,OUT,capture,save,_solve_warm

def initialize():
    _solve_warm(np.asfortranarray(np.eye(5)),np.arange(5.),.1,.5,CONFIG['a'],1.,True,1e-6,10,False,8,np.zeros(5))

def execute(job):
    return case(*job)

def main():
    DEST.mkdir(exist_ok=True)
    jobs=[(n,p,d,s) for s in CONFIG['seeds'] for n,p in CONFIG['dimensions'] for d in CONFIG['designs']]
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            ordered=sorted(rows,key=lambda r:(r['seed'],r['n'],r['p'],r['design']))
            assemble(ordered)
            print('Completed',len(rows),'of',len(jobs),flush=True)
    inputs=[ROOT/'sources/rome-99d6f07.tar.gz',ROOT/'analysis/configs/rome.json']
    inputs.extend((DEST/'cases').rglob('*.tsv'));inputs.extend((DEST/'cases').rglob('R-console.txt'))
    outputs=[DEST/'summary.json',DEST/'results.qmd',*(DEST/'cases').rglob('result.json'),*(DEST/'cases').rglob('solutions.npz')]
    save(capture(outputs,inputs=inputs,configuration={**CONFIG,'worker_processes':4,
         'parallel_note':'independent paired replications; CPU times are process CPU, not elapsed wall time'},
         seed=CONFIG['seeds']),DEST/'provenance.json')

if __name__=='__main__':main()
