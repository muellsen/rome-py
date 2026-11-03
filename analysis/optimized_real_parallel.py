"""Independent outer CV partitions on four workers; no CV-runtime claims."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
from scipy.io import loadmat
from optimized_real_data import CONFIG,DEST,run_case,RHO,A,ROOT,OUT,capture,save,standardized_path
DATA=None

def initialize():
    global DATA
    standardized_path(np.eye(6),np.arange(6.),[.5],rho=RHO,a=A,max_iter=5)
    mat=loadmat(ROOT/'sources/data/riboflavinData.mat')['riboData']
    DATA={'Glass':(np.loadtxt(ROOT/'sources/data/glass-X.tsv'),np.loadtxt(ROOT/'sources/data/glass-y.tsv')),
          'B-subtilis':(mat[1:].T,mat[0])}

def execute(job):
    name,outer=job
    return run_case(name,*DATA[name],outer)

def main():
    DEST.mkdir(exist_ok=True)
    jobs=[(name,k) for k in range(10) for name in ['Glass','B-subtilis']]
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
      for future in as_completed([pool.submit(execute,j) for j in jobs]):
        rows.append(future.result())
        save(dict(configuration=CONFIG,results=sorted(rows,key=lambda r:(r['split'],r['dataset']))),DEST/'summary.json')
    lines=['| Data | Method | Successful splits | Test normalized MSE | Nonzeros |',
           '|:--|:--|--:|--:|--:|']
    for name in ['Glass','B-subtilis']:
      for method,label in [('rome','Fixed-scale CD'),('joint','Concomitant CD')]:
        rr=[r['methods'][method] for r in rows if r['dataset']==name and r['methods'][method]['status']=='returned']
        fmt=lambda key:f"{np.mean([r[key] for r in rr]):.3f} ({np.std([r[key] for r in rr],ddof=1):.3f})" if len(rr)>1 else '—'
        lines.append(f"| {name} | {label} | {len(rr)}/10 | {fmt('normalized_mse')} | {fmt('nonzero')} |")
    (DEST/'results.qmd').write_text('\n'.join(lines)+'\n')
    inputs=[ROOT/'sources/data'/n for n in ('glass-X.tsv','glass-y.tsv','riboflavinData.mat')]
    outputs=[DEST/'summary.json',DEST/'results.qmd',*DEST.rglob('result.json'),*DEST.rglob('evaluation.npz'),*DEST.rglob('fits.npz'),*DEST.rglob('fits.json'),*DEST.rglob('anchors.json'),*DEST.rglob('lambda.tsv'),*DEST.rglob('X.tsv'),*DEST.rglob('y.tsv'),*DEST.rglob('columns.npy')]
    save(capture(outputs,inputs=inputs,configuration={**CONFIG,'worker_processes':4},seed=CONFIG['seed']),DEST/'provenance.json')

if __name__=='__main__':main()
