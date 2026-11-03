"""External classifications from saved Figure1 residual/gap audits.

The package's internal stopping flag is necessary but not sufficient for the
independent strict classification. No raw solver return is rewritten.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
from pipeline import OUT,capture,save,verify

DEST=OUT/'figure1-complete'


def main():
    manifest=DEST/'provenance.json'
    verify(json.loads(manifest.read_text()),source_paths=['analysis/figure1_complete.py',
        *['src/concomitant_huber/'+s+'.py' for s in ['qualified_path','scale_solver','scale_updates','optimized','boundary','solver']]])
    study=json.loads((DEST/'summary.json').read_text());rows=[];inputs=[manifest]
    for row in study['results']:
        item=dict(case=row['case'],n=row['n'],p=row['p'],design=row['design'],seed=row['seed'],methods={})
        for method in ('fixed','joint'):
            path=DEST/row['case']/(method+'.npz');inputs.append(path)
            z=np.load(path);positive=z['scales']>0
            passed=z['strict'].copy()
            passed[positive]&=z['coefficient_kkt'][positive]<=1e-6
            if method=='joint':passed[positive]&=z['scale_kkt'][positive]<=1e-6
            passed[~positive]&=z['normalized_gap'][~positive]<=1e-6
            item['methods'][method]=dict(strict_points=int(passed.sum()),strict_complete=bool(passed.all()),
                approximate_points=int((~passed).sum()),practical_gap_points=int((z['normalized_gap']<=1e-4).sum()),
                practical_gap_complete=bool((z['normalized_gap']<=1e-4).all()),
                maximum_normalized_gap=float(max(z['normalized_gap'])),
                median_normalized_gap=float(np.median(z['normalized_gap'])),
                internal_stopping_disagreements=int(np.sum(z['strict']!=passed)))
        rows.append(item)
    save(dict(configuration={'strict_tol':1e-6,'practical_normalized_gap_tol':1e-4,
                            'scope':'independent audit; original returns and timings preserved'},results=rows),DEST/'accuracy.json')
    save(capture([DEST/'accuracy.json'],inputs=inputs),DEST/'accuracy.provenance.json')
    print({m:sum(r['methods'][m]['strict_complete'] for r in rows) for m in ('fixed','joint')})


if __name__=='__main__':main()
