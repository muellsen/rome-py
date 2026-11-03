"""Paired kernel audit on saved difficult points; JIT excluded from timings."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
import numpy as np
from pipeline import ROOT,OUT,capture,save
sys.path.insert(0,str(ROOT/'src'))
from concomitant_huber.path import _solve_warm
from concomitant_huber.optimized import solve_warm
from scipy.special import ndtr
rho=.5;a=.5*(1-2*rho*np.exp(-rho*rho/2)/np.sqrt(2*np.pi)+2*(rho*rho-1)*(1-ndtr(rho)))
DEST=OUT/'optimization-pilot';DEST.mkdir(exist_ok=True)
cases=['100_100_Block-AR_71','500_100_Block-AR_71','100_500_AR-t2_71','100_100_Contaminated-AR_71']
small=(np.asfortranarray(np.eye(6)),np.arange(6.),.1,rho,a,1.,True,1e-6,2,False,8,np.zeros(6))
_solve_warm(*small);solve_warm(*small);solve_warm(*small,newton_every=5)
rows=[];inputs=[]
for case in cases:
 d=OUT/'revision-rome/cases'/case;z=np.load(d/'solutions.npz')
 X=np.asfortranarray(np.loadtxt(d/'X.tsv'));y=np.loadtxt(d/'y.tsv')
 finite=np.flatnonzero(np.isfinite(z['joint_scale']));k=int(finite[-1])
 args=(X,y,float(z['penalties'][k]),rho,a,float(z['joint_scale'][k]),True,1e-6,4000,False,8,z['joint_beta'][k].copy())
 record=dict(case=case,point=k,methods={});fits={}
 for name,fun in [('reference',_solve_warm),('optimized',solve_warm),('polished',lambda *args:solve_warm(*args,newton_every=5))]:
  tt=time.perf_counter();cc=time.process_time();r=fun(*args);cpu=time.process_time()-cc;wall=time.perf_counter()-tt
  fits[name]=r
  record['methods'][name]=dict(cpu_seconds=cpu,wall_seconds=wall,sweeps=int(r[2]),converged=bool(r[4]),coefficient_kkt=float(r[5]),scale_kkt=float(r[6]),objective=float(r[3][-1]),max_objective_increase=float(np.max(np.diff(r[3]))))
  if name!='reference':record['methods'][name].update(local_roots=int(r[8]),kink_sweeps=int(r[9]))
 record['maximum_coefficient_difference']=float(max(abs(fits['reference'][0]-fits['optimized'][0])))
 rows.append(record);print(record,flush=True)
 inputs.extend(d/n for n in ['solutions.npz','X.tsv','y.tsv'])
save(dict(configuration={'cases':cases,'budget':4000,'tol':1e-6,'starting_state':'saved last attempted path point','JIT_timed':False},results=rows),DEST/'summary.json')
save(capture([DEST/'summary.json'],inputs=inputs,configuration={'paired_same_start':True}),DEST/'provenance.json')
