import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
"""Re-create (without fitting any model) the calibration draws behind Fig. 3/Table 2 (protocol 'exp') and Fig. 4 (coverage experiment),
using exactly the same seeds and random-number order as bench2.py / coverage.py, and write the selected cut IDs and test-experiment IDs."""
import os, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from coverage import *
import pandas as pd, numpy as np
GID='global_path_id'
src=pd.read_csv(os.environ.get('MSM_MASTER_CSV','data/master_cutting_paths_CLEAN.csv'))[['experiment_id','path_id',GID]]
m=m.merge(src[['experiment_id','path_id',GID]],on=['experiment_id','path_id'],how='left') if GID not in m.columns else m
cal_rows=[]; split_rows=[]
# ---- Fig. 4 / coverage experiment (coverage.py, seeds 104729*rep+5), 80 repeats
for rep in range(80):
    for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]:
        rng=np.random.default_rng(104729*rep+5); T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]; pool,te=split_exp(T,rng)
        split_rows.append(dict(experiment='coverage',dir=dname,rep=rep,test_experiments=';'.join(sorted(te.experiment_id.unique())),n_pool_cuts=len(pool),n_test_cuts=len(te)))
        for k in BUD:
            for sc,par in SCHEMES:
                cal=draw(pool,sc,par,k,rng)
                if cal is None: continue
                cal_rows.append(dict(experiment='coverage',dir=dname,rep=rep,k=k,scheme=f'{sc}{par if par else ""}',n_experiments=cal.experiment_id.nunique(),n_combos=cal.combo.nunique(),cut_ids=';'.join(cal[GID])))
# ---- Fig. 3 / Table 2 'exp' protocol (bench2.py, seeds 7919*rep+13), 80 repeats
for rep in range(80):
    for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]:
        rng=np.random.default_rng(7919*rep+13); T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]; pool,te=split_exp(T,rng)
        split_rows.append(dict(experiment='budget_curve',dir=dname,rep=rep,test_experiments=';'.join(sorted(te.experiment_id.unique())),n_pool_cuts=len(pool),n_test_cuts=len(te)))
        for k in KS:
            if k>len(pool): continue
            cal=pool.iloc[rng.permutation(len(pool))[:k]]
            cal_rows.append(dict(experiment='budget_curve',dir=dname,rep=rep,k=k,scheme='random',n_experiments=cal.experiment_id.nunique(),n_combos=(cal.Tool+'|'+cal.Workpiece).nunique(),cut_ids=';'.join(cal[GID])))
pd.DataFrame(split_rows).to_csv('results/test_splits.csv',index=False); pd.DataFrame(cal_rows).to_csv('results/calibration_indices.csv',index=False)
# ---- verification against the archived result tables
C=pd.DataFrame([r for r in cal_rows if r['experiment']=='coverage']); D=pd.read_csv('results/cov_draws.csv')
J=C.merge(D,left_on=['dir','rep','k','scheme'],right_on=['dir','rep','k','name'],how='inner')
print('coverage draws re-created:',len(C),'| matched to archived rows:',len(J),'| n_experiments identical:',bool((J.n_experiments==J.n_exp).all()),'| n_combos identical:',bool((J.n_combos==J.n_combo).all()))
