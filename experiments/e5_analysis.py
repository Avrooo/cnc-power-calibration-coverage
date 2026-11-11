import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from analyze2 import boot_tables
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
D=pd.read_csv('results/e5_draws.csv'); E=pd.read_csv('results/e5_exp.csv')
print('draws',len(D),'; feasible draws by scenario/k/strategy:'); print(D.pivot_table(index=['scen','strategy'],columns='k',values='rep',aggfunc='count').to_string())
print('\n#### mean MAE (W, both directions pooled) and coverage diagnostics')
g=D.groupby(['scen','k','strategy']).agg(n_exp=('n_exp','mean'),n_combo=('n_combo','mean'),mat_cov=('mat_cov','mean'),nn=('nn','mean'),snap=('snap','mean'),T_offset=('MAE_T-offset','mean'),T_phys=('MAE_T-physics','mean'),O_phys=('MAE_O-physics','mean')).round(2)
print(g.to_string())
E['method']=E.method+'@'+E.protocol; res=boot_tables(E,'sae','n','x')
def dl(d,k1,a,k2,b):
    if (d,k1,a) not in res or (d,k2,b) not in res: return None
    x=res[(d,k1,a)][0]-res[(d,k2,b)][0]; return round(np.nanmean(x)),f'[{np.nanpercentile(x,2.5):.0f},{np.nanpercentile(x,97.5):.0f}]',round(np.nanmean(x<0),2)
print('\n#### paired cluster-bootstrap, MAE(first)-MAE(second) in W (negative = first better), 95% CI, P(first better)')
rows=[]
for scen in ['full','narrow6','narrow3']:
  for d in sorted(E['dir'].unique()):
    for k in [60,150,250]:
        for lab,(k1,a,k2,b) in {
          'planned vs random (O-physics)':(k,f'O-physics@{scen}|planned',k,f'O-physics@{scen}|random'),
          'planned vs complete-exp (O-physics)':(k,f'O-physics@{scen}|planned',k,f'O-physics@{scen}|complete_exp'),
          'planned vs planned-no-material (O-physics)':(k,f'O-physics@{scen}|planned',k,f'O-physics@{scen}|planned_nomat'),
          'random vs complete-exp (O-physics)':(k,f'O-physics@{scen}|random',k,f'O-physics@{scen}|complete_exp'),
          'LOCAL planned vs TRANSFER 10-cut offset':(k,f'O-physics@{scen}|planned',10,f'T-offset@{scen}|transfer10'),
          'LOCAL random vs TRANSFER 10-cut offset':(k,f'O-physics@{scen}|random',10,f'T-offset@{scen}|transfer10'),
          'LOCAL complete-exp vs TRANSFER 10-cut offset':(k,f'O-physics@{scen}|complete_exp',10,f'T-offset@{scen}|transfer10'),
          'LOCAL planned vs TRANSFER same-cut offset':(k,f'O-physics@{scen}|planned',k,f'T-offset@{scen}|planned')}.items():
            r=dl(d,k1,a,k2,b)
            if r: rows.append((scen,d,k,lab,*r))
T=pd.DataFrame(rows,columns=['scenario','dir','k','comparison','dMAE','95% CI','P(first better)'])
T.to_csv('results/e5_paired.csv',index=False)
for scen in ['full','narrow6','narrow3']:
    print('\n---',scen); print(T[T.scenario==scen].drop(columns='scenario').to_string(index=False))
# subgroups: where does remaining error live? (full pool, both directions pooled)
print('\n#### subgroup MAE (W), full pool, O-physics, both directions pooled')
F=E[(E.method.str.startswith('O-physics@full'))]
rows=[]
for strat in ['random','complete_exp','planned','planned_nomat']:
    for k in [60,150,250]:
        x=F[(F.protocol==f'full|{strat}')&(F.k==k)]
        if len(x)==0: continue
        rows.append((strat,k,x.sae.sum()/x.n.sum(),x.sae_h.sum()/x.n_h.sum(),x.sae_d.sum()/x.n_d.sum(),x.sae_hi.sum()/x.n_hi.sum()))
print(pd.DataFrame(rows,columns=['strategy','k','all','intact tools','damaged tools','high-engagement']).round(0).to_string(index=False))
# material coverage control: among draws with mat_cov==1, does planned still differ from complete_exp/random?
print('\n#### draws with 100% material coverage only (full pool): mean O-physics MAE and counts')
Fm=D[(D.scen=='full')&(D.mat_cov>=0.999)]
print(Fm.pivot_table(index='strategy',columns='k',values='MAE_O-physics',aggfunc=['mean','count']).round(0).to_string())
print('\nshare of draws with 100% material coverage:'); print(D[D.scen=='full'].assign(ok=lambda x:x.mat_cov>=0.999).pivot_table(index='strategy',columns='k',values='ok').round(2).to_string())
