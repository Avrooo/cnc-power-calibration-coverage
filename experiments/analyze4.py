import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from analyze2 import *
import analyze2
print('#### subgroup MAE (W) at selected k, protocol exp (both directions pooled): transfer vs target-only')
d=load('exp'); rows=[]
for sub,(s,n) in {'all':('sae','n'),'intact tools':('sae_h','n_h'),'damaged tools':('sae_d','n_d'),'high-engagement (top MRR quartile)':('sae_hi','n_hi')}.items():
    g=d.groupby(['k','method'])[[s,n]].sum(); g=(g[s]/g[n]).unstack('method')
    for k in [10,30,100,200]: rows.append((sub,k,*(g.loc[k,['T-offset (GBM)','T-physics','O-physics (reg.)','O-GP','O-GBM']].round(0).tolist())))
print(pd.DataFrame(rows,columns=['subgroup','k','T-offset','T-physics','O-physics','O-GP','O-GBM']).to_string(index=False))
print('\n#### k* by subgroup, protocol exp, rival = best-of-3 (median [q05,q95], P(never<=200))')
out=[]
for sub,(s,n) in {'intact tools':('sae_h','n_h'),'damaged tools':('sae_d','n_d'),'high-engagement':('sae_hi','n_hi')}.items():
    _,k=summarize('exp',sae=s,n=n)
    for _,r in k.iterrows(): out.append((sub,r['dir'],f"{r.k_star_median:g} [{r.k_star_q05:g},{r.k_star_q95:g}] never={r.P_never_within_200:.2f}"))
print(pd.DataFrame(out,columns=['subgroup','dir','k*']).pivot(index='subgroup',columns='dir',values='k*').to_string())
# n experiments / rows underlying damaged subgroup
dd=d[(d.k==30)&(d.method=='T-offset (GBM)')].groupby('dir')[['n','n_d','n_hi']].sum(); print('\ntest rows summed over reps (k=30):\n',dd)
