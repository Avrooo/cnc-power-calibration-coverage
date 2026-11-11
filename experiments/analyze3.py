import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from analyze2 import *
import analyze2
# (A) competitor dependence: k* when the target-only rival is restricted to one model class
print('#### k* (median [q05,q95], P(no crossover<=200)) by protocol x competitor class; transfer = source GBM + offset')
rows=[]
for comp in [['O-physics (reg.)'],['O-GP'],['O-GBM'],['O-physics (reg.)','O-GP','O-GBM']]:
    analyze2.OO=comp
    for p in ['naive','exp','exp_purged','exp_healthy']:
        _,k=summarize(p)
        for _,r in k.iterrows(): rows.append((' / '.join(c.split()[0] for c in comp) if len(comp)==1 else 'best-of-3',p,r['dir'],f"{r.k_star_median:g} [{r.k_star_q05:g},{r.k_star_q95:g}] never={r.P_never_within_200:.2f}"))
T=pd.DataFrame(rows,columns=['competitor','protocol','dir','k* [q05,q95]']); print(T.pivot_table(index=['competitor','protocol'],columns='dir',values='k* [q05,q95]',aggfunc='first').to_string())
analyze2.OO=['O-physics (reg.)','O-GP','O-GBM']
# (B) protocol inflation at fixed k (MAE, W): same method, different evaluation protocol
print('\n#### MAE (W) by protocol at fixed k (mean of both directions) -> evaluation-protocol inflation')
fr=[]
for p in ['naive','exp','exp_purged']:
    d=load(p); g=d.groupby(['k','method'])[['sae','n']].sum(); fr.append((g.sae/g.n).rename(p))
F=pd.concat(fr,axis=1).reset_index(); F=F[F.k.isin([10,30,100,200])&F.method.isin(['T-offset (GBM)','O-physics (reg.)','O-GP','O-GBM'])]
F['inflation_naive_vs_exp_%']=(100*(F.exp-F.naive)/F.exp).round(1); print(F.round(0).to_string(index=False))
# (C) experiment-level calibration vs matched path-level
import os
if os.path.exists('results/b2_expcal.csv'):
    d=load('expcal'); d['kind']=d['protocol']
    print('\n#### experiment-level calibration vs matched number of cuts drawn path-level (MAE W)')
    g=d.groupby(['dir','e','kind','method'])[['sae','n']].sum(); g['MAE']=g.sae/g.n; kk=d.groupby(['dir','e','kind']).k.mean()
    P=g.MAE.unstack('kind'); P['pen_%']=100*(P.expcal-P.matched)/P.matched
    P=P.reset_index().merge(kk.unstack('kind')[['expcal']].rename(columns={'expcal':'mean_cuts'}).reset_index(),on=['dir','e'])
    print(P[P.method.isin(['T-offset (GBM)','O-physics (reg.)','O-GP'])].round(1).to_string(index=False))
