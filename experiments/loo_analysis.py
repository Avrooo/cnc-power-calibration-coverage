import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import pandas as pd, numpy as np
D=pd.read_csv('results/loo_sensitivity.csv'); D['G']=D['exps4']-D['random']
g=D.groupby(['dir','excl'])[['random','exps4','G']].mean().reset_index()
for d in sorted(g['dir'].unique()):
    x=g[g['dir']==d]; base=x[x.excl=='none'].iloc[0]; y=x[x.excl!='none']
    print(f"\n== {d}: full-data reference (no exclusion): random {base['random']:+.0f} W | 4-experiment {base['exps4']:+.0f} W | gap {base['G']:+.0f} W  (local minus transfer MAE)")
    print(f"   {len(y)} leave-one-experiment-out reruns:")
    print(f"   random design:        range {y['random'].min():+.0f} to {y['random'].max():+.0f} W | local better in {100*(y['random']<0).mean():.0f}% of reruns")
    print(f"   4-experiment design:  range {y['exps4'].min():+.0f} to {y['exps4'].max():+.0f} W | transfer better in {100*(y['exps4']>0).mean():.0f}% of reruns")
    print(f"   gap (4-exp minus random): range {y['G'].min():+.0f} to {y['G'].max():+.0f} W | positive in {100*(y['G']>0).mean():.0f}% of reruns")
    inf=y.assign(dG=(y['G']-base['G']).abs()).sort_values('dG',ascending=False).head(3)
    print('   most influential exclusions (change in gap):',[(r.excl.replace('imi_',''),f"{r.G:+.0f}") for r in inf.itertuples()])
