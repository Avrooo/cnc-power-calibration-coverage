import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys; sys.argv=['x','all','1']
from bench import *
from sklearn.model_selection import GroupKFold
out=[]
for mc,d in m.groupby('machine'):
    d=d.reset_index(drop=True); p=np.zeros(len(d))
    for a,b in GroupKFold(8).split(d,groups=d.experiment_id):
        th=fit_phys(d.iloc[a]); u,mrr,hm,mi=prep(d.iloc[b]); p[b]=phys(th,u,mrr,hm,mi)
    d['err']=np.abs(d.watt_mean-p); d['res']=d.watt_mean-p; d['mc']=mc; out.append(d)
D=pd.concat(out)
print('overall MAE %.0f, R2 %.2f (per-machine physics, held-out experiments)'%(D.err.mean(),r2_score(D.watt_mean,D.watt_mean-D.res)))
print(D.groupby('mc').apply(lambda x: pd.Series({'MAE':x.err.mean(),'R2':r2_score(x.watt_mean,x.watt_mean-x.res)})).round(2))
print(D.groupby(['mc','Workpiece']).err.agg(['mean','size']).round(0).T.to_string())
print(D.groupby(['mc',pd.qcut(D.MRR,4,labels=['Q1 low MRR','Q2','Q3','Q4 high MRR'])]).err.mean().round(0).unstack().to_string())
print(D.groupby(['mc','star']).err.mean().round(0).unstack().to_string())
print('worst tools:',D.groupby(['mc','Tool']).err.mean().round(0).sort_values(ascending=False).head(6).to_dict())
# error share from low-MRR (baseline) vs high-MRR (cutting)
D['mrrQ']=pd.qcut(D.MRR,4,labels=False); sh=D.groupby('mrrQ').err.sum()/D.err.sum(); print('share of total abs error by MRR quartile',sh.round(2).to_dict())
# baseline-only error: air-cut-ish rows
lo=D[D.MRR<D.MRR.quantile(.1)]; print('MAE on lowest-10%% MRR rows (≈baseline): %.0f W vs mean power %.0f W'%(lo.err.mean(),lo.watt_mean.mean()))
