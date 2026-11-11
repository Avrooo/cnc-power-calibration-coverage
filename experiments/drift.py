import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from sklearn.model_selection import GroupKFold
m=load()
# is parameter order confounded with path order?
for col in ['N','MRR','ap','ae']:
    r=[np.corrcoef(d.path_id,d[col])[0,1] for e,d in m.groupby('experiment_id') if len(d)>=20 and d[col].nunique()>1]
    print(f'median within-exp corr(path_id,{col}) = {np.median(r):.2f}')
# residual drift: out-of-experiment model residuals vs path_id
p=np.zeros(len(m))
for tr,te in GroupKFold(8).split(m,groups=m.experiment_id):
    p[te]=gbm().fit(m.iloc[tr][FEATS+['star']],m.iloc[tr].watt_mean).predict(m.iloc[te][FEATS+['star']])
m['res']=m.watt_mean-p
# within-experiment OLS: watt ~ params(flexible via GBM residual) -> residual vs path order; also vs normalized position
m['pos']=m.groupby('experiment_id').path_id.rank(pct=True)
r=[np.corrcoef(d.pos,d.res)[0,1] for e,d in m.groupby('experiment_id') if len(d)>=20]
print('residual vs position (out-of-experiment): median r %.2f, share exps r<-0.3: %.0f%%, r>0.3: %.0f%%'%(np.median(r),100*np.mean(np.array(r)<-.3),100*np.mean(np.array(r)>.3)))
# direct drift estimate: pairs UP/DOWN of same treatment: power difference by direction
t=m.pivot_table(index='trt',columns='dir',values='watt_mean').dropna()
d=(t[1]-t[0]); print('UP minus DOWN same treatment: median %.0f W, mean %.0f W, share UP>DOWN %.2f, n=%d'%(d.median(),d.mean(),(d>0).mean(),len(d)))
# is the UP/DOWN difference tied to path order (UP later than DOWN or vice versa)?
po=m.pivot_table(index='trt',columns='dir',values='path_id').dropna(); print('UP path_id minus DOWN path_id median',(po[1]-po[0]).median())
# per-machine baseline drift: lowest-engagement paths power vs path_id
lo=m[m.MRR<m.MRR.quantile(.1)]
for mc,dd in lo.groupby('machine'):
    print(mc,'low-MRR: corr(path_id, watt) overall %.2f ; within-N-band median %.2f'%(np.corrcoef(dd.path_id,dd.watt_mean)[0,1], np.median([np.corrcoef(g.path_id,g.watt_mean)[0,1] for _,g in dd.groupby(pd.cut(dd.N,4)) if len(g)>8 and g.path_id.nunique()>1])))
m[['experiment_id','path_id','res','pos']].to_csv('results/oof_residuals.csv',index=False)
