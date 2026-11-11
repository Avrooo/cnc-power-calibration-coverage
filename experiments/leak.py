import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error as mae
m=load()
print('rows',len(m),'| healthy only rows',(m.star==0).sum())
# --- what are the "siblings"? ---
g=m.groupby('trt')
sz=g.size(); print('rows sharing (exp,N,ap,ae,F):',(m.trt.map(sz)>1).mean().round(3))
nd=g.dir.nunique(); print('sibling groups with BOTH directions (UP+DOWN):',(nd[sz>1]==2).mean().round(3),'| same direction repeat groups:',(nd[sz>1]==1).mean().round(3))
cv=g.watt_mean.agg(lambda x:(x.std()/x.mean())); same=m.groupby(['trt','dir']).watt_mean.agg(['size','std','mean']); same=same[same['size']>1]
print('same-direction repeats within experiment: n groups',len(same),' median CV %.4f'%(same['std']/same['mean']).median())
# across-experiment repeatability (same machine,params,tool,material,dir, different experiments)
key=['machine','N','ap','ae','F','Tool','Workpiece','dir']
ge=m.groupby(key).agg(nexp=('experiment_id','nunique'),n=('watt_mean','size'),sd=('watt_mean','std'),mu=('watt_mean','mean'))
x=ge[ge.nexp>1]; print('param-identical settings seen in >1 experiments:',len(x),' median CV %.4f  p90 CV %.4f'%((x.sd/x.mu).median(),(x.sd/x.mu).quantile(.9)))
# noise ceiling across experiments in R2 terms (between-exp replicate variance)
print('across-experiment noise ceiling R2 ~ %.3f'%(1-(x.sd**2).mean()/m.watt_mean.var()))
# drift within experiment
dr=[]
for e,d in m.groupby('experiment_id'):
    if len(d)>=20: dr.append(np.corrcoef(d.path_id,d.watt_mean)[0,1])
print('within-experiment corr(path_id, watt): median %.2f  (|r|>0.3 in %.0f%% of %d exps)'%(np.median(dr),100*np.mean(np.abs(dr)>0.3),len(dr)))
# --- leakage ladder, per machine and pooled ---
def ladder(d,label):
    out={}
    def run(splits,drop_sib=False):
        p=np.full(len(d),np.nan)
        for tr,te in splits:
            if drop_sib:
                bad=set(d.iloc[te].trt); tr=np.array([i for i in tr if d.trt.iloc[i] not in bad])
            p[te]=gbm().fit(d.iloc[tr][FEATS],d.iloc[tr].watt_mean).predict(d.iloc[te][FEATS])
        return r2_score(d.watt_mean,p),mae(d.watt_mean,p)
    rk=KFold(5,shuffle=True,random_state=0)
    out['random']=run(list(rk.split(d)))
    out['random, sibling-purged']=run(list(rk.split(d)),True)
    for gname,col in [('held-out experiment','experiment_id'),('held-out tool','Tool'),('held-out material','Workpiece'),('held-out machine','machine')]:
        if d[col].nunique()<2: continue
        out[gname]=run(list(GroupKFold(min(5,d[col].nunique())).split(d,groups=d[col])))
    return out
# first without machine-id (as before); this is pooled; add per-machine rows
rows=[]
for lab,d in [('pooled (2 machines)',m),('vm20i',m[m.machine=='imi_vm20i'].reset_index(drop=True)),('vmx30ui',m[m.machine=='imi_vmx30ui'].reset_index(drop=True))]:
    for k,(r2,e) in ladder(d,lab).items(): rows.append((lab,k,round(r2,3),round(e)))
L=pd.DataFrame(rows,columns=['scope','split','R2','MAE_W']); print(L.to_string(index=False)); L.to_csv('results/leakage_ladder.csv',index=False)
