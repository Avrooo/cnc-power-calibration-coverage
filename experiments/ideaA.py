import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# PRE-REGISTERED CRITERION (written before running): GO only if the proposed policy ('nn_mat') beats the better of
# always_T / always_L (chosen per design x budget cell, i.e. an oracle-fixed comparator), AND beats nn_only and disagree,
# with 95% paired cluster-bootstrap CI below zero in BOTH machine directions (pooled over cells, equal cell weights).
import sys, time, warnings, os
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
from e5 import *
from sklearn.metrics import roc_auc_score
COST=500.0; BUD=[60,150,250]; DESIGNS=['random','exps4','complete_exp','narrow3_random']
def nn_dist(te,cal): return cdist(DES[te._i.values],DES[cal._i.values]).min(1)
def models(cal,te,ctx):
    f=ctx['f']; off=np.median(cal.watt_mean.values-f.predict(cal[FEATS])); pT=f.predict(te[FEATS])+off
    u,mrr,hm,mi=prep(te); pL=phys(fit_reg(cal),u,mrr,hm,mi); return pT,pL
def pick_thresholds(cal,ctx,rng,maxfolds=8):
    exps=cal.experiment_id.unique()
    if len(exps)<3: return None
    P=[]
    for g in rng.permutation(exps)[:maxfolds]:
        sub=cal[cal.experiment_id!=g]; hold=cal[cal.experiment_id==g]
        if len(sub)<10 or sub.experiment_id.nunique()<2: continue
        pT,pL=models(sub,hold,ctx); y=hold.watt_mean.values
        P.append(np.c_[nn_dist(hold,sub),hold.Workpiece.isin(set(sub.Workpiece)).values.astype(float),np.abs(y-pT),np.abs(y-pL),np.abs(pL-pT)])
    if not P: return None
    P=np.vstack(P); d,mat,eT,eL,dis=P.T; th={}
    th['best_single']='T' if eT.mean()<=eL.mean() else 'L'
    def best(score_ok_fn,grid):
        errs=[np.mean(np.where(score_ok_fn(g),eL,eT)) for g in grid]; return grid[int(np.argmin(errs))]
    gd=np.r_[np.quantile(d,np.linspace(.1,1,10)),np.inf]; th['nn_only']=best(lambda g:d<=g,gd); th['nn_mat']=best(lambda g:(mat>0)&(d<=g),gd)
    gs=np.r_[np.quantile(dis,np.linspace(.1,1,10)),np.inf]; th['disagree']=best(lambda g:dis<=g,gs); return th
def policies(th,d,mat,pT,pL):
    n=len(pT); out={'always_T':pT,'always_L':pL}
    if th is None: 
        for k in ['best_single','nn_only','nn_mat','disagree']: out[k]=pT     # too few experiments to calibrate a policy -> conservative fallback
        return out
    out['best_single']=pT if th['best_single']=='T' else pL
    out['nn_only']=np.where(d<=th['nn_only'],pL,pT); out['nn_mat']=np.where((mat>0)&(d<=th['nn_mat']),pL,pT)
    out['disagree']=np.where(np.abs(pL-pT)<=th['disagree'],pL,pT); return out
def build(design,k,pool0,rng):
    if design=='random': return pool0.iloc[rng.permutation(len(pool0))[:k]]
    if design=='complete_exp': return complete_exp(pool0,k,rng)
    if design=='exps4': return draw(pool0,'exps',4,k,rng)
    if design=='narrow3_random':
        pe=rng.choice(pool0.experiment_id.unique(),3,replace=False); p=pool0[pool0.experiment_id.isin(pe)]
        return p.iloc[rng.permutation(len(p))[:k]] if len(p)>=k else None
def job(rep,s,t,ctx):
    rng=np.random.default_rng(32452843*(rep+1)+11); T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]
    pool,te=split_exp(T,rng); erows=[]; mrows=[]; y=te.watt_mean.values; ex=te.experiment_id.values
    for design in DESIGNS:
        for k in BUD:
            cal=build(design,k,pool,rng)
            if cal is None or len(cal)<k: continue
            th=pick_thresholds(cal,ctx,rng); pT,pL=models(cal,te,ctx); d=nn_dist(te,cal); mat=te.Workpiece.isin(set(cal.Workpiece)).values.astype(float)
            P=policies(th,d,mat,pT,pL); P['oracle_row']=np.where(np.abs(y-pT)<=np.abs(y-pL),pT,pL)
            for nm,p in P.items(): records(f'{design}|{nm}',dname,rep,k,0,te,{'p':p},erows)
            for nm in ['always_T','nn_mat']:   # costly-misprediction counts per test experiment
                pass
            # risk-coverage: does coverage distance flag the rows where the chosen policy is badly wrong?
            e=np.abs(y-P['nn_mat']); costly=(e>COST).astype(int); risk=np.where(mat>0,d,1e6)
            row=dict(dir=dname,rep=rep,design=design,k=k,n_exp_cal=cal.experiment_id.nunique(),mat_cov=mat.mean(),fallback=int(th is None),
                     costly_rate_T=np.mean(np.abs(y-pT)>COST),costly_rate_L=np.mean(np.abs(y-pL)>COST),costly_rate_policy=costly.mean(),
                     costly_rate_oracle=np.mean(np.abs(y-P['oracle_row'])>COST))
            row['auc_risk']=roc_auc_score(costly,risk) if 0<costly.sum()<len(costly) else np.nan
            o=np.argsort(risk); keep=o[:int(.8*len(o))]; row['mae_all']=e.mean(); row['mae_keep80']=e[keep].mean(); row['mae_drop20']=e[o[int(.8*len(o)):]].mean()
            if th: row.update({f'th_{a}':b for a,b in th.items() if not isinstance(b,str)}); row['best_single_pick']=th['best_single']
            mrows.append(row)
    return mrows,erows
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[(r,s,t,ctxs[s]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    out=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs)
    pd.DataFrame([r for o in out for r in o[0]]).to_csv('results/ideaA_draws.csv',index=False)
    pd.DataFrame([r for o in out for r in o[1]],columns=['protocol','dir','rep','k','e','method','exp','n','sae','n_h','sae_h','n_d','sae_d','n_hi','sae_hi']).to_csv('results/ideaA_exp.csv',index=False)
    print('done',round(time.time()-t0),'s')
