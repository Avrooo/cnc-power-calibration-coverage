import sys, time, warnings, os
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
from bench2 import *
from scipy.spatial.distance import cdist
BUD=[60,150,250]; SCHEMES=[('random',0),('exps',1),('exps',2),('exps',4),('exps',8),('combos',1),('combos',2),('combos',4),('aware',0)]
m['combo']=m.Tool+'|'+m.Workpiece
DES=np.c_[np.log(m.N),np.log(m.ap),np.log(m.ae),np.log(m.F),m.D,m.Z]; DES=(DES-DES.mean(0))/DES.std(0); m['_i']=np.arange(len(m))
def pred_sub(cal,te,ctx,k):
    out={}; f=ctx['f']; thS=ctx['thS']; u,mrr,hm,mi=prep(te)
    out['T-offset']=f.predict(te[FEATS])+np.median(cal.watt_mean.values-f.predict(cal[FEATS]))
    th=fit_free(cal,thS,[0] if k<10 else [0,1]); out['T-physics']=phys(th,u,mrr,hm,mi)
    out['O-physics']=phys(fit_reg(cal),u,mrr,hm,mi)
    Zc=Zf(cal); mu,sd=Zc.mean(0),Zc.std(0)+1e-9
    gp=GPR(C(1.0)*RBF([1.]*4,(0.1,50))+WhiteKernel(0.1,(1e-4,1)),normalize_y=True).fit((Zc-mu)/sd,cal.watt_mean.values); out['O-GP']=gp.predict((Zf(te)-mu)/sd)
    return out
def aware_select(pool,k,rng):
    # design-only k-center: uses tool/material identity + operating settings, never measured power
    X=DES[pool._i.values]; cb=pd.factorize(pool.combo)[0]; n=len(pool)
    def dist(i): return np.linalg.norm(X-X[i],axis=1)+3.0*(cb!=cb[i])
    sel=[int(rng.integers(n))]; md=dist(sel[0])
    while len(sel)<k:
        j=int(np.argmax(md)); sel.append(j); md=np.minimum(md,dist(j))
    return pool.iloc[sel]
def draw(pool,scheme,par,k,rng):
    if scheme=='random': return pool.iloc[rng.permutation(len(pool))[:k]]
    if scheme=='aware': return aware_select(pool,k,rng)
    col='experiment_id' if scheme=='exps' else 'combo'; keys=pool[col].unique()
    if len(keys)<par: return None
    for _ in range(40):
        ch=rng.choice(keys,par,replace=False); u=pool[pool[col].isin(ch)]
        if len(u)>=k: return u.iloc[rng.permutation(len(u))[:k]]
    return None
def diag(cal,te):
    D=cdist(DES[te._i.values],DES[cal._i.values]).min(1)
    return dict(n_exp=cal.experiment_id.nunique(),n_combo=cal.combo.nunique(),n_mat=cal.Workpiece.nunique(),
        mat_cov=te.Workpiece.isin(set(cal.Workpiece)).mean(),combo_cov=te.combo.isin(set(cal.combo)).mean(),nn=D.mean())
def job(rep,s,t,ctx):
    rng=np.random.default_rng(104729*rep+5); S=m[m.machine==s]; T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]
    pool,te=split_exp(T,rng); drows=[]; erows=[]
    for k in BUD:
        for sc,par in SCHEMES:
            cal=draw(pool,sc,par,k,rng)
            if cal is None: continue
            p=pred_sub(cal,te,ctx,k); name=f'{sc}{par if par else ""}'
            row=dict(dir=dname,rep=rep,k=k,scheme=sc,par=par,name=name,**diag(cal,te)); 
            for mth,pp in p.items(): row['MAE_'+mth]=np.mean(np.abs(te.watt_mean.values-pp))
            drows.append(row); records(name,dname,rep,k,0,te,p,erows)
    return drows,erows
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[(r,s,t,ctxs[s]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    out=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs)
    pd.DataFrame([r for o in out for r in o[0]]).to_csv('results/cov_draws.csv',index=False)
    pd.DataFrame([r for o in out for r in o[1]],columns=['protocol','dir','rep','k','e','method','exp','n','sae','n_h','sae_h','n_d','sae_d','n_hi','sae_hi']).to_csv('results/cov_exp.csv',index=False)
    print('done',round(time.time()-t0),'s')
