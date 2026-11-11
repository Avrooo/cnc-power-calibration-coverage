import sys, time, warnings, os, pickle
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from coverage import *
K=150
qN=np.quantile(m.N,np.linspace(0,1,7)[1:-1]); LM=np.log10(m.ap*m.ae*m.F+1e-3); qM=np.quantile(LM,np.linspace(0,1,7)[1:-1])
DN=['tool-material combos','speed bins covered (of 6)','SD log speed','MRR bins covered (of 6)','SD log MRR','SD log chip thickness']
def desc(cal):
    bn=np.bincount(np.digitize(cal.N.values,qN),minlength=6); lmc=np.log10(cal.ap.values*cal.ae.values*cal.F.values+1e-3); bm=np.bincount(np.digitize(lmc,qM),minlength=6)
    return np.array([cal.combo.nunique(),(bn>=5).sum(),np.std(np.log(cal.N.values)),(bm>=5).sum(),np.std(lmc),np.std(np.log10(cal.hm.values+1e-3))],float)
def make_sets(pool,rng,n_each):
    exps=pool.experiment_id.unique(); A=[];B=[]; tries=0
    while (len(A)<n_each or len(B)<n_each) and tries<400:
        tries+=1
        if len(A)<n_each:
            ch=rng.choice(exps,4,replace=False); u=pool[pool.experiment_id.isin(ch)]
            if len(u)>=K:
                cal=u.iloc[rng.permutation(len(u))[:K]]
                if cal.Workpiece.nunique()==3: A.append(cal)
        if len(B)<n_each:
            j=int(rng.choice([8,10,12])); j=min(j,len(exps)); ch=rng.choice(exps,j,replace=False); u=pool[pool.experiment_id.isin(ch)]
            if len(u)>=K:
                cal=u.iloc[rng.permutation(len(u))[:K]]
                if cal.Workpiece.nunique()==3: B.append(cal)
    return A,B
def job(rep,s,t,ctx,n_each=16):
    rng=np.random.default_rng(7001*rep+17); T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]
    pool,te=split_exp(T,rng); A,B=make_sets(pool,rng,n_each)
    ue,inv=np.unique(te.experiment_id.values,return_inverse=True); n=np.bincount(inv).astype(float); y=te.watt_mean.values
    u,mrr,hm,mi=prep(te); f=ctx['f']; fte=f.predict(te[FEATS]); out=[]
    for grp,L in [('A',A),('B',B)]:
        for cal in L:
            off=np.median(cal.watt_mean.values-f.predict(cal[FEATS])); pT=fte+off; pL=phys(fit_reg(cal),u,mrr,hm,mi)
            out.append(dict(dir=dname,rep=rep,grp=grp,desc=desc(cal),nexp=cal.experiment_id.nunique(),sL=np.bincount(inv,weights=np.abs(y-pL)),sT=np.bincount(inv,weights=np.abs(y-pT)),n=n,ue=ue,maeL=np.mean(np.abs(y-pL)),maeT=np.mean(np.abs(y-pT))))
    return out
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[(r,s,t,ctxs[s]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    res=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs); pickle.dump([o for r in res for o in r],open('results/matched_sets.pkl','wb')); print('matched sets done',round(time.time()-t0),'s',sum(len(r) for r in res))
