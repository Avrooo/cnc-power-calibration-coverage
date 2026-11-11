import sys, time, warnings, os
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
from coverage import *
from scipy.stats import qmc
BUD=[60,150,250]; NARROW=6
OPS=['N','ap','ae','F']
def ops_std(d): 
    X=np.c_[np.log(d.N),np.log(d.ap),np.log(d.ae),np.log(d.F)]; return X
ALL=ops_std(m); MU,SD=ALL.mean(0),ALL.std(0)
def planned(pool,k,rng,S,intended):
    """Preplanned space-filling design. Plan = scrambled Sobol points over the operating box known from the SOURCE machine's
    historical ranges, assigned round-robin to intended materials and to tools in the target tool crib; each planned point is then
    realised by the nearest still-unused available cut with that material+tool (fallbacks: same material, then any). No measured power used."""
    lo=ops_std(S).min(0); hi=ops_std(S).max(0)
    P=qmc.Sobol(4,scramble=True,seed=int(rng.integers(1<<31))).random(k)
    Pl=(lo+P*(hi-lo)); Pl=(Pl-MU)/SD
    Xp=(ops_std(pool)-MU)/SD; used=np.zeros(len(pool),bool); W=list(intended); sel=[]; snap=[]
    toolsby={w:list(pool[pool.Workpiece==w].Tool.unique()) for w in pool.Workpiece.unique()}
    ctr={w:0 for w in toolsby}
    for i in range(k):
        w=W[i%len(W)]
        if w not in toolsby: w=pool.Workpiece.iloc[0]
        t=toolsby[w][ctr[w]%len(toolsby[w])]; ctr[w]+=1
        for mask in [(pool.Workpiece.values==w)&(pool.Tool.values==t),(pool.Workpiece.values==w),np.ones(len(pool),bool)]:
            mk=mask&~used
            if mk.any(): break
        idx=np.where(mk)[0]; dd=np.linalg.norm(Xp[idx]-Pl[i],axis=1); j=idx[int(np.argmin(dd))]; used[j]=True; sel.append(j); snap.append(dd.min())
    return pool.iloc[sel],float(np.mean(snap))
def complete_exp(pool,k,rng):
    exps=list(rng.permutation(pool.experiment_id.unique())); parts=[]; n=0
    for e in exps:
        g=pool[pool.experiment_id==e]; parts.append(g); n+=len(g)
        if n>=k: break
    u=pd.concat(parts); 
    if len(u)<k: return None
    full=pd.concat(parts[:-1]) if len(parts)>1 else parts[0].iloc[:0]; need=k-len(full)
    return pd.concat([full,parts[-1].iloc[rng.permutation(len(parts[-1]))[:need]]])
def pred(cal,te,ctx,k,which=('T-offset','T-physics','O-physics')):
    out={}; f=ctx['f']; thS=ctx['thS']; u,mrr,hm,mi=prep(te)
    out['T-offset']=f.predict(te[FEATS])+np.median(cal.watt_mean.values-f.predict(cal[FEATS]))
    if 'T-physics' in which: out['T-physics']=phys(fit_free(cal,thS,[0] if k<10 else [0,1]),u,mrr,hm,mi)
    if 'O-physics' in which: out['O-physics']=phys(fit_reg(cal),u,mrr,hm,mi)
    return out
def job(rep,s,t,ctx):
    rng=np.random.default_rng(15485863*(rep+1)+3); S=m[m.machine==s]; T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]
    pool0,te=split_exp(T,rng); intended=sorted(te.Workpiece.unique()); drows=[]; erows=[]
    for scen,npool in [('full',None),('narrow6',6),('narrow3',3)]:
        pool=pool0
        if npool:
            pe=rng.choice(pool0.experiment_id.unique(),npool,replace=False); pool=pool0[pool0.experiment_id.isin(pe)]
        # conservative baseline: offset transfer from 10 random cuts
        c10=pool.iloc[rng.permutation(len(pool))[:10]]; records(f'{scen}|transfer10',dname,rep,10,0,te,pred(c10,te,ctx,10,which=('T-offset',)),erows)
        for k in BUD:
            if len(pool)<k: continue
            sets={}
            sets['random']=(pool.iloc[rng.permutation(len(pool))[:k]],np.nan)
            ce=complete_exp(pool,k,rng)
            if ce is not None: sets['complete_exp']=(ce,np.nan)
            sets['planned']=planned(pool,k,rng,S,intended)
            sets['planned_nomat']=planned(pool,k,rng,S,sorted(pool.Workpiece.unique()))
            for nm,(cal,snap) in sets.items():
                p=pred(cal,te,ctx,k); row=dict(scen=scen,dir=dname,rep=rep,k=k,strategy=nm,snap=snap,**diag(cal,te))
                for mth,pp in p.items(): row['MAE_'+mth]=np.mean(np.abs(te.watt_mean.values-pp))
                drows.append(row); records(f'{scen}|{nm}',dname,rep,k,0,te,p,erows)
    return drows,erows
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[(r,s,t,ctxs[s]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    out=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs)
    pd.DataFrame([r for o in out for r in o[0]]).to_csv('results/e5_draws.csv',index=False)
    pd.DataFrame([r for o in out for r in o[1]],columns=['protocol','dir','rep','k','e','method','exp','n','sae','n_h','sae_h','n_d','sae_d','n_hi','sae_hi']).to_csv('results/e5_exp.csv',index=False)
    print('done',round(time.time()-t0),'s')
