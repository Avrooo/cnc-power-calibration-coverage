import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
B=2000; rng=np.random.default_rng(0)
KS=[3,5,10,20,30,50,75,100,150,200]
TR=['T-offset (GBM)','T-physics']; OO=['O-physics (reg.)','O-GP','O-GBM']
def load(p): return pd.read_csv(f'results/b2_{p}.csv')
def boot_tables(df,sae,n,key):
    """per (dir,k,method): arrays [n_exp x B] resampled MAE using experiment-level cluster bootstrap"""
    g=df.groupby(['dir','k','method','exp'])[[sae,n]].sum().reset_index()
    res={}
    for d,gd in g.groupby('dir'):
        exps=np.array(sorted(gd.exp.unique())); idx=rng.integers(0,len(exps),(B,len(exps)))   # same resample for all methods in this dir
        pos={e:i for i,e in enumerate(exps)}
        for (k,meth),gg in gd.groupby(['k','method']):
            s=np.zeros(len(exps)); c=np.zeros(len(exps)); ii=gg.exp.map(pos).values; s[ii]=gg[sae].values; c[ii]=gg[n].values
            den=c[idx].sum(1); res[(d,k,meth)]=(s[idx].sum(1)/np.where(den>0,den,np.nan), s.sum()/max(c.sum(),1))
    return res
def kstar(delta_by_k):
    """delta_by_k: dict k-> array(B) of MAE_transfer - MAE_target. k* = smallest grid k after which target-only is better at all larger k; inf if never"""
    ks=[k for k in KS if k in delta_by_k]; out=np.full(B,np.inf)
    for b in range(B):
        for i,k in enumerate(ks):
            if all(delta_by_k[kk][b]>0 for kk in ks[i:]): out[b]=k; break
    return out
def summarize(proto,sae='sae',n='n',label=None,transfer='T-offset (GBM)'):
    df=load(proto); res=boot_tables(df,sae,n,proto); rows=[]; ks_rows=[]; dirs=sorted(df['dir'].unique())
    for d in dirs:
        dl={}
        for k in KS:
            if (d,k,transfer) not in res: continue
            cand=[(res[(d,k,o)][0],o) for o in OO if (d,k,o) in res]
            if not cand: continue
            # per-bootstrap-sample best target-only model (conservative against transfer)
            best=np.nanmin(np.vstack([c[0] for c in cand]),axis=0); pt=min((res[(d,k,o)][1],o) for _,o in cand)
            T=res[(d,k,transfer)][0]; delta=T-best; dl[k]=np.nan_to_num(delta,nan=0.0)
            rows.append(dict(proto=label or proto,dir=d,k=k,MAE_transfer=res[(d,k,transfer)][1],MAE_bestTarget=pt[0],best_is=pt[1],
                 delta=np.nanmean(delta),lo=np.nanpercentile(delta,2.5),hi=np.nanpercentile(delta,97.5),P_target_better=np.nanmean(delta>0)))
        ks=kstar(dl); fin=ks[np.isfinite(ks)]
        ks_rows.append(dict(proto=label or proto,dir=d,transfer=transfer,k_star_median=np.median(ks),P_never_within_200=np.mean(~np.isfinite(ks)),k_star_q05=np.quantile(ks,.05),k_star_q95=np.quantile(ks,.95)))
    return pd.DataFrame(rows),pd.DataFrame(ks_rows)
if __name__=='__main__':
    allr=[];allk=[]
    for p in [q for q in ['naive','exp','exp_purged','exp_healthy'] if __import__('os').path.exists(f'results/b2_{q}.csv')]:
        r,k=summarize(p); allr.append(r); allk.append(k)
    R=pd.concat(allr); K=pd.concat(allk); R.to_csv('results/a2_delta.csv',index=False); K.to_csv('results/a2_kstar.csv',index=False)
    print('\n#### crossover k* by protocol (transfer = source GBM + offset; target-only = best of physics(reg)/GP/GBM per bootstrap draw)'); print(K.round(2).to_string(index=False))
    for tr in ['T-physics']:
        K2=pd.concat([summarize(p,transfer=tr)[1] for p in ['naive','exp','exp_purged']]); print('\n#### same with transfer = physics transfer'); print(K2.round(2).to_string(index=False))
    print('\n#### paired delta MAE (transfer - best target-only), protocol exp: >0 means target-only better'); 
    x=R[R.proto=='exp']; print(x.assign(CI=x.apply(lambda r:f'[{r.lo:.0f},{r.hi:.0f}]',axis=1))[['dir','k','MAE_transfer','MAE_bestTarget','best_is','delta','CI','P_target_better']].round(2).to_string(index=False))
