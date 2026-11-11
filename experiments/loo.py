import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys, time, warnings, os
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from coverage import *
K=150
def job(dname,s,t,ei,e,R,ctx):
    T=m[m.machine==t]; T2=T if e is None else T[T.experiment_id!=e]; rows=[]; f=ctx['f']
    for rep in range(R):
        rng=np.random.default_rng(900001*(ei+2)+137*rep+3)
        try: pool,te=split_exp(T2,rng)
        except RuntimeError: continue
        u,mrr,hm,mi=prep(te); y=te.watt_mean.values; fte=f.predict(te[FEATS]); r={'dir':dname,'excl':e if e else 'none','rep':rep}
        for nm,(sc,par) in [('random',('random',0)),('exps4',('exps',4))]:
            cal=draw(pool,sc,par,K,rng)
            if cal is None: r[nm]=np.nan; continue
            pT=fte+np.median(cal.watt_mean.values-f.predict(cal[FEATS])); pL=phys(fit_reg(cal),u,mrr,hm,mi); r[nm]=np.mean(np.abs(y-pL))-np.mean(np.abs(y-pT))
        rows.append(r)
    return rows
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[]
    for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]:
        T=m[m.machine==t]; exps=[None]+sorted(T.experiment_id.unique())
        for ei,e in enumerate(exps): jobs.append((s[4:]+'->'+t[4:],s,t,ei-1,e,R,ctxs[s]))
    res=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs); pd.DataFrame([r for rr in res for r in rr]).to_csv('results/loo_sensitivity.csv',index=False); print('loo done',round(time.time()-t0),'s')
