import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys, time, warnings, os
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS']='ignore'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from coverage import *
K=150; UG=np.linspace(0.15,1.2,8)
def base(th): return th[0]+th[1]*UG+th[2]*UG**2
def job(rep,s,t,ctx):
    rng=np.random.default_rng(4242*rep+9); T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]; pool,te=split_exp(T,rng); f=ctx['f']
    ref=fit_reg(pool); u,mrr,hm,mi=prep(te); y=te.watt_mean.values; pT=f.predict(te[FEATS]); rows=[]
    for nm,(sc,par) in [('random',('random',0)),('exps8',('exps',8)),('combos4',('combos',4)),('exps4',('exps',4))]:
        cal=draw(pool,sc,par,K,rng)
        if cal is None: continue
        th=fit_reg(cal); med=np.median(cal.watt_mean.values); pL=phys(th,u,mrr,hm,mi); off=np.median(cal.watt_mean.values-f.predict(cal[FEATS])); pTr=pT+off
        present=[i for i,w in enumerate(MATS) if w in set(cal.Workpiece)]
        dK=np.mean([abs(th[4+i]-ref[4+i]) for i in present]) if present else np.nan
        K_=np.exp(th[4:4+M_]); hitK=any((K_[i]<0.2*1.02) or (K_[i]>8/1.02) for i in present); hitm=(th[3]<0.005) or (th[3]>0.595); lo,hi=0.3*med,1.05*med; hitb=(th[0]<lo*1.005) or (th[0]>hi*0.995)
        rows.append(dict(dir=dname,rep=rep,design=nm,nexp=cal.experiment_id.nunique(),nmat=len(present),maeL=np.mean(np.abs(y-pL)),maeT=np.mean(np.abs(y-pTr)),dK=dK,dm=abs(th[3]-ref[3]),dbase=float(np.mean(np.abs(base(th)-base(ref)))),
            hitK=hitK,hitm=hitm,hitb0=hitb,m=th[3],b0=th[0],**{f'logK_{w}':th[4+i] for i,w in enumerate(MATS)},**{f'ref_logK_{w}':ref[4+i] for i,w in enumerate(MATS)},ref_m=ref[3]))
    return rows
if __name__=='__main__':
    R=int(sys.argv[1]); t0=time.time(); ctxs={s:make_ctx(m[m.machine==s]) for s in ['imi_vm20i','imi_vmx30ui']}
    jobs=[(r,s,t,ctxs[s]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    res=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs); pd.DataFrame([r for rr in res for r in rr]).to_csv('results/coef_diag.csv',index=False); print('coef done',round(time.time()-t0),'s')
