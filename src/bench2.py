import sys, time
from common import *
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from joblib import Parallel, delayed
KS=[3,5,10,20,30,50,75,100,150,200]; ELIST=[1,2,3,4,6]
m=load(); m['setting']=m.N.astype(str)+'|'+m.ap.astype(str)+'|'+m.ae.astype(str)+'|'+m.F.astype(str)+'|'+m.Tool+'|'+m.Workpiece+'|'+m.dir.astype(str)
m['hi']=0
for mc in m.machine.unique(): ix=m.machine==mc; m.loc[ix,'hi']=(m.loc[ix,'MRR']>=m.loc[ix,'MRR'].quantile(.75)).astype(int)
MATS=sorted(m.Workpiece.unique()); M_=len(MATS)
def prep(d):
    u=(d.N/1e4).values; mrr=(d.ap*d.ae*d.F).values; hm=np.clip(d.hm.values,1e-3,None)
    return u,mrr,hm,np.array([MATS.index(w) for w in d.Workpiece])
def phys(t,u,mrr,hm,mi): return t[0]+t[1]*u+t[2]*u*u+np.exp(t[4:4+M_])[mi]*mrr*(hm/0.05)**(-t[3])
def fit_free(d,th0,free):
    u,mrr,hm,mi=prep(d); y=d.watt_mean.values; th=th0.copy()
    def f(z): t=th.copy(); t[free]=z; return phys(t,u,mrr,hm,mi)-y
    r=least_squares(f,th[free],loss='soft_l1',f_scale=300); th[free]=r.x; return th
def fit_full(d):   # unregularized (used for the SOURCE machine, lots of data)
    u,mrr,hm,mi=prep(d); y=d.watt_mean.values
    lo=[-np.inf]*3+[0.0]+[-5]*M_; hi=[np.inf]*3+[0.8]+[6]*M_
    t0=np.array([np.median(y)*.8,0,0,.2]+[np.log(1.5)]*M_)
    return least_squares(lambda t:phys(t,u,mrr,hm,mi)-y,t0,bounds=(lo,hi),loss='soft_l1',f_scale=300).x
def fit_reg(d):    # regularized target-only physics (bounded parameters) = fair competitor
    u,mrr,hm,mi=prep(d); y=d.watt_mean.values; med=np.median(y)
    lo=[.3*med,-5e3,-5e3,0.0]+[np.log(.2)]*M_; hi=[1.05*med,5e3,5e3,.6]+[np.log(8)]*M_
    t0=np.array([.8*med,0,0,.2]+[np.log(1.5)]*M_); t0=np.clip(t0,np.array(lo)+1e-9,np.array(hi)-1e-9)
    return least_squares(lambda t:phys(t,u,mrr,hm,mi)-y,t0,bounds=(lo,hi),loss='soft_l1',f_scale=300).x
def baseline_fit(d):
    lo=d[d.MRR<=d.MRR.quantile(.10)]; return HuberRegressor().fit(np.c_[lo.N/1e4,(lo.N/1e4)**2],lo.watt_mean)
Zf=lambda d:np.c_[d.N/1e4,np.log10(d.MRR+1e-3),d.ap_D,np.log10(d.hm+1e-3)]
def make_ctx(S):
    f=gbm().fit(S[FEATS],S.watt_mean); return dict(f=f,thS=fit_full(S))
def predict_all(cal,te,ctx,k):
    out={}; f=ctx['f']; thS=ctx['thS']; u,mrr,hm,mi=prep(te)
    fte=f.predict(te[FEATS]); off=np.median(cal.watt_mean.values-f.predict(cal[FEATS]))
    out['T-offset (GBM)']=fte+off
    th=fit_free(cal,thS,[0] if k<10 else [0,1]); out['T-physics']=phys(th,u,mrr,hm,mi)
    if k>=5: out['O-physics (reg.)']=phys(fit_reg(cal),u,mrr,hm,mi)
    if k>=10:
        out['O-GBM']=H(max_iter=120,learning_rate=0.08,min_samples_leaf=max(2,min(20,k//5)),random_state=0).fit(cal[FEATS],cal.watt_mean).predict(te[FEATS])
        if k<=250:
            Zc=Zf(cal); mu,sd=Zc.mean(0),Zc.std(0)+1e-9
            ker=C(1.0)*RBF([1.]*4,(0.1,50))+WhiteKernel(0.1,(1e-4,1))
            gp=GPR(ker,normalize_y=True).fit((Zc-mu)/sd,cal.watt_mean.values); out['O-GP']=gp.predict((Zf(te)-mu)/sd)
            rc=cal.watt_mean.values-f.predict(cal[FEATS])-off
            gp2=GPR(C(1e4)*RBF([1.]*4,(0.1,20))+WhiteKernel(5e4),normalize_y=False).fit((Zc-mu)/sd,rc)
            out['H-GP (GBM+offset+GP resid)']=fte+off+gp2.predict((Zf(te)-mu)/sd)
    return out
def records(proto,dname,rep,k,e,te,preds,rows):
    y=te.watt_mean.values; ex=te.experiment_id.values; ue,inv=np.unique(ex,return_inverse=True)
    h=(te.star.values==0).astype(float); d_=1-h; hi=te.hi.values.astype(float)
    for meth,p in preds.items():
        ae=np.abs(y-p); b=lambda w:np.bincount(inv,weights=w,minlength=len(ue))
        n=b(np.ones(len(y))); s=b(ae); nh=b(h); sh=b(ae*h); nd=b(d_); sd=b(ae*d_); nhi=b(hi); shi=b(ae*hi)
        for i,x in enumerate(ue): rows.append((proto,dname,rep,k,e,meth,x,n[i],s[i],nh[i],sh[i],nd[i],sd[i],nhi[i],shi[i]))
def split_exp(T,rng):
    exps=T.experiment_id.unique()
    for _ in range(100):
        te_e=rng.choice(exps,size=max(2,int(round(.3*len(exps)))),replace=False)
        te=T[T.experiment_id.isin(te_e)]; pool=T[~T.experiment_id.isin(te_e)]
        if len(te)>=40 and len(pool)>=250: return pool,te
    raise RuntimeError
def one(proto,rep,s,t,ctx):
    rng=np.random.default_rng(7919*rep+13); S=m[m.machine==s]; T=m[m.machine==t]; dname=s[4:]+'->'+t[4:]; rows=[]
    if proto=='naive':
        idx=rng.permutation(len(T)); n=int(.7*len(T)); pool,te=T.iloc[idx[:n]],T.iloc[idx[n:]]
    else: pool,te=split_exp(T,rng)
    if proto=='exp_purged': pool=pool[~pool.setting.isin(set(te.setting))]
    if proto=='exp_healthy': pool=pool[pool.star==0]
    if proto in ('naive','exp','exp_purged','exp_healthy'):
        assert proto=='naive' or not (set(pool.experiment_id)&set(te.experiment_id))   # separation audit
        for k in KS:
            if k>len(pool): continue
            cal=pool.iloc[rng.permutation(len(pool))[:k]]; records(proto,dname,rep,k,0,te,predict_all(cal,te,ctx,k),rows)
    elif proto=='expcal':
        pe=pool.experiment_id.unique()
        for e in ELIST:
            if e>len(pe): continue
            ch=rng.choice(pe,e,replace=False); cal=pool[pool.experiment_id.isin(ch)]; k=len(cal)
            records('expcal',dname,rep,k,e,te,predict_all(cal,te,ctx,k),rows)
            calm=pool.iloc[rng.permutation(len(pool))[:k]]   # matched number of cuts, drawn path-level
            records('matched',dname,rep,k,e,te,predict_all(calm,te,ctx,k),rows)
    return rows
if __name__=='__main__':
    proto=sys.argv[1]; R=int(sys.argv[2]); t0=time.time()
    ctxs={}; 
    for s in ['imi_vm20i','imi_vmx30ui']:
        S=m[m.machine==s]; ctxs[(s,'all')]=make_ctx(S); ctxs[(s,'healthy')]=make_ctx(S[S.star==0])
    jobs=[(proto,r,s,t,ctxs[(s,'healthy' if proto=='exp_healthy' else 'all')]) for r in range(R) for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]]
    out=Parallel(n_jobs=2)(delayed(one)(*j) for j in jobs)
    df=pd.DataFrame([r for rr in out for r in rr],columns=['protocol','dir','rep','k','e','method','exp','n','sae','n_h','sae_h','n_d','sae_d','n_hi','sae_hi'])
    df.to_csv(f'results/b2_{proto}.csv',index=False); print(proto,'done',round(time.time()-t0),'s',len(df))
