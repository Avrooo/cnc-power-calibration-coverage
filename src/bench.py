import sys, time
from common import *
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
HEALTHY = len(sys.argv)>1 and sys.argv[1]=='healthy'
R = int(sys.argv[2]) if len(sys.argv)>2 else 30
KS=[0,1,2,3,5,10,20,50,100,200]
m=load()
if HEALTHY: m=m[m.star==0].reset_index(drop=True)
MATS=sorted(m.Workpiece.unique())
def prep(d):
    u=(d.N/1e4).values; mrr=(d.ap*d.ae*d.F).values; hm=np.clip(d.hm.values,1e-3,None)
    mi=np.array([MATS.index(w) for w in d.Workpiece]); return u,mrr,hm,mi
def phys(theta,u,mrr,hm,mi):
    b0,b1,b2,m_=theta[:4]; K=np.exp(theta[4:4+len(MATS)])
    return b0+b1*u+b2*u*u+K[mi]*mrr*(hm/0.05)**(-m_)
def fit_phys(d,theta0=None,free=None,fixed=None):
    u,mrr,hm,mi=prep(d); y=d.watt_mean.values
    lo=[-np.inf]*3+[0.0]+[-5]*len(MATS); hi=[np.inf]*3+[0.8]+[6]*len(MATS)
    t0=np.array([d.watt_mean.min(),0,0,0.2]+[np.log(1.5)]*len(MATS)) if theta0 is None else theta0.copy()
    if free is None:
        t0[0]=np.median(y)*0.8
        r=least_squares(lambda t:phys(t,u,mrr,hm,mi)-y,t0,bounds=(lo,hi),loss='soft_l1',f_scale=300)
        return r.x
    th=t0.copy()
    def f(z): t=th.copy(); t[free]=z; return phys(t,u,mrr,hm,mi)-y
    r=least_squares(f,th[free],loss='soft_l1',f_scale=300); th[free]=r.x; return th
def baseline_fit(d):  # idle law on low-MRR paths
    lo=d[d.MRR<=d.MRR.quantile(.10)]; X=np.c_[lo.N/1e4,(lo.N/1e4)**2]; return HuberRegressor().fit(X,lo.watt_mean)
def bl(h,d): return h.predict(np.c_[d.N/1e4,(d.N/1e4)**2])
def metrics(y,p): return r2_score(y,p),np.mean(np.abs(y-p))
def pool_split(tgt,rng,protocol):
    exps=tgt.experiment_id.unique(); 
    if protocol=='grouped':
        for _ in range(50):
            te_e=rng.choice(exps,size=max(2,int(round(.3*len(exps)))),replace=False)
            te=tgt[tgt.experiment_id.isin(te_e)]; pool=tgt[~tgt.experiment_id.isin(te_e)]
            if len(te)>=40 and len(pool)>=30: return pool,te
    idx=rng.permutation(len(tgt)); n=int(.7*len(tgt)); return tgt.iloc[idx[:n]],tgt.iloc[idx[n:]]  # naive/row split
def one(rep,src_name,tgt_name,protocol):
    rng=np.random.default_rng(1000*rep+7); S=m[m.machine==src_name]; T=m[m.machine==tgt_name]
    pool,te=pool_split(T,rng,protocol); y=te.watt_mean.values
    # source models (trained once per call; deterministic)
    f_S=gbm().fit(S[FEATS],S.watt_mean); hS=baseline_fit(S); g_S=gbm().fit(S[FEATS],S.watt_mean-bl(hS,S))
    thS=fit_phys(S)
    fS_te=f_S.predict(te[FEATS]); gS_te=g_S.predict(te[FEATS]); uT,mT,hT,miT=prep(te); phS_te=phys(thS,uT,mT,hT,miT)
    out=[]
    for k in KS:
        if k>len(pool): continue
        cal=pool.iloc[rng.permutation(len(pool))[:k]] if k>0 else pool.iloc[:0]
        # air-cut style calibration: k lowest-MRR rows from lowest-20% pool
        low=pool[pool.MRR<=pool.MRR.quantile(.2)]; air=low.iloc[rng.permutation(len(low))[:k]] if k>0 else low.iloc[:0]
        res={}
        res['M2 direct source GBM']=fS_te
        if k==0:
            res['M3 source GBM + offset']=fS_te
            res['M5 decomp + baseline offset']=gS_te+bl(hS,te)  # assume source baseline
            res['M7 physics-transfer + offset']=phS_te
        else:
            r_c=cal.watt_mean.values-f_S.predict(cal[FEATS]); res['M3 source GBM + offset']=fS_te+np.median(r_c)
            if k>=5:
                fc=f_S.predict(cal[FEATS]); x=fc-fc.mean(); yv=cal.watt_mean.values-fc
                b=(x@(yv-np.median(yv)))/(x@x+1e6/k)
                res['M4 source GBM + affine']=fS_te+np.median(yv)+b*(fS_te-fc.mean())
            else: res['M4 source GBM + affine']=res['M3 source GBM + offset']
            cc=cal.watt_mean.values-g_S.predict(cal[FEATS]); res['M5 decomp + baseline offset']=gS_te+np.median(cc)
            ca=air.watt_mean.values-g_S.predict(air[FEATS]); res['M5b decomp + air-cut offset']=gS_te+np.median(ca)
            if k>=10:
                X=np.c_[np.ones(k),cal.N.values/1e4]; w=np.linalg.lstsq(X.T@X+1e-3*np.eye(2),X.T@cc,rcond=None)[0]
                res['M6 decomp + baseline offset&slope']=gS_te+w[0]+w[1]*te.N.values/1e4
            # physics transfer: recalibrate b0 (k<10) or b0,b1 (k>=10)
            uC,mC,hC,miC=prep(cal)
            th=fit_phys(cal,theta0=thS,free=[0] if k<10 else [0,1])
            res['M7 physics-transfer + offset']=phys(th,uT,mT,hT,miT)
            if k>=10:
                thT=fit_phys(cal); res['M1 target-only physics']=phys(thT,uT,mT,hT,miT)
                res['M0 target-only GBM']=H(max_iter=150,learning_rate=0.08,min_samples_leaf=max(2,min(20,k//5)),random_state=0).fit(cal[FEATS],cal.watt_mean).predict(te[FEATS])
                # GP hybrid on residual of M3
                Z=lambda d:np.c_[d.N/1e4,np.log10(d.MRR+1e-3),d.ap_D,np.log10(d.hm+1e-3)]
                mu,sd=Z(cal).mean(0),Z(cal).std(0)+1e-9; rc=cal.watt_mean.values-f_S.predict(cal[FEATS])-np.median(r_c)
                gp=GPR(C(1e4)*RBF([1,1,1,1],(0.1,20))+WhiteKernel(5e4),normalize_y=False,n_restarts_optimizer=0).fit((Z(cal)-mu)/sd,rc)
                res['M8 GBM+offset+GP residual']=res['M3 source GBM + offset']+gp.predict((Z(te)-mu)/sd)
                # pooled multitask GBM
                Xp=pd.concat([S[FEATS].assign(tgt=0),cal[FEATS].assign(tgt=1)]); yp=np.r_[S.watt_mean.values,cal.watt_mean.values]; wp=np.r_[np.ones(len(S)),np.full(k,5.0)]
                res['M9 pooled GBM (machine flag)']=H(max_iter=300,learning_rate=0.05,random_state=0).fit(Xp,yp,sample_weight=wp).predict(te[FEATS].assign(tgt=1))
        for name,p in res.items():
            r2,ma=metrics(y,p); out.append((protocol,src_name,tgt_name,rep,k,name,r2,ma))
    return out
if __name__=='__main__':
    t0=time.time(); jobs=[]
    for protocol in ['grouped','naive']:
        for s,t in [('imi_vm20i','imi_vmx30ui'),('imi_vmx30ui','imi_vm20i')]:
            for rep in range(R if protocol=='grouped' else max(10,R//3)): jobs.append((rep,s,t,protocol))
    rows=Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    df=pd.DataFrame([r for rr in rows for r in rr],columns=['protocol','src','tgt','rep','k','method','R2','MAE'])
    df.to_csv(f'results/bench_{"healthy" if HEALTHY else "all"}.csv',index=False); print('done',round(time.time()-t0),'s',len(df))
