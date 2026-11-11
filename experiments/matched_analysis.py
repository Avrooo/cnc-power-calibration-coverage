import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from matched import DN
S=pickle.load(open('results/matched_sets.pkl','rb')); rng=np.random.default_rng(5); B=2000; CAL=0.30
allD=np.array([s['desc'] for s in S]); SD=allD.std(0)+1e-9
dirs=['vm20i->vmx30ui','vmx30ui->vm20i']
def smd(a,b): return (a.mean(0)-b.mean(0))/np.sqrt((a.var(0)+b.var(0))/2+1e-12)
pairs=[]; unm=[]; before_A=[];before_B=[]; aft_A=[];aft_B=[]
for d in dirs:
    for rep in sorted({s['rep'] for s in S if s['dir']==d}):
        A=[s for s in S if s['dir']==d and s['rep']==rep and s['grp']=='A']; Bs=[s for s in S if s['dir']==d and s['rep']==rep and s['grp']=='B']
        if not A or not Bs: continue
        dA=np.array([a['desc'] for a in A])/SD; dB=np.array([b['desc'] for b in Bs])/SD; before_A+=list(dA*SD); before_B+=list(dB*SD)
        dist=np.sqrt(((dA[:,None,:]-dB[None,:,:])**2).mean(2)); used=set(); order=np.argsort(dist.min(1))
        for i in order:
            j=[jj for jj in np.argsort(dist[i]) if jj not in used]
            if j and dist[i,j[0]]<=CAL: used.add(j[0]); pairs.append((d,A[i],Bs[j[0]])); aft_A.append(A[i]['desc']); aft_B.append(Bs[j[0]]['desc'])
        # unmatched same-split contrast
        mA=np.mean([a['sL']-a['sT'] for a in A],0); mB=np.mean([b['sL']-b['sT'] for b in Bs],0); unm.append((d,A[0]['ue'],mA,mB,A[0]['n']))
print('sets',len(S),'| matched pairs',len(pairs),'(per direction:',{d:sum(1 for p in pairs if p[0]==d) for d in dirs},')')
print('\nBalance (standardised mean difference, 4-experiment minus 8–12-experiment sets):'); sb=smd(np.array(before_A),np.array(before_B)); sa=smd(np.array(aft_A),np.array(aft_B))
print(pd.DataFrame({'descriptor':DN,'SMD before':sb.round(2),'SMD after matching':sa.round(2)}).to_string(index=False))
print('experiments per set: 4-exp group mean %.1f | matched comparison group mean %.1f'%(np.mean([p[1]['nexp'] for p in pairs]),np.mean([p[2]['nexp'] for p in pairs])))
def agg(entries,fn):
    res={}
    for d in dirs:
        exps=sorted({e for x in entries if x[0]==d for e in x[1]}); pos={e:i for i,e in enumerate(exps)}; num=np.zeros((3,len(exps))); den=np.zeros(len(exps))
        for x in entries:
            if x[0]!=d: continue
            vals,nn=fn(x); ii=[pos[e] for e in x[1]]
            for r,v in enumerate(vals): num[r,ii]+=v
            den[ii]+=nn
        idx=rng.integers(0,len(exps),(B,len(exps))); res[d]=(num,den,idx)
    out=[]
    for r in range(3):
        arr=[];pt=[]
        for d in dirs:
            num,den,idx=res[d]; arr.append(num[r][idx].sum(1)/den[idx].sum(1)); pt.append(num[r].sum()/den.sum())
        a=np.mean(arr,0); out.append((float(np.mean(pt)),float(np.percentile(a,2.5)),float(np.percentile(a,97.5)),float(np.mean(a>0))))
    return out
matched=[(d,a['ue'],a,b) for d,a,b in pairs]
mt=agg(matched,lambda x:((x[2]['sL']-x[2]['sT'],x[3]['sL']-x[3]['sT'],(x[2]['sL']-x[2]['sT'])-(x[3]['sL']-x[3]['sT'])),x[2]['n']))
um=agg(unm,lambda x:((x[2],x[3],x[2]-x[3]),x[4]))
print('\nLocal-minus-transfer MAE at 150 cuts (W), pooled over directions, cluster bootstrap over target experiments (all three materials present in every set):')
t=pd.DataFrame([['MATCHED  4-experiment sets',*mt[0][:3]],['MATCHED  8–12-experiment sets',*mt[1][:3]],['MATCHED  difference (4-exp minus 8–12-exp)',*mt[2][:3]],
  ['UNMATCHED 4-experiment sets',*um[0][:3]],['UNMATCHED 8–12-experiment sets',*um[1][:3]],['UNMATCHED difference',*um[2][:3]]],columns=['contrast','estimate','CI low','CI high']).round(0); print(t.to_string(index=False))
print('P(difference>0) matched: %.3f | per-direction matched difference:'%mt[2][3])
for d in dirs:
    sel=[x for x in matched if x[0]==d]; num=sum(((x[2]['sL']-x[2]['sT'])-(x[3]['sL']-x[3]['sT'])).sum() for x in sel); den=sum(x[2]['n'].sum() for x in sel); print('  ',d,round(num/den),'W (n pairs %d)'%len(sel))
# within-split regression on all sets: effect of number of experiments holding coverage descriptors fixed
rows=[]
for s in S: rows.append(dict(dir=s['dir'],rep=s['rep'],nexp=s['nexp'],maeL=s['maeL'],delta=s['maeL']-s['maeT'],**{f'd{i}':v for i,v in enumerate(s['desc'])}))
df=pd.DataFrame(rows); df['split']=df['dir']+'_'+df.rep.astype(str); cols=['nexp']+[f'd{i}' for i in range(6)]
dm=df.copy(); 
for c in cols+['maeL','delta']: dm[c]=df[c]-df.groupby('split')[c].transform('mean')
X=dm[cols].values; X=X/ (X.std(0)+1e-9); sdn=df.nexp.std()
def ols(y): b=np.linalg.lstsq(X,y,rcond=None)[0]; return b[0]/ (dm['nexp'].std()+1e-9)  # per 1 experiment
splits=df.split.unique(); est=[]
for _ in range(1000):
    sel=rng.choice(splits,len(splits)); idx=np.concatenate([np.where(df.split.values==s)[0] for s in sel]); Xs=X[idx]; ys=dm['maeL'].values[idx]; b=np.linalg.lstsq(Xs,ys,rcond=None)[0]; est.append(b[0]/(dm['nexp'].std()+1e-9))
b0=np.linalg.lstsq(X,dm['maeL'].values,rcond=None)[0][0]/(dm['nexp'].std()+1e-9)
print('\nWithin-split regression, local MAE ~ number of experiments + 6 coverage descriptors: change per extra experiment = %.1f W (split-bootstrap 95%% CI %.1f to %.1f)'%(b0,np.percentile(est,2.5),np.percentile(est,97.5)))
