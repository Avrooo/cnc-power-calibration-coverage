import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import analyze2
from analyze2 import boot_tables, B
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
D=pd.read_csv('results/cov_draws.csv'); E=pd.read_csv('results/cov_exp.csv')
print('draws',len(D),'| by scheme x k feasibility:'); print(D.pivot_table(index='name',columns='k',values='rep',aggfunc='count').to_string())
print('\n#### mean MAE (W) and coverage by scheme at each budget (pooled over both directions)')
g=D.groupby(['k','name']).agg(n_exp=('n_exp','mean'),n_combo=('n_combo','mean'),mat_cov=('mat_cov','mean'),nn=('nn','mean'),T_offset=('MAE_T-offset','mean'),T_phys=('MAE_T-physics','mean'),O_phys=('MAE_O-physics','mean'),O_GP=('MAE_O-GP','mean')).round(2)
print(g.to_string())
# ---- attribution: does sampling scheme add anything beyond coverage? (grouped CV by test split)
D['grp']=D['dir']+'_'+D.rep.astype(str); D['lk']=np.log(D.k); D['dirc']=(D['dir']==D['dir'].iloc[0]).astype(int)
D['lnn']=np.log(D.nn); D['lnexp']=np.log(D.n_exp); D['lncombo']=np.log(D.n_combo)
cov=['mat_cov','combo_cov','lnn','lnexp','lncombo']; base=['lk','dirc']
sch=pd.get_dummies(D.name,drop_first=True).astype(float); X_all=pd.concat([D[base+cov],sch],axis=1)
def cvr2(cols,y):
    p=np.zeros(len(D))
    for a,b in GroupKFold(10).split(D,groups=D.grp):
        p[b]=LinearRegression().fit(X_all.iloc[a][cols],y.iloc[a]).predict(X_all.iloc[b][cols])
    return r2_score(y,p)
print('\n#### attribution: out-of-split R2 of log(MAE) (10-fold grouped by test split)')
rows=[]
for meth in ['MAE_O-physics','MAE_T-offset','MAE_O-GP']:
    y=np.log(D[meth]); sc=list(sch.columns)
    rows.append((meth,cvr2(base,y),cvr2(base+sc,y),cvr2(base+cov,y),cvr2(base+cov+sc,y),*[cvr2(base+[c],y) for c in ['mat_cov','lnn','lnexp','combo_cov']]))
print(pd.DataFrame(rows,columns=['target','budget+dir only','+scheme','+coverage','+coverage+scheme','mat_cov alone','nn alone','n_exp alone','combo_cov alone']).round(3).to_string(index=False))
# ---- strategy test with experiment-level cluster bootstrap
E['method']=E.method+'@'+E.protocol
res=boot_tables(E,'sae','n','x')
def delta(d,k,a,b):
    A=res[(d,k,a)][0]; Bv=res[(d,k,b)][0]; x=A-Bv; return np.nanmean(x),np.nanpercentile(x,2.5),np.nanpercentile(x,97.5),np.nanmean(x<0)
print('\n#### paired cluster-bootstrap: MAE(first) - MAE(second), W (negative = first better). [95% CI], P(first better)')
rows=[]
for d in sorted(E['dir'].unique()):
    for k in [60,150,250]:
        for lab,a,b in [('O-physics: aware vs random','O-physics@aware','O-physics@random'),('O-physics: aware vs 2-experiments','O-physics@aware','O-physics@exps2'),('O-physics: aware vs 4-experiments','O-physics@aware','O-physics@exps4'),('O-physics: random vs 4-experiments','O-physics@random','O-physics@exps4'),
                        ('LOCAL(aware) vs TRANSFER(aware)','O-physics@aware','T-offset@aware'),('LOCAL(random) vs TRANSFER(random)','O-physics@random','T-offset@random'),('LOCAL(4exp) vs TRANSFER(4exp)','O-physics@exps4','T-offset@exps4'),('LOCAL(2exp) vs TRANSFER(2exp)','O-physics@exps2','T-offset@exps2')]:
            if (d,k,a) in res and (d,k,b) in res:
                mu,lo,hi,p=delta(d,k,a,b); rows.append((d,k,lab,round(mu),f'[{lo:.0f},{hi:.0f}]',round(p,2)))
print(pd.DataFrame(rows,columns=['dir','k','comparison','dMAE','95% CI','P(first better)']).to_string(index=False))
# ---- when does local beat transfer? by coverage
D['local_wins']=(D['MAE_O-physics']<D['MAE_T-offset']).astype(int); D['gap']=D['MAE_O-physics']-D['MAE_T-offset']
D['matfull']=np.where(D.mat_cov>=0.999,'all test materials in cal','some test material missing')
D['nn_bin']=pd.qcut(D.nn,4,labels=['Q1 closest','Q2','Q3','Q4 farthest'])
print('\n#### local (O-physics) vs transfer (T-offset) by coverage, all draws'); 
print(D.groupby('matfull').agg(draws=('gap','size'),P_local_wins=('local_wins','mean'),mean_gap_W=('gap','mean')).round(2).to_string())
print(D[D.matfull.str.startswith('all')].groupby('nn_bin').agg(draws=('gap','size'),nn_mean=('nn','mean'),P_local_wins=('local_wins','mean'),mean_gap_W=('gap','mean')).round(2).to_string())
D.to_csv('results/cov_draws_annot.csv',index=False)
