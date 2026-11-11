import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
D=pd.read_csv('results/cov_draws_annot.csv')
print('#### WITHIN-SPLIT test: split fixed effects removed (every scheme sees the same held-out experiments)')
sch=pd.get_dummies(D.name,drop_first=True).astype(float); X=pd.concat([D[['lk','mat_cov','combo_cov','lnn','lnexp','lncombo']],sch],axis=1)
rows=[]
for meth in ['MAE_O-physics','MAE_T-offset','MAE_O-GP']:
    y=np.log(D[meth]); yd=y-y.groupby([D.grp,D.k]).transform('mean')          # demean within split x budget
    Xd=X-X.groupby([D.grp,D.k]).transform('mean')
    def cv(cols):
        p=np.zeros(len(D))
        for a,b in GroupKFold(10).split(D,groups=D.grp): p[b]=LinearRegression().fit(Xd.iloc[a][cols],yd.iloc[a]).predict(Xd.iloc[b][cols])
        return r2_score(yd,p)
    sc=list(sch.columns); cov=['mat_cov','combo_cov','lnn','lnexp','lncombo']
    rows.append((meth,cv(sc),cv(cov),cv(cov+sc),cv(['mat_cov']),cv(['lnn']),cv(['lnexp'])))
print(pd.DataFrame(rows,columns=['target','scheme only','coverage only','coverage+scheme','mat_cov alone','nn alone','n_exp alone']).round(3).to_string(index=False))
print('\n#### holding material coverage at 100% (all test materials present in calibration): O-physics MAE (W) by scheme')
F=D[D.mat_cov>=0.999]
print(F.pivot_table(index='name',columns='k',values='MAE_O-physics',aggfunc=['mean','count']).round(0).to_string())
print('\n  same subset, transfer (T-offset):'); print(F.pivot_table(index='name',columns='k',values='MAE_T-offset',aggfunc='mean').round(0).to_string())
print('\n#### P(local beats transfer) by scheme, material coverage 100% vs <100%')
D['mc']=np.where(D.mat_cov>=0.999,'mat 100%','mat <100%')
print(D.pivot_table(index='name',columns='mc',values='local_wins',aggfunc=['mean','count']).round(2).to_string())
