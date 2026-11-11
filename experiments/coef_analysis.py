import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import pandas as pd, numpy as np
from scipy.stats import spearmanr
D=pd.read_csv('results/coef_diag.csv'); D['split']=D['dir']+'_'+D.rep.astype(str)
print('draws per design:',D.groupby('design').size().to_dict(),'| share with all 3 materials:',D.groupby('design').apply(lambda x:(x.nmat==3).mean()).round(2).to_dict())
F=D[D.nmat==3].copy(); order=['random','exps8','combos4','exps4']
t=F.groupby('design').agg(n=('maeL','size'),nexp=('nexp','mean'),MAE_local=('maeL','mean'),MAE_transfer=('maeT','mean'),dlogK=('dK','median'),dm=('dm','median'),dbase=('dbase','median'),hitK=('hitK','mean'),hit_m=('hitm','mean'),hit_b0=('hitb0','mean')).loc[order].round(2); print('\nAll-material draws: coefficient error vs reference fit, bound hits, errors'); print(t.to_string())
sd=F.groupby('design').agg(sd_m=('m','std'),**{f'sd_logK_{w}':(f'logK_{w}','std') for w in ['AA 6061','AISI 1018','AISI 4140']}).loc[order].round(2); print('\nBetween-draw SD of fitted parameters:'); print(sd.to_string())
# within-split association of parameter error with local error (demean within split x across designs)
for c in ['dK','dm','dbase','maeL']: F[c+'_w']=F[c]-F.groupby('split')[c].transform('mean')
print('\nWithin-split Spearman (draw-level) between local MAE and parameter error:')
for c in ['dK','dm','dbase']:
    r,p=spearmanr(F[c+'_w'],F['maeL_w']); print(f'  {c:6s} rho={r:+.2f} (p={p:.1e})')
# design effect before/after adjusting for parameter error (log MAE, split fixed effects)
F['lm']=np.log(F.maeL); F['lm_w']=F.lm-F.groupby('split').lm.transform('mean')
for dsg in order: F['is_'+dsg]=(F.design==dsg).astype(float); F['is_'+dsg+'_w']=F['is_'+dsg]-F.groupby('split')['is_'+dsg].transform('mean')
des=['is_exps8_w','is_combos4_w','is_exps4_w']; pe=['dK_w','dm_w','dbase_w']
def fit(cols):
    X=F[cols].values; sx=X.std(0); b=np.linalg.lstsq(X/sx,F.lm_w.values,rcond=None)[0]/sx; return dict(zip(cols,b))
a=fit(des); b=fit(des+pe)
print('\nLog-MAE design effect relative to random cuts (within split):')
for c in des: print(f'  {c[3:-2]:8s} without parameter error: {100*(np.exp(a[c])-1):+.0f}%   with parameter error as covariates: {100*(np.exp(b[c])-1):+.0f}%')
