import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import HuberRegressor
import os
P=os.environ.get('MSM_MASTER_CSV','data/master_cutting_paths_CLEAN.csv')
FEATS=['N','ap','ae','F','Z','D','dir','MRR','fz','ae_D','ap_D','vc','hm']
def load():
    m=pd.read_csv(P).copy(); m=m[m.watt_mean.notna()&m.machine.isin(['imi_vm20i','imi_vmx30ui'])].reset_index(drop=True)
    m['dir']=(m.direction=='UP').astype(int); m['star']=m.Tool.str.contains(r'\*').astype(int)
    m['MRR']=m.ap*m.ae*m.F/1000; m['fz']=m.F*60/(m.N*m.Z); m['vc']=np.pi*m.D*m.N/60000
    m['ae_D']=m.ae/m.D; m['ap_D']=m.ap/m.D
    a=np.clip(m.ae_D,1e-4,1); m['hm']=m.fz*np.sqrt(a*(1-a))*2
    m['trt']=m.experiment_id+'|'+m.N.astype(str)+'|'+m.ap.astype(str)+'|'+m.ae.astype(str)+'|'+m.F.astype(str)
    return m
def gbm(): return H(max_iter=300,learning_rate=0.05,random_state=0)
