import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from analyze2 import boot_tables, B
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
D=pd.read_csv('results/ideaA_draws.csv'); E=pd.read_csv('results/ideaA_exp.csv')
E['design']=E.protocol.str.split('|').str[0]; E['policy']=E.protocol.str.split('|').str[1]
print('fallback (too few cal experiments to set thresholds) share by design:',D.groupby('design').fallback.mean().round(2).to_dict())
print('\n#### MAE (W) by policy, design, budget (both directions pooled, row-weighted)')
g=E.groupby(['design','k','policy'])[['sae','n']].sum(); T=(g.sae/g.n).unstack('policy')[['always_T','always_L','best_single','nn_only','nn_mat','disagree','oracle_row']].round(0); print(T.to_string())
# ---- pre-registered test: pooled over cells with equal cell weights, per direction
E['cell']=E.design+'@'+E.k.astype(str)
cells=sorted(E.cell.unique()); rng=np.random.default_rng(1)
def pooled(df,policy):
    """equal-weight mean over cells of cluster-bootstrap MAE; returns (B,) array and point"""
    arrs=[];pts=[]
    for c in cells:
        x=df[(df.cell==c)&(df.policy==policy)]
        if x.empty: continue
        arrs.append(x); 
    return arrs
out=[]
for d in sorted(E['dir'].unique()):
    sub=E[E['dir']==d]; exps=np.array(sorted(sub.exp.unique())); pos={e:i for i,e in enumerate(exps)}; idx=rng.integers(0,len(exps),(B,len(exps)))
    def cell_mae(c,pol):
        x=sub[(sub.cell==c)&(sub.policy==pol)]
        if x.empty: return None
        s=np.zeros(len(exps)); n=np.zeros(len(exps)); ii=x.exp.map(pos).values; s[ii]=x.groupby('exp').sae.sum().reindex(x.exp).values if False else 0
        gg=x.groupby('exp')[['sae','n']].sum(); jj=gg.index.map(pos).values; s[jj]=gg.sae.values; n[jj]=gg.n.values
        den=n[idx].sum(1); return s[idx].sum(1)/np.where(den>0,den,np.nan), s.sum()/max(n.sum(),1)
    pol=['always_T','always_L','best_single','nn_only','nn_mat','disagree','oracle_row']; M={p:[] for p in pol}; valid=[]
    for c in cells:
        r={p:cell_mae(c,p) for p in pol}
        if any(v is None for v in r.values()): continue
        valid.append(c); [M[p].append(r[p][0]) for p in pol]
    A={p:np.nanmean(np.vstack(M[p]),0) for p in pol}
    best_fixed=np.nanmean(np.vstack([np.minimum(a,b) for a,b in zip(M['always_T'],M['always_L'])]),0)   # per-cell oracle-fixed comparator
    for lab,base in [('proposed nn_mat vs best-fixed (oracle per cell)',best_fixed),('proposed nn_mat vs always_T',A['always_T']),('proposed nn_mat vs always_L',A['always_L']),('proposed nn_mat vs best_single (chosen on cal data)',A['best_single']),('proposed nn_mat vs nn_only',A['nn_only']),('proposed nn_mat vs disagree',A['disagree'])]:
        x=A['nn_mat']-base; out.append((d,lab,round(np.nanmean(x)),f'[{np.nanpercentile(x,2.5):.0f},{np.nanpercentile(x,97.5):.0f}]',round(np.nanmean(x<0),2)))
    out.append((d,'ORACLE per-row headroom (nn_mat - oracle_row)',round(np.nanmean(A['nn_mat']-A['oracle_row'])),'—','—'))
    print(f'\n[{d}] cells used: {len(valid)}; mean MAE per policy:',{p:round(float(np.nanmean(A[p]))) for p in pol})
R=pd.DataFrame(out,columns=['dir','comparison','dMAE (W)','95% CI','P(proposed better)']); print('\n#### PRE-REGISTERED TEST (negative = proposed policy better)'); print(R.to_string(index=False))
# ---- per-cell view of proposed vs best fixed (point estimates)
print('\n#### per-cell: proposed minus best-fixed (W, point estimates, both dirs pooled)')
pc=(T['nn_mat']-T[['always_T','always_L']].min(axis=1)).round(0).unstack('k'); print(pc.to_string())
# ---- costly mispredictions and risk-coverage
print('\n#### costly misprediction rate (|error|>500 W) by design (mean over draws)')
print(D.groupby('design')[['costly_rate_T','costly_rate_L','costly_rate_policy','costly_rate_oracle']].mean().round(3).to_string())
print('\n#### can coverage distance flag risky rows? AUC for predicting costly errors (0.5=useless) and MAE of kept 80% vs dropped 20%')
print(D.groupby('design')[['auc_risk','mae_all','mae_keep80','mae_drop20']].mean().round(2).to_string())
