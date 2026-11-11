import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings, json, os; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import analyze2
from analyze2 import boot_tables, B
OUT='figures/'; N={}
BLUE,ORG,AQ,GRY,INK,MUT='#2a78d6','#eb6834','#1baf7a','#6b6a66','#0b0b0b','#52514e'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
    'axes.edgecolor':'#8a8985','axes.linewidth':0.6,'xtick.major.width':0.5,'ytick.major.width':0.5,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.dpi':300,'figure.dpi':300,'axes.titlesize':7.5,'axes.titleweight':'bold','text.color':INK,'axes.labelcolor':INK,'xtick.color':MUT,'ytick.color':MUT})
SC,FW=3.45,7.1
m=pd.read_csv(os.environ.get('MSM_MASTER_CSV','data/master_cutting_paths_CLEAN.csv'))
def pooled(res,key_fn,dirs):
    arrs=[res[key_fn(d)][0] for d in dirs]; pts=[res[key_fn(d)][1] for d in dirs]; a=np.mean(arrs,0); return float(np.mean(pts)),float(np.nanpercentile(a,2.5)),float(np.nanpercentile(a,97.5)),a
def grid(ax,y=True):
    ax.grid(axis='y' if y else 'x',color='#e4e3df',lw=0.5); ax.set_axisbelow(True)
# ---------------- Table 1 -----------------
p=m[m.watt_mean.notna()&m.machine.isin(['imi_vm20i','imi_vmx30ui'])].copy(); p['star']=p.Tool.str.contains(r'\*')
T1={}
for mc,d in p.groupby('machine'):
    T1[mc]=dict(experiments=int(d.experiment_id.nunique()),paths=len(d),materials=', '.join(sorted(d.Workpiece.unique())),tools=int(d.Tool.nunique()),damaged=int(d.star.sum()),
      lab0=int((d.label==0).sum()),lab1=int((d.label==1).sum()),lab2=int((d.label==2).sum()),N=f"{int(d.N.min())}–{int(d.N.max())}",ap=f"{d.ap.min():.2f}–{d.ap.max():.1f}",ae=f"{d.ae.min():.2f}–{d.ae.max():.1f}",F=f"{d.F.min():.1f}–{d.F.max():.0f}",
      P=f"{d.watt_mean.mean():.0f} ± {d.watt_mean.std():.0f}",dur=f"{d.duration_s.median():.1f}",smp=f"{(d.n_samples/d.duration_s).median():.1f}")
N['table1']=T1; N['tmf_paths_excluded']=int((m.machine=='tmf_vf10').sum()); N['total_paths']=len(p)
# ---------------- Fig 2 leakage ladder -----------------
L=pd.read_csv('results/leakage_ladder.csv'); order=['random','random, sibling-purged','held-out experiment','held-out tool','held-out material','held-out machine']
lab={'random':'Random\ncuts','random, sibling-purged':'Random,\nno siblings','held-out experiment':'Held-out\nexperiment','held-out tool':'Held-out\ntool','held-out material':'Held-out\nmaterial','held-out machine':'Held-out\nmachine'}
pool=L[L.scope.str.startswith('pooled')].set_index('split').R2; N['ladder']=pool.to_dict()
fig,ax=plt.subplots(figsize=(SC,2.4)); x=np.arange(len(order))
ax.bar(x,[pool[o] for o in order],color=BLUE,width=0.6,zorder=2,label='Pooled, both machines')
for mc,mk,c in [('vm20i','o',ORG),('vmx30ui','s',AQ)]:
    s_=L[L.scope==mc].set_index('split').R2; xs=[i for i,o in enumerate(order) if o in s_.index]; off=-0.11 if mc=='vm20i' else 0.11
    ax.scatter([i+off for i in xs],[s_[order[i]] for i in xs],marker=mk,s=14,color=c,zorder=3,edgecolor='white',linewidth=0.5,label={'vm20i':'Within VM20i only','vmx30ui':'Within VMX30Ui only'}[mc])
for i,o in enumerate(order):
    v=pool[o]
    if v>=0.2: ax.text(i,v/2,f"{v:.2f}",ha='center',va='center',fontsize=6.5,color='white',fontweight='bold')
    elif v>=0: ax.text(i,v+0.04,f"{v:.2f}",ha='center',va='bottom',fontsize=6.5)
    else: ax.text(i,v-0.05,f"{v:.2f}",ha='center',va='top',fontsize=6.5)
ax.axhline(0,color='#8a8985',lw=0.6); ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in order],fontsize=6); ax.set_ylabel('Out-of-split R²'); ax.set_ylim(-1.45,1.0); grid(ax)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.02),ncol=1,handlelength=1.2)
fig.tight_layout(pad=0.4); fig.savefig(OUT+'fig2.png'); plt.close()
# ---------------- Fig 3 budget curves with cluster-bootstrap CIs -----------------
E=pd.read_csv('results/b2_exp.csv'); res=boot_tables(E,'sae','n','x'); dirs=['vm20i->vmx30ui','vmx30ui->vm20i']; KS=[3,5,10,20,30,50,75,100,150,200]
fig,axs=plt.subplots(1,2,figsize=(FW,2.5),sharey=True)
sty={'T-offset (GBM)':(BLUE,'o','Source model + offset (transfer)'),'O-physics (reg.)':(ORG,'s','Target-only physics'),'O-GP':(AQ,'^','Target-only GP'),'O-GBM':(GRY,'D','Target-only GBM')}
N['fig3']={}
for ax,d in zip(axs,dirs):
    for meth,(c,mk,lb) in sty.items():
        ks=[k for k in KS if (d,k,meth) in res]; pt=[res[(d,k,meth)][1] for k in ks]; lo=[np.nanpercentile(res[(d,k,meth)][0],2.5) for k in ks]; hi=[np.nanpercentile(res[(d,k,meth)][0],97.5) for k in ks]
        ax.fill_between(ks,lo,hi,color=c,alpha=0.09,lw=0); ax.plot(ks,pt,color=c,marker=mk,ms=3,lw=1.1,label=lb,zorder=3)
        N['fig3'][f'{d}|{meth}']={int(k):round(v) for k,v in zip(ks,pt)}
    ax.set_xscale('log'); ax.set_xticks([3,5,10,20,50,100,200]); ax.set_xticklabels(['3','5','10','20','50','100','200']); ax.set_xlabel('Calibration cuts on target machine (k)'); ax.set_ylim(150,800); grid(ax)
    ax.set_title(('(a) ' if d==dirs[0] else '(b) ')+d.replace('->',' → ').replace('vm20i','VM20i').replace('vmx30ui','VMX30Ui'),loc='left')
axs[0].set_ylabel('Held-out MAE (W)'); axs[0].legend(frameon=False,loc='upper right',handlelength=1.6)
fig.tight_layout(pad=0.4,w_pad=0.6); fig.savefig(OUT+'fig3.png'); plt.close()
# direct transfer MAE reference (earlier benchmark, experiment-separated, all rows)
a=pd.read_csv('results/bench_all.csv'); g=a[(a.protocol=='grouped')&(a.method=='M2 direct source GBM')]; N['direct_MAE']=round(float(g.MAE.mean())); N['direct_R2']=round(float(g.R2.mean()),2)
# ---------------- Table 2: crossover by rival x protocol -----------------
protos=['naive','exp']; rivals={'Target-only physics':['O-physics (reg.)'],'Target-only GP':['O-GP'],'Target-only GBM':['O-GBM']}; T2={}
for rname,oo in rivals.items():
    analyze2.OO=oo
    for pr in protos:
        _,K=analyze2.summarize(pr); T2[f'{rname}|{pr}']={r['dir']:(float(r.k_star_median),float(r.P_never_within_200)) for _,r in K.iterrows()}
analyze2.OO=['O-physics (reg.)','O-GP','O-GBM']
infl={}
fr=[]
for pr in ['naive','exp']:
    d=pd.read_csv(f'results/b2_{pr}.csv'); gq=d.groupby(['k','method'])[['sae','n']].sum(); fr.append((gq.sae/gq.n).rename(pr))
F=pd.concat(fr,axis=1); 
for meth,nm in [('T-offset (GBM)','Source + offset'),('O-physics (reg.)','Target-only physics'),('O-GP','Target-only GP'),('O-GBM','Target-only GBM')]:
    infl[nm]=round(100*(F.loc[(100,meth),'exp']-F.loc[(100,meth),'naive'])/F.loc[(100,meth),'exp'],0)
N['table2']={k:v for k,v in T2.items()}; N['inflation_k100']=infl
# ---------------- Fig 4 coverage reversal -----------------
C=pd.read_csv('results/cov_exp.csv'); C['method']=C.method+'@'+C.protocol; rc=boot_tables(C,'sae','n','x')
schemes=[('random','Random cuts'),('aware','Design-only\nspread'),('exps8','8 experiments'),('combos4','4 tool-material\ncombinations'),('exps4','4 experiments'),('combos2','2 combinations'),('combos1','1 combination')]
fig,axs=plt.subplots(1,2,figsize=(FW,2.25),gridspec_kw={'width_ratios':[1.5,1]})
ax=axs[0]; N['fig4a']={}
for k,mk,fc in [(150,'o',None)]:
    ys=[];lo=[];hi=[]
    for sc,_ in schemes:
        arrs=[];pts=[]
        for d in dirs:
            if (d,k,f'O-physics@{sc}') not in rc: continue
            a=rc[(d,k,f'O-physics@{sc}')][0]-rc[(d,k,f'T-offset@{sc}')][0]; arrs.append(a); pts.append(rc[(d,k,f'O-physics@{sc}')][1]-rc[(d,k,f'T-offset@{sc}')][1])
        a=np.mean(arrs,0); ys.append(float(np.mean(pts))); lo.append(float(np.nanpercentile(a,2.5))); hi.append(float(np.nanpercentile(a,97.5)))
    yy=np.arange(len(schemes))[::-1]
    for i,(y,l,h,v) in enumerate(zip(yy,lo,hi,ys)):
        c=BLUE if v<0 else ORG; ax.plot([l,h],[y,y],color=c,lw=1.4,solid_capstyle='round'); ax.scatter([v],[y],s=22,color=c,zorder=3,edgecolor='white',linewidth=0.5)
        ax.text(h+12,y,f"{v:+.0f} W",va='center',fontsize=6.3,color=INK)
        N['fig4a'][schemes[i][0]]=dict(delta=round(v),lo=round(l),hi=round(h))
ax.axvline(0,color='#8a8985',lw=0.7); ax.set_yticks(yy); ax.set_yticklabels([s[1] for s in schemes],fontsize=6.2); ax.set_xlim(-200,520); ax.set_xlabel('Local minus transfer MAE at 150 cuts (W)')
ax.text(-190,len(schemes)-0.35,'← local model better',fontsize=6.2,color=BLUE,va='bottom'); ax.text(510,len(schemes)-0.35,'transfer better →',fontsize=6.2,color=ORG,va='bottom',ha='right'); ax.set_ylim(-0.6,len(schemes)-0.1); grid(ax,y=False)
ax.set_title('(a) Relative model error by calibration design',loc='left')
D=pd.read_csv('results/cov_draws_annot.csv'); D['mc']=np.where(D.mat_cov>=0.999,'full','miss'); ax=axs[1]
bars=[('All\nmaterials\ncovered',D[D.mc=='full'].local_wins.mean(),(D.mc=='full').sum()),('A material\nmissing',D[D.mc=='miss'].local_wins.mean(),(D.mc=='miss').sum())]
F2=D[D.mc=='full']; q=F2.groupby('nn_bin').local_wins.mean(); cnt=F2.groupby('nn_bin').size()
for nm in ['Q1 closest','Q2','Q3','Q4 farthest']: bars.append((('Closest\nquartile' if nm.startswith('Q1') else 'Q2' if nm.startswith('Q2') else 'Q3' if nm.startswith('Q3') else 'Farthest\nquartile'),q[nm],cnt[nm]))
xs=np.arange(len(bars)); cols=[BLUE,ORG,BLUE,BLUE,BLUE,BLUE]
ax.bar(xs,[b[1] for b in bars],color=cols,width=0.62,zorder=2)
for i,b in enumerate(bars): ax.text(i,b[1]+0.02,f"{100*b[1]:.0f}%",ha='center',fontsize=6.3)
ax.axhline(0.5,color='#8a8985',lw=0.6,ls=(0,(3,2))); ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars],fontsize=5.6,rotation=0); ax.set_ylim(0,1.08); ax.set_ylabel('Share of draws where local beats transfer'); grid(ax)
ax.set_title('(b) What the cuts cover',loc='left'); N['fig4b']={b[0].replace('\n',' '):(round(float(b[1]),2),int(b[2])) for b in bars}
fig.tight_layout(pad=0.4,w_pad=0.8); fig.savefig(OUT+'fig4.png'); plt.close()
# ---------------- Fig 5 commissioning designs (paired vs random cuts) -----------------
E5=pd.read_csv('results/e5_exp.csv'); E5['method']=E5.method+'@'+E5.protocol; r5=boot_tables(E5,'sae','n','x')
def pdiff(kA,mA,kB,mB):
    arrs=[];pts=[]
    for d in dirs:
        arrs.append(r5[(d,kA,mA)][0]-r5[(d,kB,mB)][0]); pts.append(r5[(d,kA,mA)][1]-r5[(d,kB,mB)][1])
    a=np.mean(arrs,0); return float(np.mean(pts)),float(np.nanpercentile(a,2.5)),float(np.nanpercentile(a,97.5)),float(np.nanmean(a<0))
fig,ax=plt.subplots(figsize=(SC,2.5)); N['fig5']={}
series=[('Planned space-filling (emulated)',AQ,'s',lambda k:(k,'O-physics@full|planned',k,'O-physics@full|random')),
        ('Complete experiments',ORG,'^',lambda k:(k,'O-physics@full|complete_exp',k,'O-physics@full|random')),
        ('Offset transfer, 10 cuts',BLUE,'o',lambda k:(10,'T-offset@full|transfer10',k,'O-physics@full|random'))]
for j,(lb,c,mk,fn) in enumerate(series):
    for i,k in enumerate([60,150,250]):
        pt,lo,hi,pb=pdiff(*fn(k)); x_=i+(j-1)*0.24; N['fig5'][f'{lb}|{k}']=dict(d=round(pt),lo=round(lo),hi=round(hi))
        ax.plot([x_,x_],[lo,hi],color=c,lw=1.5,solid_capstyle='round',zorder=2); ax.scatter([x_],[pt],marker=mk,s=20,color=c,zorder=3,edgecolor='white',linewidth=0.5,label=lb if i==0 else None)
        ax.text(x_,hi+14,f"{pt:+.0f}",ha='center',fontsize=5.8)
for i,k in enumerate([60,150,250]):
    N.setdefault('fig5_random',{})[k]=round(r5[('vm20i->vmx30ui',k,'O-physics@full|random')][1]/2+r5[('vmx30ui->vm20i',k,'O-physics@full|random')][1]/2)
ax.axhline(0,color=INK,lw=0.8); ax.text(-0.46,-12,'random cuts = reference (0)',ha='left',va='top',fontsize=6,color=MUT)
ax.set_xticks(range(3)); ax.set_xticklabels(['60 cuts','150 cuts','250 cuts']); ax.set_xlim(-0.5,2.5); ax.set_ylim(-70,700); ax.set_ylabel('MAE minus MAE of random cuts (W)'); grid(ax)
ax.legend(frameon=False,loc='upper right',handlelength=1.0,fontsize=6.2)
fig.tight_layout(pad=0.4); fig.savefig(OUT+'figS1.png'); plt.close()
# ---------------- Table 3: policy vs baselines -----------------
IA=pd.read_csv('results/ideaA_exp.csv'); IA['design']=IA.protocol.str.split('|').str[0]; IA['policy']=IA.protocol.str.split('|').str[1]; IA['cell']=IA.design+'@'+IA.k.astype(str)
cells=sorted(IA.cell.unique()); rng=np.random.default_rng(1); T3={}
pol=['always_T','always_L','best_single','nn_only','nn_mat','disagree','oracle_row']
for d in dirs:
    sub=IA[IA['dir']==d]; exps=np.array(sorted(sub.exp.unique())); pos={e:i for i,e in enumerate(exps)}; idx=rng.integers(0,len(exps),(B,len(exps)))
    def cm(c,pl):
        x=sub[(sub.cell==c)&(sub.policy==pl)]
        if x.empty: return None
        gg=x.groupby('exp')[['sae','n']].sum(); s=np.zeros(len(exps)); n=np.zeros(len(exps)); jj=gg.index.map(pos).values; s[jj]=gg.sae.values; n[jj]=gg.n.values; den=n[idx].sum(1); return s[idx].sum(1)/np.where(den>0,den,np.nan)
    M={p:[] for p in pol}; used=[]
    for c in cells:
        r={p:cm(c,p) for p in pol}
        if any(v is None for v in r.values()): continue
        used.append(c); [M[p].append(r[p]) for p in pol]
    A={p:np.nanmean(np.vstack(M[p]),0) for p in pol}; bf=np.nanmean(np.vstack([np.minimum(a_,b_) for a_,b_ in zip(M['always_T'],M['always_L'])]),0)
    row={'cells':len(used)}
    for p in pol: row[p]=float(np.nanmean(A[p]))
    row['best_fixed_oracle']=float(np.nanmean(bf))
    for lab_,base in [('vs_always_T',A['always_T']),('vs_always_L',A['always_L']),('vs_best_single',A['best_single']),('vs_nn_only',A['nn_only']),('vs_disagree',A['disagree']),('vs_best_fixed_oracle',bf)]:
        x=A['nn_mat']-base; row[lab_]=(float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5)))
    T3[d]=row
N['table3']=T3
ID=pd.read_csv('results/ideaA_draws.csv'); N['auc_risk']=ID.groupby('design').auc_risk.mean().round(2).to_dict(); N['costly']=ID.groupby('design')[['costly_rate_T','costly_rate_L','costly_rate_policy']].mean().round(2).to_dict()
# per-cell proposed minus better-fixed
g=IA.groupby(['design','k','policy'])[['sae','n']].sum(); Tm=(g.sae/g.n).unstack('policy'); N['percell']={f'{i[0]}@{i[1]}':round(float(r.nn_mat-min(r.always_T,r.always_L))) for i,r in Tm.iterrows()}
# ---------------- Fig 1 workflow schematic -----------------
fig,ax=plt.subplots(figsize=(SC,2.3)); ax.set_xlim(0,100); ax.set_ylim(0,70); ax.axis('off')
def box(x,y,w,h,t,fc='#f3f2ee',ec='#8a8985',fs=6.2,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.6',fc=fc,ec=ec,lw=0.7)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=INK,fontweight='bold' if bold else 'normal',linespacing=1.15)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,lw=0.8,color='#52514e',shrinkA=1,shrinkB=1))
box(1,47,27,19,'Existing power\nmodel from the\nsource machine',fc='#e8f0fb',ec=BLUE)
box(1,4,27,19,'k calibration cuts\non the new machine\n(what do they cover?)',fc='#f3f2ee',ec='#8a8985')
box(38,47,29,19,'A: transfer the\nmodel, correct its\nmachine offset',fc='#e8f0fb',ec=BLUE)
box(38,4,29,19,'B: train a local\nmodel on the\ncalibration cuts',fc='#fdeee7',ec=ORG)
box(76,24,23,22,'Lower error\non held-out\nexperiments?',fc='#ffffff',ec=INK,bold=True)
arr(28,56,38,56); arr(28,18,38,14); arr(22,23,40,47); arr(67,56,80,46); arr(67,14,80,24)
fig.tight_layout(pad=0.2); fig.savefig(OUT+'fig1_legacy_unused.png'); plt.close()
json.dump(N,open('results/numbers.json','w'),indent=1,default=str); print('ok')
