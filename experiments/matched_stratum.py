import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
S=pickle.load(open('results/matched_sets.pkl','rb')); rng=np.random.default_rng(11); Bn=2000; dirs=['vm20i->vmx30ui','vmx30ui->vm20i']
def run(label,keepA,keepB):
    ent=[]; ns=0
    for d in dirs:
        for rep in sorted({s['rep'] for s in S if s['dir']==d}):
            A=[s for s in S if s['dir']==d and s['rep']==rep and s['grp']=='A' and keepA(s)]; B=[s for s in S if s['dir']==d and s['rep']==rep and s['grp']=='B' and keepB(s)]
            if not A or not B: continue
            ns+=1; mA=np.mean([a['sL']-a['sT'] for a in A],0); mB=np.mean([b['sL']-b['sT'] for b in B],0); ent.append((d,A[0]['ue'],mA,mB,A[0]['n'],len(A),len(B),np.mean([a['nexp'] for a in A]),np.mean([b['nexp'] for b in B])))
    if not ent: print(label,'no overlapping splits'); return
    arrs={};pts=[]
    out=[]
    for r in range(3):
        arr=[];pt=[]
        for d in dirs:
            es=[e for e in ent if e[0]==d]
            if not es: continue
            exps=sorted({x for e in es for x in e[1]}); pos={x:i for i,x in enumerate(exps)}; num=np.zeros(len(exps)); den=np.zeros(len(exps))
            for e in es:
                v=[e[2],e[3],e[2]-e[3]][r]; ii=[pos[x] for x in e[1]]; num[ii]+=v; den[ii]+=e[4]
            idx=rng.integers(0,len(exps),(Bn,len(exps))); arr.append(num[idx].sum(1)/den[idx].sum(1)); pt.append(num.sum()/den.sum())
        a=np.mean(arr,0); out.append((np.mean(pt),np.percentile(a,2.5),np.percentile(a,97.5)))
    print(f"{label}: splits with both groups {ns} (by direction {[sum(1 for e in ent if e[0]==d) for d in dirs]}); sets A {sum(e[5] for e in ent)} / B {sum(e[6] for e in ent)}; mean experiments A {np.mean([e[7] for e in ent]):.1f} vs B {np.mean([e[8] for e in ent]):.1f}")
    for nm,o in zip(['  4-experiment sets (local−transfer)','  more-experiment sets (local−transfer)','  difference (4-exp minus more-exp)'],out): print(f"{nm}: {o[0]:+.0f} W  [{o[1]:+.0f}, {o[2]:+.0f}]")
run('Exactly 4 tool–material combinations in both groups',lambda s:s['desc'][0]==4,lambda s:s['desc'][0]==4)
run('4–5 combinations in both groups',lambda s:s['desc'][0]>=4,lambda s:s['desc'][0]<=5)
run('Same speed- and MRR-bin coverage (>=5 of 6 each) AND <=5 combos in comparison group',lambda s:s['desc'][1]>=5 and s['desc'][3]>=5,lambda s:s['desc'][1]>=5 and s['desc'][3]>=5 and s['desc'][0]<=5)
