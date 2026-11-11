import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BLUE,ORG,AQ,GRY,INK,MUT='#2a78d6','#eb6834','#1baf7a','#6b6a66','#0b0b0b','#52514e'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':6.2,'savefig.dpi':300})
fig=plt.figure(figsize=(3.4,2.95)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,110); ax.axis('off')
def box(x,y,w,h,title,body,col='#f3f4f6',ec=GRY,tc=INK,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=col,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h-2.2,title,ha='center',va='top',fontsize=6.4,fontweight='bold',color=tc)
    ax.text(x+w/2,y+h-7.0,body,ha='center',va='top',fontsize=5.7,color=MUT,linespacing=1.25)
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,lw=0.7,color=INK))
ax.text(1,108.5,'(a) Benchmark construction and uncertainty assessment',fontsize=6.8,fontweight='bold',va='top')
box(1,87,46,16,'MSM data','2,508 cutting paths\n2 Hurco machines, 56 experiments',col='#eaf2fc',ec=BLUE)
box(53,87,46,16,'Target calibration cuts','random, concentrated, k-centre,\nSobol; 3–250 cuts',col='#eaf2fc',ec=BLUE)
box(1,67,98,13,'Competing strategies','offset-calibrated transfer vs target-only physics, GP and GBM models')
box(1,48,98,14,'Uncertainty and sensitivity analysis','experiment-disjoint MAE · paired ΔMAE · cluster bootstrap\nleave-one-experiment-out · coefficient diagnostic')
arrow(24,86.2,40,80.8); arrow(76,86.2,60,80.8); arrow(50,66.2,50,62.8)
ax.plot([0,100],[45.5,45.5],color=GRY,lw=0.5,ls=(0,(3,2)))
ax.text(1,44,'(b) Commissioning reporting decision',fontsize=6.8,fontweight='bold',va='top')
ax.text(1,39.2,'Calibration coverage of the intended materials and conditions',fontsize=5.7,color=MUT,va='top',style='italic')
W=31.2
box(1,1,W,35,'Covered','all materials,\nbroad conditions:\ncompare local model\nwith transfer by\ngrouped validation\n(233 vs 301 W\nat 250 cuts)',col='#e6f6ef',ec=AQ)
box(1+W+2.2,1,W,35,'Narrow','materials covered,\nfew tool–material\ncombinations:\ncollect more varied\ncuts and reassess\n(≈ parity, +13 W)',col='#fdeee6',ec=ORG)
box(1+2*(W+2.2),1,W,35,'Material missing','prefer offset\ntransfer: +12–33 W\nvs +227–252 W\nfor the physics\nmodel; add the\nmissing material',col='#eaf2fc',ec=BLUE)
ax.text(50,25.0,'',fontsize=5)
fig.savefig('figures/fig1.png'); 
