#!/usr/bin/env python3
"""The rating-study figure: what raters respond to, by defect magnitude. Usage: python3 make_fig_human.py --data ../data --out ../../img"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SIX = ['label_collisions','edge_label_crossings','node_overlap_pairs',
       'edge_node_crossings','stacked_routes','route_reversals']
INK='#1a1a1a'; GREY='#8a8a8a'; BLUE='#35618f'; RED='#a8322d'; LGREY='#d8d8d8'
plt.rcParams.update({'font.size':7.4,'font.family':'serif','axes.linewidth':.6,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.4,
    'ytick.major.size':2.4,'axes.edgecolor':INK,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.bbox':'tight'})

def load(d):
    st=pd.read_csv(d/'human_stimuli.csv'); st.columns=[c.replace('report_','') for c in st.columns]
    ra=pd.read_csv(d/'human_ratings.csv'); se=pd.read_csv(d/'human_sets.csv')
    app=se[['image_1','image_2','image_3','image_4']].stack().value_counts()
    st['appear']=st.image.map(app).fillna(0)
    st['bw']=(st.image.map(ra.best.value_counts()).fillna(0)
              -st.image.map(ra.worst.value_counts()).fillna(0))/st.appear
    st['defects']=st[SIX].sum(axis=1)
    st['strict']=(st[SIX]==0).all(axis=1)
    return st

def ci(v,seed=7,n=10000):
    rng=np.random.default_rng(seed)
    b=[rng.choice(v,len(v),replace=True).mean() for _ in range(n)]
    return np.percentile(b,2.5),np.percentile(b,97.5)

def fig_threshold(st,out):
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.12),gridspec_kw={'width_ratios':[1.02,1.0,1.04]})
    # (a) threshold curve
    bins=[-.5,.5,1.5,14.5,1e9]; lab=['0','1','2–14','≥15']
    st=st.copy(); st['bin']=pd.cut(st.defects,bins=bins,labels=lab)
    xs,ms,los,his,ns=[],[],[],[],[]
    for i,b in enumerate(lab):
        s=st[st['bin']==b]
        if len(s)<2: continue
        lo,hi=ci(s.bw.values); xs.append(i); ms.append(s.bw.mean()); los.append(lo); his.append(hi); ns.append(len(s))
    a=ax[0]
    a.axhspan(-0.2,0.2,color=LGREY,alpha=.5,lw=0)
    a.errorbar(xs,ms,yerr=[np.array(ms)-np.array(los),np.array(his)-np.array(ms)],
               fmt='o',ms=4.2,color=BLUE,ecolor=BLUE,elinewidth=1.0,capsize=2.4,zorder=3)
    a.plot(xs,ms,color=BLUE,lw=1.0,alpha=.55,zorder=2)
    for x,m,n in zip(xs,ms,ns): a.annotate(f'n={n}',(x,m),textcoords='offset points',
        xytext=(0,-13) if x<1 else ((14,-2) if x<2 else (0,11)),ha='center' if x!=1 else 'left',fontsize=6.2,color=GREY)
    a.annotate('no figure in\nthe panel here',(2,-0.42),ha='center',va='center',fontsize=6.0,
               color=GREY,style='italic')
    a.axhline(0,color=GREY,lw=.6,ls=(0,(3,2)))
    a.set_xticks(range(len(lab))); a.set_xticklabels(lab)
    a.set_xlabel('geometric defects in the figure'); a.set_ylabel('best−worst score')
    a.set_title('(a) a single defect is not seen;\nmany are seen decisively',fontsize=7.4,pad=4)
    a.annotate('strict verdict\nsplits here',xy=(0.5,1.70),xytext=(1.62,2.18),fontsize=6.2,
        color=RED,ha='center',arrowprops=dict(arrowstyle='-|>',color=RED,lw=.7,shrinkA=0,shrinkB=2))
    a.set_zorder(5)
    a.axvline(0.5,color=RED,lw=.7,ls=(0,(2,1.6)))
    a.set_ylim(-2.45,2.62); a.set_xlim(-0.45,3.45)
    # (b) what raters track
    want=[('label_collisions','label collisions','defect'),
          ('edge_node_crossings','arrows thru box','defect'),
          ('node_overlap_pairs','box overlaps','defect'),
          ('edge_edge_crossings','arrow crossings','defect'),
          ('n_nodes','box count','size'),('area','figure area','size'),
          ('aspect_deviation','aspect dev.','shape')]
    e48=st[st.condition.isin(['ours','graphviz'])]
    names,r72,r48=[],[],[]
    for k,nm,_ in want:
        names.append(nm)
        r72.append(abs(stats.spearmanr(st[k],st.bw).statistic))
        v=e48[k]
        r48.append(abs(stats.spearmanr(v,e48.bw).statistic) if len(set(v.dropna()))>=3 else np.nan)
    y=np.arange(len(names))[::-1]
    b=ax[1]
    b.barh(y+.19,r72,height=.36,color=BLUE,label='all 72 images',lw=0)
    b.barh(y-.19,[0 if np.isnan(v) else v for v in r48],height=.36,color=RED,
           label='48 engine-drawn',lw=0)
    for i,v in enumerate(r48):
        if np.isnan(v):
            k=want[i][0]; u=sorted(set(e48[k].dropna()))
            msg='all zero' if len(u)==1 else '0 or 1 only'
            b.annotate(msg,(0.012,y[i]-.19),va='center',fontsize=5.9,color=RED)
    b.set_yticks(y); b.set_yticklabels(names,fontsize=6.6)
    b.set_xlabel('|Spearman $\\rho$| with rater score'); b.set_xlim(0,.88)
    b.set_title('(b) above the floor, only size\nand shape move the raters',fontsize=7.4,pad=4)
    b.legend(frameon=False,fontsize=6.2,loc='upper center',bbox_to_anchor=(.55,-.30),
             ncol=2,handlelength=1.0,columnspacing=1.0,borderpad=0)
    # (c) contrast coverage
    c=ax[2]
    piv=st.pivot_table(index='paper_id',columns='condition',values='strict',aggfunc='first')
    pairs=[('ours','graphviz','ours vs \\texttt{dot}'),('ours','model','ours vs model coords'),
           ('graphviz','model','\\texttt{dot} vs model coords')]
    labs,cov=[],[]
    for a_,b_,nm in pairs:
        m=piv[a_].notna()&piv[b_].notna()
        labs.append(nm.replace('\\texttt{','').replace('}',''))
        cov.append(100*((piv[a_]!=piv[b_])&m).sum()/m.sum())
    yy=np.arange(len(labs))[::-1]
    c.barh(yy,cov,height=.5,color=[RED,GREY,GREY],lw=0)
    for i,v in enumerate(cov): c.annotate(f'{v:.0f}%',(v+2,yy[i]),va='center',fontsize=6.6)
    c.set_yticks(yy); c.set_yticklabels(labs,fontsize=6.6)
    c.set_xlabel('stimulus pairs differing on the verdict'); c.set_xlim(0,112)
    c.set_xticks([0,25,50,75,100]); c.set_xticklabels(['0','25','50','75','100%'])
    c.set_title('(c) contrast coverage: the engine\npair is 7 of 24, each a 1-defect flip',fontsize=7.4,pad=4)
    for a_ in ax: a_.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(wspace=.72)
    for ext in ('pdf','png'):
        fig.savefig(out/f'human_threshold.{ext}',dpi=300)
    plt.close(fig); print('wrote',out/'human_threshold.pdf')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='../data'); ap.add_argument('--out',default='../../img')
    a=ap.parse_args(); d=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    st=load(d); fig_threshold(st,out)
