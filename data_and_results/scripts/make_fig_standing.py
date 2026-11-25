#!/usr/bin/env python3
"""Figure 1: where the pipeline stands, paired against every system the
benchmark distributes, overall and split at the median gold graph size."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

INK='#1a1a1a'; GREY='#8a8a8a'; BLUE='#35618f'; RED='#a8322d'; GREEN='#2f6b4f'; LGREY='#e2e2e2'
plt.rcParams.update({'font.size':7.4,'font.family':'serif','axes.linewidth':.6,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.4,
    'ytick.major.size':2.4,'axes.edgecolor':INK,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.bbox':'tight'})

def clean(s):
    return (s.replace('$\\rightarrow$','→').replace('PixArt-$\\Sigma$','PixArt-Σ'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tables',default='../results/tables')
    ap.add_argument('--out',default='../../img'); a=ap.parse_args()
    T=Path(a.tables); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(T/'standing_readback.csv'); bs=pd.read_csv(T/'standing_by_size.csv')
    t['lab']=t.system.map(clean); bs['lab']=bs.system.map(clean)
    t=t.iloc[::-1].reset_index(drop=True)     # worst rival at the bottom

    fig,ax=plt.subplots(1,2,figsize=(7.1,2.55),gridspec_kw={'width_ratios':[1.32,1.0]})
    # ---- (a) overall forest
    A=ax[0]; y=np.arange(len(t))
    A.axvspan(-0.025,0,color=LGREY,alpha=.65,lw=0)
    for i,r in t.iterrows():
        c=GREEN if r.verdict=='ahead' else (RED if r.verdict=='behind' else GREY)
        A.plot([r.ci_lo,r.ci_hi],[i,i],color=c,lw=1.5,solid_capstyle='round',zorder=2)
        A.plot([r.delta],[i],'o',ms=4.6,color=c,zorder=3)
        A.annotate(f'{r.delta:+.3f}',(r.ci_hi+.0035,i),va='center',fontsize=6.2,color=c)
    A.axvline(0,color=INK,lw=.8,zorder=1)
    A.set_yticks(y); A.set_yticklabels([f'{l}' for l in t.lab],fontsize=6.8)
    for i,r in t.iterrows():
        if r.weights=='open':
            A.get_yticklabels()[i].set_color(BLUE); A.get_yticklabels()[i].set_fontweight('bold')
    A.set_xlabel('paired difference in read-back score (ours + Ministral $-$ system)')
    A.set_xlim(-0.035,0.135); A.set_ylim(-.7,len(t)-.3)
    A.annotate('we are behind',(-0.0125,len(t)-.45),ha='center',fontsize=6.0,color=RED,style='italic')
    A.annotate('we are ahead',(0.055,len(t)-.45),ha='center',fontsize=6.0,color=GREEN,style='italic')
    A.set_title('(a) ahead of five, tied with one, behind one',fontsize=7.6,pad=5)
    A.annotate('bold: open weights',(0.133,-0.52),ha='right',fontsize=5.9,color=BLUE,style='italic')
    A.spines[['top','right','left']].set_visible(False); A.tick_params(axis='y',length=0)
    # ---- (b) by size
    B=ax[1]
    order=[l for l in t.lab][::-1]
    sm=bs[bs.stratum.str.startswith('small')].set_index('lab').reindex(order)
    lg=bs[bs.stratum.str.startswith('large')].set_index('lab').reindex(order)
    yy=np.arange(len(order))[::-1]
    B.barh(yy+.20,sm.delta,height=.38,color=GREY,lw=0,label='small graphs')
    B.barh(yy-.20,lg.delta,height=.38,color=BLUE,lw=0,label='large graphs')
    B.axvline(0,color=INK,lw=.8)
    B.set_yticks(yy); B.set_yticklabels(order,fontsize=6.4)
    B.set_xlabel('paired difference, by gold graph size')
    B.set_title('(b) the margin widens on large graphs',fontsize=7.6,pad=5)
    B.legend(frameon=False,fontsize=6.2,loc='upper center',bbox_to_anchor=(.62,-.22),
             ncol=2,handlelength=1.0,columnspacing=1.2,borderpad=0)
    B.set_xlim(-0.03,0.135)
    B.spines[['top','right','left']].set_visible(False); B.tick_params(axis='y',length=0)
    fig.subplots_adjust(wspace=.62)
    for e in ('pdf','png'): fig.savefig(out/f'standing.{e}',dpi=300)
    print('wrote',out/'standing.pdf')

if __name__=='__main__': main()
