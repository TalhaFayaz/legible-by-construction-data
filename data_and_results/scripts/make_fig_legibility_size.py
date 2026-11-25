#!/usr/bin/env python3
"""Legibility against graph size, our engine against Graphviz dot, from
data/layout_items.csv only. Produces img/legibility_size.pdf and .png and
the CSV behind the figure. No GPU, no model call, no new run.

    python3 make_fig_legibility_size.py --data ../data --out ../../img
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS=[0,9,14,19,29,49,10**9]
LAB=['2--9','10--14','15--19','20--29','30--49','50+']
BEST='dot_tb_spline'

def paired(L):
    G=L[L.source.isin(['graphviz','graphviz_outside_development'])]
    key=['corpus','model','run','item']
    p=G.pivot_table(index=key,columns='config',values='valid')
    meta=G[G.config=='ours'].set_index(key)[['n_nodes']]
    out=pd.DataFrame({'ours':p['ours'],'dot':p[BEST]}).join(meta).dropna()
    return out.astype(float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='../data'); ap.add_argument('--out',default='../../img')
    a=ap.parse_args()
    d=paired(pd.read_csv(f"{a.data}/layout_items.csv",low_memory=False))
    d['bin']=pd.cut(d.n_nodes,bins=BINS,labels=LAB)
    g=d.groupby('bin',observed=True).agg(n=('ours','size'),ours=('ours','mean'),dot=('dot','mean'))
    # Wilson intervals
    def wilson(k,n,z=1.96):
        ph=k/n; den=1+z*z/n
        c=(ph+z*z/(2*n))/den; h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        return c-h,c+h
    lo_o,hi_o=zip(*[wilson(r.ours*r.n,r.n) for r in g.itertuples()])
    lo_d,hi_d=zip(*[wilson(r.dot*r.n,r.n) for r in g.itertuples()])
    x=np.arange(len(g))
    fig,ax=plt.subplots(figsize=(3.3,2.25),dpi=200)
    for y,lo,hi,c,m,lbl in [(g.ours*100,np.array(lo_o)*100,np.array(hi_o)*100,'#1f4e79','o','Our engine'),
                            (g['dot']*100,np.array(lo_d)*100,np.array(hi_d)*100,'#9a9a9a','s',r'Graphviz $\tt dot$')]:
        ax.fill_between(x,lo,hi,color=c,alpha=0.15,linewidth=0)
        ax.plot(x,y,marker=m,color=c,linewidth=1.6,markersize=4,label=lbl)
    for i in x:
        ax.annotate('',xy=(i,g.ours.iloc[i]*100),xytext=(i,g['dot'].iloc[i]*100),
                    arrowprops=dict(arrowstyle='-',color='#cccccc',linewidth=0.6,shrinkA=3,shrinkB=3),zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([l.replace('--','–') for l in LAB],fontsize=7)
    ax.set_xlabel('Boxes in the extracted graph',fontsize=7.5)
    ax.set_ylabel('Strictly legible (%)',fontsize=7.5)
    ax.tick_params(labelsize=7); ax.set_ylim(0,104)
    ax.grid(axis='y',color='#e8e8e8',linewidth=0.6); ax.set_axisbelow(True)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7,frameon=False,loc='lower left')
    gap=(g.ours-g['dot'])*100
    ax.annotate(f'gap {gap.iloc[0]:+.0f} pt',(0,(g.ours.iloc[0]+g['dot'].iloc[0])/2*100),
                fontsize=6.5,color='#555555',ha='center',va='center',
                bbox=dict(fc='white',ec='none',pad=0.8))
    ax.annotate(f'{gap.iloc[4]:+.0f} pt',(4,(g.ours.iloc[4]+g['dot'].iloc[4])/2*100),
                fontsize=6.5,color='#555555',ha='center',va='center',
                bbox=dict(fc='white',ec='none',pad=0.8))
    fig.tight_layout(pad=0.3)
    os.makedirs(a.out,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(a.out,f'legibility_size.{ext}'),bbox_inches='tight')
    out=g.assign(gap=gap.round(1),ours=(g.ours*100).round(1),dot=(g['dot']*100).round(1))
    out.to_csv(os.path.join(a.out,'legibility_size.csv'))
    print(out.to_string()); print("wrote legibility_size.{pdf,png,csv} to",a.out)

if __name__=='__main__': main()
