import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D,OUT=sys.argv[1],sys.argv[2]
ju=pd.read_csv(f'{D}/judge_paperbanana.csv',low_memory=False); pi=pd.read_csv(f'{D}/pipeline_items.csv',low_memory=False); dot=pd.read_csv(f'{D}/dot_items.csv',low_memory=False)
main=ju[ju.judge=='InternVL3.5-14B']; pb=pi[pi.corpus=='paperbanana']
dpb=dot[dot.corpus=='paperbanana'].copy(); dpb['pred_nodes']=pd.to_numeric(dpb.pred_nodes,errors='coerce')
rows=[]
for m in ['Qwen','Gemma','Ministral']:
    for sysn,boxes in [('dot',dpb[(dpb.variant=='dot')&(dpb.model==m)].pred_nodes.mean()),
                       ('dot_matched',dpb[(dpb.variant=='dot_matched')&(dpb.model==m)].pred_nodes.mean()),
                       ('ours',pb[(pb.variant=='evaluated')&(pb.model==m)].layout_n_nodes.mean()),
                       ('ours_coverage',pb[(pb.variant=='coverage')&(pb.model==m)].layout_n_nodes.mean())]:
        g=main[(main.system==sysn)&(main.model==m)]
        rows.append(dict(model=m,system=sysn,boxes=boxes,overall=g.overall_points.mean()))
R=pd.DataFrame(rows)
plt.rcParams.update({'font.size':7,'font.family':'serif','axes.linewidth':0.6})
fig,ax=plt.subplots(figsize=(3.15,1.95))
style={'Qwen':('o','#35618f'),'Gemma':('s','#8f5a35'),'Ministral':('^','#3f7f4f')}
for m,g in R.groupby('model'):
    g=g.sort_values('boxes'); mk,c=style[m]
    ax.plot(g.boxes,g.overall,'-',color=c,lw=0.7,alpha=0.55,zorder=1)
    for r in g.itertuples():
        filled = r.system.startswith('ours')
        ax.scatter(r.boxes,r.overall,marker=mk,s=22,zorder=3,
                   facecolor=c if filled else 'white',edgecolor=c,linewidths=0.8,label=m if r.system=='ours' else None)
x=R.boxes.values; y=R.overall.values
b=np.polyfit(x,y,1); xs=np.linspace(x.min()-1,x.max()+1,10)
ax.plot(xs,np.polyval(b,xs),'--',color='0.35',lw=0.7,zorder=2)
ax.axhline(50,color='0.6',lw=0.5,ls=':')
ax.text(41.5,51.0,'parity',color='0.45',fontsize=5.5,ha='right')
ax.set_xlabel('boxes per figure'); ax.set_ylabel('judge, overall')
ax.set_xlim(10,47); ax.set_ylim(5,78)
ax.legend(frameon=False,fontsize=5.8,loc='upper right',handletextpad=0.3,borderpad=0.1)
ax.text(0.02,0.04,r'filled: typed graph   open: DOT'+'\n'+r'$\rho=-0.86$',transform=ax.transAxes,fontsize=5.8,color='0.3')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.tick_params(width=0.6,length=2.5,labelsize=6)
fig.tight_layout(pad=0.25); fig.savefig(OUT,format='pdf')
R.to_csv(OUT.replace('.pdf','.csv'),index=False); print("wrote",OUT)
