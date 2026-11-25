#!/usr/bin/env python3
"""Instrument analyses (C1-C6). Computed from data_and_results/data/ only.
No GPU, no model call, no new run. Specified after the test runs; descriptive."""
import argparse, sys
import numpy as np, pandas as pd
from scipy import stats

def holm(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); n=len(ps); adj=np.empty(n); run=0.0
    for i,k in enumerate(o):
        run=max(run,(n-i)*ps[k]); adj[k]=min(1.0,run)
    return adj

def boot_mean_diff(a,b,B=10000,seed=0):
    d=np.asarray(a,float)-np.asarray(b,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),(B,len(d))); m=d[idx].mean(1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)

def rank_biserial(a,b):
    d=np.asarray(a,float)-np.asarray(b,float); d=d[d!=0]
    if len(d)==0: return 0.0
    r=stats.rankdata(np.abs(d)); return (r[d>0].sum()-r[d<0].sum())/r.sum()

def partial_spearman(x,y,z):
    s=pd.DataFrame({'x':x,'y':y,'z':z}).dropna()
    rx,ry,rz=(stats.rankdata(s[c]) for c in ('x','y','z'))
    B=np.c_[np.ones(len(s)),rz]
    res=lambda v: v-B@np.linalg.lstsq(B,v,rcond=None)[0]
    r,p=stats.pearsonr(res(rx),res(ry)); return r,p,len(s)

SZ_BINS=[0,9,14,19,29,49,10**9]
SZ_LAB=['2-9','10-14','15-19','20-29','30-49','50+']
BEST='dot_tb_spline'

def load(d):
    g=lambda f: pd.read_csv(f"{d}/{f}",low_memory=False)
    return (g('pipeline_items.csv'),g('dot_items.csv'),g('layout_items.csv'),
            g('judge_paperbanana.csv'),g('readback_sciflow.csv'))

def paired_layout(L):
    """One row per (corpus, model, run, item): ours and each dot config, + graph properties."""
    G=L[L.source.isin(['graphviz','graphviz_outside_development'])]
    key=['corpus','model','run','item']
    p=G.pivot_table(index=key,columns='config',values=['valid','edge_edge_crossings','label_collisions','total_edge_length'])
    meta=G[G.config=='ours'].set_index(key)[['n_nodes','arrows','loops','nested']]
    out=pd.DataFrame(index=p.index)
    for v in ['valid','edge_edge_crossings','label_collisions','total_edge_length']:
        out[('ours',v)]=p[(v,'ours')]; out[('dot',v)]=p[(v,BEST)]
    out=out.join(meta)
    out.columns=['_'.join(c) if isinstance(c,tuple) else c for c in out.columns]
    out=out.dropna(subset=['ours_valid','dot_valid'])
    for c in out.columns: out[c]=pd.to_numeric(out[c],errors='coerce').astype(float)
    return out

# ---------------------------------------------------------------- C1
def c1(P,J):
    print("="*78); print("C1  Does the PaperBananaBench judge see legibility?"); print("="*78)
    pb=P[(P.variant=='evaluated')&(P.corpus=='paperbanana')].copy(); pb['seed']=pb.seed.astype(int)
    j=(J[(J.judge=='InternVL3.5-14B')&(J.system=='ours')]
       .groupby(['model','seed','item'],as_index=False)
       [['overall_points','readability_points','conciseness_points','faithfulness_points','aesthetics_points']].mean())
    j['seed']=j.seed.astype(int)
    m=pb.merge(j,on=['model','seed','item'])
    for c in m.columns:
        if c.startswith(('layout_','overall','readab','concis','faith','aesth')): m[c]=pd.to_numeric(m[c],errors='coerce')
    geo={'layout_valid':'strict legible','layout_edge_edge_crossings':'arrow crossings',
         'layout_label_collisions':'label collisions','layout_mean_bends_per_edge':'bends per arrow',
         'layout_total_edge_length':'total arrow length','layout_area':'area'}
    dims=['readability_points','overall_points','conciseness_points']
    print(f"\n(a) raw Spearman, n={len(m)} figures (3 models x 3 seeds x 292 cases)")
    print(f"  {'geometric measure':22s}"+"".join(f"{d.split('_')[0][:12]:>14s}" for d in dims))
    raw={}
    for g,nm in geo.items():
        row=f"  {nm:22s}"
        for d in dims:
            s=m[[g,d]].dropna()
            r=stats.spearmanr(s[g],s[d]); raw[(g,d)]=r
            row+=f"{r.statistic:>+12.3f}{'*' if r.pvalue<0.01 else ' '} "
        print(row)
    print("\n(b) the same, with figure size (number of boxes) partialled out")
    print(f"  {'geometric measure':22s}"+"".join(f"{d.split('_')[0][:12]:>14s}" for d in dims))
    res={}
    for g,nm in geo.items():
        row=f"  {nm:22s}"
        for d in dims:
            r,p,n=partial_spearman(m[g],m[d],m['layout_n_nodes']); res[(g,d)]=(r,p,n)
            row+=f"{r:>+12.3f}{'*' if p<0.01 else ' '} "
        print(row)
    sn=m[['layout_n_nodes','readability_points','overall_points']].dropna()
    print("  (* p < 0.01; 'size' itself: rho(boxes, readability) = %+.3f, rho(boxes, overall) = %+.3f, n=%d)"
          %(stats.spearmanr(sn.layout_n_nodes,sn.readability_points).statistic,
            stats.spearmanr(sn.layout_n_nodes,sn.overall_points).statistic,len(sn)))
    print("\n(c) judge readability of legible vs illegible figures of the same system")
    rows=[]
    for mod,gg in m.groupby('model'):
        s=gg[['layout_valid','readability_points','overall_points']].dropna()
        a=s[s.layout_valid==1]; b=s[s.layout_valid==0]
        u=stats.mannwhitneyu(a.readability_points,b.readability_points) if len(b)>5 else None
        rows.append((mod,len(a),a.readability_points.mean(),len(b),b.readability_points.mean(),
                     u.pvalue if u else np.nan))
        print(f"  {mod:10s} legible n={len(a):4d} readability={a.readability_points.mean():5.1f} | "
              f"illegible n={len(b):4d} readability={b.readability_points.mean():5.1f} | "
              f"diff={a.readability_points.mean()-b.readability_points.mean():+5.1f} p={rows[-1][5]:.3g}")
    ps=[r[5] for r in rows if not np.isnan(r[5])]
    print(f"  Holm-corrected p: {np.round(holm(ps),4).tolist()}")
    return m,res,rows

# ---------------------------------------------------------------- C2
def c2(L):
    print("\n"+"="*78); print("C2  Layout gains outside the corpus the layout was tuned on"); print("="*78)
    p=paired_layout(L)
    print("\n  route costs and layout options were set on SciFlow-Bench development figures;")
    print("  FlowVQA and PaperBananaBench were never used for any layout choice.\n")
    print(f"  {'corpus':13s}{'model':11s}{'n':>6s}{'ours':>8s}{'dot':>8s}{'gap':>8s}{'95% CI':>20s}{'r':>7s}{'p(Holm)':>10s}")
    rows=[]
    for (cp,mod),g in p.groupby(['corpus','model']):
        a=g.ours_valid.values; b=g.dot_valid.values
        d,lo,hi=boot_mean_diff(a,b); w=stats.wilcoxon(a,b); r=rank_biserial(a,b)
        rows.append([cp,mod,len(g),a.mean(),b.mean(),d,lo,hi,r,w.pvalue])
    adj=holm([r[-1] for r in rows])
    for r,q in zip(rows,adj):
        r[-1]=q
        print(f"  {r[0]:13s}{r[1]:11s}{r[2]:6d}{r[3]*100:7.1f}%{r[4]*100:7.1f}%{r[5]*100:+7.1f}"
              f"   [{r[6]*100:+5.1f},{r[7]*100:+5.1f}]{r[8]:>7.2f}{q:>10.1e}")
    dev=[r for r in rows if r[0]=='sciflow']; out=[r for r in rows if r[0]!='sciflow']
    print(f"\n  development corpus gap: {min(x[5] for x in dev)*100:+.1f} to {max(x[5] for x in dev)*100:+.1f} points")
    print(f"  held-out corpora gap  : {min(x[5] for x in out)*100:+.1f} to {max(x[5] for x in out)*100:+.1f} points")
    print(f"  figures never reported: {sum(x[2] for x in out)} ({len(out)} corpus-model settings)")
    return rows

# ---------------------------------------------------------------- C3
def c3(L):
    print("\n"+"="*78); print("C3  How legibility scales with the size of the graph"); print("="*78)
    p=paired_layout(L); p['sz']=pd.cut(p.n_nodes,bins=SZ_BINS,labels=SZ_LAB)
    print(f"\n  {'boxes':>8s}{'n':>7s}{'ours':>9s}{'dot':>9s}{'gap':>8s}{'ours x':>9s}{'dot x':>8s}")
    rows=[]
    for sz,g in p.groupby('sz',observed=True):
        rows.append((str(sz),len(g),g.ours_valid.mean(),g.dot_valid.mean(),
                     g.ours_edge_edge_crossings.mean(),g.dot_edge_edge_crossings.mean()))
        print(f"  {sz:>8s}{len(g):7d}{g.ours_valid.mean()*100:8.1f}%{g.dot_valid.mean()*100:8.1f}%"
              f"{(g.ours_valid.mean()-g.dot_valid.mean())*100:+7.1f}"
              f"{g.ours_edge_edge_crossings.mean():9.2f}{g.dot_edge_edge_crossings.mean():8.2f}")
    print("\n  per figure (n=%d):"%len(p))
    o1=((p.ours_valid==1)&(p.dot_valid==0)).mean(); o2=((p.ours_valid==0)&(p.dot_valid==1)).mean()
    both=((p.ours_valid==1)&(p.dot_valid==1)).mean(); none=((p.ours_valid==0)&(p.dot_valid==0)).mean()
    tbl=np.array([[((p.ours_valid==1)&(p.dot_valid==1)).sum(),((p.ours_valid==1)&(p.dot_valid==0)).sum()],
                  [((p.ours_valid==0)&(p.dot_valid==1)).sum(),((p.ours_valid==0)&(p.dot_valid==0)).sum()]])
    b,c=tbl[0,1],tbl[1,0]; chi2=(abs(b-c)-1)**2/(b+c); pmc=stats.chi2.sf(chi2,1)
    print(f"    both legible {both*100:.1f}%   ours only {o1*100:.1f}%   dot only {o2*100:.1f}%   neither {none*100:.1f}%")
    print(f"    ours-only : dot-only = {b}:{c} = {b/max(c,1):.1f}:1   (McNemar chi2={chi2:.1f}, p={pmc:.1e})")
    for nm,col in [('ours','ours_valid'),('dot','dot_valid')]:
        r=stats.spearmanr(p.n_nodes,p[col]); print(f"    rho(legible, boxes) {nm:5s} = {r.statistic:+.3f} (p={r.pvalue:.1e})")
    print("\n  size at which half the figures are still legible (interpolated from the table):")
    for nm,col in [('ours','ours_valid'),('dot','dot_valid')]:
        t=p.groupby('sz',observed=True)[col].mean()
        print(f"    {nm:5s}: {' '.join(f'{k}={v*100:.0f}%' for k,v in t.items())}")
    return rows,(b,c,chi2,pmc)

# ---------------------------------------------------------------- C4
def c4(P,RB):
    print("\n"+"="*78); print("C4  Does the read-back reader see legibility?"); print("="*78)
    sf=P[(P.variant=='evaluated')&(P.corpus=='sciflow')&(P.seed==0)].copy()
    rb=(RB[RB.system=='ours'][['model','item','score','node_f1','edge_f1','pred_nodes']]
        .rename(columns={'score':'rb_score','node_f1':'rb_node','edge_f1':'rb_edge','pred_nodes':'rb_nodes'}))
    m=sf.merge(rb,on=['model','item'])
    for c in ['layout_valid','layout_n_nodes','layout_edge_edge_crossings','layout_label_collisions','score','rb_score','rb_nodes']:
        m[c]=pd.to_numeric(m[c],errors='coerce')
    print(f"\n  n={len(m)} figures (3 models x 483 papers, seed 0), one reader, one prompt")
    print(f"\n  {'measure':24s}{'raw rho':>10s}{'rho | boxes':>14s}{'p':>10s}")
    for g,nm in [('layout_valid','strict legible'),('layout_edge_edge_crossings','arrow crossings'),
                 ('layout_label_collisions','label collisions')]:
        ss=m[[g,'rb_score']].dropna().astype(float)
        raw=stats.spearmanr(ss[g].values,ss['rb_score'].values)
        r,p,n=partial_spearman(m[g],m['rb_score'],m['layout_n_nodes'])
        print(f"  {nm:24s}{raw.statistic:>+10.3f}{r:>+14.3f}{p:>10.2f}")
    print("\n  read-back score of legible vs illegible figures:")
    rows=[]
    for mod,g in m.groupby('model'):
        a=g[g.layout_valid==1]; b=g[g.layout_valid==0]
        u=stats.mannwhitneyu(a.rb_score,b.rb_score) if len(b)>5 else None
        rows.append((mod,len(a),a.rb_score.mean(),len(b),b.rb_score.mean(),u.pvalue if u else np.nan))
        print(f"    {mod:10s} legible n={len(a):3d} {a.rb_score.mean():.4f} | illegible n={len(b):3d} {b.rb_score.mean():.4f} | p={rows[-1][5]:.2f}")
    print("\n  what the reader returns: boxes read back / boxes drawn")
    for mod,g in m.groupby('model'):
        g=g.assign(rec=g.rb_nodes/g.layout_n_nodes)
        a=g[g.layout_valid==1].rec.dropna(); b=g[g.layout_valid==0].rec.dropna()
        u=stats.mannwhitneyu(a,b) if len(b)>5 else None
        print(f"    {mod:10s} all {g.rec.mean():.2f}x | legible {a.mean():.2f}x | illegible {b.mean():.2f}x | p={(u.pvalue if u else np.nan):.3f}")
    print("\n  score kept through drawing and re-reading (rb_score / own-graph score, items with score>0):")
    m['ret']=np.where(m.score>0,m.rb_score/m.score,np.nan)
    print("   "+"  ".join(f"{mod}={g.ret.mean():.2f}" for mod,g in m.groupby('model')))
    return m,rows

# ---------------------------------------------------------------- C5
def c5(P):
    print("\n"+"="*78); print("C5  Structural score and legibility, per figure"); print("="*78)
    e=P[(P.variant=='evaluated')&P.corpus.isin(['sciflow','flowvqa'])].copy()
    for c in ['score','layout_valid','layout_edge_edge_crossings','layout_n_nodes']: e[c]=pd.to_numeric(e[c],errors='coerce')
    print(f"\n  {'corpus':9s}{'model':11s}{'n':>6s}{'rho(score,legible)':>20s}{'p':>9s}{'| boxes':>10s}{'rho(score,cross)':>18s}")
    rows=[]
    for (cp,mod),g in e.groupby(['corpus','model']):
        s=g[['score','layout_valid']].dropna(); r=stats.spearmanr(s.score,s.layout_valid)
        pr,pp,_=partial_spearman(g.score,g.layout_valid,g.layout_n_nodes)
        sc=g[['score','layout_edge_edge_crossings']].dropna().astype(float)
        rc=stats.spearmanr(sc['score'].values,sc['layout_edge_edge_crossings'].values)
        rows.append((cp,mod,len(s),r.statistic,r.pvalue,pr,rc.statistic))
        print(f"  {cp:9s}{mod:11s}{len(s):6d}{r.statistic:>+20.3f}{r.pvalue:>9.3f}{pr:>+10.3f}{rc.statistic:>+18.3f}")
    a=np.abs([r[3] for r in rows]); print(f"\n  |rho| over {sum(r[2] for r in rows)} figures: max {a.max():.3f}, median {np.median(a):.3f}")
    print("\n  pooled over settings (ranks taken within corpus x model, then weighted by n):")
    def pooled(x,y,ctrl=None):
        parts=[]
        for _,g in e.groupby(['corpus','model']):
            s=g[[x,y]+([ctrl] if ctrl else [])].dropna()
            if len(s)<20: continue
            rx,ry=stats.rankdata(s[x]),stats.rankdata(s[y])
            if ctrl:
                rz=stats.rankdata(s[ctrl]); B=np.c_[np.ones(len(s)),rz]
                f=lambda v: v-B@np.linalg.lstsq(B,v,rcond=None)[0]; rx,ry=f(rx),f(ry)
            parts.append((stats.pearsonr(rx,ry)[0],len(s)))
        w=np.array([x2[1] for x2 in parts]); v=np.array([x2[0] for x2 in parts])
        return (v*w).sum()/w.sum(),v.min(),v.max(),int(w.sum())
    for c,nm in [('layout_valid','strict legible'),('layout_edge_edge_crossings','arrow crossings'),
                 ('layout_label_collisions','label collisions')]:
        e[c]=pd.to_numeric(e[c],errors='coerce')
        a1=pooled(c,'score'); b1=pooled(c,'score','layout_n_nodes')
        print(f"    {nm:18s} rho={a1[0]:+.3f} [{a1[1]:+.3f},{a1[2]:+.3f}]  rho|size={b1[0]:+.3f} [{b1[1]:+.3f},{b1[2]:+.3f}]  n={a1[3]}")
    return rows

# ---------------------------------------------------------------- C6
def c6(P,DT):
    print("\n"+"="*78); print("C6  Is the SciFlow-Bench gain carried by one area?"); print("="*78)
    e=P[P.variant=='evaluated']
    o=e[e.corpus=='sciflow'].groupby(['model','item','domain'],as_index=False).score.mean()
    name={'cv_2025_autobench':'CV','nlp_2025_autobench':'NLP','ml_2025_autobench':'ML',
          'ic_2025_autobench':'IC','robotics_2025_autobench':'Robotics'}
    rows=[]
    for var,lab in [('dot','short prompt'),('dot_matched','our prompts')]:
        d=(DT[(DT.variant==var)&(DT.corpus=='sciflow')].groupby(['model','item','domain'],as_index=False)
           .score.mean().rename(columns={'score':'base'}))
        m=o.merge(d,on=['model','item','domain'],how='left'); m['base']=m.base.fillna(0)
        print(f"\n  baseline: same model writing DOT, {lab}")
        print(f"  {'model':11s}{'area':10s}{'n':>5s}{'ours':>8s}{'DOT':>8s}{'delta':>9s}{'p(Holm)':>10s}")
        sub=[]
        for (mod,dom),g in m.groupby(['model','domain']):
            w=stats.wilcoxon(g.score.values,g.base.values)
            sub.append([mod,name[dom],len(g),g.score.mean(),g.base.mean(),g.score.mean()-g.base.mean(),w.pvalue])
        for mod in ['Qwen','Gemma','Ministral']:
            part=[r for r in sub if r[0]==mod]; adj=holm([r[6] for r in part])
            for r,q in zip(part,adj):
                r[6]=q; print(f"  {r[0]:11s}{r[1]:10s}{r[2]:5d}{r[3]:8.4f}{r[4]:8.4f}{r[5]:+9.4f}{q:>10.4f}")
        pos=sum(1 for r in sub if r[5]>0); sig=sum(1 for r in sub if r[5]>0 and r[6]<0.05)
        neg=sum(1 for r in sub if r[5]<0 and r[6]<0.05)
        print(f"  --> {pos}/15 areas favour ours, {sig} significantly; {neg} significantly against ({lab})")
        rows.append((lab,pos,sig,neg,sub))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='../data')
    ap.add_argument('--analysis',nargs='*',default=['C1','C2','C3','C4','C5','C6'])
    ap.add_argument('--latex')
    ap.add_argument('--out')
    a=ap.parse_args()
    if a.out:
        import builtins, io as _io
        buf=_io.StringIO(); _print=builtins.print
        def _tee(*args,**kw):
            kw.pop('file',None); _print(*args,**kw); _print(*args,file=buf,**kw)
        builtins.print=_tee
    P,DT,L,J,RB=load(a.data)
    out={}
    if 'C1' in a.analysis: out['C1']=c1(P,J)
    if 'C2' in a.analysis: out['C2']=c2(L)
    if 'C3' in a.analysis: out['C3']=c3(L)
    if 'C4' in a.analysis: out['C4']=c4(P,RB)
    if 'C5' in a.analysis: out['C5']=c5(P)
    if 'C6' in a.analysis: out['C6']=c6(P,DT)
    print("\nAll numbers above come from data/*.csv only: no GPU, no model call, no new run.")
    if a.latex: write_latex(a.latex,out)
    if a.out:
        import builtins
        open(a.out,'w').write(buf.getvalue())
        builtins.print=_print

def write_latex(path,out):
    import io
    f=io.StringIO()
    if 'C2' in out and 'C3' in out:
        rows=out['C2']
        f.write("% Table: generalisation outside the development corpus\n")
        f.write("\\begin{table}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\begin{tabular}{l l r r r l}\n\\toprule\n")
        f.write("\\textbf{Corpus} & \\textbf{Model} & \\textbf{Ours} & \\texttt{\\textbf{dot}} & \\textbf{$\\Delta$ [95\\% CI]} & \\textbf{$r$} \\\\\n\\midrule\n")
        order={'sciflow':'SciFlow-Bench$^\\dagger$','flowvqa':'FlowVQA','paperbanana':'PaperBanana'}
        for cp in ['sciflow','flowvqa','paperbanana']:
            sel=[r for r in rows if r[0]==cp]
            for i,r in enumerate(sel):
                lead=f"\\multirow{{3}}{{*}}{{{order[cp]}}}" if i==0 else ""
                f.write(f"{lead} & {r[1]} & \\textbf{{{r[3]*100:.1f}\\%}} & {r[4]*100:.1f}\\% & "
                        f"${r[5]*100:+.1f}$ [${r[6]*100:+.1f}$, ${r[7]*100:+.1f}$] & {r[8]:.2f} \\\\\n")
            f.write("\\midrule\n" if cp!='paperbanana' else "")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Strictly legible figures, our engine against Graphviz \\texttt{dot} (top-down, curved edges) "
                "on identical graphs. $^\\dagger$the only corpus whose figures were seen while the route costs and layout "
                "options were set; FlowVQA and PaperBananaBench were used for no layout choice. $\\Delta$: paired mean "
                "difference in percentage points with its bootstrap interval; $r$: matched-pairs rank-biserial correlation. "
                "Every row has Holm-corrected $p<10^{-30}$.}\n\\label{table:outsidedev}\n\\end{table}\n\n")
        szrows,mc=out['C3']
        f.write("% Table: legibility against graph size\n")
        f.write("\\begin{table}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{l r r r r r r}\n\\toprule\n")
        f.write(" & & \\multicolumn{3}{c}{\\textbf{Strictly legible}} & \\multicolumn{2}{c}{\\textbf{Crossings}} \\\\\n")
        f.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n")
        f.write("\\textbf{Boxes} & \\textbf{Figures} & \\textbf{Ours} & \\texttt{\\textbf{dot}} & \\textbf{$\\Delta$} & \\textbf{Ours} & \\texttt{\\textbf{dot}} \\\\\n\\midrule\n")
        for sz,n,o,d,co,cd in szrows:
            f.write(f"{sz} & {n} & \\textbf{{{o*100:.1f}}} & {d*100:.1f} & ${ (o-d)*100:+.1f}$ & {co:.2f} & \\textbf{{{cd:.2f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Legibility against the size of the extracted graph, over the 6{,}009 figures of all three corpora "
                "laid out by both engines (three models, three runs). Both engines lose legibility as graphs grow; the "
                "distance between them widens from 15.5 to 40.7 points. Per figure, ours is strictly legible and "
                f"\\texttt{{dot}} is not {mc[0]} times against {mc[1]} the other way (McNemar $\\chi^2={mc[2]:.0f}$, $p<10^{{-300}}$).}}\n")
        f.write("\\label{table:sizescaling}\n\\end{table}\n")
    open(path,'w').write(f.getvalue()); print(f"\nwrote {path}")

if __name__=='__main__': main()
