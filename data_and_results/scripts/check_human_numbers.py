#!/usr/bin/env python3
"""Every number \S6 and the abstract state about the rating study, recomputed
from data/human_*.csv and compared with the value the LaTeX source prints.
Exits non-zero if any check fails.
Usage: python3 check_human_numbers.py --data ../data_and_results/data --tex ..
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

SIX = ['label_collisions','edge_label_crossings','node_overlap_pairs',
       'edge_node_crossings','stacked_routes','route_reversals']

def load(d):
    st = pd.read_csv(d/'human_stimuli.csv')
    st.columns = [c.replace('report_','') for c in st.columns]
    ra = pd.read_csv(d/'human_ratings.csv'); se = pd.read_csv(d/'human_sets.csv')
    app = se[['image_1','image_2','image_3','image_4']].stack().value_counts()
    st['appear'] = st.image.map(app).fillna(0)
    st['bw'] = (st.image.map(ra.best.value_counts()).fillna(0)
                - st.image.map(ra.worst.value_counts()).fillna(0)) / st.appear
    st['defects'] = st[SIX].sum(axis=1)
    st['strict'] = (st[SIX] == 0).all(axis=1)
    return st, ra

def boot(v, seed=7, n=10000):
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path(__file__).resolve().parent.parent / 'data'))
    ap.add_argument('--tex', default=str(Path(__file__).resolve().parent.parent.parent))
    a = ap.parse_args()
    st, ra = load(Path(a.data))
    tex = ''.join((Path(a.tex)/f).read_text() for f in ('main.tex','appendix.tex'))
    checks, fails = [], 0

    def chk(name, got, want, tol=0.0, in_tex=None):
        nonlocal fails
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        if in_tex is not None and in_tex not in tex:
            ok = False; name += '  [string absent from LaTeX]'
        checks.append((name, got, want, ok))
        if not ok: fails += 1

    e48 = st[st.condition.isin(['ours','graphviz'])]
    m = st[st.condition == 'model']

    # floor
    cond = dict(zip(st.image, st.condition))
    worst = ra.worst.map(cond).value_counts(); best = ra.best.map(cond).value_counts()
    chk('model-coord share of least-legible picks (%)', round(100*worst['model']/len(ra),1), 79.2, 0.05, '79.2')
    chk('model-coord least-legible picks', int(worst['model']), 171, 0, '171 of the 216')
    chk('model-coord most-legible picks', int(best['model']), 11, 0)
    chk('model-coord median defects', float(m.defects.median()), 54.5, 0.001, '54.5')
    chk('model-coord max defects', int(m.defects.max()), 237, 0, '237')
    chk('model-coord strict passes', int(m.strict.sum()), 0, 0)

    # degeneracy among the 48
    zero = [c for c in SIX if set(e48[c].unique()) == {0}]
    binary = [c for c in SIX if sorted(set(e48[c].unique())) == [0,1]]
    chk('defect classes identically zero in the 48', len(zero), 4, 0, 'four defect classes are constant')
    chk('defect classes taking only 0/1 in the 48', len(binary), 2, 0, 'never exceed one')
    chk('label-collision images among the 48', int((e48.label_collisions>0).sum()), 1, 0, 'in only 1 and 8 images')
    chk('arrow-through-label images among the 48', int((e48.edge_label_crossings>0).sum()), 8, 0)

    # threshold bins
    b0 = st[st.defects == 0].bw; b1 = st[st.defects == 1].bw; bN = st[st.defects >= 15].bw
    chk('n, 0 defects', len(b0), 39, 0, '39 images with no defects')
    chk('n, 1 defect', len(b1), 9, 0, '9 with exactly one defect')
    chk('n, >=15 defects', len(bN), 24, 0, '24 with 15 or more')
    chk('n, 2-14 defects', int(((st.defects>=2)&(st.defects<=14)).sum()), 0, 0, 'no examples with 2 to 14 defects')
    chk('BW, 0 defects', round(float(b0.mean()),2), 0.72, 0.005, '+0.72')
    chk('BW, 1 defect', round(float(b1.mean()),2), 1.31, 0.005, '+1.31')
    chk('BW, >=15 defects', round(float(bN.mean()),2), -1.67, 0.005, '-1.67')
    lo, hi = boot(b0.values); chk('CI lo, 0 defects', round(lo,2), 0.49, 0.015, '[+0.49,+0.96]')
    chk('p, 0 vs 1 defect', round(float(stats.mannwhitneyu(b0,b1).pvalue),3), 0.073, 0.001, 'p=0.073')
    chk('p, 0 vs >=15 (exponent)', int(np.floor(np.log10(stats.mannwhitneyu(b0,bN).pvalue))), -10, 0, '4\\times10^{-10}')
    for lab, v, want in [('0',b0,14.6), ('1',b1,14.9), ('>=15',bN,14.7)]:
        sub = st.loc[v.index]
        chk(f'mean box count, {lab} defects', round(float(sub.n_nodes.mean()),1), want, 0.05)
    chk('box counts matched string', 1, 1, 0, 'matched on box count')

    # contrast coverage and resolution
    piv = st.pivot_table(index='paper_id', columns='condition', values='strict', aggfunc='first')
    diff = int((piv['ours'] != piv['graphviz']).sum())
    chk('engine pairs differing on the verdict', diff, 7, 0, '7 of 24')
    chk('contrast coverage (%)', round(100*diff/len(piv),1), 29.2, 0.05, '29\\%')
    # each covered pair is a single-defect flip
    dv = st.pivot_table(index='paper_id', columns='condition', values='defects', aggfunc='first')
    idx = piv.index[(piv['ours'] != piv['graphviz'])]
    maxd = int(max(dv.loc[idx, ['ours','graphviz']].max()))
    chk('max defects in a covered pair', maxd, 1, 0, 'differ by exactly one defect')
    pb = st.pivot_table(index='paper_id', columns='condition', values='bw', aggfunc='first')
    d = (pb['ours'] - pb['graphviz']).dropna()
    chk('paired BW difference', round(float(d.mean()),2), -0.27, 0.005, '-0.27')
    mde = (stats.norm.ppf(.975)+stats.norm.ppf(.80)) * d.std(ddof=1)/np.sqrt(len(d))
    chk('MDE at 80% power', round(float(mde),2), 0.58, 0.005, '0.58')
    gap = 100*(piv['ours'].mean() - piv['graphviz'].mean())
    chk('strict gap inside the panel (pp)', round(float(gap),1), 20.8, 0.05, '20.8')
    chk('ours strict passes in panel', int(piv['ours'].sum()), 22, 0, '22 of 24')
    chk('dot strict passes in panel', int(piv['graphviz'].sum()), 17, 0, '17 of 24')

    # saturation
    for col, want, s_ in [('n_nodes',-0.354,'figure size and aspect ratio'), ('area',-0.354,None), ('aspect_deviation',-0.311,'figure size and aspect ratio')]:
        r = stats.spearmanr(e48[col], e48.bw).statistic
        chk(f'rho(48), {col}', round(float(r),3), want, 0.002, s_)
    for col, want in [('label_collisions',-0.747), ('node_overlap_pairs',-0.733),
                      ('edge_node_crossings',-0.692), ('edge_edge_crossings',-0.255)]:
        r = stats.spearmanr(st[col], st.bw).statistic
        chk(f'rho(72), {col}', round(float(r),3), want, 0.002)

    # ---------------- standing (\S5) ----------------
    rb = pd.read_csv(Path(a.data)/'readback_sciflow.csv')
    om = rb[(rb.system=='ours') & (rb.model.str.contains('inistral', na=False))][['item','domain','score']] \
            .rename(columns={'score':'ours'})
    EXP = {'gemini3':(-0.0116,'behind'), 'gemini2.5':(+0.0064,'tied'),
           'seedream':(+0.0110,'ahead'), 'graphviz':(+0.0098,'ahead'),
           'qwen':(+0.0314,'ahead'), 'pixart':(+0.0945,'ahead'), 'sdxl':(+0.0908,'ahead')}
    ps, deltas, verds = [], {}, {}
    for k in EXP:
        o2 = rb[rb.system==k][['item','domain','score']].rename(columns={'score':'other'})
        mm = om.merge(o2, on=['item','domain']); dd = (mm.ours-mm.other).values
        deltas[k] = dd
        ps.append(stats.wilcoxon(mm.ours, mm.other).pvalue)
    order = list(EXP); adj = {}
    for rank, i in enumerate(np.argsort(ps)):
        adj[order[i]] = max(adj.get(order[order_i], 0) if False else 0,
                            min(1.0, (len(ps)-rank)*ps[i]))
    run = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps)-rank)*ps[i])); adj[order[i]] = run
    nah = nti = nbe = 0
    for k,(want,wv) in EXP.items():
        dd = deltas[k]; lo, hi = boot(dd)
        chk(f'standing delta vs {k}', round(float(dd.mean()),4), want, 0.0002)
        v = 'ahead' if (lo>0 and adj[k]<.05) else ('behind' if (hi<0 and adj[k]<.05) else 'tied')
        chk(f'standing verdict vs {k}', v, wv)
        nah += v=='ahead'; nti += v=='tied'; nbe += v=='behind'
    chk('systems we are ahead of', nah, 5, 0, 'ahead of five of the seven')
    chk('systems we are tied with', nti, 1, 0)
    chk('systems ahead of us', nbe, 1, 0, 'behind only Gemini~3 Pro Image')
    openw = ['qwen','pixart','sdxl']
    chk('open-weight systems beaten', sum(boot(deltas[k])[0] > 0 for k in openw), 3, 0)
    chk('gold-graph ceiling', round(float(rb[rb.system=='gold_graph_ceiling'].score.mean()),3), 0.611, 0.0005, '0.611')
    # size strata
    pi = pd.read_csv(Path(a.data)/'pipeline_items.csv', low_memory=False)
    gsz = pi[pi.corpus=='sciflow'][['item','gold_nodes']].dropna().groupby('item', as_index=False).first()
    med = gsz.gold_nodes.median()
    chk('median gold graph size', int(med), 28, 0, '28 boxes')
    big = set(gsz[gsz.gold_nodes > med].item)
    for k, want in [('seedream',0.0165), ('graphviz',0.0148), ('qwen',0.0427)]:
        o2 = rb[rb.system==k][['item','domain','score']].rename(columns={'score':'other'})
        mm = om.merge(o2, on=['item','domain']); mm = mm[mm.item.isin(big)]
        chk(f'large-graph delta vs {k}', round(float((mm.ours-mm.other).mean()),4), want, 0.0002)

    # ---------------- decomposition (\S5) ----------------
    COMP = ['node_precision','node_recall','edge_precision','edge_recall']
    od = rb[(rb.system=='ours') & (rb.model.str.contains('inistral', na=False))][['item','domain']+COMP]
    EXPD = {'qwen':   dict(node_precision=+0.0400, node_recall=+0.0709, sig=4),
            'seedream':dict(node_precision=+0.0258, node_recall=+0.0085, sig=2),
            'graphviz':dict(node_precision=-0.0041, node_recall=+0.0405, sig=2)}
    for k, want in EXPD.items():
        o2 = rb[rb.system==k][['item','domain']+COMP]
        mm = od.merge(o2, on=['item','domain'], suffixes=('_o','_b'))
        nsig = 0
        for c in COMP:
            dd = (mm[c+'_o']-mm[c+'_b']).values
            lo, hi = boot(dd)
            if lo > 0 or hi < 0: nsig += 1
            if c in want:
                chk(f'decomp {k} {c}', round(float(dd.mean()),4), want[c], 0.0002)
        chk(f'decomp {k}: components with CI excluding 0', nsig, want['sig'], 0)
    # the Graphviz margin is recall-only
    o2 = rb[rb.system=='graphviz'][['item','domain']+COMP]
    mm = od.merge(o2, on=['item','domain'], suffixes=('_o','_b'))
    precsig = any(boot((mm[c+'_o']-mm[c+'_b']).values)[0] > 0 for c in ['node_precision','edge_precision'])
    chk('Graphviz margin is recall-only (no precision win)', precsig, False, 0,
        'at level precision')

    w = max(len(c[0]) for c in checks)
    for name, got, want, ok in checks:
        print(f"{'ok ' if ok else 'FAIL'}  {name:<{w}}  got {got!s:>10}  expected {want!s:>10}")
    print(f"\n{len(checks)-fails}/{len(checks)} checks passed")
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
