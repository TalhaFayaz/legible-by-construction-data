#!/usr/bin/env python3
"""The graph-drawing criteria (B1-B8), Sections 5.1, 5.2, 5.3 and Appendix S2.

Every number comes from the per-item measurements in data_and_results/data/.
No GPU, no model call, no new run.

    python3 checks_criteria.py [--data ../data] [--analysis B1] [--out report.txt]

B1  crossing angle and angle-weighted crossings          Tables 4 and 9
B2  the graph-drawing criteria behind Table S3's text    Section 5.2
B3  the split by feedback loops and nested containers    Section 5.2
B4  FlowVQA by source: where the pipeline wins           Section 5.4, Table 1
B5  precision and recall against DOT                     the content detail in README.md
B6  PaperBananaBench: judge score against figure size    Section 6, judge panels
B7  what the raters responded to                         Section 5.3, Table S5
B8  why the DiagramEval-style columns are not reported   (negative result)
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

RNG = np.random.default_rng(0)
MODELS = ['Qwen', 'Gemma', 'Ministral']
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps)
    out = np.empty_like(ps); m = len(ps); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[idx])); out[idx] = run
    return out


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    bs = np.array([RNG.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    d, lo, hi = boot_ci(x - y)
    return dict(n=len(x), mx=x.mean(), my=y.mean(), d=d, lo=lo, hi=hi, p=p)


def load(D):
    li = pd.read_csv(f'{D}/layout_items.csv', low_memory=False)
    li['valid_b'] = li.valid.astype(str) == 'True'
    li['loops_b'] = li.loops.astype(str) == 'True'
    li['nested_b'] = li.nested.astype(str) == 'True'
    li['ang'] = li.crossing_angle.fillna(90.0)
    li['wcross'] = li.edge_edge_crossings * (1.0 - li.ang / 90.0)
    pi = pd.read_csv(f'{D}/pipeline_items.csv', low_memory=False)
    dot = pd.read_csv(f'{D}/dot_items.csv', low_memory=False)
    ju = pd.read_csv(f'{D}/judge_paperbanana.csv', low_memory=False)
    return li, pi, dot, ju


def B1(d):
    global RNG
    RNG = np.random.default_rng(0)
    li = d['li']; G = li[li.source.isin(['graphviz', 'graphviz_outside_development'])]
    head("B1  CROSSING ANGLE AND ANGLE-WEIGHTED CROSSINGS  (Tables 4 and 9)")
    say("w = crossings x (1 - angle/90): a right-angle crossing costs a reader")
    say("almost nothing, a shallow one nearly its full weight.")
    say()
    say("The engine comparison (SciFlow-Bench, 472 graphs x 3 runs), ours vs dot top-down curved:")
    rows = []
    S = li[li.source == 'graphviz']
    for m in MODELS:
        for cfg in ['ours', 'dot_tb_spline']:
            g = S[(S.model == m) & (S.config == cfg)]
            rows.append(dict(model=m, layout=cfg, cross=g.edge_edge_crossings.mean(),
                             angle=g.crossing_angle.mean(), wcross=g.wcross.mean(),
                             bends=g.mean_bends_per_edge.mean(), through=g.edge_node_crossings.mean(),
                             labels=g.label_collisions.mean(), strict=100 * g.valid_b.mean(),
                             area=g.area.median()))
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say()
    say("Crossing angles (the same 472 SciFlow-Bench graphs x 3 runs; pct_shallow_all is")
    say("over all 6,012 figures including the out-of-development ones):")
    rows = []
    for cfg in ['ours', 'dot_tb_spline', 'dot_lr_spline', 'dot_tb_ortho', 'dot_lr_ortho']:
        g = G[G.config == cfg]; s = S[S.config == cfg]
        rows.append(dict(config=cfg, n=len(s), angle=s.crossing_angle.mean(),
                         pct_figs_shallow=100 * ((s.crossing_angle < 60) & s.crossing_angle.notna()).mean(),
                         pct_shallow_all=100 * ((g.crossing_angle < 60) & g.crossing_angle.notna()).mean(),
                         through_box=s.edge_node_crossings.mean(),
                         **{f'{m}_strict/cross/w': f"{100*s[s.model==m].valid_b.mean():.1f}/"
                                                   f"{s[s.model==m].edge_edge_crossings.mean():.2f}/"
                                                   f"{s[s.model==m].wcross.mean():.2f}" for m in MODELS}))
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say()
    say("Paired tests on w, ours vs the two configurations dot draws legibly:")
    res = []
    for m in MODELS:
        for cfg in ['dot_tb_spline', 'dot_lr_spline']:
            a = G[(G.model == m) & (G.config == 'ours')].set_index(['corpus', 'run', 'item'])
            b = G[(G.model == m) & (G.config == cfg)].set_index(['corpus', 'run', 'item'])
            j = a[['wcross']].join(b[['wcross']], lsuffix='_o', rsuffix='_d', how='inner')
            st = paired(j.wcross_o, j.wcross_d); st.update(model=m, config=cfg); res.append(st)
    R = pd.DataFrame(res); R['p_holm'] = holm(R.p.values)
    say(R[['model', 'config', 'n', 'mx', 'my', 'd', 'lo', 'hi', 'p_holm']].to_string(index=False, float_format=lambda v: f"{v:.2e}" if 0 < abs(v) < 1e-4 else f"{v:.4f}"))
    say()
    say("Both orthogonal configurations cross at right angles (w = 0.000) and are")
    say("strictly legible on %.1f%% and %.1f%% of figures; ours on %.1f%%." % (
        100 * G[G.config == 'dot_tb_ortho'].valid_b.mean(),
        100 * G[G.config == 'dot_lr_ortho'].valid_b.mean(),
        100 * G[G.config == 'ours'].valid_b.mean()))


def B2(d):
    global RNG
    RNG = np.random.default_rng(0)
    li = d['li']; S = li[li.source == 'graphviz']
    head("B2  THE OTHER GRAPH-DRAWING CRITERIA  (Section 5.2)")
    crit = ['mean_bends_per_edge', 'min_separation', 'edge_label_crossings', 'frame_label_hits',
            'total_edge_length', 'edge_length_cv', 'ink_ratio', 'area']
    res = []
    for m in MODELS:
        a = S[(S.model == m) & (S.config == 'ours')].set_index(['run', 'item'])
        b = S[(S.model == m) & (S.config == 'dot_tb_spline')].set_index(['run', 'item'])
        j = a[crit].join(b[crit], lsuffix='_o', rsuffix='_d', how='inner')
        for c in crit:
            st = paired(j[c + '_o'], j[c + '_d']); st.update(model=m, measure=c); res.append(st)
    R = pd.DataFrame(res); R['p_holm'] = holm(R.p.values)
    say(R[['measure', 'model', 'n', 'mx', 'my', 'd', 'lo', 'hi', 'p_holm']].round(4).to_string(index=False))
    say()
    G = li[li.source.isin(['graphviz', 'graphviz_outside_development'])]
    nz = G[G.nested_b]
    for cfg in ['ours', 'dot_tb_spline']:
        x = nz[nz.config == cfg]
        say("  nested figures, %-14s hits per figure %.3f, figures with >=1 hit %.1f%%" % (
            cfg, x.frame_label_hits.mean(), 100 * (x.frame_label_hits > 0).mean()))


def B3(d):
    global RNG
    RNG = np.random.default_rng(0)
    li = d['li']; G = li[li.source.isin(['graphviz', 'graphviz_outside_development'])]
    head("B3  LOOPS AND NESTED CONTAINERS  (Section 5.2)")
    res = []
    for prop, col in [('feedback loop', 'loops_b'), ('nested container', 'nested_b')]:
        for val in [False, True]:
            a = G[(G.config == 'ours') & (G[col] == val)].set_index(['corpus', 'model', 'run', 'item'])
            b = G[(G.config == 'dot_tb_spline') & (G[col] == val)].set_index(['corpus', 'model', 'run', 'item'])
            j = a[['valid_b']].join(b[['valid_b']], lsuffix='_o', rsuffix='_d', how='inner')
            st = paired(j.valid_b_o.astype(float), j.valid_b_d.astype(float))
            st.update(prop=prop, value=val, strict_ours=100 * j.valid_b_o.mean(), strict_dot=100 * j.valid_b_d.mean())
            res.append(st)
    R = pd.DataFrame(res); R['p_holm'] = holm(R.p.values)
    say(R[['prop', 'value', 'n', 'strict_ours', 'strict_dot', 'd', 'lo', 'hi', 'p_holm']].round(3).to_string(index=False))
    say()
    say("Interaction (gap with the property minus gap without, bootstrap 95% CI):")
    for prop, col in [('feedback loop', 'loops_b'), ('nested container', 'nested_b')]:
        gaps = {}
        for val in [False, True]:
            a = G[(G.config == 'ours') & (G[col] == val)].set_index(['corpus', 'model', 'run', 'item'])
            b = G[(G.config == 'dot_tb_spline') & (G[col] == val)].set_index(['corpus', 'model', 'run', 'item'])
            j = a[['valid_b']].join(b[['valid_b']], lsuffix='_o', rsuffix='_d', how='inner')
            gaps[val] = (j.valid_b_o.astype(float) - j.valid_b_d.astype(float)).values
        bs = [RNG.choice(gaps[True], len(gaps[True]), True).mean() - RNG.choice(gaps[False], len(gaps[False]), True).mean()
              for _ in range(10000)]
        say("  %-18s with %+.3f, without %+.3f, difference %+.3f [%+.3f, %+.3f]" % (
            prop, gaps[True].mean(), gaps[False].mean(), gaps[True].mean() - gaps[False].mean(),
            np.percentile(bs, 2.5), np.percentile(bs, 97.5)))


def _seed_avg(df, cols, keys):
    g = df.copy()
    for c in cols:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    return g.groupby(keys, dropna=False)[cols].mean().reset_index()


def B4(d):
    global RNG
    RNG = np.random.default_rng(0)
    pi, dot = d['pi'], d['dot']
    head("B4  FLOWVQA BY SOURCE  (Section 5.4, Table 1)")
    cols = ['score']
    A = _seed_avg(pi[(pi.variant == 'evaluated') & (pi.corpus == 'flowvqa')], cols, ['model', 'item', 'domain'])
    B = _seed_avg(dot[(dot.variant == 'dot') & (dot.corpus == 'flowvqa')], cols, ['model', 'item', 'domain'])
    C = _seed_avg(dot[(dot.variant == 'dot_matched') & (dot.corpus == 'flowvqa')], cols, ['model', 'item', 'domain'])
    J = (A.merge(B, on=['model', 'item', 'domain'], suffixes=('', '_dot'))
          .merge(C.rename(columns={'score': 'score_matched'}), on=['model', 'item', 'domain']))
    J[['score', 'score_dot', 'score_matched']] = J[['score', 'score_dot', 'score_matched']].fillna(0.0)
    res = []
    for (m, dm), g in J.groupby(['model', 'domain']):
        st = paired(g.score, g.score_dot); st.update(model=m, source=dm, matched=g.score_matched.mean()); res.append(st)
    R = pd.DataFrame(res); R['p_holm'] = holm(R.p.values)
    say(R[['source', 'model', 'n', 'mx', 'my', 'matched', 'd', 'lo', 'hi', 'p_holm']].round(4).to_string(index=False))
    say()
    for dm, g in J.groupby('domain'):
        st = paired(g.score, g.score_dot)
        say("  pooled %-9s ours %.3f, DOT %.3f, matched DOT %.3f, d %+.3f [%+.3f, %+.3f], p %.1e" % (
            dm, st['mx'], st['my'], g.score_matched.mean(), st['d'], st['lo'], st['hi'], st['p']))


def B5(d):
    global RNG
    RNG = np.random.default_rng(0)
    pi, dot = d['pi'], d['dot']
    head("B5  PRECISION AND RECALL AGAINST THE SAME MODEL WRITING DOT")
    cols = ['node_precision', 'node_recall', 'node_f1', 'edge_precision', 'edge_recall', 'edge_f1']
    corp = ['sciflow', 'flowvqa']
    A = pi[(pi.variant == 'evaluated') & pi.corpus.isin(corp)].copy()
    B = dot[(dot.variant == 'dot') & dot.corpus.isin(corp)].copy()
    for f in (A, B):
        for c in cols:
            f[c] = pd.to_numeric(f[c], errors='coerce').fillna(0.0)   # a failed answer scores zero
    A = A.groupby(['corpus', 'model', 'item'])[cols].mean().reset_index()
    B = B.groupby(['corpus', 'model', 'item'])[cols].mean().reset_index()
    J = A.merge(B, on=['corpus', 'model', 'item'], suffixes=('_o', '_d'))
    res = []
    for (c, m), g in J.groupby(['corpus', 'model']):
        for k in cols:
            st = paired(g[k + '_o'], g[k + '_d']); st.update(corpus=c, model=m, measure=k); res.append(st)
    R = pd.DataFrame(res)
    R['p_holm'] = np.nan                      # Holm within each corpus and measure, as elsewhere
    for key, idx in R.groupby(['corpus', 'measure']).groups.items():
        R.loc[idx, 'p_holm'] = holm(R.loc[idx, 'p'].values)
    say(R[['corpus', 'model', 'measure', 'n', 'mx', 'my', 'd', 'p_holm']].round(4).to_string(index=False))
    say()
    for c, g in R.groupby('corpus'):
        ahead = g[(g.p_holm < 0.05) & (g.d > 0)]
        say("  %-8s ours significantly ahead in %d of %d comparisons: %s" % (
            c, len(ahead), len(g), ", ".join(sorted(set(ahead.measure)))))


def B6(d):
    global RNG
    RNG = np.random.default_rng(0)
    pi, dot, ju = d['pi'], d['dot'], d['ju']
    head("B6  PAPERBANANABENCH: THE JUDGE SCORES FIGURE SIZE  (Section 6)")
    main = ju[ju.judge == 'InternVL3.5-14B']
    pb = pi[pi.corpus == 'paperbanana']
    dpb = dot[dot.corpus == 'paperbanana'].copy()
    dpb['pred_nodes'] = pd.to_numeric(dpb.pred_nodes, errors='coerce')
    rows = []
    for m in MODELS:
        for sysn, boxes in [('dot', dpb[(dpb.variant == 'dot') & (dpb.model == m)].pred_nodes.mean()),
                            ('dot_matched', dpb[(dpb.variant == 'dot_matched') & (dpb.model == m)].pred_nodes.mean()),
                            ('ours', pb[(pb.variant == 'evaluated') & (pb.model == m)].layout_n_nodes.mean()),
                            ('ours_coverage', pb[(pb.variant == 'coverage') & (pb.model == m)].layout_n_nodes.mean())]:
            g = main[(main.system == sysn) & (main.model == m)]
            rows.append(dict(model=m, system=sysn, repr='graph' if sysn.startswith('ours') else 'DOT',
                             boxes=boxes, faithful=g.faithfulness_points.mean(), concise=g.conciseness_points.mean(),
                             readable=g.readability_points.mean(), aesthetic=g.aesthetics_points.mean(),
                             overall=g.overall_points.mean()))
    R = pd.DataFrame(rows)
    say(R.round(2).to_string(index=False))
    say()
    for col in ['concise', 'faithful', 'readable', 'aesthetic', 'overall']:
        rho, p = spearmanr(R.boxes, R[col])
        say("  rho(boxes, %-9s) = %+.3f   p = %.4f" % (col, rho, p))
    say()
    for m, g in R.groupby('model'):
        g = g.sort_values('boxes')
        say("  %-10s %s" % (m, "  <  ".join("%s (%.0f boxes, %.0f pts)" % (r.system, r.boxes, r.overall) for r in g.itertuples())))
    say()
    X = np.column_stack([np.ones(len(R)), R.boxes.values, (R.repr == 'graph').astype(float).values]
                        + [(R.model == m).astype(float).values for m in MODELS[1:]])
    b, *_ = np.linalg.lstsq(X, R.overall.values, rcond=None)
    resid = R.overall.values - X @ b; ss = (resid ** 2).sum(); dof = len(R) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * ss / dof)
    say("Least squares, overall judge points ~ boxes + representation + model:")
    for nm, c, s_ in zip(['intercept', 'boxes', 'graph (vs DOT)', 'Gemma', 'Ministral'], b, se):
        say("  %-16s %+8.2f   se %.2f   t %+.2f" % (nm, c, s_, c / s_))
    say("  R2 = %.3f, n = 12" % (1 - ss / ((R.overall.values - R.overall.mean()) ** 2).sum()))
    return R


def _bws(D):
    st = pd.read_csv(f'{D}/human_stimuli.csv'); se = pd.read_csv(f'{D}/human_sets.csv')
    ra = pd.read_csv(f'{D}/human_ratings.csv')
    cnt = {i: [0, 0, 0] for i in st.image}          # best, worst, shown
    for _, r in ra.iterrows():
        row = se[se.set == r['set']].iloc[0]
        for k in ['image_1', 'image_2', 'image_3', 'image_4']:
            cnt[row[k]][2] += 1
        cnt[r['best']][0] += 1; cnt[r['worst']][1] += 1
    st['bws'] = [(cnt[i][0] - cnt[i][1]) / max(cnt[i][2], 1) for i in st.image]
    return st


def B7(d, D):
    global RNG
    RNG = np.random.default_rng(0)
    head("B7  WHAT THE RATERS RESPONDED TO  (Section 5.3, Table S5)")
    st = _bws(D)
    o = st[st.condition == 'ours'].set_index('paper_id')
    g = st[st.condition == 'graphviz'].set_index('paper_id')
    m = st[st.condition == 'model'].set_index('paper_id')
    say("Best-worst means: ours %+.3f, dot %+.3f, model's own placement %+.3f" % (
        st[st.condition == 'ours'].bws.mean(), st[st.condition == 'graphviz'].bws.mean(),
        st[st.condition == 'model'].bws.mean()))
    dbws = (o.bws - g.bws).dropna()
    say("Ours minus dot over the 24 graphs: %+.3f; dot ahead on %d, ours on %d." % (
        dbws.mean(), int((dbws < 0).sum()), int((dbws > 0).sum())))
    say()
    say("Correlation of each measure with the best-worst score (measures negated,")
    say("so a positive value means the measure agrees with the raters):")
    for lbl, sub in [('all 72', st), ('the 48 by ours or dot', st[st.condition.isin(['ours', 'graphviz'])])]:
        say("  -- %s" % lbl)
        for c in [c for c in st.columns if c.startswith('report_')]:
            if st[c].dtype == bool or sub[c].nunique() < 3:
                continue
            rho, p = spearmanr(sub[c], sub.bws)
            say("     %-30s %+.3f (p %.3f)" % (c.replace('report_', ''), -rho, p))
    say()
    say("Paired within the 24 graphs: difference ours - dot in each measure against")
    say("the difference in best-worst score; p from 10,000 sign-flip permutations.")
    res = []
    for c in [c for c in st.columns if c.startswith('report_')]:
        if st[c].dtype == bool or st[c].nunique() < 4:
            continue
        dv = (o[c] - g[c]); ok = dv.notna() & dbws.notna()
        if dv[ok].nunique() < 4 or dv[ok].std() == 0:
            continue
        rho = spearmanr(dv[ok], dbws[ok])[0]
        null = [spearmanr(dv[ok] * RNG.choice([-1, 1], ok.sum()), dbws[ok])[0] for _ in range(10000)]
        res.append(dict(measure=c.replace('report_', ''), mean_difference=dv[ok].mean(), rho=rho,
                        p_perm=(np.abs(null) >= abs(rho)).mean()))
    R = pd.DataFrame(res).sort_values('p_perm')
    say(R.round(4).to_string(index=False))
    say()
    say("Our arrows are %.0f%% longer than dot's on these graphs." % (
        100 * (o.report_total_edge_length.mean() / g.report_total_edge_length.mean() - 1)))
    dbm = (o.bws - m.bws).dropna()
    say("Against the model's own placement: %+.3f points, ours ahead on %d of 24." % (dbm.mean(), int((dbm > 0).sum())))


def B8(d):
    global RNG
    RNG = np.random.default_rng(0)
    head("B8  WHY THE DIAGRAMEVAL-STYLE COLUMNS ARE NOT REPORTED  (negative result)")
    pi = d['pi']
    ev = pi[(pi.variant == 'evaluated') & pi.corpus.isin(['sciflow', 'flowvqa'])].copy()
    ev['comparable'] = ev.strict_path_comparable.astype(str) == 'True'
    say("pipeline_items.csv carries a second metric family under strict matching")
    say("(strict_node_f1, strict_path_f1, strict_diagrameval_score, ...). We checked")
    say("whether it could carry a second reading of the results. It cannot:")
    say()
    for c, g in ev.groupby('corpus'):
        inc = g[~g.comparable]
        say("  %-8s items whose paths are not comparable: %d of %d (%.0f%%), and their" % (
            c, len(inc), len(g), 100 * len(inc) / len(g)))
        say("           strict_path_f1 is %.3f by convention, not by measurement." % inc.strict_path_f1.mean())
    say()
    say("Since strict_diagrameval_score is the mean of node and path F1, that default")
    say("propagates: on SciFlow-Bench it correlates with the benchmark score at")
    for c, g in ev.groupby('corpus'):
        gc = g[g.comparable]
        de_all = (g.strict_node_f1 + g.strict_path_f1) / 2
        de_cmp = (gc.strict_node_f1 + gc.strict_path_f1) / 2
        say("  %-8s rho = %+.3f over all items, %+.3f over the comparable ones" % (
            c, spearmanr(g.score, de_all)[0], spearmanr(gc.score, de_cmp)[0]))
    say()
    say("The apparent disagreement between the two metric families is an artefact of")
    say("the default, and disappears once incomparable items are excluded. The family")
    say("is therefore not used in the paper. The same columns do carry one usable")
    say("check: scoring our own graphs with exact label matching instead of the")
    say("benchmark's 0.5 cosine threshold ranks items the same way,")
    for (c, m), g in ev.groupby(['corpus', 'model']):
        say("  %-8s %-10s rho = %+.3f" % (c, m, spearmanr(g.score, g.strict_sciflow_graph_score)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*', default=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    li, pi, dot, ju = load(a.data)
    d = dict(li=li, pi=pi, dot=dot, ju=ju)
    for name in a.analysis:
        if name == 'B7':
            B7(d, a.data)
        elif name == 'B6':
            B6(d)
        else:
            {'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B8': B8}[name](d)
    if a.out:
        open(a.out, 'w').write("\n".join(OUT) + "\n")
        print("\n[written %s]" % a.out)


if __name__ == '__main__':
    main()
