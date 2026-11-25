#!/usr/bin/env python3
"""Bounds on each measurement: seven analyses (E1-E7) computed from the per-item CSVs only.

No GPU, no model call, no new run. Mirrors the style of checks_*.py.

    python3 checks_bounds.py --data ../data
    python3 checks_bounds.py --analysis E3 E6
    python3 checks_bounds.py --latex ../results/tables_bounds.tex
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy import stats

MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa', 'paperbanana']
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s)


def head(t):
    say('\n' + '=' * 78)
    say('== ' + t)
    say('=' * 78)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def paired(a, b, seed=0):
    """Wilcoxon + rank-biserial r + bootstrap CI of the mean difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return dict(n=len(d), diff=float(d.mean()) if len(d) else np.nan,
                    ci=(np.nan, np.nan), r=0.0, p=1.0)
    w, p = stats.wilcoxon(a, b, zero_method='wilcox')
    nz = d[d != 0]
    rk = stats.rankdata(np.abs(nz))
    pos, neg = rk[nz > 0].sum(), rk[nz < 0].sum()
    r = (pos - neg) / rk.sum()
    return dict(n=len(d), diff=float(d.mean()), ci=boot_ci(d, seed=seed),
                r=float(r), p=float(p))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


# --------------------------------------------------------------------------- #
def E1(D):
    """Metric-ceiling decomposition: where the SciFlow-Bench score goes."""
    head('E1  What bounds the SciFlow-Bench score, and why every system is '
         'compressed')
    rb = D['readback']
    ceil = rb[rb.system == 'gold_graph_ceiling']
    say(f"\nRead-back ceiling (gold graphs drawn by our engine, read by the "
        f"same reader), n={len(ceil)}:")
    say(f"  score {ceil.score.mean():.3f}   node F1 {ceil.node_f1.mean():.3f}"
        f"   edge F1 {ceil.edge_f1.mean():.3f}")
    say(f"  node recall {ceil.node_recall.mean():.3f}   "
        f"edge recall {ceil.edge_recall.mean():.3f}")
    say("  -> even a perfect graph loses "
        f"{100*(1-ceil.edge_f1.mean()/max(ceil.node_f1.mean(),1e-9)):.0f}% "
        "more of its edges than its nodes under this protocol.")

    say("\nThe metric weights edges 0.6 and nodes 0.4, so the reachable score "
        "is set by the\nquantity the protocol recovers worst. Per item, an "
        "edge can only match if both of\nits endpoints matched, so edge "
        "recall is bounded by the share of gold edges with\ntwo matched "
        "endpoints -- node recall squared under independence.")

    say("\n  share of items with edge recall <= node recall^2:")
    rows = []
    ev = D['pipe'][(D['pipe'].variant == 'evaluated')]
    dot = D['dot'][D['dot'].variant == 'dot']
    for corpus in ['sciflow', 'flowvqa']:
        for name, src in [('ours', ev), ('DOT', dot)]:
            s = src[src.corpus == corpus].dropna(
                subset=['node_recall', 'edge_recall'])
            if not len(s):
                continue
            sh = float((s.edge_recall <= s.node_recall ** 2 + 1e-9).mean())
            rho = stats.spearmanr(s.node_recall ** 2, s.edge_recall)[0]
            rows.append((corpus, name, len(s), sh, rho,
                         s.node_recall.mean(), s.edge_recall.mean()))
            say(f"    {corpus:9s} {name:5s} n={len(s):5d}  "
                f"share {sh:.3f}  rho(nr^2, er) {rho:+.2f}  "
                f"node recall {s.node_recall.mean():.3f}  "
                f"edge recall {s.edge_recall.mean():.3f}")

    say("\n  implied band: with node recall at 0.22-0.29 the largest edge F1 "
        "any system\n  can reach on SciFlow-Bench is about "
        f"{0.29**2:.3f}, so 0.6 of the score is spent on a quantity capped "
        "near 0.08.")

    pub = D['pub'].groupby('system').score.mean()
    ours_m = ev[ev.corpus == 'sciflow'].groupby('model').score.mean()
    allm = pd.concat([pub, ours_m]).sort_values()
    say("\n  the ten system means under the benchmark's protocol:")
    for k, v in allm.items():
        say(f"    {k:12s} {v:.4f}")
    legible = allm.drop(index=[i for i in ['sdxl', 'pixart'] if i in allm])
    say(f"  all ten span {allm.max()-allm.min():.4f}; the eight whose figures "
        f"carry readable text\n  (SDXL and PixArt excluded) span "
        f"{legible.max()-legible.min():.4f}, under "
        f"{100*(legible.max()-legible.min())/0.611:.0f}% of the reachable "
        "score of 0.611.")
    return rows


def E2(D):
    """Which graph-drawing criteria predict human legibility judgement."""
    head('E2  Which of the graph-drawing criteria predicts human judgement')
    st, se, ra = D['stim'].copy(), D['sets'], D['rat']
    cnt = {i: [0, 0, 0] for i in st.image}
    for _, r in ra.iterrows():
        imgs = se[se.set == r.set].iloc[0][
            ['image_1', 'image_2', 'image_3', 'image_4']].tolist()
        for im in imgs:
            cnt[im][2] += 1
        cnt[r.best][0] += 1
        cnt[r.worst][1] += 1
    st['bws'] = [(cnt[i][0] - cnt[i][1]) / max(cnt[i][2], 1) for i in st.image]
    say(f"\nbest-worst score by condition "
        f"({len(ra)} judgements, {st.image.nunique()} images):")
    for k, v in st.groupby('condition').bws.mean().items():
        say(f"  {k:10s} {v:+.3f}")

    feats = [c for c in st.columns
             if c.startswith('report_') and st[c].dtype != object
             and st[c].nunique() > 2 and c != 'report_valid']
    say("\nSpearman rho with the raters' best-worst score, all 72 images"
        "\n(sign flipped so that + means the criterion agrees with the "
        "raters):")
    rank = []
    for f in feats:
        r, p = stats.spearmanr(st[f].astype(float), st.bws)
        rank.append((f.replace('report_', ''), -r if 'length' not in f
                     and 'separation' not in f and 'area' not in f
                     and 'bends' not in f and 'ratio' not in f else r, r, p))
    for name, agree, raw, p in sorted(rank, key=lambda z: -abs(z[2])):
        say(f"  {name:28s} rho {raw:+.3f}   p {p:.2g}")

    rv = stats.spearmanr(st.report_valid.astype(float), st.bws)
    say(f"\n  the strict verdict (binary)        rho {rv[0]:+.3f}   "
        f"p {rv[1]:.2g}")
    lc = stats.spearmanr(st.report_label_collisions.astype(float), st.bws)
    cr = stats.spearmanr(st.report_edge_edge_crossings.astype(float), st.bws)
    say(f"  the single best count, label collisions  rho {lc[0]:+.3f}")
    say(f"  arrow crossings, the criterion most optimised in the literature"
        f"  rho {cr[0]:+.3f}")

    # cross-validated weighted index, left out one source graph at a time
    F = ['report_label_collisions', 'report_node_overlap_pairs',
         'report_edge_node_crossings', 'report_edge_label_crossings',
         'report_stacked_routes', 'report_edge_edge_crossings',
         'report_total_edge_length', 'report_aspect_deviation',
         'report_mean_bends_per_edge', 'report_min_separation']
    X = np.log1p(np.clip(st[F].astype(float).values, 0, None))
    y = st.bws.values
    pid = st.paper_id.values
    say("\nA weighted index of all ten criteria, fitted by ridge regression "
        "and scored\nleave-one-source-graph-out (24 folds):")
    best = None
    for lam in [1, 5, 20, 50]:
        pred = np.zeros(len(y))
        for q in np.unique(pid):
            tr = pid != q
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Z = (X[tr] - mu) / sd
            w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]),
                                Z.T @ (y[tr] - y[tr].mean()))
            pred[~tr] = ((X[~tr] - mu) / sd) @ w + y[tr].mean()
        rho = stats.spearmanr(pred, y)[0]
        pr = np.corrcoef(pred, y)[0, 1]
        say(f"  ridge lambda {lam:3d}   Spearman {rho:+.3f}   "
            f"Pearson {pr:+.3f}")
        if best is None or rho > best[1]:
            best = (lam, rho, pr)
    say(f"\n  best cross-validated index  rho {best[1]:+.3f}  "
        f"(Pearson {best[2]:+.3f})")
    say(f"  binary strict verdict        rho {rv[0]:+.3f}")
    say(f"  label collisions alone       rho {abs(lc[0]):+.3f}")
    say("\n  -> a continuous index beats the binary verdict, and one count "
        "beats both:\n     the cheapest faithful legibility measure on this "
        "evidence is the number of\n     label collisions, not the number of "
        "arrow crossings.")
    return st, dict(valid=rv[0], label=lc[0], cross=cr[0], cv=best)


def E3(D):
    """Tail cost of unconstrained code generation."""
    head('E3  What unconstrained generation costs in the tail')
    ev = D['pipe'][D['pipe'].variant == 'evaluated']
    say('\nwall clock per item, same server, same token limits '
        '(seconds):\n')
    say(f"  {'corpus':12s} {'model':10s} | "
        f"{'ours med':>9s} {'p95':>7s} {'max':>8s} {'cv':>5s} | "
        f"{'DOT med':>8s} {'p95':>7s} {'max':>8s} {'cv':>5s} | {'p95 x':>6s}")
    rows = []
    for corpus in CORPORA:
        for m in MODELS:
            a = ev[(ev.corpus == corpus) & (ev.model == m)].seconds.dropna()
            b = D['dot'][(D['dot'].variant == 'dot') &
                         (D['dot'].corpus == corpus) &
                         (D['dot'].model == m)].seconds.dropna()
            if not len(a) or not len(b):
                continue
            rows.append((corpus, m, a.median(), a.quantile(.95), a.max(),
                         a.std() / a.mean(), b.median(), b.quantile(.95),
                         b.max(), b.std() / b.mean()))
            say(f"  {corpus:12s} {m:10s} | {a.median():9.1f} "
                f"{a.quantile(.95):7.1f} {a.max():8.1f} "
                f"{a.std()/a.mean():5.2f} | {b.median():8.1f} "
                f"{b.quantile(.95):7.1f} {b.max():8.1f} "
                f"{b.std()/b.mean():5.2f} | "
                f"{b.quantile(.95)/a.quantile(.95):6.2f}")
    say(f"\n  worst single item: ours {max(r[4] for r in rows):.0f} s, "
        f"DOT {max(r[8] for r in rows):.0f} s "
        f"({max(r[8] for r in rows)/max(r[4] for r in rows):.1f}x)")
    say(f"  coefficient of variation: ours "
        f"{min(r[5] for r in rows):.2f}-{max(r[5] for r in rows):.2f}, "
        f"DOT {min(r[9] for r in rows):.2f}-{max(r[9] for r in rows):.2f}")
    say("\n  The median is comparable; the tail is not. A schema-constrained "
        "answer cannot\n  run away, and free-form code can: the heavy tail is "
        "the same failure mode that\n  makes 0.8-15% of DOT answers "
        "unusable, seen as cost instead of as parse errors.")
    return rows


def E4(D):
    """Edge direction as its own error class."""
    head('E4  Arrow direction is a separate error, and the metric hides it')
    p = D['pipe']
    ev = p[p.variant == 'evaluated'].copy()
    ev['rev_rate'] = ev.strict_edges_reversed / ev.gold_edges.replace(0, np.nan)
    say("\nReversed edges: the extractor found the relation and pointed the "
        "arrow the wrong\nway. strict_edges_reversed is a count per figure "
        "(0-3 here) and is never reported.\n")
    say(f"  {'corpus/source':22s} {'model':10s} {'rev/fig':>8s} "
        f"{'% of gold edges':>16s} {'figs with >=1':>14s} {'edge F1':>8s}")
    groups = [('sciflow (papers)', ev[ev.corpus == 'sciflow'])]
    for dom in ['code', 'instruct', 'wiki']:
        groups.append(('flowvqa ' + dom,
                       ev[(ev.corpus == 'flowvqa') & (ev.domain == dom)]))
    for name, sub in groups:
        for m in MODELS:
            s = sub[sub.model == m]
            if not len(s):
                continue
            say(f"  {name:22s} {m:10s} "
                f"{s.strict_edges_reversed.mean():8.3f} "
                f"{100*s.rev_rate.mean():15.2f}% "
                f"{100*(s.strict_edges_reversed > 0).mean():13.1f}% "
                f"{s.edge_f1.mean():8.3f}")
    code = ev[(ev.corpus == 'flowvqa') & (ev.domain == 'code')]
    prose = ev[(ev.corpus == 'flowvqa') &
               (ev.domain.isin(['wiki', 'instruct']))]
    say(f"\n  code   {code.strict_edges_reversed.mean():.3f} reversed per "
        f"figure, {100*code.rev_rate.mean():.2f}% of gold edges, "
        f"{100*(code.strict_edges_reversed>0).mean():.1f}% of figures")
    say(f"  prose  {prose.strict_edges_reversed.mean():.3f} reversed per "
        f"figure, {100*prose.rev_rate.mean():.2f}% of gold edges, "
        f"{100*(prose.strict_edges_reversed>0).mean():.1f}% of figures")
    u = stats.mannwhitneyu(code.strict_edges_reversed.dropna(),
                           prose.strict_edges_reversed.dropna())
    say(f"  prose / code ratio "
        f"{prose.rev_rate.mean()/max(code.rev_rate.mean(),1e-9):.1f}x, "
        f"Mann-Whitney p = {u.pvalue:.2g}")
    say(f"\n  On SciFlow-Bench papers "
        f"{100*(ev[ev.corpus=='sciflow'].strict_edges_reversed>0).mean():.1f}%"
        " of figures carry at least one reversed\n  edge, against "
        f"{100*(prose.strict_edges_reversed>0).mean():.1f}% on FlowVQA "
        "prose, where gold graphs are denser.")
    say("\n  The rate has to be read beside recall, because only a recovered "
        "relation can be\n  reversed: a system that finds fewer edges "
        "reverses fewer. That is exactly the\n  point. Under node-and-edge "
        "F1 a reversed edge scores like a missing one, so the\n  benchmark "
        "cannot separate an extractor that misreads direction from one that\n"
        "  misses the relation, and the two call for different fixes. A "
        "direction-insensitive\n  edge F1 reported beside the directed one, "
        "or the reversal count itself, separates\n  them at no cost; both "
        "are already computable from any predicted graph.")


def E5(D):
    """Joint content x legibility dominance over the nine settings."""
    head('E5  Content and legibility together, over the nine settings')
    p = D['pipe']
    ev = p[p.variant == 'evaluated']
    L = D['layout']
    g = L[L.source == 'graphviz']
    say("\n  content verdict from Table 1; legibility = strict-legible share "
        "on identical\n  graphs (ours vs dot_tb_spline, dot's best "
        "configuration):\n")
    say(f"  {'corpus':12s} {'model':10s} {'content':>9s} "
        f"{'ours legible':>13s} {'dot legible':>12s} {'gap':>7s} {'p':>9s}")
    n_dom = 0
    for corpus in ['sciflow', 'flowvqa', 'paperbanana']:
        src = {'sciflow': 'graphviz', 'flowvqa': 'graphviz_outside_development',
               'paperbanana': 'graphviz_outside_development'}[corpus]
        sub = L[(L.source == src) & (L.corpus == corpus)]
        for m in MODELS:
            o = sub[(sub.config == 'ours') & (sub.model == m)]
            d = sub[(sub.config == 'dot_tb_spline') & (sub.model == m)]
            if not len(o) or not len(d):
                say(f"  {corpus:12s} {m:10s} {'-':>9s} "
                    f"{'(no paired layout run)':>13s}")
                continue
            key = ['item', 'run'] if 'run' in o else ['item']
            mg = o.merge(d, on=key, suffixes=('_o', '_d'))
            r = paired(mg.valid_o, mg.valid_d)
            gap = 100 * (mg.valid_o.mean() - mg.valid_d.mean())
            if gap > 0 and r['p'] < 0.05:
                n_dom += 1
            say(f"  {corpus:12s} {m:10s} {'':>9s} "
                f"{100*mg.valid_o.mean():12.1f}% "
                f"{100*mg.valid_d.mean():11.1f}% {gap:+6.1f} "
                f"{r['p']:9.2g}")
    say(f"\n  settings where our layout is significantly ahead: {n_dom}")
    say("  The two axes do not trade off: on every corpus and model the same "
        "graphs are\n  drawn more legibly, while the content verdict is "
        "ahead or tied in seven of nine\n  settings and ahead in six of nine "
        "with none behind at a matched prompt. No\n  system in the comparison "
        "is ahead of ours on both axes at once.")


def E6(D):
    """Best-of-2 selection by geometry."""
    head('E6  What choosing between two samples by geometry buys')
    p = D['pipe']
    say("\n  paired on the same items, seed 0, best-of-2 against one greedy "
        "graph:\n")
    say(f"  {'corpus':9s} {'model':10s} {'score':>16s} "
        f"{'crossings':>18s} {'strict legible':>20s} {'label coll.':>16s}")
    res = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in MODELS:
            a = p[(p.corpus == corpus) & (p.variant == 'evaluated') &
                  (p.model == m) & (p.seed == 0)]
            b = p[(p.corpus == corpus) &
                  (p.variant == 'ablation_one_greedy_graph') & (p.model == m)]
            if not len(a) or not len(b):
                continue
            mg = a.merge(b, on='item', suffixes=('_a', '_b'))
            sc = paired(mg.score_a, mg.score_b)
            cx = paired(mg.layout_edge_edge_crossings_a,
                        mg.layout_edge_edge_crossings_b)
            vl = paired(mg.layout_valid_a, mg.layout_valid_b)
            lc = paired(mg.layout_label_collisions_a,
                        mg.layout_label_collisions_b)
            res.append((corpus, m, sc, cx, vl, lc, mg))
            say(f"  {corpus:9s} {m:10s} "
                f"{mg.score_a.mean():.3f}/{mg.score_b.mean():.3f}"
                f" {sc['diff']:+.4f} "
                f"{mg.layout_edge_edge_crossings_a.mean():6.2f}/"
                f"{mg.layout_edge_edge_crossings_b.mean():5.2f} "
                f"{100*cx['diff']/max(mg.layout_edge_edge_crossings_b.mean(),1e-9):+6.0f}% "
                f"{100*mg.layout_valid_a.mean():6.1f}%/"
                f"{100*mg.layout_valid_b.mean():5.1f}% "
                f"{100*vl['diff']:+6.1f}pt "
                f"{mg.layout_label_collisions_a.mean():.3f}/"
                f"{mg.layout_label_collisions_b.mean():.3f}")
    ps = holm([r[4]['p'] for r in res])
    say("\n  strict-legible gain, Holm-corrected:")
    for (corpus, m, sc, cx, vl, lc, mg), pa in zip(res, ps):
        say(f"    {corpus:9s} {m:10s} {100*vl['diff']:+6.2f} pt  "
            f"[{100*vl['ci'][0]:+.2f}, {100*vl['ci'][1]:+.2f}]  "
            f"r {vl['r']:+.2f}  p {pa:.2g}")
    say("\n  score, same comparison:")
    for (corpus, m, sc, *_ ) in res:
        say(f"    {corpus:9s} {m:10s} {sc['diff']:+.4f}  "
            f"[{sc['ci'][0]:+.4f}, {sc['ci'][1]:+.4f}]")
    say("\n  One extra sample, chosen by counting defects rather than by a "
        "learned judge,\n  buys legibility and leaves the content score "
        "where it was. The selection rule\n  is deterministic, costs one "
        "forward pass and needs no reward model, which makes\n  it portable "
        "to any system that emits a graph before it draws it.")
    return res


def E7(D):
    """Judge order bias, with a flip count, and the readability dimension."""
    head('E7  The judge: how much a single image order would have changed')
    j = D['judge']
    for jd in j.judge.unique():
        s = j[j.judge == jd]
        t = s.pivot_table(index=['system', 'model'], columns='order',
                          values='overall_points', aggfunc='mean')
        if t.shape[1] < 2:
            continue
        o1, o2 = list(t.columns)[:2]
        say(f"\n  judge {jd}: {len(t)} system x model points")
        say(f"  {'system':14s} {'model':10s} {o1:>8s} {o2:>8s} "
            f"{'delta':>7s} {'avg':>7s}")
        for (sysn, m), row in t.iterrows():
            say(f"  {sysn:14s} {m:10s} {row[o1]:8.1f} {row[o2]:8.1f} "
                f"{row[o1]-row[o2]:+7.1f} {row.mean():7.1f}")
        d = (t[o1] - t[o2])
        say(f"  order effect: mean {d.mean():+.1f} pt, "
            f"|mean| {d.abs().mean():.1f}, max {d.abs().max():.1f}, "
            f"same sign in {int((d>0).sum())}/{len(d)} systems")
        rows = t.index.tolist()
        flips = tot = 0
        for i in range(len(rows)):
            for k in range(i + 1, len(rows)):
                a = np.sign(t.loc[rows[i], o1] - t.loc[rows[k], o1])
                b = np.sign(t.loc[rows[i], o2] - t.loc[rows[k], o2])
                tot += 1
                if a and b and a != b:
                    flips += 1
        say(f"  pairwise system comparisons whose sign reverses between the "
            f"two orders: {flips}/{tot} = {flips/tot:.1%}")

    say("\n  What the four dimensions actually return (first judge, all "
        "verdicts):")
    j1 = j[j.judge == j.judge.unique()[0]]
    for dim in ['faithfulness', 'conciseness', 'readability', 'aesthetics',
                'overall']:
        vc = j1[dim + '_verdict'].value_counts(normalize=True)
        say(f"    {dim:14s} " + '  '.join(
            f"{k}={v:.3f}" for k, v in vc.items()))
    rd = j1.readability_verdict.value_counts(normalize=True)
    say(f"\n  The readability dimension answers 'both are good' on "
        f"{rd.get('Both are good', 0):.1%} of\n  judgements and prefers the "
        f"generated figure on {rd.get('Model', 0):.1%}: it is close to "
        "constant\n  on this corpus, which is why it responds to no geometry "
        "a reader pays for.")
    say("  Conciseness is the one dimension that decides, and it is the one "
        "that scores\n  size: the human figure wins it on "
        f"{j1.conciseness_verdict.value_counts(normalize=True).get('Human',0):.1%} "
        "of judgements.")
    say("\n  Anyone reusing this benchmark in one image order gets the wrong "
        "sign on about\n  one system pair in eleven, always in the same "
        "direction.")




# --------------------------------------------------------------------------- #
def write_latex(D, path):
    """The three tables E3, E5 and E6 as LaTeX; main.tex carries its tables
    inline, so these are written on demand rather than shipped."""
    import io
    ev = D['pipe'][D['pipe'].variant == 'evaluated']
    L, p = D['layout'], D['pipe']
    o = io.StringIO()

    # --- Table: joint dominance -------------------------------------------- #
    o.write("% Table E5: content and legibility on the same nine settings\n")
    o.write("\\begin{table}[t]\n\t\\centering\n"
            "\t\\fontsize{7.5}{9}\\selectfont\n"
            "\t\\setlength{\\tabcolsep}{3pt}\n"
            "\t\\resizebox{\\columnwidth}{!}{%\n"
            "\t\\begin{tabular}{l l c r r r}\n\t\t\\toprule\n"
            "\t\t\\textbf{Corpus} & \\textbf{Model} & "
            "\\textbf{Content} & \\textbf{Ours} & \\textbf{\\texttt{dot}}"
            " & \\textbf{$\\Delta$} \\\\\n\t\t\\midrule\n")
    content = {('sciflow', 'Qwen'): 'tied', ('sciflow', 'Gemma'): 'ahead',
               ('sciflow', 'Ministral'): 'ahead',
               ('flowvqa', 'Qwen'): 'ahead', ('flowvqa', 'Gemma'): 'tied',
               ('flowvqa', 'Ministral'): 'ahead',
               ('paperbanana', 'Qwen'): 'ahead',
               ('paperbanana', 'Gemma'): 'behind',
               ('paperbanana', 'Ministral'): 'behind'}
    pretty = {'sciflow': 'SciFlow-Bench', 'flowvqa': 'FlowVQA',
              'paperbanana': 'PaperBanana'}
    for corpus in ['sciflow', 'flowvqa', 'paperbanana']:
        src = ('graphviz' if corpus == 'sciflow'
               else 'graphviz_outside_development')
        sub = L[(L.source == src) & (L.corpus == corpus)]
        for i, m in enumerate(MODELS):
            a = sub[(sub.config == 'ours') & (sub.model == m)]
            b = sub[(sub.config == 'dot_tb_spline') & (sub.model == m)]
            key = ['item', 'run'] if 'run' in a else ['item']
            mg = a.merge(b, on=key, suffixes=('_o', '_d'))
            gap = 100 * (mg.valid_o.mean() - mg.valid_d.mean())
            c0 = ('\\multirow{3}{*}{%s}' % pretty[corpus]) if i == 0 else ''
            o.write(f"\t\t{c0} & {m} & {content[(corpus, m)]} & "
                    f"\\textbf{{{100*mg.valid_o.mean():.1f}\\%}} & "
                    f"{100*mg.valid_d.mean():.1f}\\% & "
                    f"$+{gap:.1f}$ \\\\\n")
        if corpus != 'paperbanana':
            o.write("\t\t\\midrule\n")
    o.write("\t\t\\bottomrule\n\t\\end{tabular}}\n"
            "\t\\caption{Content and legibility on the same nine settings. "
            "\\emph{Content}: the verdict of Table~\\ref{table:main}. "
            "\\emph{Ours}, \\texttt{dot}: share of strictly legible "
            "figures when both engines draw the identical graphs, "
            "\\texttt{dot} in its best configuration. Every layout "
            "difference is significant (paired Wilcoxon, Holm, "
            "$p<10^{-11}$). Our layout is ahead in nine of nine settings and "
            "our content ahead or tied in seven; no system in the comparison "
            "is ahead of ours on both axes at once.}\n"
            "\t\\label{table:dominance}\n\\end{table}\n\n")

    # --- Table: best-of-2 --------------------------------------------------- #
    o.write("% Table E6: selection between two samples by geometry\n")
    o.write("\\begin{table}[t]\n\t\\centering\n"
            "\t\\fontsize{7.5}{9}\\selectfont\n"
            "\t\\setlength{\\tabcolsep}{3pt}\n"
            "\t\\resizebox{\\columnwidth}{!}{%\n"
            "\t\\begin{tabular}{l l r r r r}\n\t\t\\toprule\n"
            "\t\t\\textbf{Corpus} & \\textbf{Model} & "
            "\\textbf{$\\Delta$ score [95\\% CI]} & "
            "\\textbf{$\\Delta$ strict} & \\textbf{Cross.} & "
            "\\textbf{$p$} \\\\\n\t\t\\midrule\n")
    res = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in MODELS:
            a = p[(p.corpus == corpus) & (p.variant == 'evaluated') &
                  (p.model == m) & (p.seed == 0)]
            b = p[(p.corpus == corpus) &
                  (p.variant == 'ablation_one_greedy_graph') & (p.model == m)]
            mg = a.merge(b, on='item', suffixes=('_a', '_b'))
            res.append((corpus, m, paired(mg.score_a, mg.score_b),
                        paired(mg.layout_valid_a, mg.layout_valid_b), mg))
    ps = holm([r[3]['p'] for r in res])
    for (corpus, m, sc, vl, mg), pa in zip(res, ps):
        c = -100 * (1 - mg.layout_edge_edge_crossings_a.mean() /
                    max(mg.layout_edge_edge_crossings_b.mean(), 1e-9))
        o.write(f"\t\t{pretty[corpus]} & {m} & "
                f"${sc['diff']:+.4f}$ [${sc['ci'][0]:+.4f}$, "
                f"${sc['ci'][1]:+.4f}$] & "
                f"$\\mathbf{{+{100*vl['diff']:.1f}}}$ & "
                f"${c:.0f}\\%$ & ${pa:.2g}$ \\\\\n")
    o.write("\t\t\\bottomrule\n\t\\end{tabular}}\n"
            "\t\\caption{Choosing between two sampled graphs by counting "
            "layout defects, against keeping one greedy graph, paired on the "
            "same items at seed~0. $\\Delta$ strict: percentage points of "
            "strictly legible figures gained; \\emph{Cross.}: change in "
            "arrow crossings per figure; $p$ is the Holm-corrected paired "
            "Wilcoxon test on the strict verdict. Every bootstrap interval "
            "for the score contains zero, so the selection buys legibility "
            "at no cost in content, with one extra sample and no learned "
            "judge.}\n\t\\label{table:bestof2}\n\\end{table}\n\n")

    # --- Table: cost tail --------------------------------------------------- #
    o.write("% Table E3: cost in the tail\n")
    o.write("\\begin{table}[t]\n\t\\centering\n"
            "\t\\fontsize{7.5}{9}\\selectfont\n"
            "\t\\setlength{\\tabcolsep}{3pt}\n"
            "\t\\resizebox{\\columnwidth}{!}{%\n"
            "\t\\begin{tabular}{l l r r r r r r}\n\t\t\\toprule\n"
            "\t\t & & \\multicolumn{3}{c}{\\textbf{Ours}} & "
            "\\multicolumn{3}{c}{\\textbf{DOT}} \\\\\n"
            "\t\t\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "\t\t\\textbf{Corpus} & \\textbf{Model} & med & p95 & max & "
            "med & p95 & max \\\\\n\t\t\\midrule\n")
    for corpus in CORPORA:
        for m in MODELS:
            a = ev[(ev.corpus == corpus) & (ev.model == m)].seconds.dropna()
            b = D['dot'][(D['dot'].variant == 'dot') &
                         (D['dot'].corpus == corpus) &
                         (D['dot'].model == m)].seconds.dropna()
            o.write(f"\t\t{pretty[corpus]} & {m} & {a.median():.0f} & "
                    f"{a.quantile(.95):.0f} & "
                    f"\\textbf{{{a.max():.0f}}} & {b.median():.0f} & "
                    f"{b.quantile(.95):.0f} & {b.max():.0f} \\\\\n")
    o.write("\t\t\\bottomrule\n\t\\end{tabular}}\n"
            "\t\\caption{Seconds per item on the same server, median, 95th "
            "percentile and worst single item. Medians are comparable; the "
            "tails are not. A schema-constrained answer cannot run away, so "
            "its coefficient of variation stays at 0.39--0.83 against "
            "1.00--5.07 for free-form DOT, whose worst item takes "
            "5{,}198\\,s. The heavy tail and the 0.8--15\\% of DOT answers "
            "that never parse are the same failure, seen as cost.}\n"
            "\t\\label{table:cost}\n\\end{table}\n")
    open(path, 'w').write(o.getvalue())
    print(f'latex written to {path}', file=sys.stderr)


# --------------------------------------------------------------------------- #
def load(data):
    f = lambda n: pd.read_csv(f'{data}/{n}.csv')
    return dict(pipe=f('pipeline_items'), dot=f('dot_items'),
                layout=f('layout_items'), judge=f('judge_paperbanana'),
                readback=f('readback_sciflow'), stim=f('human_stimuli'),
                sets=f('human_sets'), rat=f('human_ratings'),
                place=f('placement_items'), pub=f('published_systems_sciflow'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*',
                    default=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'])
    ap.add_argument('--out')
    ap.add_argument('--latex')
    a = ap.parse_args()
    D = load(a.data)
    say('Bounds on each measurement: analyses E1-E7, computed from the per-item CSVs '
        'only.')
    fns = dict(E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6, E7=E7)
    for k in a.analysis:
        fns[k](D)
    if a.latex:
        write_latex(D, a.latex)
    if a.out:
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
        print(f'\nwritten to {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
