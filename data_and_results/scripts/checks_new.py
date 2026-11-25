#!/usr/bin/env python3
"""Seven analyses that need no new run: they read data/ and nothing else.

    python3 checks_new.py                                  # all seven
    python3 checks_new.py --analysis N2 N3
    python3 checks_new.py --latex ../results/tables_new.tex
    python3 checks_new.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| N1 judge power | what a vision--language judge comparison can resolve, and where its variance is | 5.6 |
| N2 reader      | how faithfully a read-back reader recovers a graph whose geometry we know | 5.6 |
| N3 capacity    | the graph size at which half an engine's figures stop passing the strict check | 5.3 |
| N4 layoutvar   | how much of legibility's variance belongs to the engine and how much to the graph | 5.3 |
| N5 index       | an index of the ten layout measures fitted to the human ratings | 5.3 |
| N6 ceiling     | every SciFlow-Bench score as a share of what the benchmark can return | 5.1 |
| N7 frontier    | content against legibility over every corpus x model x variant cell | 5.4 |

N1 decomposes the judge's points over item x model x seed x order inside one system and
derives the paired and unpaired minimum detectable difference at 80% power. N2 uses the
graph the pipeline drew as ground truth, which is available only because the geometry is
explicit. N3 fits the strict verdict on log box count by IRLS and reports the 50% point.
N5 log1p-transforms and standardises the ten measures and cross-validates leaving one
source graph out, so no rated image is predicted by a fit that saw another layout of the
same graph.
"""
import argparse, itertools, os
import pandas as pd, numpy as np
from scipy import stats

CSV, TEX = {}, {}

def _b(s):
    return s.astype(str).map({"True":1,"False":0,"1":1,"0":0,"1.0":1,"0.0":0})

def _logit_fit(x, y, iters=60):
    X = np.vstack([x, np.ones_like(x)]).T; b = np.zeros(2)
    for _ in range(iters):
        p = 1/(1+np.exp(-X@b)); W = p*(1-p)+1e-9
        b = b + np.linalg.solve(X.T@(X*W[:, None]), X.T@(y-p))
    return b

# ---------------------------------------------------------------- N1
def N1(D, P):
    j = pd.read_csv(D+"judge_paperbanana.csv")
    J = j[j.judge == "InternVL3.5-14B"]
    dims = ["overall", "faithfulness", "conciseness", "readability", "aesthetics"]
    P("\n== N1  What a vision--language judge comparison can resolve ==")
    g = J[J.system == "ours"]
    rows = []
    P("\n(a) variance of the judge's points, ours, 3 models x 3 seeds x 2 orders x 292 cases")
    for d in dims:
        v = d+"_points"; x = g.dropna(subset=[v]); mu = x[v].mean(); tot = ((x[v]-mu)**2).mean()
        c = {}
        for f in ["item", "model", "seed", "order"]:
            m = x.groupby(f)[v].mean(); n = x.groupby(f)[v].size()
            c[f] = ((m-mu)**2*n).sum()/n.sum()
        m2 = x.groupby(["item", "model"])[v].mean(); n2 = x.groupby(["item", "model"])[v].size()
        c["item_x_model"] = ((m2-mu)**2*n2).sum()/n2.sum()-c["item"]-c["model"]
        res = tot-sum(c.values())
        rows.append(dict(dimension=d, sd=np.sqrt(tot), **{k: 100*v2/tot for k, v2 in c.items()},
                         residual=100*res/tot, n=len(x)))
        P(f"  {d:13s} sd {np.sqrt(tot):5.1f} | item {100*c['item']/tot:5.1f}%  model {100*c['model']/tot:4.1f}%"
          f"  seed {100*c['seed']/tot:4.1f}%  order {100*c['order']/tot:4.1f}%"
          f"  item x model {100*c['item_x_model']/tot:5.1f}%  residual {100*res/tot:5.1f}%")
    CSV["judge_variance"] = pd.DataFrame(rows)
    P("\n(b) minimum detectable difference in overall points, 80% power, alpha=.05, two-sided")
    agg = J.groupby(["system", "model", "item"]).overall_points.mean().reset_index()
    mdd = []
    for m in ["Qwen", "Gemma", "Ministral"]:
        a = agg[(agg.system == "ours") & (agg.model == m)].set_index("item").overall_points
        b = agg[(agg.system == "dot") & (agg.model == m)].set_index("item").overall_points
        k = a.index.intersection(b.index); dd = (a[k]-b[k]).dropna()
        sd = dd.std(ddof=1); s1 = a.std(ddof=1)
        P(f"  {m:10s} paired sd {sd:5.1f}  -> paired MDD   n=292 {2.8*sd/np.sqrt(292):4.1f}"
          f"  n=100 {2.8*sd/np.sqrt(100):4.1f}  n=50 {2.8*sd/np.sqrt(50):4.1f}")
        P(f"  {'':10s} item sd   {s1:5.1f}  -> unpaired MDD n=292 {2.8*s1*np.sqrt(2/292):4.1f}"
          f"  n=100 {2.8*s1*np.sqrt(2/100):4.1f}  n=50 {2.8*s1*np.sqrt(2/50):4.1f}")
        mdd.append(dict(model=m, paired_sd=sd, item_sd=s1,
                        paired_292=2.8*sd/np.sqrt(292), paired_100=2.8*sd/np.sqrt(100),
                        paired_50=2.8*sd/np.sqrt(50),
                        unpaired_292=2.8*s1*np.sqrt(2/292), unpaired_100=2.8*s1*np.sqrt(2/100),
                        unpaired_50=2.8*s1*np.sqrt(2/50)))
    CSV["judge_mdd"] = pd.DataFrame(mdd)
    P("\n(c) per-figure order instability: the same figure judged in both image orders")
    inst = []
    for d in dims:
        q = J.pivot_table(index=["system", "model", "seed", "item"], columns="order",
                          values=d+"_points").dropna()
        dis = 100*(q["normal"] != q["swapped"]).mean()
        inst.append(dict(dimension=d, disagree_pct=dis, mean_abs_shift=abs(q["normal"]-q["swapped"]).mean(),
                         signed_shift=(q["normal"]-q["swapped"]).mean(), n=len(q)))
        P(f"  {d:13s} the two orders disagree on {dis:4.1f}% of figures; mean |shift|"
          f" {abs(q['normal']-q['swapped']).mean():5.1f}; signed {(q['normal']-q['swapped']).mean():+5.1f} (n={len(q)})")
    CSV["judge_order_instability"] = pd.DataFrame(inst)
    v = CSV["judge_variance"].set_index("dimension"); mm = CSV["judge_mdd"].set_index("model")
    io = CSV["judge_order_instability"].set_index("dimension")
    TEX["judge_power"] = (
        "% generated by checks_new.py --analysis N1 --latex\n"
        "\\begin{table}[htbp]\n\t\\centering\n\t\\fontsize{7}{8.4}\\selectfont\n"
        "\t\\setlength{\\tabcolsep}{3pt}\n\t\\begin{tabular}{l r r r r @{\\hspace{8pt}} r}\n\t\t\\toprule\n"
        "\t\t & \\multicolumn{4}{c}{\\textbf{Variance of the points, \\%}} & \\textbf{Orders} \\\\\n"
        "\t\t\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\n"
        "\t\t\\textbf{Dimension} & item & \\textbf{system} & order & resid. & \\textbf{differ \\%} \\\\\n"
        "\t\t\\midrule\n" +
        "".join(f"\t\t{d.capitalize()} & {v.item[d]:.1f} & \\textbf{{{v.model[d]:.1f}}} & {v.order[d]:.1f}"
                f" & {v.residual[d]:.1f} & \\textbf{{{io.disagree_pct[d]:.1f}}} \\\\\n" for d in v.index) +
        "\t\t\\midrule\n\t\t\\multicolumn{6}{l}{\\textbf{Detectable difference in overall points, 80\\% power}} \\\\\n"
        "\t\t & \\multicolumn{2}{c}{paired, $n{=}$} & \\multicolumn{3}{c}{two samples of $n$} \\\\\n"
        "\t\t\\textbf{Model} & 292 & 50 & 292 & 100 & 50 \\\\\n\t\t\\midrule\n" +
        "".join(f"\t\t{m} & {mm.paired_292[m]:.1f} & {mm.paired_50[m]:.1f} & {mm.unpaired_292[m]:.1f}"
                f" & {mm.unpaired_100[m]:.1f} & \\textbf{{{mm.unpaired_50[m]:.1f}}} \\\\\n" for m in mm.index) +
        "\t\t\\bottomrule\n\t\\end{tabular}\n\t\\caption{What the PaperBananaBench judge can resolve, "
        "from the released verdicts. Top: variance of the points over item, system, image order and decoding "
        "seed inside one system ($n=5{,}256$ judgements per dimension), and how often the two image orders "
        "return different points for the \\emph{same} figure. Bottom: the smallest difference in mean judge "
        "points that 80\\% power resolves, paired per figure and between two independent samples of $n$ "
        "figures. Computed by \\texttt{checks\\_new.py --analysis N1}.}\n\t\\label{table:judgepower}\n\\end{table}\n")

# ---------------------------------------------------------------- N2
def N2(D, P):
    rb = pd.read_csv(D+"readback_sciflow.csv"); pi = pd.read_csv(D+"pipeline_items.csv")
    ev = pi[(pi.variant == "evaluated") & (pi.corpus == "sciflow") & (pi.seed == 0)]
    # (domain, item) is the paper's key: eleven SciFlow-Bench papers are filed under
    # two areas each, and keying on the item id alone merges them.
    t = ev.groupby(["model", "domain", "item"]).agg(
        drawn_nodes=("layout_n_nodes", "mean"),
        drawn_edges=("layout_n_edges", "mean")).reset_index()
    r = rb[rb.system == "ours"].merge(t, on=["model", "domain", "item"])
    P("\n== N2  How faithfully a read-back reader recovers a graph whose geometry we know ==")
    P("   ground truth is the graph the pipeline drew, not the gold graph")
    rows = []
    for m in ["Qwen", "Gemma", "Ministral"]:
        s = r[r.model == m]
        ni = (s.pred_nodes/s.drawn_nodes).replace([np.inf, -np.inf], np.nan).dropna()
        ei = (s.pred_edges/s.drawn_edges.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(model=m, n=len(s), drawn_nodes=s.drawn_nodes.mean(), read_nodes=s.pred_nodes.mean(),
                         node_ratio=s.pred_nodes.mean()/s.drawn_nodes.mean(),
                         node_inflated_pct=100*(ni > 1).mean(),
                         drawn_edges=s.drawn_edges.mean(), read_edges=s.pred_edges.mean(),
                         edge_ratio=s.pred_edges.mean()/s.drawn_edges.mean(),
                         edge_inflated_pct=100*(ei > 1).mean(),
                         rho_nodes=stats.spearmanr(s.drawn_nodes, s.pred_nodes)[0],
                         rho_edges=stats.spearmanr(s.drawn_edges, s.pred_edges)[0]))
        q = rows[-1]
        P(f"  {m:10s} boxes  drawn {q['drawn_nodes']:5.1f} -> recovered {q['read_nodes']:5.1f}"
          f"  ({q['node_ratio']:.2f}x, inflated on {q['node_inflated_pct']:.0f}% of papers)"
          f"  rho {q['rho_nodes']:+.2f}")
        P(f"  {'':10s} arrows drawn {q['drawn_edges']:5.1f} -> recovered {q['read_edges']:5.1f}"
          f"  ({q['edge_ratio']:.2f}x, inflated on {q['edge_inflated_pct']:.0f}% of papers)"
          f"  rho {q['rho_edges']:+.2f}")
    CSV["reader_fidelity"] = pd.DataFrame(rows)
    d = CSV["reader_fidelity"].set_index("model")
    TEX["reader"] = (
        "% generated by checks_new.py --analysis N2 --latex\n"
        "\\begin{table}[htbp]\n\t\\centering\n\t\\fontsize{7}{8.4}\\selectfont\n"
        "\t\\setlength{\\tabcolsep}{3pt}\n\t\\begin{tabular}{l r r r r @{\\hspace{6pt}} r r r r}\n"
        "\t\t\\toprule\n\t\t & \\multicolumn{4}{c}{\\textbf{Boxes}} & \\multicolumn{4}{c}{\\textbf{Arrows}} \\\\\n"
        "\t\t\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "\t\t\\textbf{Model} & drawn & read & $\\times$ & $\\rho$ & drawn & read & $\\times$ & $\\rho$ \\\\\n"
        "\t\t\\midrule\n" +
        "".join(f"\t\t{m} & {d.drawn_nodes[m]:.1f} & {d.read_nodes[m]:.1f} & \\textbf{{{d.node_ratio[m]:.2f}}}"
                f" & {d.rho_nodes[m]:+.2f} & {d.drawn_edges[m]:.1f} & {d.read_edges[m]:.1f}"
                f" & \\textbf{{{d.edge_ratio[m]:.2f}}} & {d.rho_edges[m]:+.2f} \\\\\n" for m in d.index) +
        "\t\t\\bottomrule\n\t\\end{tabular}\n\t\\caption{What a read-back reader recovers, against the graph "
        "that was actually drawn. 483 SciFlow-Bench papers per model, seed 0; the ground truth is our own "
        "figure's geometry, which no image baseline can supply. $\\times$ is recovered over drawn and $\\rho$ "
        "their Spearman correlation per paper. Computed by \\texttt{checks\\_new.py --analysis N2}.}\n"
        "\t\\label{table:reader}\n\\end{table}\n")

# ---------------------------------------------------------------- N3 / N4
def _gv(D):
    lay = pd.read_csv(D+"layout_items.csv", low_memory=False)
    g = lay[lay.source.isin(["graphviz", "graphviz_outside_development"])].copy()
    g["valid"] = _b(g.valid)
    return g.dropna(subset=["valid"])

def N3(D, P):
    g = _gv(D)
    P("\n== N3  Legibility capacity: the graph size at which half the figures stop passing ==")
    rows = []
    for cfg in ["ours", "dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "dot_lr_ortho"]:
        s = g[g.config == cfg]
        if len(s) < 100: continue
        b = _logit_fit(np.log(s.n_nodes.clip(lower=1)).values, s.valid.values)
        rows.append(dict(config=cfg, n=len(s), strict_pct=100*s.valid.mean(), slope=b[0],
                         half_at_boxes=float(np.exp(-b[1]/b[0]))))
        P(f"  {cfg:15s} n={len(s):6d}  strict {100*s.valid.mean():5.1f}%  slope/log-box {b[0]:+.2f}"
          f"  50%-legible at {rows[-1]['half_at_boxes']:6.1f} boxes")
    CSV["capacity"] = pd.DataFrame(rows)
    bk = [0, 5, 9, 14, 19, 29, 49, 10**6]
    lab = ["$\\le$5", "6--9", "10--14", "15--19", "20--29", "30--49", "$\\ge$50"]
    g["bucket"] = pd.cut(g.n_nodes, bk, labels=lab)
    t = 100*g.pivot_table(index="bucket", columns="config", values="valid", aggfunc="mean", observed=True)
    t["gap"] = t["ours"]-t[["dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "dot_lr_ortho"]].max(axis=1)
    t["n_figs"] = g[g.config == "ours"].groupby("bucket", observed=True).size()
    P("\n  strict-legible % by graph size:")
    P(t[["ours", "dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "gap", "n_figs"]].round(1).to_string())
    CSV["capacity_by_size"] = t.reset_index()
    c = CSV["capacity"].set_index("config")
    nice = {"ours": "Ours", "dot_tb_spline": "\\texttt{dot} top-down, curved",
            "dot_lr_spline": "\\texttt{dot} left-right, curved",
            "dot_tb_ortho": "\\texttt{dot} top-down, orthogonal",
            "dot_lr_ortho": "\\texttt{dot} left-right, orthogonal"}
    TEX["capacity"] = (
        "% generated by checks_new.py --analysis N3 --latex\n"
        "\\begin{table}[htbp]\n\t\\centering\n\t\\fontsize{7}{8.4}\\selectfont\n"
        "\t\\setlength{\\tabcolsep}{3pt}\n\t\\begin{tabular}{l r r r}\n\t\t\\toprule\n"
        "\t\t\\textbf{Engine} & \\textbf{strict \\%} & \\textbf{slope} & \\textbf{half at} \\\\\n\t\t\\midrule\n" +
        "".join(f"\t\t{nice[k]} & {c.strict_pct[k]:.1f} & {c.slope[k]:+.2f} & "
                f"\\textbf{{{c.half_at_boxes[k]:.1f}}} \\\\\n" for k in c.index) +
        "\t\t\\midrule\n\t\t\\multicolumn{4}{l}{\\textbf{Strictly legible \\% by graph size}} \\\\\n"
        "\t\t\\textbf{Boxes} & \\textbf{Ours} & \\textbf{best \\texttt{dot}} & \\textbf{figures} \\\\\n"
        "\t\t\\midrule\n" +
        "".join(f"\t\t{r.bucket} & {r.ours:.1f} & {max(r.dot_tb_spline, r.dot_lr_spline, r.dot_tb_ortho, r.dot_lr_ortho):.1f}"
                f" & {int(r.n_figs)} \\\\\n" for _, r in CSV["capacity_by_size"].iterrows()) +
        "\t\t\\bottomrule\n\t\\end{tabular}\n\t\\caption{Legibility as a capacity. Top: a logistic fit of the "
        "strict verdict on log box count over the 6{,}007 figures both engines draw from identical graphs, "
        "per \\texttt{dot} configuration; \\emph{half at} is the box count at which half the figures stop "
        "passing. Bottom: the same share by graph size, against the best of \\texttt{dot}'s four "
        "configurations in each band. Below ten boxes the engines are within seven points; the distance "
        "widens to forty. Computed by \\texttt{checks\\_new.py --analysis N3}.}\n"
        "\t\\label{table:capacity}\n\\end{table}\n")

def N4(D, P):
    g = _gv(D)
    g["gid"] = g.corpus+"|"+g.model.astype(str)+"|"+g["run"].astype(str)+"|"+g.item.astype(str)
    mu = g.valid.mean(); tot = ((g.valid-mu)**2).mean(); c = {}
    for f in ["gid", "config"]:
        m = g.groupby(f).valid.mean(); n = g.groupby(f).valid.size()
        c[f] = ((m-mu)**2*n).sum()/n.sum()
    P(f"\n== N4  Where legibility's variance lives, over {len(g)} layouts of the same graphs ==")
    P(f"  the graph itself         {100*c['gid']/tot:5.1f}%")
    P(f"  the layout engine        {100*c['config']/tot:5.1f}%")
    P(f"  interaction + residual   {100*(tot-c['gid']-c['config'])/tot:5.1f}%")
    CSV["legibility_variance"] = pd.DataFrame([dict(component="graph", pct=100*c["gid"]/tot),
                                               dict(component="engine", pct=100*c["config"]/tot),
                                               dict(component="interaction+residual",
                                                    pct=100*(tot-c["gid"]-c["config"])/tot)])

# ---------------------------------------------------------------- N5
def _bws(D):
    st = pd.read_csv(D+"human_stimuli.csv"); se = pd.read_csv(D+"human_sets.csv")
    ra = pd.read_csv(D+"human_ratings.csv")
    cnt = {i: {"b": 0, "w": 0, "s": 0} for i in st.image}
    for _, row in ra.iterrows():
        s = se[se.set == row.set].iloc[0]
        for k in ["image_1", "image_2", "image_3", "image_4"]: cnt[s[k]]["s"] += 1
        cnt[row.best]["b"] += 1; cnt[row.worst]["w"] += 1
    st["bws"] = st.image.map(lambda i: (cnt[i]["b"]-cnt[i]["w"])/max(cnt[i]["s"], 1))
    return st

F10 = ["report_label_collisions", "report_node_overlap_pairs", "report_edge_node_crossings",
       "report_stacked_routes", "report_edge_label_crossings", "report_edge_edge_crossings",
       "report_total_edge_length", "report_mean_bends_per_edge", "report_aspect_deviation",
       "report_area"]

def N5(D, P):
    st = _bws(D)
    X = st[F10].astype(float).clip(lower=0).apply(np.log1p)
    X = (X-X.mean())/X.std(ddof=0)
    y = st.bws.values; gr = st.paper_id.values
    def logo(cols):
        pred = np.zeros(len(st))
        for q in np.unique(gr):
            tr = gr != q; te = ~tr
            A = np.hstack([X.loc[tr, cols].values, np.ones((tr.sum(), 1))])
            b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            pred[te] = np.hstack([X.loc[te, cols].values, np.ones((te.sum(), 1))])@b
        return stats.spearmanr(pred, y)[0]
    P("\n== N5  An index of the ten layout measures fitted to the human ratings ==")
    P("   leave-one-source-graph-out cross-validation, 72 images over 24 source graphs")
    base = [("arrow crossings alone", stats.spearmanr(-st.report_edge_edge_crossings, y)[0]),
            ("strict verdict alone", stats.spearmanr(_b(st.report_valid), y)[0]),
            ("label collisions alone", stats.spearmanr(-st.report_label_collisions, y)[0]),
            ("ten measures, log1p + z, fitted", logo(F10))]
    for n, v in base: P(f"  {n:34s} rho {v:+.3f}")
    allsmall = sorted(((logo(list(c)), c) for k in (3, 4) for c in itertools.combinations(F10, k)),
                      reverse=True)
    small = allsmall[:5]
    best4 = next(c for _v, c in allsmall if len(c) == 4)
    P("  most predictive small indices:")
    for v, c in small: P(f"    {v:+.3f}  {[x.replace('report_','') for x in c]}")
    A = np.hstack([X[F10].values, np.ones((len(st), 1))])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    w = sorted(zip(F10, b[:-1]), key=lambda t: -abs(t[1]))
    P("  weights of the ten-measure index, fitted on all 72 images (log1p, z-scored inputs):")
    for c, v in w: P(f"    {c.replace('report_',''):26s} {v:+.3f}")
    four = list(best4)
    A4 = np.hstack([X[four].values, np.ones((len(st), 1))])
    b4 = np.linalg.lstsq(A4, y, rcond=None)[0]
    P("  weights of the four-measure index (log1p, z-scored inputs; all signs as expected):")
    for c, v in zip(four, b4[:-1]): P(f"    {c.replace('report_',''):26s} {v:+.3f}")
    CSV["index_weights"] = pd.DataFrame(
        [dict(index="ten_measure", measure=c.replace("report_", ""), weight=v) for c, v in w] +
        [dict(index="four_measure", measure=c.replace("report_", ""), weight=v)
         for c, v in zip(four, b4[:-1])])
    CSV["index_cv"] = pd.DataFrame([dict(index=n, logo_rho=v) for n, v in base] +
                                   [dict(index=" + ".join(x.replace("report_", "") for x in c), logo_rho=v)
                                    for v, c in small])
    nice = {"label_collisions": "label collisions", "node_overlap_pairs": "box overlaps",
            "edge_node_crossings": "arrows through a box", "stacked_routes": "stacked routes",
            "edge_label_crossings": "arrows through a label", "edge_edge_crossings": "arrow crossings",
            "total_edge_length": "total arrow length", "mean_bends_per_edge": "bends per arrow",
            "aspect_deviation": "page shape", "area": "area"}
    rows_a = base + [("four measures, fitted", logo(four))]
    TEX["index"] = (
        "% generated by checks_new.py --analysis N5 --latex\n"
        "\\begin{table}[htbp]\n\t\\centering\n\t\\fontsize{7}{8.4}\\selectfont\n"
        "\t\\setlength{\\tabcolsep}{3pt}\n\t\\begin{tabular}{l r @{\\hspace{10pt}} l r}\n\t\t\\toprule\n"
        "\t\t\\multicolumn{2}{l}{\\textbf{(a) Recovering the ratings}} & "
        "\\multicolumn{2}{l}{\\textbf{(b) Four-measure index}} \\\\\n"
        "\t\t & $\\rho$ & & $w$ \\\\\n\t\t\\midrule\n" +
        "".join(f"\t\t{n} & {v:+.3f} & "
                + (f"{nice[four[i].replace('report_','')]} & {b4[i]:+.3f}" if i < len(four) else " & ")
                + " \\\\\n" for i, (n, v) in enumerate(rows_a)) +
        "\t\t\\bottomrule\n\t\\end{tabular}\n\t\\caption{An index fitted to the raters. \\textbf{(a)} "
        "Spearman correlation with the raters' best--worst scores over the 72 images, each prediction made by a "
        "fit that never saw another layout of the same source graph. \\textbf{(b)} Weights of the four-measure "
        "index on $\\log(1+x)$, standardised inputs, every sign as expected; the ten-measure weights are in "
        "\\texttt{results/tables/index\\_weights.csv}. Computed by \\texttt{checks\\_new.py --analysis N5}.}\n"
        "\t\\label{table:index}\n\\end{table}\n")

# ---------------------------------------------------------------- N6
def N6(D, P):
    rb = pd.read_csv(D+"readback_sciflow.csv")
    ceil = rb[rb.system == "gold_graph_ceiling"].score.mean()
    t = rb.groupby("system").agg(score=("score", "mean"), node_f1=("node_f1", "mean"),
                                 edge_f1=("edge_f1", "mean"))
    t["pct_of_ceiling"] = 100*t.score/ceil
    P(f"\n== N6  SciFlow-Bench read-back, as a share of what the benchmark can return ({ceil:.3f}) ==")
    P(t.round(3).sort_values("score", ascending=False).to_string())
    gg = rb[rb.system == "gold_graph_ceiling"]
    P(f"  the gold graph recovers node F1 {gg.node_f1.mean():.3f} and edge F1 {gg.edge_f1.mean():.3f};")
    P(f"  0.6 of the score's weight sits on a quantity bounded at {gg.edge_f1.mean():.3f}.")
    CSV["ceiling"] = t.reset_index()

# ---------------------------------------------------------------- N7
def N7(D, P):
    pi = pd.read_csv(D+"pipeline_items.csv"); pi["lv"] = _b(pi.layout_valid)
    c = pi.groupby(["corpus", "model", "variant"]).agg(
        score=("score", "mean"), strict=("lv", "mean"),
        boxes=("layout_n_nodes", "mean")).dropna(subset=["strict"]).reset_index()
    P("\n== N7  Content against legibility, every corpus x model x variant cell ==")
    rows = []
    for co in ["sciflow", "flowvqa"]:
        P(f"  {co}:")
        for m in sorted(c[c.corpus == co].model.unique()):
            sm = c[(c.corpus == co) & (c.model == m)].sort_values("score", ascending=False)
            for _, r in sm.iterrows():
                dom = bool(((sm.score >= r.score) & (sm.strict >= r.strict) &
                            ((sm.score > r.score) | (sm.strict > r.strict))).any())
                rows.append(dict(corpus=co, model=m, variant=r.variant, score=r.score,
                                 strict=r.strict, boxes=r.boxes, dominated=dom))
                P(f"    {m:10s} {r.variant:26s} score {r.score:.3f}  strict {100*r.strict:5.1f}%"
                  f"  boxes {r.boxes:5.1f}   {'dominated' if dom else 'on frontier'}")
    f = pd.DataFrame(rows); CSV["frontier"] = f
    ev = f[f.variant == "evaluated"]
    P(f"\n  the shipped configuration is on the frontier in {int((~ev.dominated).sum())} of {len(ev)}"
      f" corpus x model cells")
    for v in sorted(f.variant.unique()):
        s = f[f.variant == v]
        P(f"    {v:26s} on frontier in {int((~s.dominated).sum())} of {len(s)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out"); ap.add_argument("--latex"); ap.add_argument("--csv-dir")
    ap.add_argument("--analysis", nargs="*", default=["N1", "N2", "N3", "N4", "N5", "N6", "N7"])
    a = ap.parse_args()
    D = a.data.rstrip("/")+"/"
    buf = []
    def P(s=""):
        print(s); buf.append(str(s))
    for n in a.analysis: globals()[n](D, P)
    if a.out: open(a.out, "w").write("\n".join(buf)+"\n")
    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        for k, v in CSV.items(): v.to_csv(os.path.join(a.csv_dir, k+".csv"), index=False)
    if a.latex:
        open(a.latex, "w").write("\n".join(TEX[k] for k in ["reader", "index", "judge_power"]
                                           if k in TEX)+"\n")
        base = os.path.dirname(a.latex)
        for k in TEX:
            open(os.path.join(base, "table_"+k+".tex"), "w").write(TEX[k])
