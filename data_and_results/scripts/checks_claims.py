#!/usr/bin/env python3
"""What the released data say about each claim this paper makes.

    python3 checks_claims.py                               # all five
    python3 checks_claims.py --analysis K1 K2
    python3 checks_claims.py --latex ../results/tables_claims.tex
    python3 checks_claims.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| K1 guarantee | what the lenient invariant is worth next to Graphviz, not next to the model | 5.1, Table S3 |
| K2 raters    | how much of the rater result is the model-placed condition | 5.3, Table S5 |
| K3 floor     | where the SciFlow-Bench score sits between its floor and its ceiling | 5.4, Table S6 |
| K4 resolve   | every reported difference against what its instrument can resolve | 6, 7 |
| K5 register  | every number in the abstract: instrument, pre-specified or not, bound | 6, Table S4 |

All five read the per-item CSVs in data/ and the tables written by the other
checks_*.py scripts. None adds a run, a condition or a model call. K1, K2 and K3
re-express measurements the other scripts already compute over a different
comparison set, which is the whole point of them: a guarantee is worth what the
alternative anyone would actually use does not give, a rater result is worth what
survives dropping the condition nobody would ship, and a score is worth its
distance from both ends of its own range.
"""
import argparse, os, re
import pandas as pd, numpy as np
from scipy import stats

CSV, TEX = {}, {}

LENIENT = ["node_overlap_pairs", "edge_node_crossings", "route_reversals"]
STRICT_EXTRA = ["label_collisions", "edge_label_crossings", "stacked_routes"]


def _b(s):
    return s.astype(str).map({"True": 1, "False": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})


def _tex(v):
    """Plain text from the register into LaTeX. The CSV keeps the plain form."""
    out = v.replace("%", r"\%").replace("\u00a75.6", r"\S\ref{sec:validity}")
    out = re.sub(r"\|rho\|<=\s*([0-9.]+)", r"$|\\rho|\\le\1$", out)
    out = re.sub(r"rho\s*=\s*([0-9.]+)", r"$\\rho=\1$", out)
    out = re.sub(r"p\s*<\s*1e-([0-9]+)", "\x00p\\1\x00", out)
    out = out.replace("<=", r"$\le$").replace("<", "$<$")
    out = re.sub(r"\x00p([0-9]+)\x00", r"$p<10^{-\1}$", out)
    return out


def _eng(c):
    if c.startswith("dot_"):
        return "\\texttt{dot} "+c[4:].replace("_", " ")
    if c.startswith("placement: "):
        return "placement: "+c[11:]
    return c


def _p(v):
    if v >= 0.001:
        return f"{v:.3f}"
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _boot(d, rng, n=10000):
    bs = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return np.percentile(bs, [2.5, 97.5])


def _paper_keyed(df, col="score"):
    """One value per paper, keyed as the paper keys it.

    SciFlow-Bench lists eleven arXiv papers under two area folders each, so an
    item id alone is not a paper: keying on it merges two different papers and
    drops the corpus from 483 to 472. Every comparison in the paper keys on
    (domain, item), which is what make_table_systems.py does for Table 2 and Table S7.
    """
    s = df.groupby(["domain", "item"])[col].mean()
    s.index = pd.MultiIndex.from_tuples(list(s.index), names=["domain", "item"])
    return s


def _rank_biserial(d):
    """Matched-pairs rank-biserial correlation, zeros dropped."""
    nz = np.asarray([x for x in d if x != 0], dtype=float)
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    return float(2*ranks[nz > 0].sum()/ranks.sum() - 1)


# ---------------------------------------------------------------- K1
def K1(D, P):
    """The lenient verdict is a guarantee. Against what?"""
    lay = pd.read_csv(D+"layout_items.csv", low_memory=False)
    g = lay[lay.source.isin(["graphviz", "graphviz_outside_development"])].copy()
    for c in LENIENT+STRICT_EXTRA+["valid"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["valid"])
    g["lenient"] = (g[LENIENT] == 0).all(axis=1)

    P("\n== K1  What the lenient guarantee is worth next to the engine anyone would use ==")
    P("\n(a) identical graphs, our engine and every Graphviz configuration")
    rows = []
    order = ["ours", "dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "dot_lr_ortho"]
    for c in [c for c in order if c in set(g.config)]:
        s = g[g.config == c]
        rows.append(dict(config=c, n=len(s), lenient_pct=100*s.lenient.mean(),
                         strict_pct=100*s.valid.mean(),
                         box_overlaps=s.node_overlap_pairs.mean(),
                         arrows_through_box=s.edge_node_crossings.mean(),
                         route_reversals=s.route_reversals.mean(),
                         label_collisions=s.label_collisions.mean()))
    t = pd.DataFrame(rows)
    P(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # the placement ablation, for the same three counts
    pl = pd.read_csv(D+"placement_items.csv", low_memory=False)
    pl = pl.dropna(subset=["layout_valid"])
    pl["lenient"] = (pl[["layout_"+c for c in LENIENT]] == 0).all(axis=1)
    prows = []
    for c in sorted(pl.condition.unique()):
        s = pl[pl.condition == c]
        prows.append(dict(config="placement: "+c, n=len(s), lenient_pct=100*s.lenient.mean(),
                          strict_pct=100*_b(s.layout_valid.astype(str)).mean(),
                          box_overlaps=s.layout_node_overlap_pairs.mean(),
                          arrows_through_box=s.layout_edge_node_crossings.mean(),
                          route_reversals=s.layout_route_reversals.mean(),
                          label_collisions=s.layout_label_collisions.mean()))
    pt = pd.DataFrame(prows)
    P("\n(b) the same extracted graph, our code or the model emitting the coordinates")
    P(pt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    full = pd.concat([t, pt], ignore_index=True)
    CSV["guarantee_in_context"] = full

    best = t[t.config != "ours"].lenient_pct.max()
    P(f"\n  Reading: the lenient verdict holds on 100% of our figures and on {best:.1f}% of")
    P("  Graphviz's best configuration, which also never overlaps a box; the gap the three")
    P("  lenient counts open is therefore against a model placing boxes, not against an")
    P("  established layout engine. The verdict that separates the two engines is the strict")
    P(f"  one: {t[t.config=='ours'].strict_pct.iloc[0]:.1f}% against "
      f"{t[t.config!='ours'].strict_pct.max():.1f}%.")

    TEX["guarantee"] = (
        "% generated by checks_claims.py --analysis K1 --latex\n"
        "\\begin{table}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n"
        "\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{l r r r r r r}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{\\textbf{The three lenient counts}} & "
        "\\multicolumn{2}{c}{\\textbf{Verdict, \\%}} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
        "\\textbf{Engine} & $n$ & \\makecell{\\textbf{Box}\\\\\\textbf{overl.}} & "
        "\\makecell{\\textbf{Thr.}\\\\\\textbf{box}} & \\makecell{\\textbf{Route}\\\\"
        "\\textbf{rev.}} & \\textbf{Lenient} & \\textbf{Strict} \\\\\n\\midrule\n"
        + "".join(
            f"{_eng(r.config)} & {r.n} & "
            f"{r.box_overlaps:.2f} & {r.arrows_through_box:.2f} & {r.route_reversals:.2f} & "
            f"{r.lenient_pct:.1f} & {r.strict_pct:.1f} \\\\\n"
            for r in full.itertuples())
        + "\\bottomrule\n\\end{tabular}}\n\\caption{The lenient guarantee in context, over the "
          "identical graphs both engines draw. Graphviz never overlaps a box either and its "
          "spline configurations satisfy all three lenient criteria on about 99\\% of figures, "
          "so the three criteria our schema makes invariant are nearly free for an established "
          "layout engine as well; what separates the two is the \\emph{strict} verdict. The "
          "6.5\\% is what a model does with the same graph, which is a measurement of the "
          "model, not a layout baseline (\\S\\ref{sec:graphviz}).}\n"
          "\\label{table:guarantee}\n\\end{table}\n")


# ---------------------------------------------------------------- K2
def _bws(D):
    st = pd.read_csv(D+"human_stimuli.csv")
    se = pd.read_csv(D+"human_sets.csv")
    ra = pd.read_csv(D+"human_ratings.csv")
    cnt = {i: {"b": 0, "w": 0, "s": 0} for i in st.image}
    for _, row in ra.iterrows():
        s = se[se.set == row.set].iloc[0]
        for k in ["image_1", "image_2", "image_3", "image_4"]:
            cnt[s[k]]["s"] += 1
        cnt[row.best]["b"] += 1
        cnt[row.worst]["w"] += 1
    st["bws"] = st.image.map(lambda i: (cnt[i]["b"]-cnt[i]["w"])/max(cnt[i]["s"], 1))
    st["n_shown"] = st.image.map(lambda i: cnt[i]["s"])
    st["n_best"] = st.image.map(lambda i: cnt[i]["b"])
    st["n_worst"] = st.image.map(lambda i: cnt[i]["w"])
    return st, se, ra


def K2(D, P):
    """How much of the rater result is the condition nobody would ship."""
    st, se, ra = _bws(D)
    cond = st.set_index("image").condition.to_dict()
    val = st.set_index("image").report_valid.astype(str).map(
        {"True": 1, "False": 0, "1": 1, "0": 0}).to_dict()

    P("\n== K2  The rater result, with and without the model-placed condition ==")
    P("\n(a) where the choices go")
    b = ra.best.map(cond).value_counts()
    w = ra.worst.map(cond).value_counts()
    rows = []
    for c in ["ours", "graphviz", "model"]:
        rows.append(dict(condition=c, n_images=int((st.condition == c).sum()),
                         best_picks=int(b.get(c, 0)), worst_picks=int(w.get(c, 0)),
                         bws=st[st.condition == c].bws.mean(),
                         strict_pct=100*st[st.condition == c].report_valid.astype(str).map(
                             {"True": 1, "False": 0}).mean()))
    ct = pd.DataFrame(rows)
    P(ct.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    P(f"\n  {int(w.get('model',0))} of {len(ra)} 'least legible' choices are the model-placed"
      " image.")

    P("\n(b) the odds ratio of the strict verdict, on every panel and on the two engines alone")
    orows = []
    for label, drop in [("all three conditions", ()), ("the two engines only", ("model",))]:
        bv = sum(1 for x in ra.best if cond[x] not in drop and val[x] == 1)
        bn = sum(1 for x in ra.best if cond[x] not in drop and val[x] == 0)
        wv = sum(1 for x in ra.worst if cond[x] not in drop and val[x] == 1)
        wn = sum(1 for x in ra.worst if cond[x] not in drop and val[x] == 0)
        table = np.array([[bv, bn], [wv, wn]])
        orr = (bv*wn)/(bn*wv) if bn and wv else np.nan
        p = stats.fisher_exact(table)[1]
        orows.append(dict(sample=label, images=72-24*len(drop),
                          best_pass=bv, best_fail=bn, worst_pass=wv, worst_fail=wn,
                          odds_ratio=orr, p=p))
        P(f"  {label:22s} best {bv:3d} pass / {bn:3d} fail, worst {wv:3d} pass / {wn:3d} fail"
          f"  OR={orr:.2f}  p={p:.2g}")
    ot = pd.DataFrame(orows)

    P("\n(c) the two engines, per source graph")
    piv = st.pivot_table(index="paper_id", columns="condition", values="bws")
    d = (piv["ours"]-piv["graphviz"]).dropna()
    rng = np.random.default_rng(0)
    lo, hi = _boot(d.values, rng)
    P(f"  ours ahead on {int((d>0).sum())} of {len(d)} graphs, behind on {int((d<0).sum())},"
      f" tied on {int((d==0).sum())}")
    P(f"  mean best--worst difference {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    P(f"  raters picked Graphviz most legible {int(b.get('graphviz',0))} times against"
      f" {int(b.get('ours',0))} for ours")

    CSV["rater_sensitivity"] = ot
    CSV["rater_conditions"] = ct

    P("\n  Reading: the strict verdict separates a broken layout from a drawn one, which is")
    P("  what the odds ratio over the full panel measures. Between the two engines it does")
    P("  not predict the raters at all, and the raters do not prefer our engine. The study")
    P("  therefore validates the counts as a detector of unusable figures, and settles")
    P("  nothing about either engine.")

    TEX["ratersens"] = (
        "% generated by checks_claims.py --analysis K2 --latex\n"
        "\\begin{table}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{l r r r r}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Most legible} & \\textbf{Least legible} & "
        "\\textbf{OR} & $p$ \\\\\n"
        " & \\emph{pass / fail} & \\emph{pass / fail} & & \\\\\n\\midrule\n"
        + "".join(
            f"{r.sample} & {r.best_pass} / {r.best_fail} & {r.worst_pass} / {r.worst_fail} & "
            f"{r.odds_ratio:.2f} & {_p(r.p)} \\\\\n" for r in ot.itertuples())
        + "\\bottomrule\n\\end{tabular}\n"
          "\\caption{The strict verdict against the raters' choices, over all 216 choices and "
          "over the choices that fall on the 48 images the two engines drew. The verdict tells "
          "an unusable figure from a drawn one; between two engines whose defect counts barely "
          "vary it carries no signal, and the raters put \\texttt{dot} slightly ahead "
          "(\\S\\ref{sec:human}).}\n\\label{table:ratersens}\n\\end{table}\n")


# ---------------------------------------------------------------- K3
def K3(D, P):
    """Where the SciFlow-Bench score sits between its floor and its ceiling."""
    p = pd.read_csv(D+"pipeline_items.csv", low_memory=False)
    d = pd.read_csv(D+"dot_items.csv", low_memory=False)
    r = pd.read_csv(D+"readback_sciflow.csv").dropna(subset=["score"])
    ceiling = r[r.system == "gold_graph_ceiling"].score.mean()

    P("\n== K3  The SciFlow-Bench score between its floor and its ceiling ==")
    P(f"\n  the gold graphs reach {ceiling:.3f} under this protocol")
    P("\n(a) how often each half of the score is exactly zero")
    rows = []
    for corpus in ["sciflow", "flowvqa"]:
        ours = (p[(p.variant == "evaluated") & (p.corpus == corpus)]
                .groupby(["model", "domain", "item"])[["node_f1", "edge_f1", "score"]]
                .mean().reset_index())
        dot = (d[(d.corpus == corpus) & (d.variant == "dot")]
               .groupby(["model", "domain", "item"])[["node_f1", "edge_f1", "score"]]
               .mean().reset_index())
        for name, s in [("ours", ours), ("dot", dot)]:
            for m in ["Qwen", "Gemma", "Ministral"]:
                x = s[s.model == m]
                rows.append(dict(corpus=corpus, system=name, model=m, n=len(x),
                                 score=x.score.mean(),
                                 node_f1=x.node_f1.mean(), edge_f1=x.edge_f1.mean(),
                                 edge_zero_pct=100*(x.edge_f1 == 0).mean(),
                                 node_zero_pct=100*(x.node_f1 == 0).mean()))
    t = pd.DataFrame(rows)
    P(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    CSV["score_floor"] = t

    sf = t[(t.corpus == "sciflow")]
    P(f"\n  On SciFlow-Bench the edge half of the score, which carries 0.6 of its weight, is")
    P(f"  exactly zero on {sf.edge_zero_pct.min():.0f}--{sf.edge_zero_pct.max():.0f}% of items"
      f" for every system compared, ours included, and its edge F1 stays at"
      f" {sf.edge_f1.min():.2f}--{sf.edge_f1.max():.2f}.")
    P("  The share of the ceiling belongs to the symmetric read-back of (b), which is the"
      "\n  protocol that ceiling is measured under. On FlowVQA the same half is zero on")
    P(f"  {t[(t.corpus=='flowvqa')].edge_zero_pct.max():.0f}% of items at"
      " most, which is where the content comparison has room to resolve anything.")

    P("\n(b) the symmetric read-back: every figure drawn and read by the same reader")
    rng = np.random.default_rng(0)
    base = {m: _paper_keyed(r[(r.system == "ours") & (r.model == m)])
            for m in ["Qwen", "Gemma", "Ministral"]}
    others = ["gemini3", "gemini2.5", "seedream", "graphviz", "qwen", "sdxl", "pixart"]
    prows = []
    for m in ["Qwen", "Gemma", "Ministral"]:
        for s in others:
            b = _paper_keyed(r[r.system == s])
            idx = sorted(set(base[m].index) & set(b.index))
            x = base[m][idx].values
            y = b[idx].values
            dif = x-y
            lo, hi = _boot(dif, rng)
            pv = float(stats.wilcoxon(dif).pvalue)
            prows.append(dict(model=m, against=s, n=len(idx), diff=dif.mean(), lo=lo, hi=hi,
                              p=pv, r=_rank_biserial(dif),
                              diff_pct_ceiling=100*dif.mean()/ceiling))
    # the paper's protocol: Holm over the seven shipped systems within a model,
    # the verdict following the sign of the rank-biserial r at alpha = 0.05
    for m in ["Qwen", "Gemma", "Ministral"]:
        block = [row for row in prows if row["model"] == m]
        for row, ph in zip(block, _holm([row["p"] for row in block])):
            row["p_holm"] = ph
            row["verdict"] = ("tied" if ph >= 0.05
                              else "ahead" if row["r"] > 0 else "behind")
    for row in prows:
        P(f"  ours+{row['model']:<10s} vs {row['against']:<10s} n={row['n']:3d}"
          f" d={row['diff']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"
          f" ({row['diff_pct_ceiling']:+.1f}% of ceiling)"
          f" p_holm={row['p_holm']:.2g}  {row['verdict']}")
    pt = pd.DataFrame(prows)
    CSV["readback_pairs"] = pt
    gv = pt[pt.against == "graphviz"].set_index("model")
    P("\n  Reading: against the DOT convention this pipeline replaces, the symmetric read-back")
    P("  puts " + ", ".join(
        f"{m} {gv.loc[m, 'diff_pct_ceiling']:+.1f}% of the ceiling ({gv.loc[m, 'verdict']})"
        for m in ["Ministral", "Gemma", "Qwen"]) + ".")
    P("  The three margins are small beside the 0.611 the gold graphs reach, and six of the")
    P("  21 differences do not clear what their own samples resolve (K8), so the ordering is")
    P("  reported as standing on this instrument and not as a content advantage.")
    for m in ["Qwen", "Gemma", "Ministral"]:
        g = pt[pt.model == m]
        P(f"  ours + {m}: ahead of {(g.verdict == 'ahead').sum()} of the seven shipped systems,"
          f" tied with {(g.verdict == 'tied').sum()}, behind {(g.verdict == 'behind').sum()}")

    TEX["floor"] = (
        "% generated by checks_claims.py --analysis K3 --latex\n"
        "\\begin{table}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n"
        "\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{l l r r r r}\n\\toprule\n"
        "\\textbf{Corpus} & \\textbf{System} & \\textbf{Score} & "
        "\\textbf{Node F1} & \\textbf{Edge F1} & \\makecell{\\textbf{Edge F1}\\\\"
        "\\textbf{$=0$, \\%}} \\\\\n\\midrule\n"
        + "".join(
            f"{'SciFlow' if r.corpus=='sciflow' else 'FlowVQA'} & {r.system}+{r.model} & "
            f"{r.score:.3f} & "
            f"{r.node_f1:.3f} & {r.edge_f1:.3f} & {r.edge_zero_pct:.0f} \\\\\n"
            for r in t.itertuples())
        + "\\bottomrule\n\\end{tabular}}\n"
          "\\caption{Both halves of the content metric, and how much of its range is in use. "
          "On SciFlow-Bench the edge half---0.6 of the score's weight---is exactly zero for most "
          "items of every system compared, so differences there are differences between two "
          "systems near the floor of the instrument. FlowVQA is where the metric has room "
          "(\\S\\ref{sec:main}).}\n\\label{table:floor}\n\\end{table}\n")


# ---------------------------------------------------------------- K4
def K4(D, P):
    """Every reported difference against what its instrument can resolve."""
    T = D.replace("data/", "results/tables/")
    res = pd.read_csv(T+"resolution.csv")
    sg = pd.read_csv(T+"sign_errors.csv")
    jm = pd.read_csv(T+"judge_mdd.csv")
    P("\n== K4  Every reported difference against what its instrument can resolve ==")

    mde = {c: res[(res.corpus == c) & res.mde.notna()].set_index("n").mde.to_dict()
           for c in res.corpus.unique()}
    P("\n(a) content, paired per item, against the paired resolution of the same sample")
    rows = []
    for _, r in sg.iterrows():
        full = max(mde[r.corpus])
        rows.append(dict(corpus=r.corpus, model=r.model, instrument="benchmark score",
                         difference=r.true, unpaired_mde_50=mde[r.corpus][50.0],
                         sign_error_pct_50=100*r.wrong50,
                         sign_error_pct_full=100*float(
                             res[(res.corpus == r.corpus) & (res.model == r.model)]
                             .wrongfull.iloc[0])))
    for _, r in jm.iterrows():
        rows.append(dict(corpus="paperbanana", model=r.model, instrument="judge",
                         difference=np.nan, unpaired_mde_50=r.unpaired_50,
                         sign_error_pct_50=np.nan, sign_error_pct_full=np.nan))
    t = pd.DataFrame(rows)
    P(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    CSV["claim_resolution"] = t
    worst = t.dropna(subset=["sign_error_pct_full"]).sort_values(
        "sign_error_pct_full", ascending=False).iloc[0]
    P(f"\n  Reading: the largest sign-error rate at the full sample is"
      f" {worst.sign_error_pct_full:.1f}% ({worst.corpus}, {worst.model}),")
    P("  so a difference that small is reported as tied and read as no evidence either way.")
    P("  Differences we do call are those whose bootstrap interval clears zero at the full")
    P("  sample; none of them is claimed to hold at a sample of fifty items.")


# ---------------------------------------------------------------- K5
def K5(D, P):
    """Every number in the abstract: instrument, status, bound.

    This list is the paper's Table S4, row for row, so the release and the paper
    cannot disagree about what is confirmatory and what bounds each claim.
    """
    reg = [
        ("27,493 figures our engine laid out satisfy the three lenient criteria",
         "geometry", "confirmatory",
         "exact count over the figures our layout code drew, no test; the 895 model-placed figures of the placement ablation are not covered by it (D1); Graphviz reaches 99.1% on the same graphs (K1)"),
        ("Graphviz clears the same three on 99.1% of identical graphs",
         "geometry", "confirmatory",
         "K1, paired on identical graphs; bounds what the invariant is worth"),
        ("two of the three hold with our layout stages off",
         "geometry", "exploratory",
         "A3, ablation read after the test runs; the third needs the router"),
        ("model placement leaves 6.5% lenient and buys no content",
         "geometry + score", "confirmatory",
         "pre-specified ablation; paired per item"),
        ("86.9% strict against 54.0% for dot's best single configuration",
         "geometry", "confirmatory",
         "Holm-corrected, identical graphs, every setting's 95% interval clear of zero over source graphs (I2); not a matched comparison -- our layout is the best of several variants chosen by these counts and dot's is one fixed configuration (A6)"),
        ("matched, the strict difference is 21 points: both engines choosing, or neither",
         "geometry", "exploratory",
         "A6, specified after the test runs; +21.0 points, 95% CI [+19.5, +22.6] resampling the 1,064 source graphs, cluster permutation p<10e-4 (I1); this is the difference the paper reports"),
        ("the difference is the two label criteria; dot leads by 2.2-2.8 without them",
         "geometry", "exploratory",
         "A2/A6, specified after the test runs; -2.2 points, 95% CI [-2.9, -1.6] over source graphs (I1), and the reversal widens when the selection effort is matched; 4 of the 6 criteria survive Holm within their family (K7)"),
        ("it widens with size, 15 points at 8 boxes to 41 above 30",
         "geometry", "exploratory",
         "A5, size bins read after the test runs; all five bins survive Holm within their family (K7)"),
        ("an arrow across a label is the only defect left in 62% of our residual failures and 47% of dot's",
         "geometry", "exploratory",
         "A7, specified after the test runs; 95% CI [88.1,92.7] for its presence in our failures, resampling source graphs; over the 6,007 graphs both engines drew, and the three lenient criteria never fail for our engine (D1)"),
        ("half our figures pass at 53 boxes against 23",
         "geometry", "exploratory",
         "N3, logistic fit specified after the test runs; 95% CI [49, 57] against [22, 24] over source graphs (I6)"),
        ("content: ahead in 3 of 6 settings where the metric has room, and in no setting behind on both measurements",
         "benchmark score + judge", "confirmatory",
         "Holm-corrected; the three SciFlow-Bench score comparisons are at the instrument's floor (edge F1 0.03, zero on 60-76% of items, 16-20% of the ceiling) and no content advantage is read from them (K3); tied verdicts bounded in 7"),
        ("content against the shipped systems: ahead of five of the seven under the symmetric read-back, tied with one, behind Gemini 3 Pro Image",
         "score + reader", "exploratory",
         "paired Wilcoxon, Holm-corrected over the seven systems within a model; 6 of the 21 differences do not clear the 0.009-0.017 their samples resolve at 80% power (K8), the column reaches 17.4% of the 0.611 ceiling, and the reader behind it returns 1.21-1.30 boxes and 1.66-1.85 arrows per one drawn (N2), so it bounds standing on this instrument and not a content advantage"),
        ("no instrument tracks geometry at fixed size, |rho|<=0.11",
         "all three", "exploratory",
         "C1/C4/C5, correlations within our own figures; the same holds pooled over the placement ablation, where the strict share runs from 4% to 77%"),
        ("the read-back reader returns 1.21-1.30 boxes and 1.66-1.85 arrows per one drawn",
         "read-back reader", "exploratory",
         "N2, descriptive; ground truth is our own geometry; 6 of the 21 read-back differences do not clear their own resolution and none decides a verdict (K8)"),
        ("its error is not removable by calibration: fitted slope 0.26-0.45, and a corrected count beats the corpus mean by about one arrow",
         "read-back reader", "exploratory",
         "V6, specified after the test runs; fitted on half the papers and scored on the other half, one reader and one corpus; the coefficients do not transfer between models' figures, so the statement is about this reader on these figures and the procedure, not the numbers, is what carries"),
        ("a reversed arrow is an error node-and-edge F1 merges with a missing relation: 2.1% of papers and 13.5% of prose items carry one",
         "benchmark score", "exploratory",
         "E4/V5, specified after the test runs; 95% CI [1.7, 2.5] and [11.9, 15.1] over items; measurable only on predicted graphs, so it is reported for our own extractor and not for any baseline; the credit at stake is +0.001 to +0.010 of edge F1"),
        ("3% of judge variance is the system it judges",
         "judge", "exploratory",
         "N1, variance decomposition"),
        ("changing the judge moves a result by up to 22 points",
         "judge", "confirmatory",
         "second judge fixed before the runs; R1; reported as a range; the per-dimension signs are not stable across the two judges"),
        ("the verdict is one dimension: conciseness carries both losses, and its sign is the only one a change of judge keeps",
         "judge", "exploratory",
         "V7, specified after the test runs; paired per item over the 292 cases and Holm-corrected inside the four dimensions of one judge and model; two judges on one corpus, and four dimensions emitted in one answer are not four independent measurements"),
        ("the strict verdict predicts rater choices at OR 10.4 (10.9 unstratified)",
         "human raters", "exploratory",
         "D5; 0.52 on the two engines alone (K2), so it detects an unusable figure and does not rank two drawn layouts"),
        ("at the 8-20 boxes rated it separates the engines not at all",
         "human raters", "exploratory",
         "K2/A5; dot collides 0.04 labels per figure at those sizes"),
        ("a seed moves a setting's strict share by at most 4.7 points and an individual figure's verdict on up to 54% of them",
         "geometry", "exploratory",
         "V8, specified after the test runs; three seeds of one pipeline on three corpora, so it bounds these counts under this extractor; the rank link to what one extra sample buys is six settings (rho=+0.83, p=0.042)"),
        ("label collisions track the raters at rho=0.75 over the three conditions, and no defect count does over the 48 images the two engines drew",
         "human raters", "exploratory",
         "N5/R5/K6, leave-one-graph-out, 24 graphs, 3 raters; the rho=0.75 separates the model-placed condition from the drawn ones, and none of the 16 measures that vary among the 48 engine-drawn images survives Holm within that family (K7)"),
    ]
    t = pd.DataFrame(reg, columns=["claim", "instrument", "status", "bound"])
    P("\n== K5  Every number in the abstract ==")
    P(t.to_string(index=False))
    CSV["claim_register"] = t
    n_c = int((t.status == "confirmatory").sum())
    P(f"\n  {n_c} of {len(t)} are pre-specified; the rest are descriptive and marked as such")
    P("  in the abstract and in the text that reports them. K7 corrects each descriptive")
    P("  family of tests inside its own family, which is what these rows are bounded by.")

    TEX["claims"] = (
        "% generated by checks_claims.py --analysis K5 --latex\n"
        "\\begin{table*}[t]\n\\centering\n\\fontsize{7.5}{9}\\selectfont\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{p{0.33\\textwidth} l l p{0.33\\textwidth}}\n\\toprule\n"
        "\\textbf{Claim} & \\textbf{Instrument} & \\textbf{Status} & \\textbf{What bounds it}"
        " \\\\\n\\midrule\n"
        + "".join(
            f"{_tex(r.claim)} & {r.instrument} & {r.status} & {_tex(r.bound)} \\\\\n"
            for r in t.itertuples())
        + "\\bottomrule\n\\end{tabular}\n"
          "\\caption{Every number in the abstract, the instrument it comes from, whether it was "
          "fixed before the test runs or specified after them, and what bounds it. Only the "
          "confirmatory rows rest on a corrected test; the exploratory rows are descriptive and "
          "are reported as such throughout (\\S\\ref{sec:protocol}, \\S\\ref{sec:validity}).}\n"
          "\\label{table:claims}\n\\end{table*}\n")


def K6(D, P):
    """Which criterion tracks the raters once the broken condition is dropped."""
    st, se, ra = _bws(D)
    cols = [c for c in st.columns
            if c.startswith("report_") and c != "report_valid"
            and pd.api.types.is_numeric_dtype(st[c]) and st[c].nunique() > 1]
    eng = st[st.condition.isin(["ours", "graphviz"])]
    P("\n== K6  Which criterion tracks the raters, with and without the model-placed "
      "condition ==")
    P("\n  All 72 images: three conditions, one of which (the model placing its own boxes)")
    P("  fails every criterion by one to two orders of magnitude. Any measure that")
    P("  separates that condition from the other two correlates with the ratings, whether")
    P("  or not it says anything about two drawn layouts. The second column is the same")
    P("  correlation over the 48 images the two real engines drew.")
    rows = []
    for c in cols:
        a = stats.spearmanr(st[c].astype(float), st.bws)
        b = (stats.spearmanr(eng[c].astype(float), eng.bws)
             if eng[c].nunique() > 1 else None)
        rows.append(dict(measure=c.replace("report_", ""),
                         rho_all72=a.statistic, p_all72=a.pvalue,
                         rho_engines48=b.statistic if b else np.nan,
                         p_engines48=b.pvalue if b else np.nan,
                         constant_within_engines=b is None))
    t = pd.DataFrame(rows)
    # Benjamini-Hochberg within each family of correlations
    for suf in ["all72", "engines48"]:
        p = t["p_"+suf].values
        ok = ~np.isnan(p)
        q = np.full(len(p), np.nan)
        idx = np.argsort(p[ok])
        m = ok.sum()
        run = 1.0
        for rank, i in enumerate(idx[::-1]):
            run = min(run, m*p[ok][i]/(m-rank))
            q[np.flatnonzero(ok)[i]] = run
        t["q_"+suf] = q
    t = t.sort_values("rho_all72")
    P()
    P(t.round(3).to_string(index=False))
    CSV["criteria_vs_raters_within"] = t

    def g(name, col):
        r = t[t.measure == name]
        return np.nan if r.empty else float(r.iloc[0][col])
    lc, ec = g("label_collisions", "rho_all72"), g("edge_edge_crossings", "rho_all72")
    lc2, ec2 = g("label_collisions", "rho_engines48"), g("edge_edge_crossings", "rho_engines48")
    sv = stats.spearmanr(_b(eng.report_valid), eng.bws)
    P(f"\n  label collisions   all 72: rho {lc:+.2f}   the 48 engine-drawn: {lc2:+.2f} "
      f"(q={g('label_collisions','q_engines48'):.2f})")
    P(f"  arrow crossings    all 72: rho {ec:+.2f}   the 48 engine-drawn: {ec2:+.2f} "
      f"(q={g('edge_edge_crossings','q_engines48'):.2f})")
    P(f"  strict verdict     the 48 engine-drawn: rho {sv.statistic:+.2f} (p={sv.pvalue:.2f})")
    surv = t[(t.q_engines48 < 0.05) & t.rho_engines48.notna()]
    P(f"\n  Of {int(t.rho_engines48.notna().sum())} measures that vary among the 48 engine-drawn"
      f" images, {len(surv)} track the raters")
    P("  after Benjamini-Hochberg correction:")
    for r in surv.sort_values("rho_engines48").itertuples():
        P(f"    {r.measure:26s} rho {r.rho_engines48:+.2f}  q={r.q_engines48:.3f}")
    # the fitted index, cross-validated, on the images the two engines drew
    F10 = ["report_label_collisions", "report_node_overlap_pairs",
           "report_edge_node_crossings", "report_edge_label_crossings",
           "report_stacked_routes", "report_edge_edge_crossings",
           "report_total_edge_length", "report_aspect_deviation",
           "report_mean_bends_per_edge", "report_min_separation"]

    def logo(df, use):
        X = np.log1p(np.clip(df[use].astype(float).values, 0, None))
        X = (X-X.mean(0))/(X.std(0)+1e-9)
        y, gr, pred = df.bws.values, df.paper_id.values, np.zeros(len(df))
        for q in np.unique(gr):
            tr = gr != q
            A = np.hstack([X[tr], np.ones((tr.sum(), 1))])
            b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            pred[~tr] = np.hstack([X[~tr], np.ones(((~tr).sum(), 1))])@b
        return pred, y
    pr, y = logo(st, F10)
    m = st.condition.isin(["ours", "graphviz"]).values
    e = eng.reset_index(drop=True)
    use = [c for c in F10 if e[c].nunique() > 1]
    pr2, y2 = logo(e, use)
    irows = [dict(index="ten measures, fitted on all 72, leave-one-source-graph-out",
                  scored_on="all 72 images", rho=stats.spearmanr(pr, y).statistic,
                  p=stats.spearmanr(pr, y).pvalue),
             dict(index="ten measures, fitted on all 72, leave-one-source-graph-out",
                  scored_on="the 48 the two engines drew",
                  rho=stats.spearmanr(pr[m], y[m]).statistic,
                  p=stats.spearmanr(pr[m], y[m]).pvalue),
             dict(index=f"{len(use)} measures that vary, fitted on the 48, "
                        "leave-one-source-graph-out",
                  scored_on="the 48 the two engines drew",
                  rho=stats.spearmanr(pr2, y2).statistic,
                  p=stats.spearmanr(pr2, y2).pvalue)]
    it = pd.DataFrame(irows)
    CSV["index_on_engine_images"] = it
    P("\n  the fitted index of N5, cross-validated, on the same two samples:")
    P(it.round(3).to_string(index=False))
    P("  In sample the index reaches 0.65 on those 48 images; cross-validated it does not")
    P("  predict them, which is what a ten-parameter fit on 48 points would predict.")

    P("\n  Reading: on the 48 images drawn by a real engine, no defect count tracks the")
    P("  raters -- label collisions point the other way and do not survive correction,")
    P("  and so does the strict verdict. What the raters follow there is the size and")
    P("  shape of the drawing. The rho=0.75 of the full panel is therefore evidence that")
    P("  these counts detect a figure nobody can read, and not evidence that label")
    P("  occlusion is the criterion a reader pays for; the paper claims only the first.")
    return t


def _holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0]*len(ps), 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(ps)-rank)*ps[i])
        adj[i] = min(1.0, run)
    return adj


def K7(D, P):
    """Every descriptive family of tests, corrected inside its own family."""
    LEN = ["node_overlap_pairs", "edge_node_crossings", "route_reversals"]
    SIX = LEN + ["label_collisions", "edge_label_crossings", "stacked_routes"]
    DOTS = ["dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "dot_lr_ortho"]
    K2 = ["corpus", "model", "run", "item"]
    li = pd.read_csv(D+"layout_items.csv", low_memory=False)
    for c in SIX + ["valid", "n_nodes"]:
        li[c] = pd.to_numeric(li[c], errors="coerce")
    li = li.dropna(subset=["valid"])
    gv = li[li.source.isin(["graphviz", "graphviz_outside_development"])]

    def piv(v):
        return gv.pivot_table(index=K2, columns="config", values=v, aggfunc="first")
    strict = piv("valid")
    idx = strict.dropna(subset=["ours"]+DOTS).index
    best = "dot_tb_spline"
    fams, rows = {}, []

    # family 1: the six strict criteria, ours against dot's best configuration
    f1 = []
    for c in SIX:
        w = piv(c).loc[idx]
        a, b = (w["ours"] == 0).astype(int), (w[best] == 0).astype(int)
        b01 = int(((a == 1) & (b == 0)).sum())
        b10 = int(((a == 0) & (b == 1)).sum())
        p = 1.0 if b01+b10 == 0 else float(stats.binomtest(b01, b01+b10, 0.5).pvalue)
        f1.append((c, (a.mean()-b.mean())*100, p))
    fams["six strict criteria (A1): ours vs dot's best configuration"] = f1

    # family 2: the five size bins of A5
    w = piv("valid").loc[idx]
    nn = piv("n_nodes").loc[idx]["ours"]
    bins, labs = [0, 8, 12, 20, 30, 10**6], ["8 or fewer", "9-12", "13-20", "21-30", "over 30"]
    cut = pd.cut(nn, bins, labels=labs)
    f2 = []
    for lab in labs:
        m = cut == lab
        a, b = w.loc[m, "ours"], w.loc[m, best]
        b01 = int(((a == 1) & (b == 0)).sum())
        b10 = int(((a == 0) & (b == 1)).sum())
        p = 1.0 if b01+b10 == 0 else float(stats.binomtest(b01, b01+b10, 0.5).pvalue)
        f2.append((f"{lab} boxes (n={int(m.sum())})", (a.mean()-b.mean())*100, p))
    fams["five size bins (A5): where the strict difference lives"] = f2

    # family 3 and 4: the rater correlations, on all 72 images and on the 48 drawn
    st, se, ra = _bws(D)
    cols = [c for c in st.columns if c.startswith("report_") and c != "report_valid"
            and pd.api.types.is_numeric_dtype(st[c]) and st[c].nunique() > 1]
    eng = st[st.condition.isin(["ours", "graphviz"])]
    f3 = [(c.replace("report_", ""),) + tuple(stats.spearmanr(st[c].astype(float), st.bws))[:2]
          for c in cols]
    fams["rater correlations (R5), all 72 images"] = [(a, b, c) for a, b, c in f3]
    f4 = [(c.replace("report_", ""),) + tuple(stats.spearmanr(eng[c].astype(float), eng.bws))[:2]
          for c in cols if eng[c].nunique() > 1]
    fams["rater correlations (R5/K6), the 48 the two engines drew"] = [(a, b, c) for a, b, c in f4]

    P("\n== K7  Every descriptive family of tests, corrected inside its own family ==")
    P("\n  The confirmatory comparisons of the protocol are Holm-corrected already. The")
    P("  analyses specified after the test runs were reported as descriptive and")
    P("  uncorrected, which is not enough where a family of tests is read as one finding:")
    P("  which of six criteria carries a difference, in which of five size bins it lives,")
    P("  which of twenty-one measures a rater follows. Holm within each family below.")
    for fam, items in fams.items():
        ps = [p for _n, _e, p in items]
        adj = _holm(ps)
        P(f"\n  {fam}  ({len(items)} tests)")
        for (n, e, p), q in sorted(zip(items, adj), key=lambda t: t[0][2]):
            mark = "*" if q < 0.05 else " "
            P(f"    {mark} {n:28s} effect {e:+7.2f}   p={p:.2g}   Holm p={q:.2g}")
            rows.append(dict(family=fam, test=n, effect=e, p=p, p_holm=q,
                             survives=bool(q < 0.05)))
        P(f"    {sum(q < 0.05 for q in adj)} of {len(items)} survive correction")
    t = pd.DataFrame(rows)
    CSV["exploratory_families"] = t
    P("\n  Reading: the two label criteria and every size bin survive correction, so the")
    P("  attribution of the strict difference and its growth with graph size stand as")
    P("  descriptive results. The rater correlations survive on the full panel of three")
    P("  conditions and none of them survives on the 48 images the two engines drew, which")
    P("  is the sample a claim about two drawn layouts would need (K6).")
    return t


def K8(D, P):
    """What the symmetric read-back can support, beside what the reader does."""
    rb = pd.read_csv(D+"readback_sciflow.csv").dropna(subset=["score"])
    pi = pd.read_csv(D+"pipeline_items.csv", low_memory=False)
    ceiling = rb[rb.system == "gold_graph_ceiling"].score.mean()
    ev = pi[(pi.variant == "evaluated") & (pi.corpus == "sciflow") & (pi.seed == 0)]
    drawn = ev.groupby(["model", "domain", "item"]).agg(
        drawn_nodes=("layout_n_nodes", "mean"), drawn_edges=("layout_n_edges", "mean")
    ).reset_index()
    P("\n== K8  What the symmetric read-back can support ==")
    P("\n  Every number in the read-back column is a graph this reader recovered from an")
    P("  image. N2 measures the reader against the figure it was looking at, whose")
    P("  geometry we know exactly:")
    fid = []
    for m in ["Qwen", "Gemma", "Ministral"]:
        s = rb[(rb.system == "ours") & (rb.model == m)].merge(
            drawn, on=["model", "domain", "item"])
        fid.append(dict(model=m, n=len(s), node_ratio=s.pred_nodes.mean()/s.drawn_nodes.mean(),
                        edge_ratio=s.pred_edges.mean()/s.drawn_edges.mean()))
        P(f"    {m:10s} boxes {fid[-1]['node_ratio']:.2f}x drawn, "
          f"arrows {fid[-1]['edge_ratio']:.2f}x drawn")
    others = ["gemini3", "gemini2.5", "seedream", "graphviz", "qwen", "sdxl", "pixart"]
    rows = []
    for m in ["Qwen", "Gemma", "Ministral"]:
        a = _paper_keyed(rb[(rb.system == "ours") & (rb.model == m)])
        for s in others:
            b = _paper_keyed(rb[rb.system == s])
            idx = sorted(set(a.index) & set(b.index))
            dif = (a[idx]-b[idx]).values
            mde = 2.80*dif.std(ddof=1)/np.sqrt(len(dif))
            rows.append(dict(model=m, against=s, n=len(idx), diff=dif.mean(),
                             diff_pct_ceiling=100*dif.mean()/ceiling,
                             paired_mde=mde, mde_pct_ceiling=100*mde/ceiling,
                             clears_its_resolution=bool(abs(dif.mean()) > mde)))
    t = pd.DataFrame(rows)
    CSV["readback_bound"] = t
    P("\n  every read-back difference beside what this sample can resolve"
      " (paired, 80% power):")
    P(t.round(4).to_string(index=False))
    cl = t[t.clears_its_resolution]
    P(f"\n  {len(cl)} of {len(t)} differences clear their own resolution. The ceiling of the")
    P(f"  protocol is {ceiling:.3f}, so the whole table lives in the bottom fifth of the")
    P("  range, and the reader that produced it returns about a quarter more boxes and")
    P("  two thirds more arrows than were drawn.")
    P("\n  Reading: this column orders what a 14B reader recovers from an image, and the")
    P("  figure is only one of the things that decides that. It is reported as a")
    P("  measurement of the protocol, descriptively; no verdict in this paper rests on")
    P("  it, and the rankings in it are not comparable to any published figure.")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out")
    ap.add_argument("--latex")
    ap.add_argument("--csv-dir")
    ap.add_argument("--analysis", nargs="*", default=["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"])
    a = ap.parse_args()
    D = a.data.rstrip("/")+"/"
    buf = []

    def P(s=""):
        print(s)
        buf.append(str(s))
    for n in a.analysis:
        globals()[n](D, P)
    if a.out:
        open(a.out, "w").write("\n".join(buf)+"\n")
    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        for k, v in CSV.items():
            v.to_csv(os.path.join(a.csv_dir, k+".csv"), index=False)
    if a.latex:
        open(a.latex, "w").write("\n".join(
            TEX[k] for k in ["guarantee", "ratersens", "floor", "claims"] if k in TEX)+"\n")
        base = os.path.dirname(a.latex)
        for k in TEX:
            open(os.path.join(base, "table_"+k+".tex"), "w").write(TEX[k])
