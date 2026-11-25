"""Rebuild the numbers of every table in the paper from the CSVs in ../data.

Needs Python 3.9+ with pandas, numpy and scipy; no GPU, no other code.

    python3 make_tables.py                 # all tables
    python3 make_tables.py --table 2 5     # only some

Each table is printed in the order of the paper. The statistics are the ones the
paper states: per item, our value is averaged over the three seeds; paired
Wilcoxon signed-rank test; matched-pairs rank-biserial r as effect size; 95 %
percentile bootstrap of the mean difference (10,000 resamples, seed 0); Holm
correction within the family named in each table; a verdict follows the sign
of r at alpha = 0.05.
"""
from __future__ import annotations

import argparse
import itertools
import random
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = ("Qwen", "Gemma", "Ministral")
SHIPPED = {"gemini3": "Gemini 3 Pro Image", "gemini2.5": "Gemini 2.5 Flash Image",
           "seedream": "Seedream 4.0", "graphviz": "Graphviz (GPT-4o->DOT)",
           "qwen": "Qwen-Image", "pixart": "PixArt-Sigma", "sdxl": "SDXL"}
DIMS = ("faithfulness", "conciseness", "readability", "aesthetics", "overall")
ALPHA = 0.05


# --------------------------------------------------------------------------
# statistics (identical to the paper's code)

def rank_biserial(diffs):
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(d) for d in nonzero])
    plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    return (plus - minus) / (plus + minus)


def bootstrap_ci(diffs, seed=0, n=10_000):
    rng = np.random.default_rng(seed)
    values = np.asarray(diffs)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare(ours: dict, theirs: dict):
    """Paired comparison on the items both cover (ours minus theirs)."""
    shared = sorted(set(ours) & set(theirs))
    diffs = [ours[k] - theirs[k] for k in shared]
    decided = [d for d in diffs if d != 0]
    if len(decided) < 2:
        return None
    return {"n": len(shared), "diff": st.fmean(diffs), "ci": bootstrap_ci(diffs),
            "r": rank_biserial(diffs), "p": float(wilcoxon(diffs).pvalue)}


def verdict(r, p_holm):
    return "tied" if p_holm >= ALPHA else ("ahead" if r > 0 else "behind")


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


# --------------------------------------------------------------------------
# data

def load(name):
    return pd.read_csv(DATA / name, low_memory=False)


def scores(df, col):
    """Failed items score zero, as in the paper."""
    return df[col].fillna(0.0).astype(float)


def keyed(df, col):
    return dict(zip(zip(df["domain"], df["item"]), scores(df, col)))


def runs_of(pipe, corpus, model, variant="evaluated"):
    sel = pipe[(pipe.variant == variant) & (pipe.corpus == corpus) & (pipe.model == model)]
    return [g for _, g in sorted(sel.groupby("seed"))]


def across_runs(runs, col):
    means = [scores(r, col).mean() for r in runs]
    return st.fmean(means), (st.stdev(means) if len(means) > 1 else 0.0)


def seed_averaged(runs, col):
    by = [keyed(r, col) for r in runs]
    shared = set.intersection(*(set(b) for b in by))
    return {k: st.fmean(b[k] for b in by) for k in shared}


def judge_items(judge, judge_name, system, model, seeds=None):
    """Per test case, each dimension averaged over the given files (seed x order)."""
    sel = judge[(judge.judge == judge_name) & (judge.system == system) & (judge.model == model)]
    if seeds is not None:
        sel = sel[sel.seed.isin(seeds)]
    groups = [g.set_index("item") for _, g in sel.groupby(["seed", "order"], dropna=False)]
    ids = set(groups[0].index)
    return {i: {d: st.fmean(float(g.loc[i, f"{d}_points"]) for g in groups) for d in DIMS}
            for i in sorted(ids)}, len(groups)


# --------------------------------------------------------------------------
# tables

def table2(pipe, dot, pub, judge):
    out = ["Data table 2  ours vs the same model writing DOT, each corpus's standard metric",
           f"{'corpus':<12}{'model':<10}{'ours':>8}{'sd':>7}{'DOT':>8}{'diff':>9}{'95% CI':>20}"
           f"{'':>8}{'matched':>9}{'':>8}"]
    for corpus in ("sciflow", "flowvqa"):
        matched = {}
        rows = {}
        for m in MODELS:
            runs = runs_of(pipe, corpus, m)
            mean, sd = across_runs(runs, "score")
            ours = seed_averaged(runs, "score")
            d = dot[(dot.variant == "dot") & (dot.corpus == corpus) & (dot.model == m)]
            rivals = {"DOT": keyed(d, "score")}
            if corpus == "sciflow":
                for s in SHIPPED:
                    rivals[s] = keyed(pub[pub.system == s], "score")
            res = {k: compare(ours, v) for k, v in rivals.items()}
            res = {k: v for k, v in res.items() if v}
            adj = dict(zip(res, holm([v["p"] for v in res.values()])))
            r = res["DOT"]
            dm = dot[(dot.variant == "dot_matched") & (dot.corpus == corpus) & (dot.model == m)]
            matched[m] = (scores(dm, "score").mean(), compare(ours, keyed(dm, "score")))
            rows[m] = (mean, sd, scores(d, "score").mean(), r, verdict(r["r"], adj["DOT"]))
        madj = dict(zip(MODELS, holm([matched[m][1]["p"] for m in MODELS])))
        for m in MODELS:
            mean, sd, dmean, r, v = rows[m]
            mm, mr = matched[m]
            out.append(f"{corpus:<12}{m:<10}{mean:>8.3f}{sd:>7.3f}{dmean:>8.3f}{r['diff']:>+9.3f}"
                       f"{ci(r['ci']):>20}{v:>8}{mm:>9.3f}{verdict(mr['r'], madj[m]):>8}")
    # PaperBananaBench
    res, matched, rows = {}, {}, {}
    for m in MODELS:
        ours, _ = judge_items(judge, "InternVL3.5-14B", "ours", m, seeds=[0, 1, 2])
        dots, _ = judge_items(judge, "InternVL3.5-14B", "dot", m)
        mat, _ = judge_items(judge, "InternVL3.5-14B", "dot_matched", m)
        o = {k: v["overall"] for k, v in ours.items()}
        res[m] = compare(o, {k: v["overall"] for k, v in dots.items()})
        matched[m] = compare(o, {k: v["overall"] for k, v in mat.items()})
        rows[m] = (st.fmean(o.values()), st.fmean(v["overall"] for v in dots.values()),
                   st.fmean(v["overall"] for v in mat.values()))
    adj = dict(zip(MODELS, holm([res[m]["p"] for m in MODELS])))
    madj = dict(zip(MODELS, holm([matched[m]["p"] for m in MODELS])))
    for m in MODELS:
        o, d, mm = rows[m]
        r = res[m]
        out.append(f"{'paperbanana':<12}{m:<10}{o:>8.1f}{'':>7}{d:>8.1f}{r['diff']:>+9.1f}"
                   f"{'[%+.1f, %+.1f]' % r['ci']:>20}{verdict(r['r'], adj[m]):>8}{mm:>9.1f}"
                   f"{verdict(matched[m]['r'], madj[m]):>8}")
    out.append("SciFlow-Bench: Holm over the 7 shipped systems and DOT per model; FlowVQA and "
               "PaperBananaBench: Holm over the three models; matched: Holm over the three models.")
    out.append("The SciFlow-Bench rows run at the floor of their metric: its edge half, 0.6 of "
               "the weight, is")
    out.append("exactly zero on 60-76% of items for every system in them and no system reaches a "
               "fifth of the")
    out.append("0.611 the gold graphs reach, so the verdicts in those three rows are not read as "
               "a content")
    out.append("advantage (checks_claims.py --analysis K3). FlowVQA and PaperBananaBench are "
               "where the two")
    out.append("content instruments have room; ours is ahead in 3 of those 6 settings, tied in 1 "
               "and behind in 2.")
    return "\n".join(out)


def table3(pipe, pub, rb):
    out = ["Data table 3  SciFlow-Bench, every system; left: own graph or graph recovered by the "
           "benchmark; right: every figure read back by one reader",
           f"{'system':<26}{'n':>5}{'edge F1':>9}{'score':>8}{'node F1':>9}{'read-back':>11}{'n':>5}"]
    for m in MODELS:
        runs = runs_of(pipe, "sciflow", m)
        e, s, n = (across_runs(runs, c)[0] for c in ("edge_f1", "score", "node_f1"))
        r = rb[(rb.system == "ours") & (rb.model == m)]
        out.append(f"{'Ours + ' + m:<26}{len(runs[0]):>5}{e:>9.3f}{s:>8.3f}{n:>9.3f}"
                   f"{scores(r, 'score').mean():>11.3f}{len(r):>5}")
    for key, name in SHIPPED.items():
        p = pub[pub.system == key]
        r = rb[rb.system == key]
        out.append(f"{name:<26}{len(p):>5}{scores(p, 'edge_f1').mean():>9.3f}"
                   f"{scores(p, 'score').mean():>8.3f}{scores(p, 'node_f1').mean():>9.3f}"
                   f"{scores(r, 'score').mean():>11.3f}{len(r):>5}")
    g = rb[rb.system == "gold_graph_ceiling"]
    ok = g[g.failed == False]  # noqa: E712
    out.append(f"\nceiling: gold graphs drawn by our engine and read back: {scores(g, 'score').mean():.3f} "
               f"over {len(g)}, {scores(ok, 'score').mean():.3f} over the {len(ok)} read within the limit")
    out.append("\nEvery number in the read-back column is a graph this one 14B reader recovered "
               "from an image;")
    out.append("measured against the figures whose geometry we know exactly, it returns 1.2-1.3 "
               "boxes and")
    out.append("1.7-1.9 arrows per one drawn (checks_new.py --analysis N2), and 6 of the 21 "
               "paired differences")
    out.append("below do not clear what this sample can resolve (checks_claims.py --analysis K8).")
    out.append("The column is reported as a measurement of the protocol, not as a ranking of "
               "systems, and")
    out.append("no verdict in the paper rests on it.")
    out.append("\nread-back, paired (ours seed 0 minus each system; Holm over the 7 systems per model)")
    for m in MODELS:
        ours = keyed(rb[(rb.system == "ours") & (rb.model == m)], "score")
        res = {k: compare(ours, keyed(rb[rb.system == k], "score")) for k in SHIPPED}
        res = {k: v for k, v in res.items() if v}
        adj = dict(zip(res, holm([v["p"] for v in res.values()])))
        out.append(f"  Ours + {m}: " + ", ".join(
            f"{SHIPPED[k]} {verdict(v['r'], adj[k])} (r {v['r']:+.2f})" for k, v in res.items()))
    return "\n".join(out)


def table4(judge):
    out = ["Data table 4  PaperBananaBench judge per dimension (50 = parity with the authors' figure); "
           "Holm over the three models per dimension",
           f"{'system':<24}" + "".join(f"{d[:12]:>14}" for d in DIMS)]
    data = {}
    for m in MODELS:
        data[m] = (judge_items(judge, "InternVL3.5-14B", "ours", m, seeds=[0, 1, 2])[0],
                   judge_items(judge, "InternVL3.5-14B", "dot", m)[0])
    verdicts = {}
    for d in DIMS:
        res = {m: compare({k: v[d] for k, v in data[m][0].items()},
                          {k: v[d] for k, v in data[m][1].items()}) for m in MODELS}
        adj = dict(zip(MODELS, holm([res[m]["p"] for m in MODELS])))
        verdicts[d] = {m: verdict(res[m]["r"], adj[m]) for m in MODELS}
    for m in MODELS:
        ours, dots = data[m]
        out.append(f"{'Ours + ' + m:<24}" + "".join(
            f"{st.fmean(v[d] for v in ours.values()):>14.1f}" for d in DIMS))
        out.append(f"{m + ' writing DOT':<24}" + "".join(
            f"{st.fmean(v[d] for v in dots.values()):>14.1f}" for d in DIMS))
        out.append(f"{'  ours is':<24}" + "".join(f"{verdicts[d][m]:>14}" for d in DIMS))
    return "\n".join(out)


def _layout_summary(rows):
    ok = rows[rows["error"].isna()] if "error" in rows else rows
    return {"strict": 100.0 * ok["valid"].astype(bool).mean(),
            "crossings": ok["edge_edge_crossings"].mean(),
            "thru_box": ok["edge_node_crossings"].mean(),
            "thru_label": ok["edge_label_crossings"].mean(),
            "label_coll": ok["label_collisions"].mean(),
            "area": ok["area"].median(), "figures": len(ok)}


def _over_runs(df):
    per_run = [_layout_summary(g) for _, g in df.groupby("run")]
    return {k: st.fmean(s[k] for s in per_run) for k in per_run[0]}


def table5_12(lay):
    gv = lay[(lay.source == "graphviz") & (lay.corpus == "sciflow")]
    out = ["Data table 5  the same SciFlow-Bench graphs laid out by our engine and by Graphviz dot "
           "(mean over three runs; area median)",
           f"{'graphs':<10}{'layout':<16}{'cross.':>8}{'thr.box':>9}{'lbl.coll':>9}{'area':>7}{'strict%':>9}"]
    names = {"ours": "ours", "dot_tb_spline": "dot TB curved", "dot_lr_spline": "dot LR curved",
             "dot_tb_ortho": "dot TB ortho", "dot_lr_ortho": "dot LR ortho"}
    summ = {(m, c): _over_runs(gv[(gv.model == m) & (gv.config == c)]) for m in MODELS for c in names}
    for m in MODELS:
        for c in ("ours", "dot_tb_spline"):
            s = summ[(m, c)]
            out.append(f"{m:<10}{names[c]:<16}{s['crossings']:>8.2f}{s['thru_box']:>9.2f}"
                       f"{s['label_coll']:>9.2f}{s['area']:>7.0f}{s['strict']:>8.1f}%")
    out.append("\nData table 12  strict legibility (%) / arrow crossings for every dot configuration")
    out.append(f"{'layout':<16}" + "".join(f"{m:>16}" for m in MODELS))
    for c, name in names.items():
        out.append(f"{name:<16}" + "".join(
            f"{summ[(m, c)]['strict']:>9.1f} / {summ[(m, c)]['crossings']:.2f}" for m in MODELS))
    od = lay[lay.source == "graphviz_outside_development"]
    out.append("\nText, Section 5.2: graphs outside development (seed 0): strict % and crossings")
    for corpus in ("flowvqa", "paperbanana"):
        for m in MODELS:
            sub = od[(od.corpus == corpus) & (od.model == m)]
            a, b = _over_runs(sub[sub.config == "ours"]), _over_runs(sub[sub.config == "dot_tb_spline"])
            out.append(f"  {corpus:<12}{m:<10} ours {a['strict']:5.1f}% / {a['crossings']:.2f}   "
                       f"dot {b['strict']:5.1f}% / {b['crossings']:.2f}")
    return "\n".join(out)


PLACEMENT = [("With overlapping boxes (%)", lambda d: 100.0 * (d["layout_node_overlap_pairs"] > 0)),
             ("Arrows through a box", lambda d: d["layout_edge_node_crossings"]),
             ("Arrow crossings", lambda d: d["layout_edge_edge_crossings"]),
             ("Label collisions", lambda d: d["layout_label_collisions"]),
             ("Strict legible (%)", lambda d: 100.0 * d["layout_valid"].astype(bool)),
             ("SciFlow score", lambda d: d["score"].fillna(0.0))]


def table6(pl):
    out = ["Data table 6  who places the boxes: ours vs the model's own positions, 100 SciFlow-Bench "
           "papers, mean over three seeds; p: Wilcoxon on seed-averaged values",
           f"{'measure':<28}" + "".join(f"{m + ' ours':>16}{'model':>9}{'p':>9}" for m in MODELS)]
    per_model = {}
    for m in MODELS:
        seeds = []
        for _, g in pl[pl.model == m].groupby("seed"):
            pairs = g.pivot_table(index=["domain", "item"], columns="condition", values="failed",
                                  aggfunc="first")
            both = pairs[(pairs["ours"] == False) & (pairs["model"] == False)].index  # noqa: E712
            g = g.set_index(["domain", "item"])
            seeds.append((g[g.condition == "ours"].loc[both], g[g.condition == "model"].loc[both]))
        per_model[m] = seeds
    for name, f in PLACEMENT:
        cells = []
        for m in MODELS:
            seeds = per_model[m]
            mo = st.fmean(float(f(o).mean()) for o, _ in seeds)
            mm = st.fmean(float(f(md).mean()) for _, md in seeds)
            shared = sorted(set.intersection(*(set(o.index) for o, _ in seeds)))
            a = [st.fmean(float(f(o.loc[[k]]).iloc[0]) for o, _ in seeds) for k in shared]
            b = [st.fmean(float(f(md.loc[[k]]).iloc[0]) for _, md in seeds) for k in shared]
            d = [x - y for x, y in zip(a, b) if x != y]
            p = wilcoxon(d).pvalue if len(d) > 1 else float("nan")
            cells.append(f"{mo:>16.3f}{mm:>9.3f}{p:>9.2g}")
        out.append(f"{name:<28}" + "".join(cells))
    return "\n".join(out)


def table7(pipe):
    out = ["Data table 7  pipeline with one component changed, against the full pipeline (seed 0); "
           "score: Holm over the three changes; crossings: mean per figure on items laid out in both"]
    changes = {"- best-of-2": "ablation_one_greedy_graph", "- constrained decoding": "ablation_unconstrained",
               "+ coverage call": "coverage"}
    for corpus in ("sciflow", "flowvqa"):
        for m in MODELS:
            full = pipe[(pipe.variant == "evaluated") & (pipe.corpus == corpus) & (pipe.model == m)
                        & (pipe.seed == 0)].set_index(["domain", "item"])
            res = {}
            for label, v in changes.items():
                run = pipe[(pipe.variant == v) & (pipe.corpus == corpus) & (pipe.model == m)
                           & (pipe.seed == 0)].set_index(["domain", "item"])
                shared = sorted(set(run.index) & set(full.index))
                r = compare({k: float(np.nan_to_num(run.loc[k, "score"])) for k in shared},
                            {k: float(np.nan_to_num(full.loc[k, "score"])) for k in shared})
                laid = [k for k in shared if pd.notna(run.loc[k, "layout_edge_edge_crossings"])
                        and pd.notna(full.loc[k, "layout_edge_edge_crossings"])]
                cf = st.fmean(full.loc[k, "layout_edge_edge_crossings"] for k in laid)
                cv = st.fmean(run.loc[k, "layout_edge_edge_crossings"] for k in laid)
                cr = compare({k: run.loc[k, "layout_edge_edge_crossings"] for k in laid},
                             {k: full.loc[k, "layout_edge_edge_crossings"] for k in laid})
                failed = int(run.loc[shared, "failed"].astype(bool).sum())
                res[label] = (r, cf, cv, cr, failed)
            adj = holm([v[0]["p"] if v[0] else 1.0 for v in res.values()])
            out.append(f"\n{corpus} {m}")
            for (label, (r, cf, cv, cr, failed)), p in zip(res.items(), adj):
                score = "same" if (r is None or p >= ALPHA) else ("better" if r["r"] > 0 else "worse")
                pc = f"p {cr['p']:.2g}" if cr else "identical"
                out.append(f"  {label:<24} score {score:<7} crossings {cf:.2f} -> {cv:.2f} "
                           f"({100 * (cv - cf) / cf:+.0f}%, {pc}), failed {failed}")
    return "\n".join(out)


def table8(lay):
    la = lay[lay.source == "layout_ablation"]
    names = {"full": "Full engine", "no_detours": "- forward detours",
             "no_label_placement": "- label placement after routing",
             "no_spread_ends": "- spread arrow ends", "no_sideways": "- sideways tracks",
             "no_variants": "- layout variants", "no_variants_no_pull": "- variants and pull-later",
             "none": "- all six"}
    out = ["Data table 8  layout ablation on identical graphs, mean of three runs: crossings / median area / strict %",
           f"{'configuration':<34}" + "".join(f"{m:>24}" for m in MODELS)]
    for c, name in names.items():
        cells = []
        for m in MODELS:
            s = _over_runs(la[(la.model == m) & (la.config == c)])
            cells.append(f"{s['crossings']:>8.2f}{s['area']:>8.0f}{s['strict']:>8.1f}")
        out.append(f"{name:<34}" + "".join(cells))
    out.append("\ncrossings, full engine vs each component off (papers averaged over runs; "
               "Holm over the configurations per model)")
    for m in MODELS:
        def per_paper(c):
            sub = la[(la.model == m) & (la.config == c)]
            return sub.groupby("item")["edge_edge_crossings"].mean().to_dict()
        full = per_paper("full")
        res = {c: compare({k: -v for k, v in full.items()}, {k: -v for k, v in per_paper(c).items()})
               for c in names if c != "full"}
        adj = holm([r["p"] for r in res.values()])
        out.append(f"  {m}: " + ", ".join(f"{c} p_Holm {p:.2g}" for c, p in zip(res, adj)))
    return "\n".join(out)


HUMAN = [("Label collisions", "report_label_collisions", -1),
         ("Overlapping box pairs", "report_node_overlap_pairs", -1),
         ("Arrows through a box", "report_edge_node_crossings", -1),
         ("Arrows through a label", "report_edge_label_crossings", -1),
         ("Strict legible (yes/no)", "report_valid", 1),
         ("Arrow crossings", "report_edge_edge_crossings", -1),
         ("Distance from target page shape", "report_aspect_deviation", -1)]


def _bws(ratings, sets):
    shown, best, worst = {}, {}, {}
    for _, j in ratings.iterrows():
        for member in sets[j["set"]]:
            shown[member] = shown.get(member, 0) + 1
        best[j["best"]] = best.get(j["best"], 0) + 1
        worst[j["worst"]] = worst.get(j["worst"], 0) + 1
    return {s: (best.get(s, 0) - worst.get(s, 0)) / n for s, n in shown.items() if n}


def _spearman(xs, ys):
    if len(set(xs)) < 2:
        return float("nan")
    rx, ry = rankdata(xs), rankdata(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall_w(per, ids):
    ranks = np.array([rankdata([per[r][k] for k in ids]) for r in per])
    m, n = ranks.shape
    s = ((ranks.sum(0) - ranks.sum(0).mean()) ** 2).sum()
    return 12 * s / (m ** 2 * (n ** 3 - n))


def table9(stim, sets_df, ratings):
    sets = {int(r["set"]): [r[f"image_{i}"] for i in range(1, 5)] for _, r in sets_df.iterrows()}
    stim = stim.set_index("image")
    human = _bws(ratings, sets)
    ids_all = sorted(human)
    ids_two = [k for k in ids_all if stim.loc[k, "condition"] in ("ours", "graphviz")]
    out = ["Data table 9  human best-worst study: Spearman rho of each measure with the raters' "
           "BWS scores (counts negated: positive = agreement)",
           f"{'measure':<34}{'all 72':>9}{'ours, dot 48':>14}"]
    for name, col, sign in HUMAN:
        cells = []
        for ids in (ids_all, ids_two):
            xs = [sign * float(stim.loc[k, col]) for k in ids]
            rho = _spearman(xs, [human[k] for k in ids])
            cells.append("---" if rho != rho else f"{rho:+.2f}")
        out.append(f"{name:<34}{cells[0]:>9}{cells[1]:>14}")
    cond = {k: stim.loc[k, "condition"] for k in ids_all}
    for c in ("ours", "graphviz", "model"):
        out.append(f"best-worst mean, {c}: {st.fmean(v for k, v in human.items() if cond[k] == c):+.2f}")
    by_paper = {}
    for k, v in human.items():
        by_paper.setdefault(stim.loc[k, "paper_id"], {})[cond[k]] = v
    for a, b in (("ours", "model"), ("graphviz", "ours")):
        pairs = [(v[a], v[b]) for v in by_paper.values() if a in v and b in v]
        better, worse = sum(x > y for x, y in pairs), sum(x < y for x, y in pairs)
        p = binomtest(better, better + worse).pvalue
        out.append(f"{a} preferred to {b} on {better} of {len(pairs)} graphs, {b} on {worse}, "
                   f"tied {len(pairs) - better - worse} (sign test p = {p:.2g})")
    raters = sorted(ratings["rater"].unique())
    per = {r: _bws(ratings[ratings.rater == r], sets) for r in raters}
    rng, values = random.Random(0), []
    for _ in range(50):
        shuffled = raters[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        sa = _bws(ratings[ratings.rater.isin(shuffled[:half])], sets)
        sb = _bws(ratings[ratings.rater.isin(shuffled[half:])], sets)
        common = sorted(set(sa) & set(sb))
        values.append(_spearman([sa[c] for c in common], [sb[c] for c in common]))
    out.append(f"split-half reliability {st.median(values):.2f}; Kendall's W all 72 "
               f"{_kendall_w(per, ids_all):.2f}, ours + dot 48 {_kendall_w(per, ids_two):.2f}")
    return "\n".join(out)


def table11(pipe, dot, pub):
    out = ["Data table 11  node and edge F1, ours vs the same model writing DOT "
           "(SciFlow: Holm over 7 shipped systems and DOT per model)",
           f"{'corpus':<10}{'model':<10}{'node ours':>10}{'DOT':>7}{'':>8}{'edge ours':>10}{'DOT':>7}{'':>8}"]
    for corpus in ("sciflow", "flowvqa"):
        for m in MODELS:
            runs = runs_of(pipe, corpus, m)
            d = dot[(dot.variant == "dot") & (dot.corpus == corpus) & (dot.model == m)]
            cells = []
            for col in ("node_f1", "edge_f1"):
                ours = seed_averaged(runs, col)
                rivals = {"DOT": keyed(d, col)}
                if corpus == "sciflow":
                    rivals.update({s: keyed(pub[pub.system == s], col) for s in SHIPPED})
                res = {k: compare(ours, v) for k, v in rivals.items()}
                res = {k: v for k, v in res.items() if v}
                adj = dict(zip(res, holm([v["p"] for v in res.values()])))
                cells.append(f"{across_runs(runs, col)[0]:>10.3f}{scores(d, col).mean():>7.3f}"
                             f"{verdict(res['DOT']['r'], adj['DOT']):>8}")
            out.append(f"{corpus:<10}{m:<10}" + "".join(cells))
    return "\n".join(out)


def table13(pipe, judge):
    out = ["Data table 13  test sets without / with the coverage call (Holm over the three models)"]
    rows = {"PaperBanana judge": [], "FlowVQA score": [], "FlowVQA edge F1": [], "SciFlow score": []}
    for label, corpus, col in (("FlowVQA score", "flowvqa", "score"), ("FlowVQA edge F1", "flowvqa", "edge_f1"),
                               ("SciFlow score", "sciflow", "score")):
        res = {}
        for m in MODELS:
            a, b = runs_of(pipe, corpus, m), runs_of(pipe, corpus, m, "coverage")
            res[m] = (across_runs(a, col)[0], across_runs(b, col)[0],
                      compare(seed_averaged(a, col), seed_averaged(b, col)))
        adj = holm([res[m][2]["p"] for m in MODELS])
        rows[label] = [(x, y, verdict(r["r"], p)) for (x, y, r), p in zip(res.values(), adj)]
    res = {}
    for m in MODELS:
        a = judge_items(judge, "InternVL3.5-14B", "ours", m, seeds=[0, 1, 2])[0]
        b = judge_items(judge, "InternVL3.5-14B", "ours_coverage", m, seeds=[0, 1, 2])[0]
        av, bv = {k: v["overall"] for k, v in a.items()}, {k: v["overall"] for k, v in b.items()}
        res[m] = (st.fmean(av.values()), st.fmean(bv.values()), compare(av, bv))
    adj = holm([res[m][2]["p"] for m in MODELS])
    rows["PaperBanana judge"] = [(x, y, verdict(r["r"], p)) for (x, y, r), p in zip(res.values(), adj)]
    out.append(f"{'':<20}" + "".join(f"{m:>26}" for m in MODELS))
    for label, cells in rows.items():
        fmt = "{:.1f} / {:.1f} ({})" if label.startswith("Paper") else "{:.3f} / {:.3f} ({})"
        out.append(f"{label:<20}" + "".join(f"{fmt.format(*c):>26}" for c in cells))
    out.append("(verdict: without the call vs with it)")
    return "\n".join(out)


def table14(pipe, dot):
    out = ["Data table 14  FlowVQA score per source: ours / DOT / DOT from our prompts "
           "(verdict ours vs DOT and ours vs matched; Holm over the three models per source)"]
    for source in ("wiki", "instruct", "code"):
        res, cells = {}, {}
        for m in MODELS:
            runs = [r[r.domain == source] for r in runs_of(pipe, "flowvqa", m)]
            ours = seed_averaged(runs, "score")
            d = dot[(dot.corpus == "flowvqa") & (dot.model == m) & (dot.domain == source)]
            plain, mat = d[d.variant == "dot"], d[d.variant == "dot_matched"]
            res[m] = (compare(ours, keyed(plain, "score")), compare(ours, keyed(mat, "score")))
            cells[m] = (st.fmean(ours.values()), scores(plain, "score").mean(), scores(mat, "score").mean())
        a1 = holm([res[m][0]["p"] for m in MODELS])
        a2 = holm([res[m][1]["p"] for m in MODELS])
        out.append(f"{source:<14}" + "".join(
            f"   {m}: {o:.3f} / {d:.3f} / {mm:.3f} ({verdict(res[m][0]['r'], p1)}, {verdict(res[m][1]['r'], p2)})"
            for m, (o, d, mm), p1, p2 in zip(MODELS, cells.values(), a1, a2)))
    return "\n".join(out)


def table15(judge):
    out = ["Data table 15  PaperBananaBench seed 0, ours minus the same model's DOT figures "
           "(Holm over the three models per dimension)"]
    for judge_name, dims in (("InternVL3.5-14B", ("overall",)), ("Qwen2.5-VL-32B", DIMS)):
        data = {m: (judge_items(judge, judge_name, "ours", m, seeds=[0])[0],
                    judge_items(judge, judge_name, "dot", m)[0]) for m in MODELS}
        for d in dims:
            res = {m: compare({k: v[d] for k, v in data[m][0].items()},
                              {k: v[d] for k, v in data[m][1].items()}) for m in MODELS}
            adj = holm([res[m]["p"] for m in MODELS])
            out.append(f"  {judge_name:<16}{d:<14}" + "".join(
                f"{m:>10} {res[m]['diff']:+6.1f}{'*' if p < ALPHA else ' '}" for m, p in zip(MODELS, adj)))
    out.append("(* significant)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="*", type=int)
    want = set(ap.parse_args().table or [])
    show = lambda n: not want or n in want  # noqa: E731
    pipe = load("pipeline_items.csv")
    dot = load("dot_items.csv")
    pub = load("published_systems_sciflow.csv")
    judge = load("judge_paperbanana.csv")
    blocks = []
    if show(2):
        blocks.append(table2(pipe, dot, pub, judge))
    if show(3):
        blocks.append(table3(pipe, pub, load("readback_sciflow.csv")))
    if show(4):
        blocks.append(table4(judge))
    lay = load("layout_items.csv") if (show(5) or show(8) or show(12)) else None
    if show(5) or show(12):
        blocks.append(table5_12(lay))
    if show(6):
        blocks.append(table6(load("placement_items.csv")))
    if show(7):
        blocks.append(table7(pipe))
    if show(8):
        blocks.append(table8(lay))
    if show(9):
        blocks.append(table9(load("human_stimuli.csv"), load("human_sets.csv"), load("human_ratings.csv")))
    if show(11):
        blocks.append(table11(pipe, dot, pub))
    if show(13):
        blocks.append(table13(pipe, judge))
    if show(14):
        blocks.append(table14(pipe, dot))
    if show(15):
        blocks.append(table15(judge))
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
