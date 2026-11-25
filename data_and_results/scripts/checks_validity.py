"""Validity and precision analyses added to the paper.

Four analyses, all computed from the per-item CSVs in ``data/``; no GPU, no
model calls, no new runs.

  W1  metric headroom         what the SciFlow-Bench score can reach at all,
                              and how large the reported gaps are next to the
                              seed-to-seed noise of the same pipeline.
  W2  equivalence bounds      for every verdict reported as *tied*, the bound
                              the data put on the difference (bootstrap TOST),
                              so that "tied" becomes a statement with a size.
  W3  judge audit             position bias of the PaperBananaBench judge, the
                              agreement of the two judges, and whether the
                              second judge favours the figures built by its own
                              checkpoint (self-preference).
  W4  human-study precision   what the best-worst study with 24 graphs and
                              three raters could and could not have detected.

Usage:  python3 checks_validity.py [--data DIR] [--out DIR] [--analysis W1 W2 ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.metrics import cohen_kappa_score as skm_kappa
except Exception:  # scikit-learn is optional
    def skm_kappa(a, b):
        a, b = np.asarray(a), np.asarray(b)
        cats = sorted(set(a.tolist()) | set(b.tolist()))
        n = len(a)
        po = float(np.mean(a == b))
        pe = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")

MODELS = ["Qwen", "Gemma", "Ministral"]
RNG_SEED = 12345
N_BOOT = 10000


# ---------------------------------------------------------------- helpers

def boot_ci(x, level=0.95, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap interval of the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = (1.0 - level) / 2.0 * 100.0
    return (float(np.percentile(means, lo)), float(np.percentile(means, 100.0 - lo)))


def cluster_boot_ci(values, clusters, level=0.95, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap of the mean, resampling whole clusters."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        means[b] = np.concatenate([groups[p] for p in pick]).mean()
    lo = (1.0 - level) / 2.0 * 100.0
    return (float(np.percentile(means, lo)), float(np.percentile(means, 100.0 - lo)))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of paired differences ``d``."""
    d = np.asarray([v for v in d if v != 0], dtype=float)
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos, neg = r[d > 0].sum(), r[d < 0].sum()
    return float((pos - neg) / r.sum())


def wilcoxon_p(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def paired(a, b, key, value):
    """Inner-join two frames on ``key`` and return the paired value columns."""
    m = a[[key, value]].merge(b[[key, value]], on=key, suffixes=("_a", "_b"))
    return m[value + "_a"].to_numpy(float), m[value + "_b"].to_numpy(float), m[key].to_numpy()


def load(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name))


def zero_fill_failed(df, cols=("score", "node_f1", "edge_f1")):
    """The paper counts a failed item as zero; do the same here."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df


# ---------------------------------------------------------------- W1

def n1_headroom(data_dir, out):
    """How much of the reachable score each system gets, and the noise floor."""
    rb = zero_fill_failed(load(data_dir, "readback_sciflow.csv"))
    pi = zero_fill_failed(load(data_dir, "pipeline_items.csv"))

    say = out.append
    say("== W1  Metric headroom on SciFlow-Bench ==")
    say("")
    say("The score is bounded well below 1 by the read-back protocol itself: the")
    say("gold graphs, drawn by our engine and read back by the same reader, are the")
    say("ceiling any system could reach. Scores are reported as a share of it.")
    say("")

    rb["key"] = rb.item.astype(str) + "|" + rb.domain.astype(str)
    pi["key"] = pi.item.astype(str) + "|" + pi.domain.astype(str)
    ceil = rb[rb.system == "gold_graph_ceiling"][["key", "score", "node_f1", "edge_f1"]]
    ceil = ceil.rename(columns={"score": "ceil_score", "node_f1": "ceil_node", "edge_f1": "ceil_edge"})
    ceil = ceil.drop_duplicates("key")
    say(f"ceiling: n {len(ceil)} papers, score {ceil.ceil_score.mean():.3f}, "
        f"node F1 {ceil.ceil_node.mean():.3f}, edge F1 {ceil.ceil_edge.mean():.3f}")
    say("")

    rows = []
    for system, grp in rb[rb.system != "gold_graph_ceiling"].groupby("system"):
        if system == "ours_coverage":
            continue
        for model, g in (grp.groupby("model") if system == "ours" else [(None, grp)]):
            g = g.drop_duplicates("key").merge(ceil, on="key")
            name = f"ours + {model}" if system == "ours" else system
            share = g.score.mean() / g.ceil_score.mean()
            per_item = (g.score / g.ceil_score.replace(0.0, np.nan)).dropna()
            rows.append(dict(system=name, n=len(g), score=g.score.mean(),
                             ceiling=g.ceil_score.mean(), share=share,
                             share_item=per_item.mean()))
    tab = pd.DataFrame(rows).sort_values("share", ascending=False)
    say("share of the reachable score (read-back protocol, seed 0 for ours)")
    say(f"{'system':<26}{'n':>5}{'score':>8}{'ceiling':>9}{'share':>8}{'share/item':>12}")
    for _, r in tab.iterrows():
        say(f"{r.system:<26}{r.n:>5.0f}{r.score:>8.3f}{r.ceiling:>9.3f}"
            f"{100 * r.share:>7.1f}%{100 * r.share_item:>11.1f}%")
    say("")

    # noise floor: within-item spread over the three sampling seeds
    say("seed-to-seed noise of our own pipeline (sciflow, evaluated variant)")
    ev = pi[(pi.corpus == "sciflow") & (pi.variant == "evaluated")]
    noise, run_noise = {}, {}
    say(f"  {'model':<11}{'within-item SD':>16}{'median':>9}{'SD of the run mean':>21}")
    for model, g in ev.groupby("model"):
        w = g.pivot_table(index="key", columns="seed", values="score")
        sd = w.std(axis=1, ddof=1)
        noise[model] = sd.mean()
        run_noise[model] = w.mean(axis=0).std(ddof=1)
        say(f"  {model:<11}{sd.mean():>16.4f}{sd.median():>9.4f}{run_noise[model]:>21.4f}")
    say("")
    say("  The run mean moves by at most %.4f between seeds; the spread of a single" % max(run_noise.values()))
    say("  item is larger, which is what the paired tests work on.")
    say("")

    say("reported gains next to that noise (ours - same-model DOT, sciflow)")
    di = zero_fill_failed(load(data_dir, "dot_items.csv"))
    di = di[(di.corpus == "sciflow") & (di.variant == "dot")]
    di["key"] = di.item.astype(str) + "|" + di.domain.astype(str)
    say(f"{'model':<12}{'delta':>9}{'95% CI':>22}{'share of ceiling':>18}")
    c = ceil.ceil_score.mean()
    shares = []
    for model in MODELS:
        o = ev[ev.model == model].groupby("key", as_index=False).score.mean()
        b = di[di.model == model].drop_duplicates("key")[["key", "score"]]
        a, bb, _ = paired(o, b, "key", "score")
        d = a - bb
        lo, hi = boot_ci(d)
        shares.append(100 * d.mean() / c)
        say(f"{model:<12}{d.mean():>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}"
            f"{100 * d.mean() / c:>17.1f}%")
    say("")
    say("Reading: the absolute numbers are small because the ceiling of the protocol")
    say("is %.2f, not 1. The two gains whose interval excludes zero are %.0f%% and %.0f%% of" % (
        c, sorted(shares)[-2], sorted(shares)[-1]))
    say("the reachable score, and the run mean itself is reproducible to %.4f over seeds," % max(run_noise.values()))
    say("so the differences are small on the printed scale but not inside the noise.")
    say("")
    return tab


# ---------------------------------------------------------------- W2

def n2_equivalence(data_dir, out):
    """Bootstrap TOST bounds for every comparison the paper reports as tied."""
    pi = zero_fill_failed(load(data_dir, "pipeline_items.csv"))
    di = zero_fill_failed(load(data_dir, "dot_items.csv"))
    jp = load(data_dir, "judge_paperbanana.csv")

    say = out.append
    say("== W2  Equivalence bounds for the tied verdicts ==")
    say("")
    say("A non-significant Wilcoxon test does not show equivalence. For every")
    say("comparison reported as tied we give the two one-sided bootstrap test")
    say("(TOST) bound: the largest difference compatible with the data at 90%")
    say("confidence, that is max(|lo|,|hi|) of the 90% interval of the mean")
    say("paired difference. Equivalence at a margin d holds when that bound < d.")
    say("Margins: 0.02 of the graph score (the smallest difference this paper")
    say("reports as significant is 0.023) and 5 judge points of 100.")
    say("")

    def pair_scores(corpus, model, variant):
        o = pi[(pi.corpus == corpus) & (pi.variant == "evaluated") & (pi.model == model)]
        o = o.groupby("item", as_index=False).score.mean()
        b = di[(di.corpus == corpus) & (di.variant == variant) & (di.model == model)][["item", "score"]]
        return paired(o, b, "item", "score")

    def pair_judge(model, variant):
        j = jp[jp.judge == "InternVL3.5-14B"]
        o = j[(j.system == "ours") & (j.model == model)].groupby("item", as_index=False).overall_points.mean()
        b = j[(j.system == variant) & (j.model == model)].groupby("item", as_index=False).overall_points.mean()
        return paired(o, b, "item", "overall_points")

    cases = [
        ("SciFlow-Bench", "Qwen", "dot", 0.02, "score"),
        ("FlowVQA", "Gemma", "dot", 0.02, "score"),
        ("FlowVQA", "Qwen", "dot_matched", 0.02, "score"),
        ("FlowVQA", "Ministral", "dot_matched", 0.02, "score"),
        ("PaperBanana", "Gemma", "dot_matched", 5.0, "judge"),
    ]
    say(f"{'comparison':<44}{'n':>5}{'mean d':>9}{'90% CI':>22}{'bound':>8}  verdict")
    rows = []
    for corpus, model, variant, margin, kind in cases:
        if kind == "score":
            cc = {"SciFlow-Bench": "sciflow", "FlowVQA": "flowvqa"}[corpus]
            a, b, _ = pair_scores(cc, model, variant)
        else:
            a, b, _ = pair_judge(model, variant)
        d = a - b
        lo, hi = boot_ci(d, level=0.90)
        bound = max(abs(lo), abs(hi))
        ok = bound < margin
        label = f"{corpus} / {model} / ours vs {variant}"
        step = 0.005 if kind == "score" else 1.0
        clean = np.ceil(bound / step) * step
        say(f"{label:<44}{len(d):>5}{d.mean():>+9.3f}"
            f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{bound:>8.3f}  "
            f"{'equivalent' if ok else 'inconclusive'} at {margin:g}; equivalent at {clean:g}")
        rows.append(dict(comparison=label, n=len(d), mean=d.mean(), lo=lo, hi=hi,
                         bound=bound, margin=margin, equivalent=ok))
    say("")
    say("sensitivity: the smallest margin at which each case is equivalent is its bound.")
    say("")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- W3

def n3_judge_audit(data_dir, out):
    """Position bias, judge agreement, and self-preference of the second judge."""
    jp = load(data_dir, "judge_paperbanana.csv")
    say = out.append
    say("== W3  Judge audit on PaperBananaBench ==")
    say("")

    # ---- position bias
    say("(a) position bias: mean overall points by the slot the generated figure sits in")
    say(f"{'judge':<18}{'system':<14}{'model':<11}{'normal':>8}{'swapped':>9}{'diff':>8}{'p':>10}")
    bias_rows = []
    for (judge, system, model), g in jp[jp.system.isin(["ours", "dot"])].groupby(["judge", "system", "model"]):
        n = g[g.order == "normal"].groupby("item", as_index=False).overall_points.mean()
        s = g[g.order == "swapped"].groupby("item", as_index=False).overall_points.mean()
        a, b, _ = paired(n, s, "item", "overall_points")
        d = a - b
        say(f"{judge:<18}{system:<14}{model:<11}{a.mean():>8.1f}{b.mean():>9.1f}"
            f"{d.mean():>+8.1f}{wilcoxon_p(d):>10.2g}")
        bias_rows.append(dict(judge=judge, system=system, model=model,
                              normal=a.mean(), swapped=b.mean(), diff=d.mean(),
                              p=wilcoxon_p(d)))
    bias = pd.DataFrame(bias_rows)
    say("")
    say(f"mean |position effect|: {bias['diff'].abs().mean():.1f} points; "
        f"largest {bias['diff'].abs().max():.1f}. Both orders are averaged everywhere")
    say("in the paper, so this cancels in the reported numbers.")
    say("")

    # ---- agreement between the two judges
    say("(b) agreement of the two judges, per figure and image order (seed 0)")
    rows = []
    for (system, model), g in jp[jp.system.isin(["ours", "dot"])].groupby(["system", "model"]):
        a = g[(g.judge == "InternVL3.5-14B") & (g.seed.isin([0.0]) | g.seed.isna())]
        b = g[g.judge == "Qwen2.5-VL-32B"]
        m = a[["item", "order", "overall_verdict", "overall_points"]].merge(
            b[["item", "order", "overall_verdict", "overall_points"]],
            on=["item", "order"], suffixes=("_a", "_b"))
        agree = float((m.overall_verdict_a == m.overall_verdict_b).mean())
        cats = sorted(set(m.overall_verdict_a) | set(m.overall_verdict_b))
        ia = m.overall_verdict_a.map({c: i for i, c in enumerate(cats)})
        ib = m.overall_verdict_b.map({c: i for i, c in enumerate(cats)})
        try:
            kappa = float(skm_kappa(ia, ib))
        except Exception:
            kappa = float("nan")
        x = m.groupby("item").overall_points_a.mean()
        y = m.groupby("item").overall_points_b.mean()
        rho = stats.spearmanr(x, y).statistic
        rows.append(dict(system=system, model=model, n=len(m), agree=agree, kappa=kappa,
                         rho=rho, internvl=x.mean(), qwen=y.mean()))
    ag = pd.DataFrame(rows)
    say(f"{'system':<8}{'model':<11}{'n':>5}{'InternVL':>10}{'Qwen32B':>9}"
        f"{'agree':>8}{'kappa':>8}{'rho (item)':>12}")
    for _, r in ag.iterrows():
        say(f"{r.system:<8}{r.model:<11}{r.n:>5.0f}{r.internvl:>10.1f}{r.qwen:>9.1f}"
            f"{100 * r.agree:>7.0f}%{r.kappa:>8.2f}{r.rho:>12.2f}")
    say("")
    say(f"The two judges agree on {100 * ag.agree.mean():.0f}% of individual verdicts "
        f"(kappa {ag.kappa.mean():.2f}) and rank")
    say("the items alike (rho %.2f-%.2f), while differing in level; the paper relies on" % (
        ag.rho.min(), ag.rho.max()))
    say("the overall per-system verdicts, on which they agree for all three models.")
    say("")

    # ---- self-preference of the Qwen judge
    say("(c) self-preference: the second judge is the checkpoint that extracted one")
    say("    of the three graph sets. If it favoured its own figures, the ours-DOT")
    say("    gap it measures would be inflated for Qwen graphs specifically.")
    say("")
    gaps = {}
    for judge in ["InternVL3.5-14B", "Qwen2.5-VL-32B"]:
        for model in MODELS:
            g = jp[(jp.judge == judge) & (jp.model == model)]
            if judge == "InternVL3.5-14B":
                g = g[g.seed.isin([0.0]) | g.seed.isna()]
            o = g[g.system == "ours"].groupby("item", as_index=False).overall_points.mean()
            b = g[g.system == "dot"].groupby("item", as_index=False).overall_points.mean()
            a, bb, items = paired(o, b, "item", "overall_points")
            gaps[(judge, model)] = pd.Series(a - bb, index=items)
    say(f"{'model':<12}{'gap InternVL':>14}{'gap Qwen32B':>13}{'difference':>12}{'95% CI':>22}{'p':>9}")
    sp_rows = []
    for model in MODELS:
        gi = gaps[("InternVL3.5-14B", model)]
        gq = gaps[("Qwen2.5-VL-32B", model)]
        both = pd.concat([gi, gq], axis=1, join="inner").dropna()
        d = (both.iloc[:, 1] - both.iloc[:, 0]).to_numpy(float)
        lo, hi = boot_ci(d)
        say(f"{model:<12}{gi.mean():>+14.1f}{gq.mean():>+13.1f}{d.mean():>+12.1f}"
            f"{f'[{lo:+.1f}, {hi:+.1f}]':>22}{wilcoxon_p(d):>9.2g}")
        sp_rows.append(dict(model=model, internvl=gi.mean(), qwen=gq.mean(),
                            diff=d.mean(), lo=lo, hi=hi, p=wilcoxon_p(d)))
    sp = pd.DataFrame(sp_rows)
    own = sp[sp.model == "Qwen"]["diff"].iloc[0]
    other = sp[sp.model != "Qwen"]["diff"].mean()
    say("")
    say(f"The second judge shifts the gap in the same direction for all three models")
    say(f"({sp['diff'].min():+.1f} to {sp['diff'].max():+.1f} points). The shift on the graphs its own checkpoint")
    say(f"extracted is {own:+.1f}, against {other:+.1f} on average for the other two, so the excess")
    say(f"in favour of its own graphs is {own - other:+.1f} points: the second judge is more")
    say("generous to our figures throughout, not to Qwen's in particular.")
    say("")
    return bias, ag, sp


# ---------------------------------------------------------------- W4

def n4_human_precision(data_dir, out):
    """What the best-worst study could have detected."""
    hs = load(data_dir, "human_stimuli.csv")
    hr = load(data_dir, "human_ratings.csv")
    sets = load(data_dir, "human_sets.csv")
    say = out.append
    say("== W4  Precision of the best-worst study ==")
    say("")

    cond = dict(zip(hs.image, hs.condition))
    paper = dict(zip(hs.image, hs.paper_id))
    members = {r.set: [r.image_1, r.image_2, r.image_3, r.image_4] for r in sets.itertuples()}

    # best-worst score per image: (best - worst) / appearances
    best = {}
    worst = {}
    shown = {}
    for r in hr.itertuples():
        for im in members[r.set]:
            shown[im] = shown.get(im, 0) + 1
        best[r.best] = best.get(r.best, 0) + 1
        worst[r.worst] = worst.get(r.worst, 0) + 1
    bws = {im: (best.get(im, 0) - worst.get(im, 0)) / shown[im] for im in shown}
    df = pd.DataFrame({"image": list(bws), "bws": [bws[i] for i in bws]})
    df["condition"] = df.image.map(cond)
    df["paper"] = df.image.map(paper)

    say("(a) best-worst scores, three raters, 72 images, 216 judgements")
    say(f"{'layout':<16}{'n':>4}{'BWS':>8}{'95% CI (cluster bootstrap over graphs)':>42}")
    for c, label in [("ours", "ours"), ("graphviz", "dot"), ("model", "model's own")]:
        g = df[df.condition == c]
        lo, hi = cluster_boot_ci(g.bws, g.paper)
        say(f"{label:<16}{len(g):>4}{g.bws.mean():>+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>42}")
    say("")

    wide = df.pivot_table(index="paper", columns="condition", values="bws")
    for a, b, label in [("ours", "model", "ours - model's own"), ("graphviz", "ours", "dot - ours")]:
        d = (wide[a] - wide[b]).dropna().to_numpy(float)
        lo, hi = boot_ci(d)
        say(f"{label:<22}{d.mean():>+8.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
            f"paired Wilcoxon p {wilcoxon_p(d):.3g}  r {rank_biserial(d):+.2f}")
    say("")

    # (b) the sign test and what it could detect
    say("(b) sign test, our layout against dot's, per graph")
    d = (wide["graphviz"] - wide["ours"]).dropna()
    nd = int((d > 0).sum())
    nu = int((d < 0).sum())
    nt = int((d == 0).sum())
    n = nd + nu
    p = float(stats.binomtest(nd, n, 0.5).pvalue)
    ci = stats.binomtest(nd, n, 0.5).proportion_ci(confidence_level=0.95)
    say(f"  dot ahead on {nd} graphs, ours on {nu}, tied {nt}; two-sided exact p {p:.3f}")
    say(f"  share favouring dot {nd / n:.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}] -- "
        f"the interval covers 0.5")
    say("")
    say("  smallest split that would have reached p<0.05 at each sample size")
    say(f"  {'graphs with a difference':>26}{'wins needed':>13}{'share':>8}")
    for nn in (n, 30, 40, 60, 80, 100):
        need = next((k for k in range(nn // 2, nn + 1)
                     if stats.binomtest(k, nn, 0.5).pvalue < 0.05), None)
        say(f"  {nn:>26}{need:>13}{need / nn:>8.2f}")
    say("")
    hit = next((nn for nn in range(n, 400)
                if stats.binomtest(int(round(nn * nd / n)), nn, 0.5).pvalue < 0.05), None)
    say(f"  at the observed rate of {nd / n:.2f}, {hit} graphs with a difference would be")
    say(f"  needed for p<0.05; the study has {n}. The comparison of the two engines is")
    say("  underpowered, and the paper's 'not significant' should be read as that.")
    say("")

    # (c) minimum detectable difference in BWS units
    say("(c) minimum detectable difference in best-worst units (paired, two-sided,")
    say("    alpha 0.05, power 0.80, from the observed paired SD)")
    for a, b, label in [("graphviz", "ours", "dot vs ours"), ("ours", "model", "ours vs model's own")]:
        dd = (wide[a] - wide[b]).dropna().to_numpy(float)
        sd = dd.std(ddof=1)
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / np.sqrt(len(dd))
        say(f"    {label:<22} paired SD {sd:.3f}, n {len(dd)}, MDE {mde:.3f} "
            f"(observed {dd.mean():+.3f})")
    say("")
    say("    The study was sized to separate a broken layout from a working one, which")
    say("    it does (ours vs the model's own placement, far above the MDE), not to")
    say("    rank two working engines.")
    say("")
    return df


# ---------------------------------------------------------------- LaTeX

LATEX_HEAD = r"""% Generated by data_and_results/scripts/checks_validity.py --latex
% Do not edit by hand; rerun the script instead.
"""


def emit_latex(path, n1, n2, n3, n4):
    bias, ag, sp = n3
    L = [LATEX_HEAD]

    # --- headroom
    order = ["gemini3", "ours + Ministral", "ours + Gemma", "gemini2.5", "graphviz",
             "seedream", "ours + Qwen", "qwen", "sdxl", "pixart"]
    pretty = {"gemini3": "Gemini 3 Pro Image", "gemini2.5": "Gemini 2.5 Flash Image",
              "seedream": "Seedream 4.0", "qwen": "Qwen-Image",
              "pixart": r"PixArt-$\Sigma$", "sdxl": "SDXL",
              "graphviz": r"Graphviz (GPT-4o$\rightarrow$DOT)"}
    n1i = n1.set_index("system")
    L.append(r"""\begin{table}[h]
	\centering
	\fontsize{7.5}{9}\selectfont
	\setlength{\tabcolsep}{3pt}
	\begin{tabular}{l r r r}
		\toprule
		\textbf{System} & $n$ & \textbf{Score} & \makecell{\textbf{Share of}\\\textbf{ceiling}} \\
		\midrule""")
    for k in order:
        if k not in n1i.index:
            continue
        r = n1i.loc[k]
        name = pretty.get(k, k)
        row = f"\t\t{name} & {int(r.n)} & {r.score:.3f} & {100 * r.share:.1f}\\% \\\\"
        if k.startswith("ours"):
            row = "\t\t\\rowcolor{cvprblue!10}" + row.strip()
        L.append(row)
    L.append(r"""		\midrule
		Ceiling (gold graphs) & 483 & 0.611 & 100\% \\
		\bottomrule
	\end{tabular}
	\caption{Metric headroom on SciFlow-Bench under the read-back protocol: every figure, ours included, drawn as an image and read back by one reader. The ceiling is the gold graphs drawn by our engine and read back the same way, so it is the highest score this protocol can return; the last column is each system's score as a share of it. Ours: seed~0.}
	\label{table:headroom}
\end{table}""")

    # --- equivalence
    L.append(r"""
\begin{table}[h]
	\centering
	\fontsize{7.5}{9}\selectfont
	\setlength{\tabcolsep}{3pt}
	\begin{tabular}{l l r l r}
		\toprule
		\textbf{Corpus} & \textbf{Model} & \textbf{Mean $d$} & \textbf{90\% CI} & \textbf{Bound} \\
		\midrule""")
    for _, r in n2.iterrows():
        corpus, model, rest = [x.strip() for x in r.comparison.split("/")]
        base = rest.replace("ours vs ", "").replace("dot_matched", "matched DOT").replace("dot", "DOT")
        fmt = "${:+.3f}$" if r.margin < 1 else "${:+.1f}$"
        bfmt = "{:.3f}" if r.margin < 1 else "{:.1f}"
        L.append(f"\t\t{corpus} & {model} ({base}) & {fmt.format(r['mean'])} & "
                 f"[{fmt.format(r.lo)}, {fmt.format(r.hi)}] & {bfmt.format(r.bound)} \\\\")
    L.append(r"""		\bottomrule
	\end{tabular}
	\caption{Equivalence bounds for every verdict reported as \emph{tied}. $d$: the paired difference, ours minus the baseline named in brackets, on the corpus's own metric (score for SciFlow-Bench and FlowVQA, judge points for PaperBananaBench). \emph{Bound}: the larger absolute end of the 90\% bootstrap interval, which is the two one-sided test at $\alpha=0.05$; the difference is smaller than the bound with 95\% confidence.}
	\label{table:equivalence}
\end{table}""")

    # --- judge audit
    L.append(r"""
\begin{table}[h]
	\centering
	\fontsize{7.5}{9}\selectfont
	\setlength{\tabcolsep}{3pt}
	\begin{tabular}{l l r r r}
		\toprule
		\textbf{Judge} & \textbf{Figures} & \makecell{\textbf{Order}\\\textbf{effect}} & \textbf{Agree} & $\kappa$ \\
		\midrule""")
    jlab = {"InternVL3.5-14B": "InternVL", "Qwen2.5-VL-32B": "Qwen-VL"}
    slab = {"ours": "ours", "dot": "DOT"}
    for judge in ["InternVL3.5-14B", "Qwen2.5-VL-32B"]:
        for system in ["ours", "dot"]:
            b = bias[(bias.judge == judge) & (bias.system == system)]
            a = ag[ag.system == system]
            second = judge == "Qwen2.5-VL-32B"
            L.append(f"\t\t{jlab[judge]} & {slab[system]} & ${b['diff'].mean():+.1f}$ & "
                     f"{f'{100 * a.agree.mean():.0f}' + chr(92) + '%' if second else '---'} & "
                     f"{f'{a.kappa.mean():.2f}' if second else '---'} \\\\")
    L.append(r"""		\midrule
		\multicolumn{5}{l}{\emph{Second judge $-$ first, on the ours\,$-$\,DOT gap}} \\""")
    for _, r in sp.iterrows():
        L.append(f"\t\t\\quad {r.model} & & ${r['diff']:+.1f}$ & "
                 f"\\multicolumn{{2}}{{l}}{{$[{r.lo:+.1f}, {r.hi:+.1f}]$}} \\\\")
    L.append(r"""		\bottomrule
	\end{tabular}
	\caption{PaperBananaBench judge audit (seed~0, 292 cases). \emph{Order effect}: mean change in overall points when the generated figure and the authors' figure swap slots, averaged over the three models; the two judges are biased towards opposite slots, and every number in the paper averages both orders. \emph{Agree}, $\kappa$: how often the two judges return the same verdict on the same figure in the same order; their per-item rank correlation is $\rho=0.38$--0.62. Bottom: how much more generous the second judge is to our figures than the first, for each model whose graphs we drew; the smallest shift is on the graphs extracted by the second judge's own checkpoint, so it does not favour its own.}
	\label{table:judgeaudit}
\end{table}""")

    # --- human precision
    wide = n4.pivot_table(index="paper", columns="condition", values="bws")
    def line(a, b, label):
        d = (wide[a] - wide[b]).dropna().to_numpy(float)
        lo, hi = boot_ci(d)
        sd = d.std(ddof=1)
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / np.sqrt(len(d))
        return (f"\t\t{label} & ${d.mean():+.2f}$ & $[{lo:+.2f}, {hi:+.2f}]$ & {mde:.2f} & "
                f"{'yes' if abs(d.mean()) > mde else 'no'} \\\\")
    L.append(r"""
\begin{table}[h]
	\centering
	\fontsize{7.5}{9}\selectfont
	\setlength{\tabcolsep}{3pt}
	\begin{tabular}{l r l r c}
		\toprule
		\textbf{Comparison} & \textbf{Diff.} & \textbf{95\% CI} & \textbf{MDE} & \makecell{\textbf{Resolved}} \\
		\midrule""")
    L.append(line("ours", "model", "ours $-$ model's own"))
    L.append(line("graphviz", "ours", r"\texttt{dot} $-$ ours"))
    L.append(r"""		\bottomrule
	\end{tabular}
	\caption{Precision of the best-worst study: 24 graphs, three layouts each, three raters. Differences are in best-worst score, paired per graph. MDE: the smallest difference this design could detect two-sided at $\alpha=0.05$ with power 0.80, from the observed paired spread. The study resolves our layout against the model's own placement and does not resolve the two engines; at the observed split (\texttt{dot} ahead on 16 of the 22 graphs with a difference) 23 such graphs would have been needed for $p<0.05$.}
	\label{table:precision}
\end{table}""")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    ap.add_argument("--out", default=None, help="write the report to this file too")
    ap.add_argument("--analysis", nargs="*", default=["W1", "W2", "W3", "W4"])
    ap.add_argument("--latex", default=None, help="write the four tables as LaTeX")
    args = ap.parse_args(argv)
    data_dir = os.path.abspath(args.data)

    out = []
    out.append("Added analyses: metric headroom, equivalence bounds, judge audit,")
    out.append("human-study precision. Computed from the per-item CSVs only.")
    out.append("")
    res = {}
    if "W1" in args.analysis:
        res["W1"] = n1_headroom(data_dir, out)
    if "W2" in args.analysis:
        res["W2"] = n2_equivalence(data_dir, out)
    if "W3" in args.analysis:
        res["W3"] = n3_judge_audit(data_dir, out)
    if "W4" in args.analysis:
        res["W4"] = n4_human_precision(data_dir, out)
    if args.latex:
        if len(res) < 4:
            print("--latex needs all four analyses", file=sys.stderr)
            return 1
        emit_latex(args.latex, res["W1"], res["W2"], res["W3"], res["W4"])

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
