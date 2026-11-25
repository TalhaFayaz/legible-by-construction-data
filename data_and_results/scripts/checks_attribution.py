#!/usr/bin/env python3
"""Where the legibility difference actually comes from.

    python3 checks_attribution.py                      # all seven
    python3 checks_attribution.py --analysis A1 A2
    python3 checks_attribution.py --latex ../results/tables_attribution.tex
    python3 checks_attribution.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| A1 criteria | which of the six strict criteria carries the difference to `dot`     | 5.2, Table S2a |
| A2 blind    | the strict verdict with the two label criteria removed               | 5.2, Table S2a |
| A3 engine   | how much of the difference is the layout stages, not the schema       | 5.2, Table S2b |
| A4 select   | how much is choosing a candidate by the counts the verdict reports    | 5.2, 5.4 |
| A5 size     | the graph sizes the rater study covers, and where the difference lives | 5.3 |
| A6 matched  | the same comparison with the choice between candidates matched        | 5.2, Table S2a |
| A7 residual | what is left to fail, for each engine, and at which graph sizes       | 5.2, 8 |

All seven read `layout_items.csv` and `pipeline_items.csv` and nothing else: no
run, no condition and no model call is added, and none decides a reported
verdict. They answer the question a reader asks of any geometric comparison
between two engines -- which criterion is doing the work, and whether the
engine reporting the counts is also the engine selecting on them.

A1 and A2 split the strict verdict into its six criteria over the figures for
which all four `dot` configurations and our own layout are present, so every
row is the identical graph drawn five ways. A3 walks our own layout ablation
down to `none`, all six stages off, and puts it on the same axis as the `dot`
configurations. A4 compares the pipeline with candidate selection removed. A5
puts the 24 rated graphs on the size axis of A1, which is where the rater study
of 5.3 and the engine comparison of 5.2 turn out to be measuring different
parts of the same curve. A6 repeats the headline comparison with the selection
effort equal on both sides -- both engines choosing among candidates, or
neither -- because our shipped layout is the best of several variants picked by
the counts the verdict reports and a single dot configuration is not.
"""
import argparse, os
import pandas as pd, numpy as np
from scipy import stats

CSV, TEX = {}, {}

LENIENT = ["node_overlap_pairs", "edge_node_crossings", "route_reversals"]
LABEL = ["label_collisions", "edge_label_crossings"]
STRICT_EXTRA = LABEL + ["stacked_routes"]
NUMERIC = LENIENT + STRICT_EXTRA + ["valid", "n_nodes", "n_edges", "arrows",
                                    "loops", "nested", "total_edge_length"]
DOTS = ["dot_tb_spline", "dot_lr_spline", "dot_tb_ortho", "dot_lr_ortho"]
KEY = ["source", "corpus", "model", "run", "item"]
NAMES = {"ours": "ours", "dot_tb_spline": "dot top-down, curved",
         "dot_lr_spline": "dot left-right, curved",
         "dot_tb_ortho": "dot top-down, orthogonal",
         "dot_lr_ortho": "dot left-right, orthogonal"}
STAGES = {"full": "all six stages", "no_sideways": "no sideways pull",
          "no_variants_no_pull": "no variants, no pull", "no_variants": "no layout variants",
          "no_label_placement": "no label placement", "no_detours": "no obstacle detours",
          "no_spread_ends": "no spread arrow ends", "none": "all six stages off"}


def _load(D):
    d = pd.read_csv(D+"layout_items.csv", low_memory=False)
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["valid"]).copy()
    d["lenient"] = (d.node_overlap_pairs == 0) & (d.edge_node_crossings == 0) \
        & (d.route_reversals == 0)
    d["strict"] = d.lenient & (d.label_collisions == 0) \
        & (d.edge_label_crossings == 0) & (d.stacked_routes == 0)
    # label-blind: the strict verdict without the two criteria that need a label stage
    d["blind"] = d.lenient & (d.stacked_routes == 0)
    return d


def _paired(d):
    """The figures for which our layout and all four dot configurations exist."""
    gv = d[d.source.isin(["graphviz", "graphviz_outside_development"])]
    out = {}
    for v in ["strict", "blind", "lenient"] + LENIENT + STRICT_EXTRA:
        out[v] = gv.pivot_table(index=KEY, columns="config", values=v, aggfunc="first")
    idx = out["strict"].dropna(subset=["ours"]+DOTS).index
    return {k: v.loc[idx] for k, v in out.items()}, len(idx)


def _mcnemar(a, b):
    """Exact McNemar on two paired boolean columns."""
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    if b01+b10 == 0:
        return b01, b10, 1.0
    return b01, b10, float(stats.binomtest(b01, b01+b10, 0.5).pvalue)


def A1(D, P):
    w, n = _paired(_load(D))
    P(f"A1  which criterion carries the difference  ({n} identical graphs, five layouts each)")
    P()
    rows = []
    for c in ["ours"]+DOTS:
        r = {"config": NAMES[c], "strict_pct": w["strict"][c].mean()*100}
        for k in LENIENT+STRICT_EXTRA:
            r[k+"_pass_pct"] = (w[k][c] == 0).mean()*100
        r["mean_label_collisions"] = w["label_collisions"][c].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    CSV["attribution_criteria"] = t
    P(t.round(1).to_string(index=False))
    P()
    P("  The two criteria that need a label-placement stage -- no label collision and")
    P("  no arrow through a label -- are the only ones on which the two engines are")
    P("  more than 3.0 points apart (22.1 and 23.3 points). Graphviz has no label-")
    P("  placement stage: dot attaches a label to its edge's midpoint and never moves it.")
    P()
    return t


def A2(D, P):
    w, n = _paired(_load(D))
    P(f"A2  the strict verdict with the two label criteria removed  ({n} graphs)")
    P()
    rows = []
    for c in ["ours"]+DOTS:
        rows.append({"config": NAMES[c], "strict_pct": w["strict"][c].mean()*100,
                     "label_blind_pct": w["blind"][c].mean()*100,
                     "lenient_pct": w["lenient"][c].mean()*100})
    best = max(DOTS, key=lambda c: w["strict"][c].mean())
    orc_s = w["strict"][DOTS].max(axis=1).mean()*100
    orc_b = w["blind"][DOTS].max(axis=1).mean()*100
    rows.append({"config": "dot, oracle over its four", "strict_pct": orc_s,
                 "label_blind_pct": orc_b,
                 "lenient_pct": w["lenient"][DOTS].max(axis=1).mean()*100})
    t = pd.DataFrame(rows)
    CSV["attribution_label_blind"] = t
    P(t.round(1).to_string(index=False))
    P()
    s01, s10, ps = _mcnemar(w["strict"]["ours"], w["strict"][best])
    b01, b10, pb = _mcnemar(w["blind"]["ours"], w["blind"][best])
    P(f"  against {NAMES[best]}, the configuration with the highest strict share:")
    P(f"    strict      ours {w['strict']['ours'].mean()*100:.1f} vs "
      f"{w['strict'][best].mean()*100:.1f}  "
      f"(ours only {s01}, dot only {s10}, McNemar p={ps:.3g})")
    P(f"    label-blind ours {w['blind']['ours'].mean()*100:.1f} vs "
      f"{w['blind'][best].mean()*100:.1f}  "
      f"(ours only {b01}, dot only {b10}, McNemar p={pb:.3g})")
    P()
    P("  The sign reverses. Removing the two label criteria removes the whole")
    P("  difference and leaves dot ahead by 2.2 points (2.4 with the selection effort")
    P("  matched, A6), because our router stacks")
    P("  two routes on 3.4% of figures and dot almost never does. The difference")
    P("  this paper reports is a difference in label occlusion and nothing else.")
    P()
    return t


def A3(D, P):
    d = _load(D)
    ab = d[d.source == "layout_ablation"]
    gv = d[(d.source == "graphviz") & (d.config.isin(DOTS))]
    n = ab[ab.config == "full"].shape[0]
    P(f"A3  the schema or the layout stages  ({n} graphs, our engine stage by stage)")
    P()
    rows = []
    for c, lab in STAGES.items():
        s = ab[ab.config == c]
        if not len(s):
            continue
        rows.append({"layout": lab, "config": c, "strict_pct": s.strict.mean()*100,
                     "label_blind_pct": s.blind.mean()*100,
                     "lenient_pct": s.lenient.mean()*100,
                     "no_box_overlap_pct": (s.node_overlap_pairs == 0).mean()*100,
                     "no_arrow_through_box_pct": (s.edge_node_crossings == 0).mean()*100,
                     "no_route_reversal_pct": (s.route_reversals == 0).mean()*100})
    for c in DOTS:
        s = gv[gv.config == c]
        rows.append({"layout": NAMES[c], "config": c, "strict_pct": s.strict.mean()*100,
                     "label_blind_pct": s.blind.mean()*100,
                     "lenient_pct": s.lenient.mean()*100,
                     "no_box_overlap_pct": (s.node_overlap_pairs == 0).mean()*100,
                     "no_arrow_through_box_pct": (s.edge_node_crossings == 0).mean()*100,
                     "no_route_reversal_pct": (s.route_reversals == 0).mean()*100})
    t = pd.DataFrame(rows).sort_values("strict_pct", ascending=False)
    CSV["attribution_stages"] = t
    P(t.round(1).to_string(index=False))
    P()
    none = t[t.config == "none"].iloc[0]
    best = t[t.config == "dot_tb_spline"].iloc[0]
    P(f"  With all six layout stages off our own engine passes the strict check on")
    P(f"  {none.strict_pct:.1f}% of the identical graphs, against {best.strict_pct:.1f}% for")
    P(f"  dot top-down with curved edges. The coordinate-free schema is what makes the")
    P(f"  geometry computable; it is the layout stages that make it legible.")
    P(f"  Two of the three lenient criteria survive every ablation -- no box overlaps")
    P(f"  ({none.no_box_overlap_pct:.1f}%) and no route doubles back")
    P(f"  ({none.no_route_reversal_pct:.1f}%) hold with all six stages off, so they follow")
    P(f"  from laying out on a grid and marking back edges. The third does not:")
    P(f"  without the obstacle-aware router {100-none.no_arrow_through_box_pct:.1f}% of")
    P(f"  figures run an arrow through a box.")
    P()
    return t


def A4(D, P):
    d = _load(D)
    ab = d[d.source == "layout_ablation"]
    gv = d[(d.source == "graphviz") & (d.config == "dot_tb_spline")]
    P("A4  what choosing a candidate by the reported counts is worth")
    P()
    full = ab[ab.config == "full"]
    nov = ab[ab.config == "no_variants"]
    rows = [{"setting": "our engine, layout variants chosen by defect count",
             "strict_pct": full.strict.mean()*100, "label_blind_pct": full.blind.mean()*100},
            {"setting": "our engine, first layout kept",
             "strict_pct": nov.strict.mean()*100, "label_blind_pct": nov.blind.mean()*100},
            {"setting": "dot top-down, curved (no selection of any kind)",
             "strict_pct": gv.strict.mean()*100, "label_blind_pct": gv.blind.mean()*100}]
    p = pd.read_csv(D+"pipeline_items.csv", low_memory=False)
    p["layout_valid"] = pd.to_numeric(p.layout_valid, errors="coerce")
    for v, lab in [("evaluated", "pipeline, better of two sampled graphs"),
                   ("ablation_one_greedy_graph", "pipeline, one greedy graph")]:
        s = p[(p.variant == v) & (p.seed == 0)].dropna(subset=["layout_valid"])
        if len(s):
            rows.append({"setting": lab, "strict_pct": s.layout_valid.mean()*100,
                         "label_blind_pct": np.nan})
    t = pd.DataFrame(rows)
    CSV["attribution_selection"] = t
    P(t.round(1).to_string(index=False))
    P()
    P("  Keeping the first layout instead of the best of the variants costs")
    P(f"  {full.strict.mean()*100-nov.strict.mean()*100:.1f} points of the strict share, so part of the")
    P("  difference to dot is our engine selecting on the counts it then reports.")
    P("  dot is given the same chance in the oracle column of A2.")
    P()
    return t


def A5(D, P):
    d = _load(D)
    gv = d[d.source.isin(["graphviz", "graphviz_outside_development"])]
    gv = gv[gv.config.isin(["ours", "dot_tb_spline"])]
    bins, labs = [0, 8, 12, 20, 30, 10**6], ["8 or fewer", "9-12", "13-20", "21-30", "over 30"]
    b = pd.cut(gv.n_nodes, bins, labels=labs)
    g = gv.groupby([b, "config"], observed=True)
    strict = (g.strict.mean()*100).unstack()
    lab = g.label_collisions.mean().unstack()
    n = g.size().unstack()
    t = pd.DataFrame({"boxes": strict.index.astype(str),
                      "n_graphs": n["ours"].values,
                      "ours_strict_pct": strict["ours"].values,
                      "dot_strict_pct": strict["dot_tb_spline"].values,
                      "ours_label_collisions": lab["ours"].values,
                      "dot_label_collisions": lab["dot_tb_spline"].values})
    t["difference_pts"] = t.ours_strict_pct - t.dot_strict_pct
    CSV["attribution_by_size"] = t
    P("A5  where the difference lives, and what the rater study covered")
    P()
    P(t.round(2).to_string(index=False))
    P()
    st = pd.read_csv(D+"human_stimuli.csv")
    st["lenient"] = (st.report_node_overlap_pairs == 0) & (st.report_edge_node_crossings == 0) \
        & (st.report_route_reversals == 0)
    st["blind"] = st.lenient & (st.report_stacked_routes == 0)
    st["strict"] = st.blind & (st.report_label_collisions == 0) \
        & (st.report_edge_label_crossings == 0)
    eng = st[st.condition.isin(["ours", "graphviz"])]
    P(f"  the 24 rated graphs have {int(st.report_n_nodes.min())} to "
      f"{int(st.report_n_nodes.max())} boxes, median "
      f"{int(st.report_n_nodes.median())}")
    P("  over the two engine-drawn conditions of the rater study:")
    P(eng.groupby("condition")[["lenient", "blind", "strict"]].mean().round(3).to_string())
    P(eng.groupby("condition")[["report_label_collisions", "report_total_edge_length"]]
      .mean().round(2).to_string())
    P()
    P("  At the sizes rated, dot collides a label on 0.04 figures on average, so the")
    P("  strict verdict has almost nothing to separate there and the raters are")
    P("  deciding on something else -- our arrows are 2.7 times longer. The engine")
    P("  difference the strict verdict reports grows with graph size and is largest")
    P("  above 30 boxes, which the rater study did not cover. The two results are")
    P("  measuring different parts of the same curve, and neither licenses a ranking")
    P("  of the engines at the sizes the other covers.")
    P()
    return t


def A6(D, P):
    """The comparison with the selection effort matched on both sides."""
    d = _load(D)
    gv = d[d.source.isin(["graphviz", "graphviz_outside_development"])]
    ab = d[d.source == "layout_ablation"]
    K2 = ["corpus", "model", "run", "item"]   # across sources: the identical graph
    P("A6  the same comparison with the choice between candidates matched")
    P()

    def piv(df, v):
        return df.pivot_table(index=K2, columns="config", values=v, aggfunc="first")
    rows, tests = [], []
    for vname in ["strict", "label_blind"]:
        v = "strict" if vname == "strict" else "blind"
        g, a = piv(gv, v), piv(ab, v)
        full = g.dropna(subset=["ours"]+DOTS)
        orc = full[DOTS].max(axis=1)
        best = max(DOTS, key=lambda c: full[c].mean())
        rows.append(dict(verdict=vname, comparison="as reported: our layout vs dot's best "
                         "single configuration", n=len(full),
                         ours_pct=full["ours"].mean()*100, dot_pct=full[best].mean()*100,
                         matched="no: we choose among our variants, dot gets one"))
        rows.append(dict(verdict=vname, comparison="both choose: our variants vs dot's oracle "
                         "over its four", n=len(full),
                         ours_pct=full["ours"].mean()*100, dot_pct=orc.mean()*100,
                         matched="yes"))
        tests.append((vname, "both choose", full["ours"].values, orc.values, len(full)))
        idx = full.index.intersection(a.dropna(subset=["full", "no_variants"]).index)
        if len(idx):
            one = a.loc[idx, "no_variants"]
            rows.append(dict(verdict=vname, comparison="neither chooses: our first layout vs "
                             "dot's best single configuration", n=len(idx),
                             ours_pct=one.mean()*100, dot_pct=full.loc[idx, best].mean()*100,
                             matched="yes"))
            tests.append((vname, "neither chooses", one.values,
                          full.loc[idx, best].values, len(idx)))
    t = pd.DataFrame(rows)
    t["difference_pts"] = t.ours_pct - t.dot_pct
    CSV["attribution_matched"] = t
    # the same matched difference per corpus and model, for Table 1.
    # dot's best single configuration is chosen once over the pooled set, as
    # Table 1, Table S2a and checks_inference.py I2 all do, not per cell.
    cell_best = max(DOTS, key=lambda c: piv(gv, "strict")[c].mean())
    cells = []
    for (corpus, model), g in gv.groupby(["corpus", "model"]):
        w = g.pivot_table(index=["run", "item"], columns="config", values="strict",
                          aggfunc="first").dropna(subset=["ours"]+DOTS)
        if not len(w):
            continue
        cells.append(dict(corpus=corpus, model=model, n=len(w),
                          ours_pct=w["ours"].mean()*100,
                          dot_best_pct=w[cell_best].mean()*100,
                          dot_oracle_pct=w[DOTS].max(axis=1).mean()*100))
    c = pd.DataFrame(cells)
    c["difference_vs_best"] = c.ours_pct - c.dot_best_pct
    c["difference_vs_oracle"] = c.ours_pct - c.dot_oracle_pct
    CSV["attribution_matched_cells"] = c
    P(t.round(1).to_string(index=False))
    P()
    got = {}
    for vname, lab, x, y, n in tests:
        b01, b10, p = _mcnemar(pd.Series(x, dtype=float), pd.Series(y, dtype=float))
        got[(vname, lab)] = (100*(x.mean()-y.mean()), p, n)
        P(f"  {vname:11s} {lab:16s} n={n:5d}  "
          f"difference {100*(x.mean()-y.mean()):+6.1f} pts  McNemar p={p:.3g}")
    P()
    sb = got[("strict", "both choose")]
    sn = got.get(("strict", "neither chooses"))
    bb = got[("label_blind", "both choose")]
    bn = got.get(("label_blind", "neither chooses"))
    P("  Our shipped layout is the best of several variants, chosen by the very defect")
    P("  counts the verdict then reports; dot's best single configuration is one layout")
    P("  with no choice at all. The difference between those two is not a difference")
    P("  between two engines run the same way. With the choice matched the strict")
    P(f"  difference is {sb[0]:+.1f} points when both sides choose"
      + (f" and {sn[0]:+.1f} when neither does," if sn else ",")
      + " against")
    P("  the 32.9 points the unmatched pair shows, so about a third of the unmatched")
    P("  figure is our selection and not our layout. Matching does not rescue the")
    P(f"  label-blind result either: it widens it, dot leading by {-bb[0]:.1f} points when both")
    P(f"  choose" + (f" and by {-bn[0]:.1f} when neither does." if bn else "."))
    P("  The graph-level best-of-two is not part of this comparison: both engines draw")
    P("  the identical graph in every row here.")
    P()
    P("  the same difference per corpus and model, against dot's best single configuration")
    P("  and against its oracle over the four:")
    P(c.round(1).to_string(index=False))
    P(f"\n  matched, the nine cells run from {c.difference_vs_oracle.min():+.1f} to "
      f"{c.difference_vs_oracle.max():+.1f} points, against "
      f"{c.difference_vs_best.min():+.1f} to {c.difference_vs_best.max():+.1f} unmatched.")
    P()
    return t


def _tab(t, cols, head, cap, label):
    body = "".join(" & ".join(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols)+" \\\\\n" for _, r in t.iterrows())
    return ("\\begin{table}[t]\n\\centering\n\\fontsize{8}{9.6}\\selectfont\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l"+"r"*(len(cols)-1)+"}\n"
            "\\toprule\n"+" & ".join(head)+" \\\\\n\\midrule\n"+body +
            "\\bottomrule\n\\end{tabular}\n\\caption{"+cap+"}\n\\label{table:"+label+"}\n"
            "\\end{table}\n")


def _cluster_boot(vals, cl, B=4000, seed=0):
    """Percentile interval of a mean, resampling whole source graphs, as I1 does."""
    arr = np.asarray(vals, float)
    gs = {}
    for i, g in enumerate(np.asarray(cl)):
        gs.setdefault(g, []).append(i)
    keys = list(gs)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        sel = np.concatenate([gs[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        out.append(arr[sel].mean())
    return np.percentile(out, [2.5, 97.5])


def A7(D, P):
    """What is left to fail, for each engine: the residual-defect profile."""
    d = _load(D)
    gv = d[d.source.isin(["graphviz", "graphviz_outside_development"])].copy()
    gv["cl"] = gv.corpus.astype(str) + "|" + gv.item.astype(str)
    KEY7 = ["corpus", "model", "run", "item", "cl"]
    SIX = LENIENT + STRICT_EXTRA
    piv = {v: gv.pivot_table(index=KEY7, columns="config", values=v, aggfunc="first")
           for v in SIX + ["valid", "n_nodes"]}
    idx = piv["valid"].dropna(subset=["ours"]+DOTS).index
    piv = {k: v.loc[idx] for k, v in piv.items()}
    cl = np.array([i[-1] for i in idx])
    P(f"A7  what is left to fix: the residual-defect profile of each engine  "
      f"({len(idx)} identical graphs from {len(set(cl))} source graphs)")
    P()
    rows = []
    for cfg in ["ours", "dot_tb_spline"]:
        C = pd.DataFrame({k: (piv[k][cfg] > 0).values for k in SIX})
        fail = C.any(axis=1).values
        sub = C[fail]
        one = (sub.sum(axis=1) == 1).values.astype(float)
        lo, hi = _cluster_boot(one, cl[fail])
        P(f"  {NAMES[cfg]}: {int(fail.sum())} of {len(C)} figures fail the strict check "
          f"({100*fail.mean():.1f}%)")
        P(f"    exactly one of the six criteria fails on {100*one.mean():.1f}% "
          f"[{lo*100:.1f},{hi*100:.1f}] of those failures")
        for k in SIX:
            inv = sub[k].values.astype(float)
            sole = ((sub[k]) & (sub.sum(axis=1) == 1)).values.astype(float)
            l1, h1 = _cluster_boot(inv, cl[fail])
            rows.append(dict(config=NAMES[cfg], criterion=k,
                             share_of_failures_pct=100*inv.mean(), lo=100*l1, hi=100*h1,
                             sole_cause_pct=100*sole.mean(),
                             share_of_all_figures_pct=100*C[k].values.mean()))
            P(f"      {k:22s} in {100*inv.mean():5.1f}% [{100*l1:5.1f},{100*h1:5.1f}] of "
              f"failures, sole cause in {100*sole.mean():5.1f}%, "
              f"fails on {100*C[k].values.mean():5.1f}% of all figures")
        P()
    t = pd.DataFrame(rows)
    CSV["residual_profile"] = t
    # the same profile by graph size, for our engine
    C = pd.DataFrame({k: (piv[k]["ours"] > 0).values for k in SIX})
    C["boxes"] = piv["n_nodes"]["ours"].values
    srows = []
    P("  ours, by graph size:")
    for lo_, hi_ in [(0, 8), (9, 12), (13, 20), (21, 30), (31, 10**6)]:
        s = C[(C.boxes >= lo_) & (C.boxes <= hi_)]
        if not len(s):
            continue
        f = s[SIX].any(axis=1)
        srows.append(dict(size_bin=f"{lo_}-{hi_ if hi_ < 10**6 else ''}", n=len(s),
                          strict_fail_pct=100*f.mean(),
                          arrow_through_label_pct=100*s.edge_label_crossings.mean(),
                          label_collision_pct=100*s.label_collisions.mean(),
                          stacked_routes_pct=100*s.stacked_routes.mean()))
        q = srows[-1]
        P(f"    {q['size_bin']:>7s} boxes  n={len(s):5d}  strict failure "
          f"{q['strict_fail_pct']:5.1f}%   arrow through a label "
          f"{q['arrow_through_label_pct']:5.1f}%   label collision "
          f"{q['label_collision_pct']:4.1f}%   stacked routes {q['stacked_routes_pct']:5.1f}%")
    CSV["residual_profile_by_size"] = pd.DataFrame(srows)
    P()
    P("  One criterion bounds both engines. An arrow crossing a label is in 90.6% of our")
    P("  remaining failures and the only thing wrong with 62.4% of them, and in 76.3% of")
    P("  dot's; the three lenient criteria never fail for our engine, and dot fails one of")
    P("  them on 1.8% of its failures. So the defect class that separates the engines (A1)")
    P("  is also the one that limits the better of them, and it is a label stage that may")
    P("  move a label off its own route -- a leader line or a reserved channel -- that")
    P("  addresses it, not a better placement of boxes. Nothing else is left above 26%.")
    P()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out")
    ap.add_argument("--latex")
    ap.add_argument("--csv-dir")
    ap.add_argument("--analysis", nargs="*", default=["A1", "A2", "A3", "A4", "A5", "A6", "A7"])
    a = ap.parse_args()
    D = a.data.rstrip("/")+"/"
    buf = []

    def P(s=""):
        print(s)
        buf.append(str(s))
    res = {}
    for n in a.analysis:
        res[n] = globals()[n](D, P)
    if a.out:
        open(a.out, "w").write("\n".join(buf)+"\n")
    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        for k, v in CSV.items():
            v.to_csv(os.path.join(a.csv_dir, k+".csv"), index=False)
    if a.latex:
        if "A2" in res:
            TEX["labelblind"] = _tab(
                res["A2"], ["config", "lenient_pct", "label_blind_pct", "strict_pct"],
                ["\\textbf{Layout}", "\\textbf{Lenient}", "\\textbf{Label-blind}",
                 "\\textbf{Strict}"],
                "The strict verdict split by what it counts, over the 6{,}007 graphs both "
                "engines drew. \\emph{Lenient}: no box overlap, no arrow through a box, no "
                "route doubling back. \\emph{Label-blind}: lenient and no stacked routes --- "
                "the strict verdict without its two label criteria. \\emph{Strict}: all six. "
                "The difference between the engines lives entirely in the two label criteria, "
                "and Graphviz has no label-placement stage.", "labelblind")
        if "A3" in res:
            TEX["stages"] = _tab(
                res["A3"], ["layout", "lenient_pct", "label_blind_pct", "strict_pct"],
                ["\\textbf{Layout}", "\\textbf{Lenient}", "\\textbf{Label-blind}",
                 "\\textbf{Strict}"],
                "Our own engine stage by stage, on the same axis as the Graphviz "
                "configurations. The coordinate-free schema is constant down the table; "
                "only the layout stages change.", "stages")
        open(a.latex, "w").write("\n".join(TEX.values())+"\n")
        base = os.path.dirname(a.latex)
        for k in TEX:
            open(os.path.join(base, "table_"+k+".tex"), "w").write(TEX[k])
