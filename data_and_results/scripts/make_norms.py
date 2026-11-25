#!/usr/bin/env python3
"""Writes the two normalisation tables the standalone scorer needs.

  results/tables/index_norms.csv        log1p mean and sd of the ten layout
                                        measures over the 72 rated images,
                                        the scale the index weights are on
  results/tables/engine_rule_norms.csv  mean and sd of the five graph features
                                        over the 6,012 figures the engine rule
                                        is fitted on

Both are derived from the released per-item CSVs and from the same helpers the
analysis scripts use, so legibility.py scores a figure on exactly the scale the
paper fitted. Run from this directory:  python3 make_norms.py
"""
import importlib, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data") + os.sep
T = os.path.join(HERE, "..", "results", "tables") + os.sep
if HERE not in sys.path:
    sys.path.insert(0, HERE)

F5 = ["lbox", "larr", "density", "loops", "nested"]
F10 = ["label_collisions", "node_overlap_pairs", "edge_node_crossings",
       "stacked_routes", "edge_label_crossings", "edge_edge_crossings",
       "total_edge_length", "mean_bends_per_edge", "aspect_deviation", "area"]


def main():
    cj = importlib.import_module("checks_judge")
    li = pd.read_csv(D + "layout_items.csv", low_memory=False)
    df = cj._layout_wide({"layout": li})
    df["lbox"] = np.log(df.n_nodes + 1)
    df["larr"] = np.log(df.n_edges + 1)
    X = df[F5].astype(float).values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    pd.DataFrame([dict(feature=f, mean=m, sd=s) for f, m, s in zip(F5, mu, sd)]
                 ).to_csv(T + "engine_rule_norms.csv", index=False)
    print(f"engine_rule_norms.csv  fitted on {len(df)} figures")

    st = pd.read_csv(D + "human_stimuli.csv")
    L = st[["report_" + c for c in F10]].astype(float).clip(lower=0).apply(np.log1p)
    pd.DataFrame([dict(measure=c, log1p_mean=L["report_" + c].mean(),
                       log1p_sd=L["report_" + c].std(ddof=0)) for c in F10]
                 ).to_csv(T + "index_norms.csv", index=False)
    print(f"index_norms.csv        fitted on {len(st)} rated images")


if __name__ == "__main__":
    main()
