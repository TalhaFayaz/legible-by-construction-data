#!/usr/bin/env python3
"""S4: where each standing margin comes from.

The SciFlow-Bench score is 0.4*F1_node + 0.6*F1_edge, so a margin can arise from
recovering more of the gold graph (recall) or from recovering it more cleanly
(precision). This decomposes every paired margin into its four components.
Usage: python3 checks_decomposition.py --data ../data --out ../results/tables
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

SEED = 20260301
COMP = ['node_precision','node_recall','edge_precision','edge_recall']
SYSTEMS = [('gemini3','Gemini 3 Pro Image'), ('gemini2.5','Gemini 2.5 Flash Image'),
           ('seedream','Seedream 4.0'), ('graphviz','Graphviz (GPT-4o$\\rightarrow$DOT)'),
           ('qwen','Qwen-Image'), ('pixart','PixArt-$\\Sigma$'), ('sdxl','SDXL')]

def boot_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = np.sort([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(s[int(.025*n)]), float(s[int(.975*n)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data'); ap.add_argument('--out', default='../results/tables')
    ap.add_argument('--model', default='ministral')
    a = ap.parse_args()
    d = pd.read_csv(Path(a.data)/'readback_sciflow.csv')
    o = d[(d.system=='ours') & (d.model.str.contains(a.model, case=False, na=False))]
    o = o[['item','domain','score','pred_nodes']+COMP]
    rows = []
    for key, lab in SYSTEMS:
        b = d[d.system==key][['item','domain','score','pred_nodes']+COMP]
        m = o.merge(b, on=['item','domain'], suffixes=('_o','_b'))
        if len(m) < 20: continue
        r = dict(system=lab, n=len(m),
                 boxes_ours=round(float(m.pred_nodes_o.mean()),1),
                 boxes_other=round(float(m.pred_nodes_b.mean()),1),
                 delta_score=round(float((m.score_o-m.score_b).mean()),4))
        sig = []
        for c in COMP:
            dd = (m[c+'_o']-m[c+'_b']).values
            lo, hi = boot_ci(dd)
            r[c] = round(float(dd.mean()),4)
            r[c+'_lo'], r[c+'_hi'] = round(lo,4), round(hi,4)
            if lo > 0: sig.append(c)
        prec = any(c.endswith('precision') for c in sig)
        rec  = any(c.endswith('recall') for c in sig)
        r['margin_from'] = ('precision and recall' if prec and rec else
                            'precision' if prec else 'recall' if rec else 'neither')
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(Path(a.out)/'standing_decomposition.csv', index=False)
    cols = ['system','n','boxes_ours','boxes_other','delta_score']+COMP+['margin_from']
    print(t[cols].to_string(index=False))

if __name__ == '__main__':
    main()
