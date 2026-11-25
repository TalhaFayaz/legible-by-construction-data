#!/usr/bin/env python3
"""S1-S3: where the pipeline stands among the systems SciFlow-Bench distributes.

All comparisons are paired on the paper, on the symmetric read-back half: every
figure, ours included, is drawn as an image and recovered by the same reader.
  S1  overall paired differences, Holm-corrected   -> standing_readback.csv
  S2  the same, split at the median gold size      -> standing_by_size.csv
  S3  open-weight subset                           -> standing_openweight.csv
Usage: python3 checks_standing.py --data ../data --out ../results/tables
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260301
SYSTEMS = [('gemini3','Gemini 3 Pro Image','closed'),
           ('gemini2.5','Gemini 2.5 Flash Image','closed'),
           ('seedream','Seedream 4.0','closed'),
           ('graphviz','Graphviz (GPT-4o$\\rightarrow$DOT)','closed'),
           ('qwen','Qwen-Image','open'),
           ('pixart','PixArt-$\\Sigma$','open'),
           ('sdxl','SDXL','open')]

def boot_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = np.sort([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(s[int(.025*n)]), float(s[int(.975*n)])

def holm(ps):
    idx = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps)-k)*ps[i])); out[i] = run
    return out

def ours(d, model):
    return d[(d.system=='ours') & (d.model.str.contains(model, case=False, na=False))]

def compare(o, d, items=None):
    rows = []
    for key, lab, wt in SYSTEMS:
        b = d[d.system==key][['item','domain','score']].rename(columns={'score':'other'})
        m = o.merge(b, on=['item','domain'])
        if items is not None: m = m[m.item.isin(items)]
        if len(m) < 20: continue
        diff = (m.ours - m.other).values
        lo, hi = boot_ci(diff)
        r = stats.wilcoxon(m.ours, m.other)
        # matched-pairs rank-biserial
        nz = diff[diff != 0]; rk = stats.rankdata(np.abs(nz))
        rb = (rk[nz>0].sum() - rk[nz<0].sum()) / rk.sum() if len(nz) else 0.0
        rows.append(dict(system=lab, weights=wt, n=len(m),
                         ours=round(float(m.ours.mean()),4),
                         other=round(float(m.other.mean()),4),
                         delta=round(float(diff.mean()),4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4),
                         rank_biserial=round(float(rb),3), p_raw=float(r.pvalue)))
    t = pd.DataFrame(rows)
    t['p_holm'] = holm(t.p_raw.values)
    t['verdict'] = np.where((t.ci_lo>0) & (t.p_holm<.05), 'ahead',
                   np.where((t.ci_hi<0) & (t.p_holm<.05), 'behind', 'tied'))
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data'); ap.add_argument('--out', default='../results/tables')
    ap.add_argument('--model', default='ministral')
    a = ap.parse_args()
    d = pd.read_csv(Path(a.data)/'readback_sciflow.csv')
    o = ours(d, a.model)[['item','domain','score']].rename(columns={'score':'ours'})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    t = compare(o, d)
    t.to_csv(out/'standing_readback.csv', index=False)
    print('=== S1 ours + %s against every system the benchmark ships ===' % a.model.title())
    print(t.to_string(index=False))
    print('\nahead of %d, tied with %d, behind %d'
          % ((t.verdict=='ahead').sum(), (t.verdict=='tied').sum(), (t.verdict=='behind').sum()))

    # ceiling
    g = d[d.system=='gold_graph_ceiling'].score.mean()
    print('gold-graph ceiling under the same protocol: %.4f' % g)

    # S2 by gold size
    p = pd.read_csv(Path(a.data)/'pipeline_items.csv', low_memory=False)
    gs = p[p.corpus=='sciflow'][['item','gold_nodes']].dropna().groupby('item', as_index=False).first()
    med = gs.gold_nodes.median()
    small = set(gs[gs.gold_nodes <= med].item); large = set(gs[gs.gold_nodes > med].item)
    parts = []
    for nm, it in [('small (gold <= %d boxes)' % med, small), ('large (gold > %d boxes)' % med, large)]:
        tt = compare(o, d, items=it); tt.insert(0, 'stratum', nm); parts.append(tt)
    bysize = pd.concat(parts, ignore_index=True)
    bysize.to_csv(out/'standing_by_size.csv', index=False)
    print('\n=== S2 split at the median gold graph size ===')
    print(bysize[['stratum','system','n','delta','ci_lo','ci_hi','verdict']].to_string(index=False))

    # S3 open weights only
    ow = t[t.weights=='open'].copy()
    ow.to_csv(out/'standing_openweight.csv', index=False)
    print('\n=== S3 open-weight subset ===')
    print(ow[['system','delta','ci_lo','ci_hi','p_holm','verdict']].to_string(index=False))
    print('\nwrote tables to', out.resolve())

if __name__ == '__main__':
    main()
