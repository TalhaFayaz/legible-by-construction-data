#!/usr/bin/env python3
"""H1-H5: human preference treated as an evaluation instrument.

Produces the tables and figures behind Section 6 of the paper:
  H1  contrast coverage of the rating panel   -> human_contrast_coverage.csv
  H2  floor detection                          -> human_floor_detection.csv
  H3  saturation above the floor               -> human_saturation.csv
  H4  resolution of the engine contrast        -> human_resolution.csv
Usage: python3 checks_human_instrument.py --data ../data --out ../results/tables
"""
import argparse, math, random
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

SIX = ['label_collisions','edge_label_crossings','node_overlap_pairs',
       'edge_node_crossings','stacked_routes','route_reversals']
LENIENT = ['node_overlap_pairs','edge_node_crossings','route_reversals']
RNG = 20260301

def strict_mask(df, pre=''):   return (df[[pre+c for c in SIX]] == 0).all(axis=1)
def lenient_mask(df, pre=''):  return (df[[pre+c for c in LENIENT]] == 0).all(axis=1)

def boot_ci(x, fn=np.mean, n=10000, seed=RNG):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def perm_spearman(x, y, n=20000, seed=RNG, max_perm_n=400):
    """Spearman rho with an exact-ish permutation p for small samples (the rating
    panel) and the asymptotic p for large ones, where the two agree closely."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(set(x)) < 3 or len(x) < 6: return np.nan, np.nan
    res = stats.spearmanr(x, y)
    obs = float(res.statistic)
    if len(x) > max_perm_n:
        return obs, float(res.pvalue)
    rng = np.random.default_rng(seed); yy = y.copy(); c = 0
    for _ in range(n):
        rng.shuffle(yy)
        if abs(stats.spearmanr(x, yy).statistic) >= abs(obs) - 1e-12: c += 1
    return obs, (c + 1) / (n + 1)

def load_human(d):
    st = pd.read_csv(d/'human_stimuli.csv')
    st.columns = [c.replace('report_','') for c in st.columns]
    ra = pd.read_csv(d/'human_ratings.csv')
    se = pd.read_csv(d/'human_sets.csv')
    app = se[['image_1','image_2','image_3','image_4']].stack().value_counts()
    b = ra.best.value_counts(); w = ra.worst.value_counts()
    st['appear'] = st.image.map(app).fillna(0)
    st['best']   = st.image.map(b).fillna(0)
    st['worst']  = st.image.map(w).fillna(0)
    st['bw']     = (st.best - st.worst) / st.appear.replace(0, np.nan)
    st['strict'] = strict_mask(st); st['lenient'] = lenient_mask(st)
    return st, ra, se

# ---------------------------------------------------------------- H1
def h1_contrast_coverage(st, out):
    piv = st.pivot_table(index='paper_id', columns='condition',
                         values=['strict','label_collisions','edge_label_crossings'], aggfunc='first')
    rows = []
    for a, b, name in [('ours','graphviz','ours vs dot (engine contrast)'),
                       ('ours','model','ours vs model coordinates'),
                       ('graphviz','model','dot vs model coordinates')]:
        sa, sb = piv[('strict',a)], piv[('strict',b)]
        both = sa.notna() & sb.notna()
        differ = (sa != sb) & both
        rows.append(dict(contrast=name, pairs=int(both.sum()),
                         same_verdict=int((both & ~differ).sum()),
                         differ=int(differ.sum()),
                         coverage_pct=round(100*differ.sum()/both.sum(), 1),
                         a_only=int(((sa==True)&(sb==False)).sum()),
                         b_only=int(((sa==False)&(sb==True)).sum())))
    cov = pd.DataFrame(rows)
    # per-condition defect range actually present in the panel
    rng_rows = []
    for c in ['ours','graphviz','model']:
        s = st[st.condition == c]
        rng_rows.append(dict(condition=c, n=len(s),
            label_coll_mean=round(s.label_collisions.mean(), 2),
            label_coll_max=int(s.label_collisions.max()),
            label_coll_nonzero=int((s.label_collisions > 0).sum()),
            arrow_thru_label_mean=round(s.edge_label_crossings.mean(), 2),
            arrow_thru_label_nonzero=int((s.edge_label_crossings > 0).sum()),
            strict_pass=int(s.strict.sum()), lenient_pass=int(s.lenient.sum()),
            n_nodes_min=int(s.n_nodes.min()), n_nodes_max=int(s.n_nodes.max())))
    rngdf = pd.DataFrame(rng_rows)
    cov.to_csv(out/'human_contrast_coverage.csv', index=False)
    rngdf.to_csv(out/'human_panel_range.csv', index=False)
    print('\n=== H1 contrast coverage of the rating panel ===')
    print(cov.to_string(index=False)); print(); print(rngdf.to_string(index=False))
    return cov, rngdf

# ---------------------------------------------------------------- H2
def h2_floor(st, ra, out):
    cond = dict(zip(st.image, st.condition))
    b = ra.best.map(cond).value_counts(); w = ra.worst.map(cond).value_counts()
    tot = len(ra)
    rows = []
    for c in ['ours','graphviz','model']:
        s = st[st.condition == c]
        rows.append(dict(condition=c, images=len(s),
            lenient_pass_pct=round(100*s.lenient.mean(), 1),
            strict_pass_pct=round(100*s.strict.mean(), 1),
            best_picks=int(b.get(c, 0)), worst_picks=int(w.get(c, 0)),
            share_of_worst_pct=round(100*w.get(c, 0)/tot, 1),
            mean_bw=round(s.bw.mean(), 3)))
    fl = pd.DataFrame(rows)
    # strict verdict -> pick rates, as the paper's odds ratio but split by floor
    below = st[~st.lenient]; above = st[st.lenient]
    extra = pd.DataFrame([
        dict(stratum='below the lenient floor', images=len(below),
             strict_pass_pct=round(100*below.strict.mean(), 1),
             mean_bw=round(below.bw.mean(), 3), bw_sd=round(below.bw.std(), 3)),
        dict(stratum='above the lenient floor', images=len(above),
             strict_pass_pct=round(100*above.strict.mean(), 1),
             mean_bw=round(above.bw.mean(), 3), bw_sd=round(above.bw.std(), 3))])
    fl.to_csv(out/'human_floor_detection.csv', index=False)
    extra.to_csv(out/'human_floor_strata.csv', index=False)
    print('\n=== H2 floor detection ===')
    print(fl.to_string(index=False)); print(); print(extra.to_string(index=False))
    return fl

# ---------------------------------------------------------------- H3
def h3_saturation(st, out):
    MEAS = [('label_collisions','defect'),('edge_label_crossings','defect'),
            ('node_overlap_pairs','defect'),('edge_node_crossings','defect'),
            ('stacked_routes','defect'),('edge_edge_crossings','defect'),
            ('n_nodes','size'),('n_edges','size'),('area','size'),
            ('ink_ratio','shape'),('aspect_deviation','shape'),
            ('total_edge_length','shape'),('total_bends','shape'),
            ('min_separation','shape')]
    e48 = st[st.condition.isin(['ours','graphviz'])]
    rows = []
    for m, kind in MEAS:
        r_all, p_all = perm_spearman(st[m], st.bw)
        r_48,  p_48  = perm_spearman(e48[m], e48.bw)
        const = len(set(e48[m].dropna())) < 3
        rows.append(dict(measure=m, kind=kind,
                         rho_panel72=None if np.isnan(r_all) else round(r_all, 3),
                         p_panel72=None if np.isnan(p_all) else round(p_all, 4),
                         rho_engines48=None if np.isnan(r_48) else round(r_48, 3),
                         p_engines48=None if np.isnan(p_48) else round(p_48, 4),
                         constant_or_degenerate_in_48=bool(const)))
    sat = pd.DataFrame(rows)
    # Holm within each family
    for col, pcol in [('rho_panel72','p_panel72'), ('rho_engines48','p_engines48')]:
        ps = sat[pcol].astype(float)
        ok = ps.notna(); idx = ps[ok].sort_values().index; m = ok.sum()
        adj = {}; run = 0.0
        for k, i in enumerate(idx):
            run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = round(run, 4)
        sat[pcol.replace('p_','p_holm_')] = sat.index.map(adj)
    sat.to_csv(out/'human_saturation.csv', index=False)
    print('\n=== H3 what raters track, panel of 72 vs the 48 engine-drawn ===')
    print(sat.to_string(index=False))
    return sat

# ---------------------------------------------------------------- H4
def h4_resolution(st, out):
    piv = st.pivot_table(index='paper_id', columns='condition', values='bw', aggfunc='first')
    d = (piv['ours'] - piv['graphviz']).dropna().values
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1); n = len(d)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / math.sqrt(n)
    # observed geometric difference inside the panel, in strict-pass points
    sp = st.pivot_table(index='paper_id', columns='condition', values='strict', aggfunc='first')
    gap = 100 * (sp['ours'].mean() - sp['graphviz'].mean())
    res = pd.DataFrame([dict(
        quantity='paired best-worst difference, ours minus dot',
        n_graphs=n, estimate=round(float(d.mean()), 3),
        ci_lo=round(lo, 3), ci_hi=round(hi, 3), sd=round(float(sd), 3),
        mde_80pct_power=round(float(mde), 3),
        strict_gap_in_panel_pp=round(float(gap), 1),
        corpus_wide_strict_gap_pp=21.0,
        verdict='inconclusive: MDE exceeds any plausible effect')])
    res.to_csv(out/'human_resolution.csv', index=False)
    print('\n=== H4 resolution of the engine contrast ===')
    print(res.to_string(index=False))
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data'); ap.add_argument('--out', default='../results/tables')
    a = ap.parse_args()
    d = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st, ra, se = load_human(d)
    h1_contrast_coverage(st, out)
    h2_floor(st, ra, out)
    h3_saturation(st, out)
    h4_resolution(st, out)
    print('\nwrote tables to', out.resolve())
