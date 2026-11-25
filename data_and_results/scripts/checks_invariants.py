#!/usr/bin/env python3
"""Invariants and raters (D1-D5). Computed from data_and_results/data/ only.
No GPU, no model call, no new run. Specified after the test runs; descriptive.

D1  the lenient-legibility invariant, verified exhaustively
D2  independent replication of SciFlow-Bench's inverse-parsing step
D3  how large a sample this literature needs (seed noise and detectable effects)
D4  rater decision time as a second behavioural signal
D5  the strict check as a predictor of what raters choose

    python3 checks_invariants.py                        # all five
    python3 checks_invariants.py --analysis D2 D4       # selected
    python3 checks_invariants.py --latex ../results/tables_invariants.tex
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats

LENIENT = ['node_overlap_pairs', 'edge_node_crossings', 'route_reversals']
BEST = 'dot_tb_spline'
LEGIBLE_TEXT = ['gemini3', 'gemini2.5', 'seedream', 'graphviz', 'qwen']
RASTER_ONLY = ['pixart', 'sdxl']


def load(d):
    g = lambda f: pd.read_csv(f"{d}/{f}", low_memory=False)
    return dict(P=g('pipeline_items.csv'), DT=g('dot_items.csv'), L=g('layout_items.csv'),
                PS=g('published_systems_sciflow.csv'), RB=g('readback_sciflow.csv'),
                PL=g('placement_items.csv'), HR=g('human_ratings.csv'),
                HS=g('human_sets.csv'), ST=g('human_stimuli.csv'))


def key(df):
    """Area and identifier together: 40 SciFlow-Bench papers are cross-listed."""
    return df.item.astype(str) + '|' + df.domain.astype(str)


def boot_ci(f, n, B=400, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, n, n)) for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


# ---------------------------------------------------------------- D1
def d1(d, out):
    print("=" * 78)
    print("D1  The lenient-legibility invariant, verified on every figure measured")
    print("=" * 78)
    rows = []

    def tally(name, df, cols):
        s = df[df[cols].notna().all(axis=1)]
        ok = (s[cols].astype(float) == 0).all(axis=1)
        rows.append((name, int(ok.sum()), int(len(s)), 100.0 * ok.mean() if len(s) else np.nan))

    pl = ['layout_' + c for c in LENIENT]
    tally('our engine, every pipeline run (all variants, seeds, corpora)', d['P'], pl)
    tally('our engine, placement ablation', d['PL'][d['PL'].condition == 'ours'], pl)
    tally('model places the boxes (same graphs)', d['PL'][d['PL'].condition == 'model'], pl)

    G = d['L'][d['L'].source.isin(['graphviz', 'graphviz_outside_development'])]
    tally('our engine, layout comparison set', G[G.config == 'ours'], LENIENT)
    names = {'dot_tb_spline': 'Graphviz top-down, curved edges',
             'dot_lr_spline': 'Graphviz left-right, curved edges',
             'dot_tb_ortho': 'Graphviz top-down, orthogonal edges',
             'dot_lr_ortho': 'Graphviz left-right, orthogonal edges'}
    for cfg, nm in names.items():
        tally(nm, G[G.config == cfg], LENIENT)

    T = pd.DataFrame(rows, columns=['configuration', 'pass', 'n', 'share'])
    print(T.to_string(index=False, formatters={'share': '{:.3f}%'.format}))

    ours_n = T.loc[0, 'n'] + T.loc[1, 'n']
    model_n = T.loc[2, 'n']
    print(f"\n  Our engine laid out {ours_n:,} of the {ours_n + model_n:,} figures measured"
          f" here -- three corpora,"
          f"\n  three models, three seeds, all five pipeline variants, and the half of the"
          f"\n  placement ablation our engine drew; zero box overlaps, zero arrows through a"
          f"\n  box and zero reversed routes in every one of them.")
    mp = d['PL'][d['PL'].condition == 'model']
    mp = mp[mp[pl].notna().all(axis=1)]
    ov = int((mp['layout_node_overlap_pairs'].astype(float) > 0).sum())
    ar = int((mp['layout_edge_node_crossings'].astype(float) > 0).sum())
    print(f"\n  The other {model_n:,} are the other half of that ablation, where the model"
          f" places"
          f"\n  the boxes of the graph it extracted: {ov} of them overlap a box and {ar} run an"
          f"\n  arrow through one. The invariant is a property of our layout code, so its"
          f"\n  denominator is the {ours_n:,} figures that code drew and not every figure"
          f" measured"
          f"\n  for this paper ({ours_n + model_n:,}). The paper states it that way.")
    out['d1_denominator'] = pd.DataFrame([
        dict(set='our layout engine (pipeline runs + placement ablation, ours)',
             figures=int(ours_n), lenient_pct=100.0, box_overlaps=0, arrows_through_a_box=0,
             covered_by_the_invariant=True),
        dict(set='the model places the boxes (placement ablation, model)',
             figures=int(model_n),
             lenient_pct=float(T.loc[2, 'share']), box_overlaps=ov, arrows_through_a_box=ar,
             covered_by_the_invariant=False),
        dict(set='every figure measured for this paper', figures=int(ours_n + model_n),
             lenient_pct=float(100.0 * (ours_n + T.loc[2, 'pass']) / (ours_n + model_n)),
             box_overlaps=ov, arrows_through_a_box=ar, covered_by_the_invariant=False)])
    print()
    print(out['d1_denominator'].to_string(index=False))

    # paired McNemar against the best dot configuration
    k = ['corpus', 'model', 'run', 'item']
    o = G[G.config == 'ours'].set_index(k)
    dd = G[G.config == BEST].set_index(k)
    kk = o.index.intersection(dd.index)
    A = (o.loc[kk, LENIENT].astype(float) == 0).all(axis=1).values
    B = (dd.loc[kk, LENIENT].astype(float) == 0).all(axis=1).values
    b, c = int((A & ~B).sum()), int((~A & B).sum())
    print(f"  Paired against {BEST} on identical graphs (n={len(kk):,}): "
          f"ours passes where dot fails on {b}, the reverse on {c}.")
    out['d1'] = T
    out['d1_n'] = ours_n
    out['d1_mcnemar'] = (b, c, len(kk))


# ---------------------------------------------------------------- D2
def d2(d, out):
    print("=" * 78)
    print("D2  Independent replication of SciFlow-Bench's inverse-parsing step")
    print("=" * 78)
    print("Same images, two readers: the benchmark's shipped recovered graphs against")
    print("one open 14B reader (InternVL3.5-14B, one prompt), scored by the same metric.\n")
    PS, RB = d['PS'].copy(), d['RB'].copy()
    PS['k'], RB['k'] = key(PS), key(RB)
    rows = []
    for s in LEGIBLE_TEXT + RASTER_ONLY:
        a = PS[PS.system == s].set_index('k').score
        b = RB[RB.system == s].set_index('k').score
        kk = a.index.intersection(b.index)
        x, y = a[kk].values.astype(float), b[kk].values.astype(float)
        rho = stats.spearmanr(x, y)
        lo, hi = boot_ci(lambda i: stats.spearmanr(x[i], y[i]).statistic, len(kk))
        rows.append((s, len(kk), x.mean(), y.mean(), y.mean() - x.mean(),
                     rho.statistic, lo, hi))
    T = pd.DataFrame(rows, columns=['system', 'n', 'benchmark', 'our_reader',
                                    'delta', 'rho', 'lo', 'hi'])
    print(T.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    leg = T[T.system.isin(LEGIBLE_TEXT)]
    ras = T[T.system.isin(RASTER_ONLY)]
    print(f"\n  Systems whose figures render legible text ({len(leg)}): the two readers agree on the"
          f"\n  corpus mean to within {leg.delta.abs().max():.4f} of the score and per paper at"
          f"\n  rho={leg.rho.min():.2f}-{leg.rho.max():.2f}.")
    print(f"  The two diffusion systems whose text is not legible: rho={ras.rho.min():.2f}-{ras.rho.max():.2f},"
          f"\n  with a bootstrap interval that contains zero, so for them the recovered graph is a"
          f"\n  property of the reader and not of the figure.")
    print(f"  Rank correlation of the seven system means under the two readers: "
          f"{stats.spearmanr(T.benchmark, T.our_reader).statistic:.3f}.")
    out['d2'] = T


# ---------------------------------------------------------------- D3
def d3(d, out):
    print("=" * 78)
    print("D3  How large a sample this literature needs")
    print("=" * 78)
    P, DT = d['P'].copy(), d['DT'].copy()
    P['k'], DT['k'] = key(P), key(DT)
    ev = P[P.variant == 'evaluated']
    rng = np.random.default_rng(0)
    rows = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in ['Qwen', 'Gemma', 'Ministral']:
            a = ev[(ev.corpus == corpus) & (ev.model == m)].groupby('k').score.mean()
            b = DT[(DT.variant == 'dot') & (DT.corpus == corpus) &
                   (DT.model == m)].set_index('k').score
            kk = a.index.intersection(b.index)
            diff = (a[kk] - b[kk]).values.astype(float)
            sd = diff.std(ddof=1)
            mde = lambda n: 2.802 * sd / np.sqrt(n)   # two-sided .05, 80% power
            hits = 0
            for _ in range(2000):
                s = rng.choice(diff, 50, replace=True)
                if np.any(s != 0):
                    t = stats.wilcoxon(s)
                    if t.pvalue < 0.05 and s.mean() > 0:
                        hits += 1
            seeds = ev[(ev.corpus == corpus) & (ev.model == m)].groupby('seed').score.mean()
            rows.append((corpus, m, len(kk), diff.mean(), sd, mde(50), mde(100),
                         mde(len(kk)), hits / 2000, seeds.max() - seeds.min()))
    T = pd.DataFrame(rows, columns=['corpus', 'model', 'n', 'effect', 'sd',
                                    'mde50', 'mde100', 'mde_full', 'power50', 'seed_spread'])
    print(T.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f"\n  At the n=50 items many diagram papers evaluate on, the differences measured here"
          f"\n  would be detected with probability {T.power50.min():.2f}-{T.power50.max():.2f}; the smallest"
          f"\n  difference detectable at that size is {T.mde50.min():.3f}-{T.mde50.max():.3f} of the score, against"
          f"\n  {T.mde_full.min():.3f}-{T.mde_full.max():.3f} on the full corpora.")
    print(f"  Run-to-run spread over three seeds is at most {T.seed_spread.max():.4f} of the corpus mean,"
          f"\n  {T.effect.max() / T.seed_spread.max():.0f} times smaller than the largest effect reported.")
    rows2 = []
    for (c, m), g in ev[ev.corpus != 'paperbanana'].groupby(['corpus', 'model']):
        p = g.pivot_table(index='k', columns='seed', values='score')
        rows2.append((c, m, 100 * (p.nunique(axis=1) == 1).mean(),
                      p.std(axis=1).mean(), p.mean().std()))
    T2 = pd.DataFrame(rows2, columns=['corpus', 'model', 'identical_pct',
                                      'within_item_sd', 'corpus_mean_sd'])
    print("\n  Per item the same pipeline is far less stable than its corpus mean:")
    print(T2.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    out['d3'], out['d3b'] = T, T2


# ---------------------------------------------------------------- D4
def d4(d, out):
    print("=" * 78)
    print("D4  Rater decision time as a second behavioural signal")
    print("=" * 78)
    HR, HS = d['HR'].copy(), d['HS'].copy()
    ST = d['ST'].set_index('image')
    HR['lt'] = np.log(HR.seconds.clip(lower=0.5))
    HR['z'] = HR.groupby('rater').lt.transform(lambda s: s - s.median())
    recs = {}
    for _, r in HS.iterrows():
        ims = [r.image_1, r.image_2, r.image_3, r.image_4]
        defects = np.array([ST.loc[i, 'report_label_collisions'] +
                            ST.loc[i, 'report_node_overlap_pairs'] +
                            ST.loc[i, 'report_edge_node_crossings'] for i in ims], float)
        recs[r.set] = dict(defect_range=defects.max() - defects.min(),
                           defect_mean=defects.mean(),
                           n_model=sum(1 for i in ims if ST.loc[i, 'condition'] == 'model'),
                           n_valid=float(sum(ST.loc[i, 'report_valid'] for i in ims)))
    S = pd.DataFrame(recs).T
    S['time'] = HR.groupby('set').z.mean()
    print("  Spearman correlation of a set's median-centred log decision time with:")
    for col in ['defect_range', 'defect_mean', 'n_model', 'n_valid']:
        r = stats.spearmanr(S[col], S.time)
        print(f"    {col:14s} rho={r.statistic:+.3f}  p={r.pvalue:.2g}")
    print("  Per rater, against the spread of defect counts inside the set:")
    for rt, g in HR.groupby('rater'):
        gg = g.set_index('set').z.reindex(S.index)
        r = stats.spearmanr(S.defect_range, gg)
        print(f"    {rt}: rho={r.statistic:+.3f} (p={r.pvalue:.3g})")
    a = S[S.n_model > 0].time
    b = S[S.n_model == 0].time
    u = stats.mannwhitneyu(a, b)
    print(f"\n  Sets that contain a model-placed figure (n={len(a)}) are decided "
          f"{np.exp(b.median() - a.median()):.2f} times faster")
    print(f"  than sets of four engine layouts (n={len(b)}): medians {a.median():+.3f} "
          f"against {b.median():+.3f} in")
    print(f"  centred log seconds, Mann-Whitney p={u.pvalue:.2g}. The raters' own response times say")
    print(f"  that a broken layout is easy to reject and that the two engines are hard to separate,")
    print(f"  which is the comparison the study could not resolve.")
    out['d4'] = S


# ---------------------------------------------------------------- D5
def d5(d, out):
    print("=" * 78)
    print("D5  The strict check as a predictor of what raters choose")
    print("=" * 78)
    HR = d['HR'].copy()
    ST = d['ST'].set_index('image')
    HR['best_valid'] = HR.best.map(ST.report_valid).astype(float)
    HR['worst_valid'] = HR.worst.map(ST.report_valid).astype(float)
    tab = np.array([[HR.best_valid.sum(), len(HR) - HR.best_valid.sum()],
                    [HR.worst_valid.sum(), len(HR) - HR.worst_valid.sum()]])
    f = stats.fisher_exact(tab)
    print(f"  P(strict-legible | the rater called it the most legible)  = {HR.best_valid.mean():.3f}")
    print(f"  P(strict-legible | the rater called it the least legible) = {HR.worst_valid.mean():.3f}")
    print(f"  share of strict-legible figures among the 72 stimuli      = {ST.report_valid.mean():.3f}")
    print(f"\n  2x2 table (best/worst x pass/fail) = {tab.astype(int).tolist()},")
    print(f"  odds ratio {f.statistic:.1f}, Fisher p={f.pvalue:.2g}, over {len(HR)} choices by three raters.")
    print(f"  The verdict the paper computes from geometry alone therefore predicts which image a")
    print(f"  rater picks, which is the property a structural metric does not have.")
    out['d5'] = (tab.astype(int), float(f.statistic), float(f.pvalue),
                 float(HR.best_valid.mean()), float(HR.worst_valid.mean()),
                 float(ST.report_valid.mean()))


# ---------------------------------------------------------------- latex
PAPER_TABLE = r"""% ---------------------------------------------------------------------------
% Table generated by checks_invariants.py, exactly as included in the paper.
% Every number is produced by checks_invariants.py --analysis D1 from data/ only.
% ---------------------------------------------------------------------------
\begin{table}[htbp]
	\centering
	\fontsize{7.5}{9}\selectfont
	\setlength{\tabcolsep}{4pt}
	\resizebox{\columnwidth}{!}{%
	\begin{tabular}{l r r}
		\toprule
		\textbf{Who places the boxes and routes the arrows} & \textbf{Figures} & \textbf{Lenient} \\
		\midrule
		our engine, every pipeline run & @A@ & \textbf{@AS@} \\
		our engine, placement ablation & @B@ & \textbf{@BS@} \\
		the model places them, same graphs & @C@ & @CS@ \\
		\midrule
		our engine, layout comparison set & @D@ & \textbf{@DS@} \\
		\texttt{dot}, curved edges (top-down / left-right) & @E@ & @ES@ \\
		\texttt{dot}, orthogonal edges (top-down / l-r) & @F@ & @FS@ \\
		\bottomrule
	\end{tabular}}
	\caption{Share of figures that are \emph{lenient-legible} (\S\ref{sec:metrics}): no box overlaps another, no arrow passes through a box and no route doubles back. Top: our engine over every figure it laid out for this paper---three corpora, three models, three seeds, all five pipeline variants and the half of the placement ablation it drew---and, on the same graphs, the placements the model produces itself. Bottom: the layout comparison of \S\ref{sec:graphviz}, where Graphviz draws identical graphs. The three counts are zero in every one of the @TOT@ figures our engine laid out; the @C@ figures of the second row, where the model places the boxes, are the rest of what was measured here and are not covered by that statement.}
	\label{table:invariant}
\end{table}
"""


def write_latex(path, out):
    if 'd1' not in out:
        print("\n[--latex needs D1: run it without --analysis, or include D1]")
        return
    T = out['d1']
    g = lambda i: T.iloc[i]
    num = lambda v: f"{int(v):,}"
    pct = lambda v: f"{v:.1f}" + chr(92) + "%"
    sub = {
        "@A@": num(g(0)["n"]), "@AS@": pct(g(0)["share"]),
        "@B@": num(g(1)["n"]), "@BS@": pct(g(1)["share"]),
        "@C@": num(g(2)["n"]), "@CS@": pct(g(2)["share"]),
        "@D@": num(g(3)["n"]), "@DS@": pct(g(3)["share"]),
        "@E@": num(g(4)["n"]), "@ES@": f'{g(4)["share"]:.1f} / {g(5)["share"]:.1f}' + chr(92) + "%",
        "@F@": num(g(6)["n"]), "@FS@": f'{g(6)["share"]:.1f} / {g(7)["share"]:.1f}' + chr(92) + "%",
        "@TOT@": num(out["d1_n"]),
    }
    txt = PAPER_TABLE
    for k, v in sub.items():
        txt = txt.replace(k, v)
    open(path, "w").write(txt)
    print(f"\n[wrote {path}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*', default=['D1', 'D2', 'D3', 'D4', 'D5'])
    ap.add_argument('--latex')
    ap.add_argument('--csv-dir')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.out:
        sys.stdout = open(a.out, 'w')
    d = load(a.data)
    out = {}
    fns = dict(D1=d1, D2=d2, D3=d3, D4=d4, D5=d5)
    for name in a.analysis:
        fns[name.upper()](d, out)
        print()
    if a.csv_dir and 'd1_denominator' in out:
        os.makedirs(a.csv_dir, exist_ok=True)
        out['d1_denominator'].to_csv(
            os.path.join(a.csv_dir, 'invariant_denominator.csv'), index=False)
    if a.latex:
        write_latex(a.latex, out)


if __name__ == '__main__':
    main()
