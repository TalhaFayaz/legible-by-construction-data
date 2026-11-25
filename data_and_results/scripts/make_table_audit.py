#!/usr/bin/env python3
"""The audit at a glance: what each instrument returns when it is shown a
legibility difference of known size.

This script builds one table and nothing else. Every figure in it is read from
the per-item measurements in `data/` or from a table another script in this
folder already wrote to `results/tables/`; the two quantities that need a
computation (the partial correlation at fixed figure size, and the agreement of
the two judges) reuse the helpers of `checks_inference.py` and
`checks_judge.py`, so the numbers here are the same objects those scripts print
and cannot drift from them.

    python3 make_table_audit.py                       # print the table
    python3 make_table_audit.py --latex               # the LaTeX of the table
    python3 make_table_audit.py --csv-dir ../results/tables
    python3 make_table_audit.py --out ../results/printed_outputs/make_table_audit.txt

Columns: the instrument, what the literature uses it to claim, what it returns
when the difference is present, and where that reading comes from.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from checks_inference import (cluster_boot, engines, load, partial_spearman)  # noqa: E402
from checks_judge import kappa  # noqa: E402

OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s)


def _tables(d):
    g = lambda f: pd.read_csv(os.path.join(d, f))
    return {n: g(n + '.csv') for n in
            ('reader_fidelity', 'reader_calibration', 'criteria_vs_raters',
             'criteria_independence', 'judge_dimension_tie_rates',
             'judge_dimension_stability')}


# --------------------------------------------------------------- the readings
def score_reading(D):
    """The benchmark score against the strict verdict, figure size held fixed."""
    P = D['P'][D['P'].variant == 'evaluated'].dropna(
        subset=['score', 'layout_valid', 'layout_n_nodes']).copy()
    P['cl'] = P.corpus.astype(str) + '|' + P.item.astype(str)
    x = P.score.values
    y = P.layout_valid.values.astype(float)
    z = P.layout_n_nodes.values
    r, _, n = partial_spearman(x, y, z)
    ci = cluster_boot(lambda i: partial_spearman(x[i], y[i], z[i])[0], P.cl.values, B=1000)
    a, b = x[y == 1], x[y == 0]
    auc = float((pd.Series(np.concatenate([a, b])).rank().values[:len(a)].sum()
                 - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
    return dict(rho=r, lo=ci[0], hi=ci[1], n=n, graphs=P.cl.nunique(), auc=auc)


def judge_reading(D, T):
    """The judge's own readability dimension, on its own terms."""
    j = D['J']
    j1 = j[(j.judge == 'InternVL3.5-14B') & (j.seed == 0)]
    j2 = j[(j.judge == 'Qwen2.5-VL-32B') & (j.seed == 0)]
    out = {}
    for dim in ('readability', 'conciseness'):
        a = j1.groupby(['system', 'model', 'item'])[dim + '_points'].mean()
        b = j2.groupby(['system', 'model', 'item'])[dim + '_points'].mean()
        c = a.index.intersection(b.index)
        sa, sb = np.sign(a.loc[c] - 50), np.sign(b.loc[c] - 50)
        out[dim] = dict(kappa=float(kappa(sa, sb)), n=len(c))
    tie = T['judge_dimension_tie_rates'].set_index('dimension')
    st = T['judge_dimension_stability'].set_index('dimension')
    out['readability']['tie'] = float(tie.loc['readability', 'tie_pct'])
    out['readability']['sign_kept'] = int(st.loc['readability', 'sign_kept'])
    out['readability']['models'] = int(st.loc['readability', 'models'])
    out['conciseness']['tie'] = float(tie.loc['conciseness', 'tie_pct'])
    out['conciseness']['sign_kept'] = int(st.loc['conciseness', 'sign_kept'])
    return out


def reader_reading(T):
    """The read-back reader against the geometry that was actually drawn."""
    f, c = T['reader_fidelity'], T['reader_calibration']
    # slope and R2 are reported over all six fits (three models x boxes and
    # arrows); the arrow-only range, which the reader over-returns most, is
    # carried alongside because that is the one \S5 quotes
    ca = c[c.target == 'arrows']
    return dict(node_lo=f.node_ratio.min(), node_hi=f.node_ratio.max(),
                edge_lo=f.edge_ratio.min(), edge_hi=f.edge_ratio.max(),
                slope_lo=c.slope.min(), slope_hi=c.slope.max(),
                r2_lo=c.r2.min(), r2_hi=c.r2.max(),
                arrow_slope_lo=ca.slope.min(), arrow_slope_hi=ca.slope.max(),
                arrow_r2_lo=ca.r2.min(), arrow_r2_hi=ca.r2.max())


def counts_reading(D, T):
    """The geometric counts against the raters -- the instrument that responds."""
    cr = T['criteria_vs_raters'].set_index('measure')
    ind = T['criteria_independence'].iloc[0]
    G = engines(D['L']).dropna(subset=['edge_edge_crossings', 'label_collisions', 'n_nodes'])
    r2, _, n2 = partial_spearman(G.edge_edge_crossings.values,
                                 G.label_collisions.values, G.n_nodes.values)
    ci2 = cluster_boot(lambda i: partial_spearman(
        G.edge_edge_crossings.values[i], G.label_collisions.values[i],
        G.n_nodes.values[i])[0], G.cl.values, B=1000)
    return dict(labels=abs(float(cr.loc['label_collisions', 'rho'])),
                crossings=abs(float(cr.loc['edge_edge_crossings', 'rho']))
                if 'edge_edge_crossings' in cr.index else np.nan,
                mutual=r2, mutual_lo=ci2[0], mutual_hi=ci2[1],
                layouts=n2, graphs=int(G.cl.nunique()),
                partial_committed=float(ind.crossings_vs_collisions_partial))


# ------------------------------------------------------------------ the table
def build(D, T):
    s, j, rd, ct = (score_reading(D), judge_reading(D, T),
                    reader_reading(T), counts_reading(D, T))
    rows = [
        dict(instrument='Graph score recovered from the image',
             claimed='the figure can be read',
             reading=(f"rho = {s['rho']:+.3f} with the strict verdict at fixed figure size, "
                      f"95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}] over source graphs "
                      f"(n={s['n']:,} figures, {s['graphs']} graphs): equivalent to independence. "
                      f"Unconditionally it carries a little signal (AUC {s['auc']:.3f}), "
                      f"so what it tracks is how much the figure holds"),
             source='pipeline_items.csv; checks_inference.py --analysis I7'),
        dict(instrument="VLM judge, its own readability dimension",
             claimed='the figure is readable',
             reading=(f"calls the two figures equally good on {j['readability']['tie']:.1f}% of "
                      f"judgements, is the least reproducible of the four dimensions "
                      f"(kappa = {j['readability']['kappa']:.3f} against "
                      f"{j['conciseness']['kappa']:.3f} for conciseness) and keeps its sign for "
                      f"{j['readability']['sign_kept']} of {j['readability']['models']} models "
                      f"when the judge is replaced"),
             source='judge_paperbanana.csv; checks_judge.py --analysis G2'),
        dict(instrument='Read-back reader (graph recovered by a model)',
             claimed='what the figure contains',
             reading=(f"returns {rd['node_lo']:.2f}--{rd['node_hi']:.2f} boxes and "
                      f"{rd['edge_lo']:.2f}--{rd['edge_hi']:.2f} arrows per one drawn, and the "
                      f"correction that would repair it has slope "
                      f"{rd['slope_lo']:.2f}--{rd['slope_hi']:.2f} rather than 1 over the six "
                      f"fits (R2 {rd['r2_lo']:.2f}--{rd['r2_hi']:.2f}; on arrows alone, slope "
                      f"{rd['arrow_slope_lo']:.2f}--{rd['arrow_slope_hi']:.2f} and R2 "
                      f"{rd['arrow_r2_lo']:.2f}--{rd['arrow_r2_hi']:.2f}), so the bias is "
                      f"bounded and not removable"),
             source='reader_fidelity.csv, reader_calibration.csv'),
        dict(instrument='Geometric counts over known geometry (this bench)',
             claimed='the figure can be read',
             reading=(f"label collisions track the raters at rho = {ct['labels']:.2f} against "
                      f"{ct['crossings']:.2f} for the arrow crossings this literature reports, "
                      f"and the two are near-orthogonal at fixed size "
                      f"(rho = {ct['mutual']:+.3f}, 95% CI [{ct['mutual_lo']:+.3f}, "
                      f"{ct['mutual_hi']:+.3f}], {ct['layouts']:,} layouts, "
                      f"{ct['graphs']:,} graphs)"),
             source='criteria_vs_raters.csv, criteria_independence.csv; checks_inference.py I7'),
    ]
    return pd.DataFrame(rows), dict(score=s, judge=j, reader=rd, counts=ct)


def _r2(x):
    """round to two decimals, half away from zero, as the paper's prose does"""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _th(n):
    """thousands separator LaTeX writes inside and outside maths alike"""
    return f'{int(n):,}'.replace(',', '{,}')


def latex(t, v):
    s, j, rd, ct = v['score'], v['judge'], v['reader'], v['counts']
    L = [
        r'\begin{table}[t]',
        '\t' + r'\centering',
        '\t' + r'\fontsize{6.8}{8.2}\selectfont',
        '\t' + r'\setlength{\tabcolsep}{3pt}',
        '\t' + r'\renewcommand{\arraystretch}{1.12}',
        '\t' + r'\begin{tabular}{@{}p{0.235\columnwidth} p{0.20\columnwidth} '
               r'p{0.50\columnwidth}@{}}',
        '\t\t' + r'\toprule',
        '\t\t' + r'\textbf{Instrument} & \textbf{Used to claim} & '
                 r'\textbf{What it returns when the difference is present} \\',
        '\t\t' + r'\midrule',
        '\t\t' + (r'Graph score recovered from the image & the figure can be read & '
                  rf'$\rho={s["rho"]:+.3f}$ with the strict verdict at fixed size, 95\% CI '
                  rf'$[{s["lo"]:+.3f},{s["hi"]:+.3f}]$ over source graphs '
                  rf'($n={_th(s["n"])}$ figures, {s["graphs"]} graphs): equivalent to '
                  rf'independence. It does respond to how much the figure holds '
                  rf'(AUC {s["auc"]:.2f}) \\'),
        '\t\t' + (r'VLM judge, its own \emph{readability} dimension & the figure is readable & '
                  rf'ties on {j["readability"]["tie"]:.1f}\% of judgements; least reproducible '
                  rf'of its four dimensions ($\kappa={j["readability"]["kappa"]:.3f}$ against '
                  rf'{j["conciseness"]["kappa"]:.3f} for \emph{{conciseness}}); keeps its sign '
                  rf'for {j["readability"]["sign_kept"]} of '
                  rf'{j["readability"]["models"]} models when the judge is replaced \\'),
        '\t\t' + (r'Read-back reader & what the figure contains & '
                  rf'returns {rd["node_lo"]:.2f}--{rd["node_hi"]:.2f} boxes and '
                  rf'{rd["edge_lo"]:.2f}--{rd["edge_hi"]:.2f} arrows per one drawn, and the '
                  rf'correction that would repair it is flat rather than straight: slope '
                  rf'{rd["slope_lo"]:.2f}--{rd["slope_hi"]:.2f} over the six fits, and it does '
                  rf'not transfer between figure sets \\'),
        '\t\t' + r'\midrule',
        '\t\t' + (r'Geometric counts over known geometry \emph{(this bench)} & the figure can '
                  r'be read & '
                  rf'label collisions track the raters at $\rho={ct["labels"]:.2f}$ against '
                  rf'{ct["crossings"]:.2f} for arrow crossings, and the two are near-orthogonal '
                  rf'at fixed size ($\rho={ct["mutual"]:+.2f}$, '
                  rf'$[{_r2(ct["mutual_lo"]):+.2f},{_r2(ct["mutual_hi"]):+.2f}]$, '
                  rf'{_th(ct["layouts"])} layouts) \\'),
        '\t\t' + r'\bottomrule',
        '\t' + r'\end{tabular}',
        '\t' + r'\caption{The audit at a glance. Each instrument is shown the same stimulus---'
               r'the $+21.0$-point legibility difference of \S\ref{sec:graphviz}, attributed to '
               r'one defect class and sized to a point---and asked whether it responds. None of '
               r'the three the literature relies on does; the counts this bench makes possible '
               r'do (\S\ref{sec:human}). Built by '
               r'\texttt{make\_table\_audit.py} from the per-item measurements '
               r'(\S\ref{sec:validity}).}',
        '\t' + r'\label{table:audit}',
        r'\end{table}',
    ]
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, '..', 'data'))
    ap.add_argument('--tables', default=os.path.join(HERE, '..', 'results', 'tables'))
    ap.add_argument('--csv-dir', help='write audit_summary.csv here')
    ap.add_argument('--latex', action='store_true', help='print the LaTeX table')
    ap.add_argument('--out', help='write the printed output here')
    a = ap.parse_args()

    D, T = load(a.data), _tables(a.tables)
    table, vals = build(D, T)

    say('The audit at a glance: what each instrument returns when it is shown a')
    say('legibility difference of known size (+21.0 points, one defect class).')
    say('')
    for _, r in table.iterrows():
        say(f"  {r.instrument}")
        say(f"    used to claim : {r.claimed}")
        say(f"    returns       : {r.reading}")
        say(f"    from          : {r.source}")
        say('')

    if a.latex:
        print(latex(table, vals))
    if a.csv_dir:
        p = os.path.join(a.csv_dir, 'audit_summary.csv')
        table.to_csv(p, index=False)
        say(f'wrote {p}')
    if a.out:
        with open(a.out, 'w', encoding='utf8') as f:
            f.write('\n'.join(OUT) + '\n')
