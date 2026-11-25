#!/usr/bin/env python3
"""Two panels on legibility as a capacity and on the content--legibility frontier.

    python3 make_fig_capacity.py --data ../data --out ../../img

Writes capacity_frontier.{pdf,png} and the two CSVs behind the panels, so each
panel can be checked without rerunning anything.

(a) the share of strictly legible figures against graph size, for our engine and
    for each of Graphviz's four configurations, over the 6,007 figures both
    engines draw from identical graphs, with the logistic fit and the box count
    at which half the figures stop passing.
(b) content against legibility over every corpus x model x variant cell: the
    configuration the paper evaluates against the coverage call and the two
    ablations, with the Pareto frontier of each corpus drawn through them.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GREY = '#444444'
CFG = [('ours', 'Ours', '#2f6f9f', '-', 'o'),
       ('dot_tb_spline', 'dot td curved', '#d94801', '--', 's'),
       ('dot_lr_spline', 'dot lr curved', '#fd8d3c', '--', '^'),
       ('dot_tb_ortho', 'dot td orthogonal', '#969696', ':', 'v'),
       ('dot_lr_ortho', 'dot lr orthogonal', '#cccccc', ':', 'D')]
NAMES = {'ours': 'Ours', 'dot_tb_spline': 'dot, top-down curved',
         'dot_lr_spline': 'dot, left-right curved',
         'dot_tb_ortho': 'dot, top-down orthogonal',
         'dot_lr_ortho': 'dot, left-right orthogonal'}
VNAME = {'evaluated': 'evaluated', 'coverage': 'coverage call',
         'coverage_no_control_flow': 'no control-flow prompt',
         'ablation_one_greedy_graph': 'one greedy graph',
         'ablation_unconstrained': 'unconstrained'}


def tf(s):
    return s.astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})


def logit_fit(x, y, iters=60):
    X = np.vstack([x, np.ones_like(x)]).T
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        b = b + np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
    return b


def panel_a(ax, data):
    lay = pd.read_csv(f'{data}/layout_items.csv', low_memory=False)
    g = lay[lay.source.isin(['graphviz', 'graphviz_outside_development'])].copy()
    g['valid'] = tf(g.valid)
    g = g.dropna(subset=['valid'])
    bk = [0, 5, 9, 14, 19, 29, 49, 10 ** 6]
    ctr = [4, 7.5, 12, 17, 24, 39, 60]
    g['bucket'] = pd.cut(g.n_nodes, bk, labels=ctr)
    rows = []
    xs = np.linspace(np.log(3), np.log(70), 200)
    for key, _lab, col, ls, mk in CFG:
        s = g[g.config == key]
        if len(s) < 100:
            continue
        b = logit_fit(np.log(s.n_nodes.clip(lower=1)).values, s.valid.values)
        half = float(np.exp(-b[1] / b[0]))
        ax.plot(np.exp(xs), 100 / (1 + np.exp(-(b[0] * xs + b[1]))), ls,
                color=col, lw=1.4, zorder=2)
        m = s.groupby('bucket', observed=True).valid.mean() * 100
        ax.plot([float(i) for i in m.index], m.values, mk, color=col, ms=3.4,
                mec='white', mew=.5, zorder=3,
                label=f'{NAMES[key]}  ({half:.0f})')
        rows.append(dict(config=key, half_at_boxes=half, slope=b[0],
                         strict_pct=100 * s.valid.mean(), n=len(s)))
        if key in ('ours', 'dot_tb_spline'):
            ax.plot([half, half], [0, 50], color=col, lw=.6, alpha=.5, zorder=1)
    ax.axhline(50, color=GREY, lw=.5, alpha=.4, zorder=1)
    ax.set_xscale('log')
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(['4', '8', '16', '32', '64'], fontsize=8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('boxes in the figure', fontsize=8)
    ax.set_ylabel('strictly legible, %', fontsize=8)
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.4, loc='lower left', frameon=False,
              title='engine (half-legible at)', title_fontsize=6.4)
    ax.set_title('(a) legibility is a capacity', fontsize=8.5, loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return pd.DataFrame(rows)


def panel_b(ax, data):
    pi = pd.read_csv(f'{data}/pipeline_items.csv', low_memory=False)
    pi['lv'] = tf(pi.layout_valid)
    c = (pi.groupby(['corpus', 'model', 'variant'])
           .agg(score=('score', 'mean'), strict=('lv', 'mean'),
                boxes=('layout_n_nodes', 'mean'))
           .dropna(subset=['strict']).reset_index())
    rows = []
    style = {'sciflow': ('o', '#2f6f9f', 'SciFlow-Bench'),
             'flowvqa': ('s', '#d94801', 'FlowVQA')}
    for corp in ['sciflow', 'flowvqa']:
        mark, col, _lab = style[corp]
        for mdl in sorted(c[c.corpus == corp].model.unique()):
            sm = c[(c.corpus == corp) & (c.model == mdl)]
            base = sm[sm.variant == 'evaluated'].iloc[0]
            for _, r in sm.iterrows():
                dom = bool(((sm.score >= r.score) & (sm.strict >= r.strict) &
                            ((sm.score > r.score) | (sm.strict > r.strict))).any())
                rows.append(dict(corpus=corp, model=mdl, variant=r.variant,
                                 score=r.score, strict=r.strict, boxes=r.boxes,
                                 dominated=dom,
                                 d_score=r.score - base.score,
                                 d_strict=100 * (r.strict - base.strict)))
                if r.variant == 'evaluated':
                    continue
                ax.plot(r.score - base.score, 100 * (r.strict - base.strict),
                        mark, ms=4.4, mfc='white', mec=col, mew=1.1, zorder=3)
    ax.axhspan(0, 40, xmin=.70, color='#d94801', alpha=.08, zorder=0)
    ax.axhline(0, color=GREY, lw=.7, zorder=1)
    ax.axvline(0, color=GREY, lw=.7, zorder=1)
    ax.plot(0, 0, '*', ms=12, mfc='#2f6f9f', mec='white', mew=.8, zorder=4)
    ax.annotate('the configuration\nwe evaluate', (0, 0), (-16, -26),
                textcoords='offset points', fontsize=6.6, color='#2f6f9f',
                ha='right')
    ax.annotate('no configuration\nis in here', (.0035, 13), fontsize=6.6,
                color='#d94801', ha='left')
    ax.set_xlim(-.07, .03)
    ax.set_ylim(-36, 22)
    ax.set_xlabel('benchmark score, against the shipped configuration', fontsize=8)
    ax.set_ylabel('strictly legible, points', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title('(b) no configuration is better on both', fontsize=8.5, loc='left')
    h = [plt.Line2D([], [], marker='o', ls='', mfc='white', mec='#2f6f9f',
                    ms=4.4, label='SciFlow-Bench, other variant'),
         plt.Line2D([], [], marker='s', ls='', mfc='white', mec='#d94801',
                    ms=4.4, label='FlowVQA, other variant')]
    ax.legend(handles=h, fontsize=6.4, loc='lower left', frameon=False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--out', default='../../img')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'text.usetex': False,
                         'axes.edgecolor': GREY, 'axes.labelcolor': GREY,
                         'xtick.color': GREY, 'ytick.color': GREY,
                         'text.color': GREY})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    ca = panel_a(axes[0], a.data)
    cb = panel_b(axes[1], a.data)
    fig.tight_layout(pad=.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}/capacity_frontier.{ext}', dpi=200, bbox_inches='tight')
    ca.to_csv(f'{a.out}/capacity_frontier_a.csv', index=False)
    cb.to_csv(f'{a.out}/capacity_frontier_b.csv', index=False)
    print(ca.round(3).to_string(index=False))
    print()
    print(f'evaluated on frontier in {int((~cb[cb.variant=="evaluated"].dominated).sum())}'
          f' of {len(cb[cb.variant=="evaluated"])} corpus x model cells')
