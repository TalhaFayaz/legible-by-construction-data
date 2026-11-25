#!/usr/bin/env python3
"""Two appendix panels in one figure, from data/ only.

(a) the PaperBananaBench judge against figure size over twelve systems (6)
(b) which geometric criterion tracks the human raters (5.3)

legibility_size.csv, the per-size legibility of both engines, is still written
here; the panel that drew it is now the fitted version in make_fig_capacity.py,
so the two figures do not show the same thing twice.
No GPU, no model call, no new run.

    python3 make_fig_panels.py --data ../data --out ../../img
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS = [0, 9, 14, 19, 29, 49, 10 ** 9]
LAB = ['2–9', '10–14', '15–19', '20–29', '30–49', '50+']
BEST = 'dot_tb_spline'
STYLE = {'Qwen': ('o', '#35618f'), 'Gemma': ('s', '#8f5a35'),
         'Ministral': ('^', '#3f7f4f')}


def panel_a(ax, D):
    ju = pd.read_csv(f'{D}/judge_paperbanana.csv', low_memory=False)
    pi = pd.read_csv(f'{D}/pipeline_items.csv', low_memory=False)
    dot = pd.read_csv(f'{D}/dot_items.csv', low_memory=False)
    main = ju[ju.judge == 'InternVL3.5-14B']
    pb = pi[pi.corpus == 'paperbanana']
    dpb = dot[dot.corpus == 'paperbanana'].copy()
    dpb['pred_nodes'] = pd.to_numeric(dpb.pred_nodes, errors='coerce')
    rows = []
    for m in STYLE:
        for sysn, boxes in [
                ('dot', dpb[(dpb.variant == 'dot') & (dpb.model == m)].pred_nodes.mean()),
                ('dot_matched', dpb[(dpb.variant == 'dot_matched') & (dpb.model == m)].pred_nodes.mean()),
                ('ours', pb[(pb.variant == 'evaluated') & (pb.model == m)].layout_n_nodes.mean()),
                ('ours_coverage', pb[(pb.variant == 'coverage') & (pb.model == m)].layout_n_nodes.mean())]:
            g = main[(main.system == sysn) & (main.model == m)]
            rows.append(dict(model=m, system=sysn, boxes=boxes,
                             overall=g.overall_points.mean()))
    R = pd.DataFrame(rows)
    for m, g in R.groupby('model'):
        g = g.sort_values('boxes'); mk, c = STYLE[m]
        ax.plot(g.boxes, g.overall, '-', color=c, lw=0.7, alpha=0.55, zorder=1)
        for r in g.itertuples():
            filled = r.system.startswith('ours')
            ax.scatter(r.boxes, r.overall, marker=mk, s=20, zorder=3,
                       facecolor=c if filled else 'white', edgecolor=c,
                       linewidths=0.8, label=m if r.system == 'ours' else None)
    x, y = R.boxes.values, R.overall.values
    b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 1, x.max() + 1, 10)
    ax.plot(xs, np.polyval(b, xs), '--', color='0.35', lw=0.7, zorder=2)
    ax.axhline(50, color='0.6', lw=0.5, ls=':')
    ax.text(45.5, 51.2, 'parity', color='0.45', fontsize=5.5, ha='right')
    ax.set_xlabel('Boxes per figure', fontsize=7.5)
    ax.set_ylabel('Judge, overall', fontsize=7.5)
    ax.set_xlim(10, 47); ax.set_ylim(5, 80)
    ax.legend(frameon=False, fontsize=5.8, loc='upper right',
              handletextpad=0.3, borderpad=0.1)
    ax.text(0.02, 0.05, 'filled: typed graph   open: DOT\n' r'$\rho=-0.86$',
            transform=ax.transAxes, fontsize=5.8, color='0.3')
    return R


def panel_b(ax, D):
    L = pd.read_csv(f'{D}/layout_items.csv', low_memory=False)
    G = L[L.source.isin(['graphviz', 'graphviz_outside_development'])]
    key = ['corpus', 'model', 'run', 'item']
    p = G.pivot_table(index=key, columns='config', values='valid')
    meta = G[G.config == 'ours'].set_index(key)[['n_nodes']]
    meta = meta[~meta.index.duplicated()]
    d = pd.DataFrame({'ours': p['ours'], 'dot': p[BEST]}).join(meta).dropna()
    d = d.astype(float)
    d['bin'] = pd.cut(d.n_nodes, bins=BINS, labels=LAB)
    g = d.groupby('bin', observed=True).agg(n=('ours', 'size'),
                                            ours=('ours', 'mean'),
                                            dot=('dot', 'mean'))

    def wilson(k, n, z=1.96):
        ph = k / n; den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    lo_o, hi_o = zip(*[wilson(r.ours * r.n, r.n) for r in g.itertuples()])
    lo_d, hi_d = zip(*[wilson(r.dot * r.n, r.n) for r in g.itertuples()])
    x = np.arange(len(g))
    for y, lo, hi, c, m, lbl in [
            (g.ours * 100, np.array(lo_o) * 100, np.array(hi_o) * 100,
             '#1f4e79', 'o', 'Our engine'),
            (g['dot'] * 100, np.array(lo_d) * 100, np.array(hi_d) * 100,
             '#9a9a9a', 's', r'Graphviz $\tt dot$')]:
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, y, marker=m, color=c, linewidth=1.4, markersize=3.4,
                label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=6.0, rotation=30,
                       ha='right', rotation_mode='anchor')
    ax.set_xlabel('Boxes in the extracted graph', fontsize=7.5)
    ax.set_ylabel('Strictly legible (%)', fontsize=7.5)
    ax.set_ylim(0, 104)
    ax.grid(axis='y', color='#e8e8e8', linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=6.3, frameon=False, loc='lower left')
    gap = (g.ours - g['dot']) * 100
    for i in (0, 4):
        ax.annotate(f'{gap.iloc[i]:+.0f} pt',
                    (i, (g.ours.iloc[i] + g['dot'].iloc[i]) / 2 * 100),
                    fontsize=6.2, color='#555555', ha='center', va='center',
                    bbox=dict(fc='white', ec='none', pad=0.8))
    return g.assign(gap=gap.round(1))


# the criterion-vs-rater correlations of 5.3, as checks_bounds.py E2 computes
CRIT = [('Label collisions', 0.75), ('Box overlaps', 0.73),
        ('Arrows through a box', 0.69), ('Arrows through a label', 0.52),
        ('Strict verdict (yes/no)', 0.50), ('Arrow crossings', 0.26),
        ('Page shape', 0.19), ('Arrow length', -0.32)]


def panel_c(ax):
    names = [c[0] for c in CRIT][::-1]
    vals = [c[1] for c in CRIT][::-1]
    cols = ['#1f4e79' if v >= 0.5 else ('#9a9a9a' if v >= 0 else '#8f5a35')
            for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.axvline(0, color='0.3', lw=0.6)
    ax.set_xlabel(r'Agreement with raters ($\rho$)', fontsize=7.5)
    ax.set_xlim(-0.42, 0.92)
    for yi, v in zip(y, vals):
        ax.text(v + 0.03 if v >= 0 else 0.03, yi, f'{v:+.2f}',
                va='center', ha='left', fontsize=6)
    ax.grid(axis='x', color='#e8e8e8', linewidth=0.6); ax.set_axisbelow(True)
    return pd.DataFrame(CRIT, columns=['criterion', 'rho'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--out', default='../../img')
    a = ap.parse_args()
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif',
                         'axes.linewidth': 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.05))
    ra = panel_a(axes[0], a.data)
    rc = panel_c(axes[1])
    hidden, hax = plt.subplots(1, 1)
    rb = panel_b(hax, a.data)       # computed and released, drawn in img/capacity_frontier.pdf instead
    plt.close(hidden)
    for ax, t in zip(axes, ['(a)', '(b)']):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(width=0.6, length=2.5, labelsize=6.5)
        ax.set_title(t, fontsize=8, loc='left', pad=2)
    fig.tight_layout(pad=0.3, w_pad=1.1)
    os.makedirs(a.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'appendix_panels.{ext}'),
                    bbox_inches='tight')
    ra.to_csv(os.path.join(a.out, 'dose_response.csv'), index=False)
    rb.to_csv(os.path.join(a.out, 'legibility_size.csv'))
    rc.to_csv(os.path.join(a.out, 'criteria_ranking.csv'), index=False)
    print('wrote appendix_panels.{pdf,png} and the three CSVs to', a.out)


if __name__ == '__main__':
    main()
