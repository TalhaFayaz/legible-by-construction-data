#!/usr/bin/env python3
"""Three released panels on resolution, drawn from the per-item CSVs only.

    python3 make_fig_resolution.py --data ../data --out ../results/figures

Writes resolution_panels.{pdf,png} and the three CSVs behind the panels, so
each panel can be checked without rerunning anything.

(a) where the variance of the benchmark score is: item, model, their
    interaction and the decoding seed, per corpus.
(b) what that leaves a comparison: the rate at which a sample of n items
    returns the wrong sign for a paired difference that is in fact positive,
    against the size of that difference.
(c) the same extracted graphs under the benchmark's weighted score and under
    normalised graph edit distance: paired effect size per model pair.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa']
NICE = {'sciflow': 'SciFlow-Bench', 'flowvqa': 'FlowVQA'}
KEY = ['domain', 'item']
GREY = '#444444'


def num(s):
    return pd.to_numeric(s, errors='coerce')


def load(data):
    p = pd.read_csv(f'{data}/pipeline_items.csv', low_memory=False)
    e = p[p.variant == 'evaluated'].copy()
    e['score'] = num(e['score']).fillna(0.0)
    d = pd.read_csv(f'{data}/dot_items.csv', low_memory=False)
    d = d[d.variant == 'dot'].copy()
    d['score'] = num(d['score']).fillna(0.0)
    return e, d


def panel_a(ax, e):
    rows = []
    for corp in CORPORA:
        d = e[e.corpus == corp]
        grand = d.score.mean()
        tot = ((d.score - grand) ** 2).mean()
        item_m = d.groupby(KEY, dropna=False).score.transform('mean')
        model_m = d.groupby('model').score.transform('mean')
        cell = d.groupby(KEY + ['model'], dropna=False).score.transform('mean')
        v = dict(
            item=((d.groupby(KEY, dropna=False).score.mean() - grand) ** 2).mean(),
            model=((d.groupby('model').score.mean() - grand) ** 2).mean(),
            inter=((cell - item_m - model_m + grand) ** 2).mean(),
            seed=((d.score - cell) ** 2).mean())
        rows.append(dict(corpus=corp, **{k: 100 * x / tot for k, x in v.items()}))
    df = pd.DataFrame(rows)
    labs = ['Item', 'Item $\\times$ model', 'Decoding seed', 'Model']
    keys = ['item', 'inter', 'seed', 'model']
    cols = ['#2f6f9f', '#7fb0cf', '#c9dcea', '#d94801']
    y = np.arange(len(CORPORA))
    left = np.zeros(len(CORPORA))
    for k, lab, c in zip(keys, labs, cols):
        w = df[k].values
        ax.barh(y, w, left=left, color=c, label=lab, height=.5,
                edgecolor='white', linewidth=.6)
        for i, (ww, ll) in enumerate(zip(w, left)):
            if ww > 4:
                ax.text(ll + ww / 2, y[i], f'{ww:.0f}', ha='center',
                        va='center', fontsize=7,
                        color='white' if k == 'item' else GREY)
        left = left + w
    for i, corp in enumerate(CORPORA):
        ax.annotate(f'{df[df.corpus == corp]["model"].iloc[0]:.1f}%',
                    (100.5, y[i]), fontsize=7, va='center', color='#d94801')
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[c] for c in CORPORA], fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel('share of the variance of the score (%)', fontsize=8)
    ax.legend(fontsize=6.6, loc='upper center', bbox_to_anchor=(.5, -.22),
              frameon=False, ncol=4, handlelength=1.1, columnspacing=1.0)
    ax.set_title('(a) which model wrote the graph is 1% of it',
                 fontsize=8.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    return df


def panel_b(ax, e, d):
    rng = np.random.default_rng(0)
    ns = [25, 50, 100, 200, 483]
    rows = []
    for corp in CORPORA:
        for m in MODELS:
            a = e[(e.corpus == corp) & (e.model == m)].groupby(
                KEY, dropna=False).score.mean()
            b = d[(d.corpus == corp) & (d.model == m)].groupby(
                KEY, dropna=False).score.mean()
            j = pd.concat([a.rename('o'), b.rename('d')], axis=1).dropna()
            diff = (j.o - j.d).values
            true = diff.mean()
            for n in ns:
                if n > len(diff):
                    continue
                s = rng.choice(diff, (4000, n), replace=True).mean(1)
                rows.append(dict(corpus=corp, model=m, n=n, true=true,
                                 wrong=float((np.sign(s) != np.sign(true)).mean())))
    df = pd.DataFrame(rows)
    for (corp, m), g in df.groupby(['corpus', 'model']):
        small = abs(g.true.iloc[0]) < 0.01
        ax.plot(g.n, 100 * g.wrong, marker='o', ms=3, lw=1.2,
                color='#d94801' if small else '#9ecae1',
                zorder=3 if small else 2,
                label=(f'{m}, {NICE[corp]} ($\\Delta$={g.true.iloc[0]:+.3f})'
                       if small else None))
    ax.axvline(50, color=GREY, ls=':', lw=.8)
    ax.annotate('50 items', (52, 42), fontsize=7, color=GREY)
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=7)
    ax.set_xlabel('items drawn', fontsize=8)
    ax.set_ylabel('wrong sign (%)', fontsize=8)
    ax.set_ylim(-2, 48)
    ax.legend(fontsize=6.4, frameon=False, loc='upper right',
              handlelength=1.2)
    ax.set_title('(b) a true advantage under 0.01 is a coin toss at 50 items',
                 fontsize=8.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    return df


def panel_c(ax, e):
    rows = []
    for corp in CORPORA:
        d = e[e.corpus == corp].dropna(subset=['strict_normalised_ged'])
        pv = {k: d.pivot_table(index=KEY, columns='model', values=v,
                               aggfunc='mean', dropna=False)
              for k, v in [('score', 'score'),
                           ('nGED', 'strict_normalised_ged')]}
        for i in range(len(MODELS)):
            for k in range(i + 1, len(MODELS)):
                a, b = MODELS[i], MODELS[k]
                rec = dict(corpus=corp, pair=f'{a}\n{b}')
                for nm in ('score', 'nGED'):
                    x = pv[nm][[a, b]].dropna()
                    diff = x[a] - x[b]
                    rec[nm] = abs(diff.mean() / diff.std())
                    rec[nm + '_p'] = stats.wilcoxon(x[a], x[b]).pvalue
                rows.append(rec)
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    h = .34
    ax.barh(y + h / 2, df.score, height=h, color='#9ecae1',
            label='benchmark score', edgecolor='white', linewidth=.5)
    ax.barh(y - h / 2, df.nGED, height=h, color='#2f6f9f',
            label='nGED', edgecolor='white', linewidth=.5)
    for i, r in enumerate(df.itertuples()):
        if r.score_p >= .05:
            ax.annotate('tied', (r.score + .04, y[i] + h / 2), fontsize=6.6,
                        va='center', color='#d94801')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r.pair.replace(chr(10), " vs ")}\n'
                        f'{NICE[r.corpus]}' for r in df.itertuples()],
                       fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel('paired effect size $|d_z|$', fontsize=8)
    ax.legend(fontsize=6.6, frameon=False, loc='lower right',
              handlelength=1.1)
    ax.set_title('(c) the same graphs, two structural measures',
                 fontsize=8.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--out', default='../results/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'text.usetex': False,
                         'axes.edgecolor': GREY, 'axes.labelcolor': GREY,
                         'xtick.color': GREY, 'ytick.color': GREY,
                         'text.color': GREY, 'mathtext.fontset': 'dejavuserif'})
    e, d = load(a.data)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.3))
    da = panel_a(axes[0], e)
    db = panel_b(axes[1], e, d)
    dc = panel_c(axes[2], e)
    fig.tight_layout(w_pad=2.2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}/resolution_panels.{ext}', dpi=200,
                    bbox_inches='tight')
    da.to_csv(f'{a.out}/resolution_variance.csv', index=False)
    db.to_csv(f'{a.out}/resolution_sign_error.csv', index=False)
    dc.to_csv(f'{a.out}/resolution_effect_size.csv', index=False)
    print(f'written to {a.out}/resolution_panels.pdf and three CSVs')


if __name__ == '__main__':
    main()
