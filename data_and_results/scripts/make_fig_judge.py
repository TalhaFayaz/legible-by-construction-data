#!/usr/bin/env python3
"""Four panels for the judge and selection analyses, from data/ only.

(a) what the benchmark's node matching rewards: the same graphs under its 0.5
    cosine threshold and under exact label matching (G1)
(b) the PaperBananaBench judge's position bias per dimension, both judges (G2)
(c) agreement between the two judges per dimension (G2)
(d) the ceiling of any selector over our engine and 'dot' (G3)

Released alongside the per-item CSVs; not a float in the paper. Same style as
make_fig_panels.py. No GPU, no model call, no new run.

    python3 make_fig_judge.py --data ../data --out ../results/figures
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa']
DOTCFG = ['dot_tb_spline', 'dot_lr_spline', 'dot_tb_ortho', 'dot_lr_ortho']
DIMS = ['faithfulness', 'conciseness', 'readability', 'aesthetics', 'overall']
COL = {'Qwen': '#35618f', 'Gemma': '#8f5a35', 'Ministral': '#3f7f4f'}
GREY = '#9a9a9a'


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


# --------------------------------------------------------------------------- #
def panel_a(ax, D, out):
    pi = pd.read_csv(f'{D}/pipeline_items.csv', low_memory=False)
    ev = pi[(pi.variant == 'evaluated') & pi.strict_node_f1.notna()]
    rows = []
    for corp in CORPORA:
        for m in MODELS:
            s = ev[(ev.corpus == corp) & (ev.model == m)]
            rows.append(dict(corpus=corp, model=m, lenient=s.node_f1.mean(),
                             exact=s.strict_node_f1.mean(),
                             share=1 - s.strict_node_recall.mean() / s.node_recall.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(f'{out}/seventh_paraphrase.csv', index=False)
    x = np.arange(len(R))
    ax.bar(x - 0.19, R.lenient, width=0.36, color=[COL[m] for m in R.model],
           label='cosine 0.5 (reported)')
    ax.bar(x + 0.19, R.exact, width=0.36, color='white', edgecolor=[COL[m] for m in R.model],
           linewidth=0.9, hatch='///', label='exact labels')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m[:4]}\n{"SciFlow" if c == "sciflow" else "FlowVQA"}'
                        for c, m in zip(R.corpus, R.model)], fontsize=5.6,
                       rotation=30, ha='right')
    ax.set_ylabel('Node F1', fontsize=7.5)
    ax.set_ylim(0, 1.02)
    for xi, (le, sh) in enumerate(zip(R.lenient, R.share)):
        ax.text(xi, le + 0.025, f'{100*sh:.0f}%', ha='center', fontsize=5.6,
                color='0.3')
    ax.legend(frameon=False, fontsize=5.8, loc='upper center', ncol=1)
    ax.grid(axis='y', color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title('(a) half of every match is a paraphrase', fontsize=7.0)


def panel_b(ax, D, out):
    j = pd.read_csv(f'{D}/judge_paperbanana.csv', low_memory=False)
    rows = []
    for jd, c in [('InternVL3.5-14B', '#35618f'), ('Qwen2.5-VL-32B', '#8f5a35')]:
        J = j[j.judge == jd]
        for d in DIMS:
            pv = J.pivot_table(index=['system', 'model', 'seed', 'item'],
                               columns='order', values=d + '_points',
                               aggfunc='mean').dropna()
            diff = (pv.iloc[:, 0] - pv.iloc[:, 1]).values
            lo, hi = boot(diff)
            rows.append(dict(judge=jd, dim=d, bias=diff.mean(), lo=lo, hi=hi,
                             colour=c))
    R = pd.DataFrame(rows)
    R.drop(columns='colour').to_csv(f'{out}/seventh_judge_order.csv', index=False)
    y = np.arange(len(DIMS))
    for k, (jd, g) in enumerate(R.groupby('judge', sort=False)):
        off = -0.19 if k == 0 else 0.19
        g = g.set_index('dim').loc[DIMS]
        ax.barh(y + off, g.bias, height=0.36, color=g.colour.iloc[0],
                label=jd.split('-')[0])
        ax.errorbar(g.bias, y + off, xerr=[g.bias - g.lo, g.hi - g.bias],
                    fmt='none', ecolor='0.25', elinewidth=0.7, capsize=1.3)
    ax.axvline(0, color='0.3', lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(DIMS, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlabel('Points gained by image order', fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.8, loc='upper right')
    ax.grid(axis='x', color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title('(b) position bias flips sign by dimension', fontsize=7.0)


def panel_c(ax, D, out):
    j = pd.read_csv(f'{D}/judge_paperbanana.csv', low_memory=False)
    j1 = j[(j.judge == 'InternVL3.5-14B') & (j.seed == 0)]
    j2 = j[(j.judge == 'Qwen2.5-VL-32B') & (j.seed == 0)]
    rows = []
    for d in DIMS:
        a = j1.groupby(['system', 'model', 'item'])[d + '_points'].mean()
        b = j2.groupby(['system', 'model', 'item'])[d + '_points'].mean()
        c = a.index.intersection(b.index)
        sa, sb = np.sign(a.loc[c] - 50).values, np.sign(b.loc[c] - 50).values
        po = np.mean(sa == sb)
        cats = np.unique(np.concatenate([sa, sb]))
        pe = sum((sa == k).mean() * (sb == k).mean() for k in cats)
        rows.append(dict(dim=d, kappa=(po - pe) / (1 - pe), agree=100 * po,
                         n=len(c)))
    R = pd.DataFrame(rows)
    R.to_csv(f'{out}/seventh_judge_kappa.csv', index=False)
    y = np.arange(len(R))
    cols = ['#b4473c' if d == 'readability' else GREY for d in R.dim]
    ax.barh(y, R.kappa, height=0.6, color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels(R.dim, fontsize=6.3)
    ax.invert_yaxis()
    for yi, k in enumerate(R.kappa):
        ax.text(k + 0.006, yi, f'{k:.3f}', va='center', fontsize=5.8,
                color='0.3')
    ax.set_xlim(0, 0.37)
    ax.set_xlabel(r'Agreement of the two judges ($\kappa$)', fontsize=7.5)
    ax.grid(axis='x', color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title('(c) and readability agrees least', fontsize=7.0)


def panel_d(ax, D, out):
    li = pd.read_csv(f'{D}/layout_items.csv', low_memory=False)
    for c in ['valid']:
        li[c] = li[c].astype(str).str.lower().isin(['true', '1'])
    gv = li[li.source.isin(['graphviz', 'graphviz_outside_development'])]
    key = ['corpus', 'model', 'run', 'item']
    piv = gv.pivot_table(index=key, columns='config', values='valid',
                         aggfunc='first').dropna(subset=['ours'] + DOTCFG)
    df = piv.reset_index()
    for c in ['ours'] + DOTCFG:
        df[c] = df[c].astype(bool)
    df['dot_oracle'] = df[DOTCFG].any(axis=1)
    rows = []
    for corp in ['sciflow', 'flowvqa', 'paperbanana']:
        for m in MODELS:
            s = df[(df.corpus == corp) & (df.model == m)]
            if not len(s):
                continue
            rows.append(dict(corpus=corp, model=m, n=len(s),
                             ours=100 * s.ours.mean(),
                             dot_oracle=100 * s.dot_oracle.mean(),
                             both=100 * (s.ours | s.dot_oracle).mean()))
    R = pd.DataFrame(rows)
    R['headroom'] = R.both - R.ours
    R.to_csv(f'{out}/seventh_selection_ceiling.csv', index=False)
    x = np.arange(len(R))
    ax.bar(x, R.dot_oracle, width=0.62, color='#dcdcdc',
           label="best of dot's four")
    ax.bar(x, R.ours - R.dot_oracle, bottom=R.dot_oracle, width=0.62,
           color=[COL[m] for m in R.model], label='ours')
    ax.bar(x, R.headroom, bottom=R.ours, width=0.62, color='white',
           edgecolor='0.35', linewidth=0.8, hatch='...',
           label='headroom of any selector')
    ax.set_xticks(x)
    NICE = {'sciflow': 'SciFlow', 'flowvqa': 'FlowVQA', 'paperbanana': 'PaperB'}
    ax.set_xticklabels([f'{m[:4]} {NICE[c]}' for c, m in zip(R.corpus, R.model)],
                       fontsize=5.4, rotation=35, ha='right')
    ax.set_ylabel('Strictly legible (%)', fontsize=7.5)
    ax.set_ylim(0, 108)
    for xi, h in enumerate(R.headroom):
        ax.text(xi, R.both.iloc[xi] + 1.5, f'+{h:.1f}', ha='center',
                fontsize=5.4, color='0.3')
    ax.legend(frameon=False, fontsize=5.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), ncol=3, columnspacing=1.0)
    ax.grid(axis='y', color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title('(d) switching engines gains $\\leq$5.3 points', fontsize=7.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--out', default='../results/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    panel_a(axes[0][0], a.data, a.out)
    panel_b(axes[0][1], a.data, a.out)
    panel_c(axes[1][0], a.data, a.out)
    panel_d(axes[1][1], a.data, a.out)
    fig.tight_layout(w_pad=2.0, h_pad=1.8)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}/judge_panels.{ext}', dpi=240,
                    bbox_inches='tight')
    print(f'wrote {a.out}/judge_panels.pdf and .png, and three CSVs')


if __name__ == '__main__':
    main()
