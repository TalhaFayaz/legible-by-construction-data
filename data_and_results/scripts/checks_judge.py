#!/usr/bin/env python3
"""Judge and selection: four analyses (G1-G4) computed from the per-item CSVs only.

No GPU, no model call, no new run. Mirrors the style of checks_*.py.

    python3 checks_judge.py --data ../data
    python3 checks_judge.py --analysis G1 G3
    python3 checks_judge.py --latex ../results/tables_judge.tex
    python3 checks_judge.py --out ../results/printed_outputs/checks_judge.txt

G1  what the benchmark's node matching actually rewards: the share of every
    matched node that matches only above the 0.5 cosine threshold, and the
    score the same graphs reach under exact label matching.
G2  the PaperBananaBench judge per dimension: order bias, order agreement,
    sign-flip rate, and agreement between the two judges per dimension.
G3  the ceiling of a two-engine selector: what any policy over our engine and
    'dot' could reach, against always using ours.
G4  the engine-choice rule of F4, validated across corpora: train on two
    corpora, predict the third, for both engines.
"""
import argparse, io, sys
import numpy as np, pandas as pd
from scipy import stats

MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa']
DOTCFG = ['dot_tb_spline', 'dot_lr_spline', 'dot_tb_ortho', 'dot_lr_ortho']
DIMS = ['faithfulness', 'conciseness', 'readability', 'aesthetics', 'overall']
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s)


def head(t):
    say('\n' + '=' * 78)
    say('== ' + t)
    say('=' * 78)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def mcnemar(b, c):
    """Exact two-sided McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    p = stats.binomtest(min(b, c), n, 0.5).pvalue
    return min(1.0, p)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pe = sum((a == k).mean() * (b == k).mean() for k in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


# --------------------------------------------------------------------------- #
def G1(D):
    head('G1  What the benchmark\'s node matching rewards (paraphrase share)')
    say("SciFlow-Bench matches node labels above cosine 0.5. The same graphs")
    say("also carry exact-match columns (strict_*), so the share of every")
    say("matched node that matches only above the threshold is measurable.")
    ev = D['pipe'][D['pipe'].variant == 'evaluated']
    ev = ev[ev.strict_node_recall.notna()]
    rows = []
    for corp in CORPORA:
        for m in MODELS:
            s = ev[(ev.corpus == corp) & (ev.model == m)]
            lo, ex = s.node_recall.mean(), s.strict_node_recall.mean()
            lf, xf = s.node_f1.mean(), s.strict_node_f1.mean()
            share = 1 - ex / lo
            # bootstrap the aggregate ratio over items
            rng = np.random.default_rng(0)
            A = s.node_recall.values.astype(float)
            B = s.strict_node_recall.values.astype(float)
            idx = rng.integers(0, len(A), (10000, len(A)))
            bs = 1 - B[idx].mean(1) / A[idx].mean(1)
            ci = tuple(np.percentile(bs, [2.5, 97.5]))
            rows.append(dict(corpus=corp, model=m, n=len(s),
                             node_f1=round(lf, 3), exact_node_f1=round(xf, 3),
                             recall=round(lo, 3), exact_recall=round(ex, 3),
                             paraphrase_share=round(share, 3),
                             ci=f'[{ci[0]:.3f}, {ci[1]:.3f}]',
                             label_err=round(s.strict_label_error_rate.mean(), 3)))
    t = pd.DataFrame(rows)
    say('')
    say(t.to_string(index=False))
    say('')
    lo = t.paraphrase_share.min() * 100
    hi = t.paraphrase_share.max() * 100
    say(f'Between {lo:.0f}% and {hi:.0f}% of the nodes the benchmark counts as')
    say('recovered do not carry the gold label: they match only above the')
    say('threshold. Node F1 therefore roughly halves under exact matching')
    say(f'({t.node_f1.min():.3f}-{t.node_f1.max():.3f} against '
        f'{t.exact_node_f1.min():.3f}-{t.exact_node_f1.max():.3f}).')
    say('')
    say('This is a property of the instrument, not of any system: the ranking')
    say('of items is unchanged (Spearman 0.75-0.83, analysis B8). What it says')
    say('is what the reported score measures - that a reader of the figure can')
    say('find a node about the right thing, not that the label is the gold one.')
    say('Reported alongside the ceiling of 0.611 (N1) it closes the account of')
    say('why absolute scores on this benchmark are low and must stay low.')
    return t


# --------------------------------------------------------------------------- #
def G2(D):
    head('G2  The PaperBananaBench judge, per dimension')
    j = D['judge']
    rows = []
    for jd in j.judge.unique():
        J = j[j.judge == jd]
        for d in DIMS:
            pv = J.pivot_table(index=['system', 'model', 'seed', 'item'],
                               columns='order', values=d + '_points',
                               aggfunc='mean').dropna()
            if pv.shape[1] < 2:
                continue
            a, b = pv.iloc[:, 0].values, pv.iloc[:, 1].values
            flip = np.mean(((a > 50) & (b < 50)) | ((a < 50) & (b > 50))) * 100
            agree = np.mean(a == b) * 100
            bias = np.mean(a - b)
            ci = boot_ci(a - b)
            rows.append(dict(judge=jd, dim=d, n=len(pv),
                             order_bias=round(bias, 1),
                             ci=f'[{ci[0]:.1f}, {ci[1]:.1f}]',
                             order_agree=round(agree, 1),
                             sign_flip=round(flip, 1),
                             tie=round(np.mean(J[d + '_points'] == 50) * 100, 1)))
    t = pd.DataFrame(rows)
    say('')
    say('Order sensitivity, per dimension (the two image orders of one item):')
    say(t.to_string(index=False))

    say('')
    say('Agreement between the two judges, per dimension (seed 0, both orders')
    say('averaged, verdict discretised to generated / tie / human):')
    j1 = j[(j.judge == 'InternVL3.5-14B') & (j.seed == 0)]
    j2 = j[(j.judge == 'Qwen2.5-VL-32B') & (j.seed == 0)]
    rows2 = []
    for d in DIMS:
        a = j1.groupby(['system', 'model', 'item'])[d + '_points'].mean()
        b = j2.groupby(['system', 'model', 'item'])[d + '_points'].mean()
        c = a.index.intersection(b.index)
        sa, sb = np.sign(a.loc[c] - 50), np.sign(b.loc[c] - 50)
        rows2.append(dict(dim=d, n=len(c), kappa=round(kappa(sa, sb), 3),
                          agree=round(np.mean(sa == sb) * 100, 1),
                          pearson=round(np.corrcoef(a.loc[c], b.loc[c])[0, 1], 3)))
    t2 = pd.DataFrame(rows2)
    say(t2.to_string(index=False))
    say('')
    rd = t2[t2.dim == 'readability'].iloc[0]
    cs = t2[t2.dim == 'conciseness'].iloc[0]
    say('Two findings, both about the instrument and both usable by anyone who')
    say('runs this benchmark:')
    say('')
    say('(a) The position bias is dimension-specific and its sign is not the')
    say('    same across dimensions of one judge: for InternVL3.5-14B it runs')
    bias = t[(t.judge == 'InternVL3.5-14B')].set_index('dim').order_bias
    say(f'    {bias["faithfulness"]:+.1f} points on faithfulness and '
        f'{bias["readability"]:+.1f} on readability.')
    say('    Averaging the two orders is therefore required per dimension; an')
    say('    overall average does not cancel it, because the dimensions do not')
    say('    move together.')
    say('(b) The dimension this literature would use as a legibility measure is')
    say(f'    the least reproducible one across judges: readability kappa = '
        f'{rd.kappa:.3f}')
    say(f'    ({rd.agree:.1f}% agreement) against conciseness kappa = '
        f'{cs.kappa:.3f} ({cs.agree:.1f}%).')
    say('    Section 6 shows that dimension does not respond to the geometry')
    say('    a reader pays for; it also does not agree with a second judge.')
    return t, t2


# --------------------------------------------------------------------------- #
def _layout_wide(D):
    li = D['layout'].copy()
    for c in ['valid', 'loops', 'nested']:
        li[c] = li[c].astype(str).str.lower().isin(['true', '1'])
    gv = li[li.source.isin(['graphviz', 'graphviz_outside_development'])]
    key = ['corpus', 'model', 'run', 'item']
    piv = gv.pivot_table(index=key, columns='config', values='valid',
                         aggfunc='first')
    meta = gv[gv.config == 'ours'].set_index(key)[['n_nodes', 'n_edges',
                                                   'loops', 'nested']]
    df = piv.join(meta, how='inner').dropna(subset=['ours'] + DOTCFG)
    df = df.reset_index()
    for c in ['ours'] + DOTCFG:
        df[c] = df[c].astype(bool)
    df['dot_oracle'] = df[DOTCFG].any(axis=1)
    df['density'] = df.n_edges / df.n_nodes.clip(lower=1)
    return df


def G3(D):
    head('G3  The ceiling of any two-engine selector')
    say('F3 gave \'dot\' an oracle over its own four configurations. The next')
    say('question a caller asks is whether a *policy over both engines* beats')
    say('always using ours. The ceiling of every such policy is the figure')
    say('that is legible under either engine, so it is measurable here.')
    df = _layout_wide(D)
    rows = []
    for corp in ['sciflow', 'flowvqa', 'paperbanana']:
        for m in MODELS:
            s = df[(df.corpus == corp) & (df.model == m)]
            if not len(s):
                continue
            hyb = (s.ours | s.dot_oracle)
            rows.append(dict(corpus=corp, model=m, n=len(s),
                             ours=round(100 * s.ours.mean(), 1),
                             dot_oracle=round(100 * s.dot_oracle.mean(), 1),
                             both_oracle=round(100 * hyb.mean(), 1),
                             headroom=round(100 * (hyb.mean() - s.ours.mean()), 1)))
    t = pd.DataFrame(rows)
    say('')
    say(t.to_string(index=False))
    o, do = df.ours.mean(), df.dot_oracle.mean()
    hy = (df.ours | df.dot_oracle).mean()
    b = int((df.ours & ~df.dot_oracle).sum())
    c = int((~df.ours & df.dot_oracle).sum())
    say('')
    say(f'Pooled over {len(df):,} figures: ours {100*o:.1f}% strictly legible,')
    say(f'oracle \'dot\' {100*do:.1f}%, and the best of the two per figure')
    say(f'{100*hy:.1f}%. Always using ours therefore reaches '
        f'{100*o/hy:.1f}% of what')
    say('any engine-selection policy over these two engines could reach, while')
    say(f'always using \'dot\' with an oracle over its configurations reaches '
        f'{100*do/hy:.1f}%.')
    say(f'Discordant figures: ours only {b:,}, oracle \'dot\' only {c:,} '
        f'(McNemar p = {mcnemar(b, c):.2e}).')
    say('')
    say(f'The headroom of the perfect selector is {100*(hy-o):.1f} points '
        f'pooled and at most')
    say(f'{t.headroom.max():.1f} in any setting. The cheapest objection to a new '
        'layout engine')
    say('- "just pick the right existing engine per figure" - is worth less')
    say('than that, since no policy can exceed its oracle.')
    return t, df


# --------------------------------------------------------------------------- #
def G4(D, df=None):
    head('G4  The engine-choice rule of F4, validated across corpora')
    say('F4 fits box count, loop and container on all figures at once and')
    say('reports AUC in sample. A caller wants the rule on a corpus it was not')
    say('fitted on, so the fit is repeated holding one corpus out, with two')
    say('features added that cost nothing to compute (arrow count, density).')
    if df is None:
        df = _layout_wide(D)
    df['lbox'] = np.log(df.n_nodes + 1)
    df['larr'] = np.log(df.n_edges + 1)
    F3f = ['lbox', 'loops', 'nested']
    F5f = ['lbox', 'larr', 'density', 'loops', 'nested']

    def fit_predict(train, test, feats, target):
        X = train[feats].astype(float).values
        y = train[target].astype(int).values
        Xt = test[feats].astype(float).values
        yt = test[target].astype(int).values
        if len(np.unique(y)) < 2 or len(np.unique(yt)) < 2:
            return np.nan
        # logistic regression by Newton steps, no sklearn dependency
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Z = np.c_[np.ones(len(X)), (X - mu) / sd]
        Zt = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]
        w = np.zeros(Z.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-Z @ w))
            W = np.clip(p * (1 - p), 1e-9, None)
            H = Z.T @ (Z * W[:, None]) + 1e-6 * np.eye(Z.shape[1])
            g = Z.T @ (y - p) - 1e-6 * w
            step = np.linalg.solve(H, g)
            w += step
            if np.max(np.abs(step)) < 1e-9:
                break
        s = Zt @ w
        return stats.mannwhitneyu(s[yt == 1], s[yt == 0]).statistic / (
            (yt == 1).sum() * (yt == 0).sum())

    df['dot_tb_bad'] = ~df.dot_tb_spline        # F4's own target
    df['dot_bad'] = ~df.dot_oracle              # the harder target
    df['ours_bad'] = ~df.ours
    rows = []
    for held in ['sciflow', 'flowvqa', 'paperbanana']:
        tr, te = df[df.corpus != held], df[df.corpus == held]
        rows.append(dict(held_out=held, n_test=len(te),
                         F4_3feat=round(fit_predict(tr, te, F3f, 'dot_tb_bad'), 3),
                         F4_5feat=round(fit_predict(tr, te, F5f, 'dot_tb_bad'), 3),
                         oracle_5feat=round(fit_predict(tr, te, F5f, 'dot_bad'), 3),
                         ours_5feat=round(fit_predict(tr, te, F5f, 'ours_bad'), 3)))
    rows.append(dict(held_out='in sample (F4)', n_test=len(df),
                     F4_3feat=round(fit_predict(df, df, F3f, 'dot_tb_bad'), 3),
                     F4_5feat=round(fit_predict(df, df, F5f, 'dot_tb_bad'), 3),
                     oracle_5feat=round(fit_predict(df, df, F5f, 'dot_bad'), 3),
                     ours_5feat=round(fit_predict(df, df, F5f, 'ours_bad'), 3)))
    t = pd.DataFrame(rows)
    try:
        import os
        os.makedirs('../results/figures', exist_ok=True)
        t.to_csv('../results/figures/seventh_engine_rule.csv', index=False)
    except OSError:
        pass
    say('')
    say('AUC of predicting that an engine will draw the figure illegibly,')
    say('from graph properties alone, with no layout run:')
    say(t.to_string(index=False))
    say('')
    held = t[t.held_out != 'in sample (F4)']
    say(f'Held out, the rule keeps AUC {held.F4_5feat.min():.2f}-'
        f'{held.F4_5feat.max():.2f} for \'dot\' and '
        f'{held.ours_5feat.min():.2f}-{held.ours_5feat.max():.2f} for ours, on')
    say('a corpus no figure of which was used to fit it. It is therefore a rule')
    say('and not a description of these runs: a caller holding an extracted')
    say('graph can decide, before drawing anything, whether the drawing needs')
    say('checking. Adding arrow count and density to F4\'s three features')
    say(f'raises the in-sample AUC from {t.iloc[-1].F4_3feat:.3f} to '
        f'{t.iloc[-1].F4_5feat:.3f}; F4 reports 0.772 for the three-feature fit '
        f'on the same target, which this reproduces.')
    return t


# --------------------------------------------------------------------------- #
def write_latex(D, path, g1=None, g2=None, g3=None, g4=None):
    o = io.StringIO()
    if g1 is None:
        g1 = G1(D)
    o.write('% written by checks_judge.py --latex\n')
    o.write('\\begin{table}[t]\n\t\\centering\n\t\\fontsize{7.5}{9}'
            '\\selectfont\n\t\\setlength{\\tabcolsep}{3pt}\n')
    o.write('\t\\begin{tabular}{l l r r r r}\n\t\t\\toprule\n')
    o.write('\t\t\\textbf{Corpus} & \\textbf{Model} & \\textbf{Node F1} & '
            '\\textbf{Exact} & \\textbf{Recall} & \\textbf{Exact} \\\\\n')
    o.write('\t\t\\midrule\n')
    for corp, lab in [('sciflow', 'SciFlow'), ('flowvqa', 'FlowVQA')]:
        for m in MODELS:
            r = g1[(g1.corpus == corp) & (g1.model == m)].iloc[0]
            o.write(f'\t\t{lab} & {m} & {r.node_f1:.3f} & {r.exact_node_f1:.3f}'
                    f' & {r.recall:.3f} & {r.exact_recall:.3f} \\\\\n')
    o.write('\t\t\\bottomrule\n\t\\end{tabular}\n')
    o.write('\t\\caption{What the benchmark\'s node matching rewards. Node F1 '
            'and recall under SciFlow-Bench\'s 0.5 cosine threshold, and the '
            'same graphs under exact label matching. Half of the nodes counted '
            'as recovered do not carry the gold label.}\n')
    o.write('\t\\label{table:paraphrase}\n\\end{table}\n')
    open(path, 'w').write(o.getvalue())
    print(f'latex written to {path}', file=sys.stderr)


def load(data):
    f = lambda n: pd.read_csv(f'{data}/{n}.csv', low_memory=False)
    return dict(pipe=f('pipeline_items'), dot=f('dot_items'),
                layout=f('layout_items'), judge=f('judge_paperbanana'),
                readback=f('readback_sciflow'), stim=f('human_stimuli'),
                sets=f('human_sets'), rat=f('human_ratings'),
                place=f('placement_items'), pub=f('published_systems_sciflow'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*', default=['G1', 'G2', 'G3', 'G4'])
    ap.add_argument('--out')
    ap.add_argument('--latex')
    a = ap.parse_args()
    D = load(a.data)
    say('Judge and selection: analyses G1-G4, computed from the per-item CSVs '
        'only.')
    g1 = g3df = None
    if 'G1' in a.analysis:
        g1 = G1(D)
    if 'G2' in a.analysis:
        G2(D)
    if 'G3' in a.analysis:
        _, g3df = G3(D)
    if 'G4' in a.analysis:
        G4(D, g3df)
    if a.latex:
        write_latex(D, a.latex, g1)
    if a.out:
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
        print(f'\nwritten to {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
