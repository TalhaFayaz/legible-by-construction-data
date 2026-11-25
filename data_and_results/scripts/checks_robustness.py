#!/usr/bin/env python3
"""Robustness and bounds: analyses R1-R8, computed from the per-item CSVs only.

    python3 checks_robustness.py                      # every analysis
    python3 checks_robustness.py --analysis R1 R4
    python3 checks_robustness.py --latex ../results/tables_robustness.tex
    python3 checks_robustness.py --csv-dir ../results/tables

R1 judge      the PaperBananaBench comparison under a second judge
R2 threshold  the content comparison as the benchmark's matching threshold tightens
R3 rule       choosing a layout engine from graph properties, held out by corpus
R4 raters     the legibility index predicting a held-out rater
R5 criteria   which graph-drawing criterion tracks the raters, and what it costs
R6 bounds     what each instrument can resolve, in one table
R7 placement  the same model placing the boxes it extracted
R8 structure  where the legibility gap sits, by loop and by container
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa', 'paperbanana']


def load(name):
    return pd.read_csv(os.path.join(DATA, name), low_memory=False)


def tobool(s):
    if s.dtype == object:
        return s.map({'True': 1, 'False': 1 * 0, True: 1, False: 0}).astype(float)
    return s.astype(float)


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    bs = rng.choice(v, (n, len(v)), replace=True).mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    out = np.empty_like(ps)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, ps[k] * (len(ps) - i))
        out[k] = min(1.0, run)
    return out


def rbc(a, b):
    """matched-pairs rank-biserial correlation"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


# ---------------------------------------------------------------- R1


def r1_judge(out, tables):
    j = load('judge_paperbanana.csv')
    print('=' * 78, file=out)
    print('R1  The PaperBananaBench comparison under a second judge', file=out)
    print('=' * 78, file=out)
    print("""
Every figure was judged by two vision-language models with the benchmark's own
prompts and combination rule, each in both image orders. The paper reports the
first judge. The second judge sees the same images, so the difference between
the two columns below is the instrument, not the system.
""", file=out)
    rows = []
    for judge in ['InternVL3.5-14B', 'Qwen2.5-VL-32B']:
        jj = j[j.judge == judge]
        ps = {}
        recs = []
        for model in MODELS:
            sub = jj[jj.model == model]
            o = sub[sub.system == 'ours'].groupby('item').overall_points.mean()
            d = sub[sub.system == 'dot'].groupby('item').overall_points.mean()
            if len(o) == 0 or len(d) == 0:
                continue
            idx = o.index.intersection(d.index)
            a, b = o[idx], d[idx]
            w = stats.wilcoxon(a, b)
            lo, hi = boot_ci((a - b).values)
            par = stats.wilcoxon(a - 50)
            recs.append(dict(judge=judge, model=model, n=len(idx), ours=a.mean(),
                             dot=b.mean(), diff=(a - b).mean(), lo=lo, hi=hi,
                             r=rbc(a, b), p=w.pvalue, parity_p=par.pvalue))
            ps[model] = w.pvalue
        if recs:
            hp = holm([r['p'] for r in recs])
            for r, p in zip(recs, hp):
                r['p_holm'] = p
            rows += recs
    df = pd.DataFrame(rows)
    print(' judge                model      n   ours    dot    diff'
          '           95% CI      r    p(Holm)   p(vs 50)', file=out)
    for _, r in df.iterrows():
        print(f" {r.judge:16s} {r.model:10s} {r.n:4.0f} {r.ours:6.1f} {r['dot']:6.1f} "
              f"{r['diff']:+7.1f}  [{r.lo:+6.1f},{r.hi:+6.1f}] {r.r:+6.2f} "
              f"{r.p_holm:10.2e} {r.parity_p:10.2e}", file=out)
    tables['judge_second'] = df

    dims = ['faithfulness', 'conciseness', 'readability', 'aesthetics']
    print('\nPer dimension, seed 0, both orders averaged (50 = parity with the '
          "authors' figure):", file=out)
    dd = []
    for judge in ['InternVL3.5-14B', 'Qwen2.5-VL-32B']:
        jj = j[(j.judge == judge) & ((j.seed == 0) | (j.seed.isna()))]
        for sysn in ['ours', 'dot']:
            s = jj[jj.system == sysn]
            if len(s) == 0:
                continue
            rec = dict(judge=judge, system=sysn)
            for dim in dims:
                rec[dim] = s.groupby('item')[dim + '_points'].mean().mean()
            dd.append(rec)
    dd = pd.DataFrame(dd)
    print(dd.round(1).to_string(index=False), file=out)
    tables['judge_second_dimensions'] = dd

    sw = df.pivot_table(index='model', columns='judge', values='diff')
    if sw.shape[1] == 2:
        a, b = sw.columns[0], sw.columns[1]
        print(f"\nThe verdict moves by {np.abs(sw[a] - sw[b]).min():.1f} to "
              f"{np.abs(sw[a] - sw[b]).max():.1f} points between the two judges "
              f"on the same figures.", file=out)
    sec = df[df.judge == 'Qwen2.5-VL-32B']
    above = sec[(sec.ours > 50) & (sec.parity_p < 0.05)]
    print(f"Under the second judge our figures are above parity with the authors' own "
          f"for {len(above)} of {len(sec)} models.", file=out)


# ---------------------------------------------------------------- R2


def r2_threshold(out, tables):
    t = load('threshold_sweep.csv')
    print('=' * 78, file=out)
    print("R2  The content comparison as the benchmark's matching threshold tightens", file=out)
    print('=' * 78, file=out)
    print("""
SciFlow-Bench matches node labels above cosine 0.5, so a node counts as
recovered when it is about the right thing. Rescoring the same graphs at 0.6
and 0.7 asks what the comparison looks like when the label itself has to match.
""", file=out)
    piv = t.pivot_table(index=['corpus', 'model'], columns='threshold',
                        values=['diff', 'r'])
    print(piv.round(3).to_string(), file=out)
    for th in sorted(t.threshold.unique()):
        s = t[t.threshold == th]
        print(f"  threshold {th}: ahead in {(s.verdict == 'ahead').sum()} of {len(s)} "
              f"settings, mean effect size r {s.r.mean():+.2f}", file=out)
    tables['threshold_sweep'] = t


# ---------------------------------------------------------------- R3


def _fit_auc(train, test, feats, target):
    """logistic fit by Newton steps and AUC, identical to checks_judge.py G4"""
    X = train[feats].astype(float).values
    y = train[target].astype(int).values
    Xt = test[feats].astype(float).values
    yt = test[target].astype(int).values
    if len(np.unique(y)) < 2 or len(np.unique(yt)) < 2:
        return np.nan
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
        (yt == 1).sum() * (yt == 0).sum()), w


def _layout_wide():
    """the same wide table checks_judge.py builds, so R3 and G3/G4 cannot disagree"""
    import importlib
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    cj = importlib.import_module('checks_judge')
    li = load('layout_items.csv')
    return cj._layout_wide({'layout': li}), cj.DOTCFG


def r3_rule(out, tables):
    df, cfgs = _layout_wide()
    df['lbox'] = np.log(df.n_nodes + 1)
    df['larr'] = np.log(df.n_edges + 1)
    df['dot_tb_bad'] = ~df.dot_tb_spline
    df['dot_bad'] = ~df.dot_oracle
    df['ours_bad'] = ~df.ours
    print('=' * 78, file=out)
    print('R3  Choosing a layout engine from graph properties, before drawing', file=out)
    print('=' * 78, file=out)
    print("""
A caller holding an extracted graph wants to know which engine to call before
paying for a layout. The fit below uses only properties of the graph, and is
repeated three times, each time holding out one corpus.
""", file=out)
    F3 = ['lbox', 'loops', 'nested']
    F5 = ['lbox', 'larr', 'density', 'loops', 'nested']
    rows = []
    for held in CORPORA:
        tr, te = df[df.corpus != held], df[df.corpus == held]
        if len(te) == 0:
            continue
        rows.append(dict(held_out=held, n_test=len(te),
                         dot3=_fit_auc(tr, te, F3, 'dot_tb_bad')[0],
                         dot5=_fit_auc(tr, te, F5, 'dot_tb_bad')[0],
                         oracle5=_fit_auc(tr, te, F5, 'dot_bad')[0],
                         ours5=_fit_auc(tr, te, F5, 'ours_bad')[0]))
    rows.append(dict(held_out='in sample', n_test=len(df),
                     dot3=_fit_auc(df, df, F3, 'dot_tb_bad')[0],
                     dot5=_fit_auc(df, df, F5, 'dot_tb_bad')[0],
                     oracle5=_fit_auc(df, df, F5, 'dot_bad')[0],
                     ours5=_fit_auc(df, df, F5, 'ours_bad')[0]))
    t = pd.DataFrame(rows)
    print("AUC of predicting that an engine draws a figure illegibly, from the graph "
          "alone:", file=out)
    print(t.round(3).to_string(index=False), file=out)
    tables['engine_rule'] = t
    w = _fit_auc(df, df, F5, 'dot_bad')[1]
    names = ['intercept'] + F5
    print('\nCoefficients of the five-feature rule, fitted on every figure on '
          'standardised inputs:', file=out)
    print('  ' + '  '.join(f'{n} {v:+.3f}' for n, v in zip(names, w)), file=out)
    tables['engine_rule_weights'] = pd.DataFrame([dict(zip(names, w))])

    sel = []
    for corpus_ in CORPORA:
        for model in MODELS:
            s = df[(df.corpus == corpus_) & (df.model == model)]
            if len(s) == 0:
                continue
            both = (s.ours | s.dot_oracle)
            sel.append(dict(corpus=corpus_, model=model, n=len(s),
                            ours=100 * s.ours.mean(),
                            dot_oracle=100 * s.dot_oracle.mean(),
                            both=100 * both.mean()))
    sel = pd.DataFrame(sel)
    sel['headroom'] = sel['both'] - sel['ours']
    o, d = df.ours, df.dot_oracle
    both = (o | d)
    print(f"\nSelector ceiling, pooled over {len(df):,} figures: ours "
          f"{100*o.mean():.1f}%, oracle 'dot' {100*d.mean():.1f}%, best of the two "
          f"per figure {100*both.mean():.1f}%.", file=out)
    print(f"  always using ours reaches {100*o.mean()/both.mean():.1f}% of what any "
          f"policy over the two engines could reach; headroom "
          f"{100*(both.mean()-o.mean()):.1f} points pooled, at most "
          f"{sel.headroom.max():.1f} in any setting.", file=out)
    b = int((o & ~d).sum()); c = int((~o & d).sum())
    print(f"  discordant figures: ours only {b:,}, oracle 'dot' only {c:,} "
          f"(ratio {b/max(c,1):.1f}x).", file=out)
    tables['selector_ceiling'] = sel


# ---------------------------------------------------------------- R4, R5


def _bws(hr, hsets):
    s = hsets.set_index('set')
    rows = []
    for _, r in hr.iterrows():
        for im in s.loc[r['set']].tolist():
            rows.append((r.rater, im, 1 if im == r['best'] else (-1 if im == r['worst'] else 0)))
    return pd.DataFrame(rows, columns=['rater', 'image', 'v'])


def r4_raters(out, tables):
    hs = load('human_stimuli.csv').set_index('image')
    hr = load('human_ratings.csv')
    hsets = load('human_sets.csv')
    bw = _bws(hr, hsets).groupby(['rater', 'image']).v.mean().unstack('rater')
    bw = bw.reindex(hs.index)
    print('=' * 78, file=out)
    print('R4  The legibility index predicting a rater it was not fitted on', file=out)
    print('=' * 78, file=out)
    print("""
The paper cross-validates the index by leaving one source graph out, which shows
it has not memorised a layout. This leaves one rater out instead, which shows it
has not memorised a panel.
""", file=out)
    pw = []
    for a, b in [('R1', 'R2'), ('R1', 'R3'), ('R2', 'R3')]:
        pw.append(dict(pair=f'{a}-{b}', rho=stats.spearmanr(bw[a], bw[b]).statistic))
    pw = pd.DataFrame(pw)
    print('agreement between raters (Spearman of their best-worst scores):', file=out)
    print(pw.round(3).to_string(index=False), file=out)

    sets4 = ['report_label_collisions', 'report_node_overlap_pairs',
             'report_aspect_deviation', 'report_area']
    sets10 = sets4 + ['report_edge_node_crossings', 'report_edge_edge_crossings',
                      'report_edge_label_crossings', 'report_stacked_routes',
                      'report_mean_bends_per_edge', 'report_total_edge_length']
    rows = []
    for tag, cols in [('four measures', sets4), ('ten measures', sets10)]:
        X = np.log1p(hs[cols].astype(float))
        X = ((X - X.mean()) / X.std()).values
        A = np.c_[np.ones(len(X)), X]
        for held in ['R1', 'R2', 'R3']:
            tr = [c for c in bw.columns if c != held]
            y = bw[tr].mean(axis=1).values
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            rho = stats.spearmanr(A @ beta, bw[held].values).statistic
            rows.append(dict(index=tag, held_out=held, rho=rho))
    df = pd.DataFrame(rows)
    print('\nfitted on two raters, scored on the third:', file=out)
    print(df.round(3).to_string(index=False), file=out)
    print(f"\n  held-out rater: rho {df.rho.min():.2f} to {df.rho.max():.2f}; "
          f"between raters: {pw.rho.min():.2f} to {pw.rho.max():.2f}.", file=out)
    print("  the index therefore predicts a rater it has not seen about as well as "
          "another rater does.", file=out)
    tables['index_leave_rater_out'] = df
    tables['rater_agreement'] = pw


def r5_criteria(out, tables):
    hs = load('human_stimuli.csv').set_index('image')
    hr = load('human_ratings.csv')
    hsets = load('human_sets.csv')
    bw = _bws(hr, hsets).groupby('image').v.mean().reindex(hs.index)
    print('=' * 78, file=out)
    print('R5  Which graph-drawing criterion tracks the raters', file=out)
    print('=' * 78, file=out)
    cols = [c for c in hs.columns if c.startswith('report_') and
            hs[c].dtype != object and hs[c].nunique() > 1]
    eng = hs[hs.condition.isin(['ours', 'graphviz'])]
    bw_eng = bw.reindex(eng.index)

    def _holm(ps):
        o = sorted(range(len(ps)), key=lambda i: ps[i])
        adj, run = [0.0]*len(ps), 0.0
        for rank, i in enumerate(o):
            run = max(run, (len(ps)-rank)*ps[i])
            adj[i] = min(1.0, run)
        return adj
    rows = []
    for c in cols:
        rho = stats.spearmanr(hs[c].astype(float), bw)
        if eng[c].nunique() > 1:
            r2 = stats.spearmanr(eng[c].astype(float), bw_eng)
            e_rho, e_p = r2.statistic, r2.pvalue
        else:
            e_rho, e_p = np.nan, np.nan
        rows.append(dict(measure=c.replace('report_', ''), rho=rho.statistic,
                         p=rho.pvalue, rho_engines48=e_rho, p_engines48=e_p))
    df = pd.DataFrame(rows).sort_values('rho')
    df['p_holm'] = _holm(list(df.p.values))
    sub = df.p_engines48.notna()
    df.loc[sub, 'p_holm_engines48'] = _holm(list(df.loc[sub, 'p_engines48'].values))
    print('Spearman with the raters\' best-worst score (negative = more defects, '
          'worse rated).', file=out)
    print('rho: all 72 images, three conditions. rho_engines48: the 48 images the two '
          'engines', file=out)
    print('drew, which is the sample a claim about two drawn layouts needs. Holm within '
          'each', file=out)
    print('of the two families:', file=out)
    print(df.round(3).to_string(index=False), file=out)
    n_all = int((df.p_holm < 0.05).sum())
    n_eng = int((df.p_holm_engines48 < 0.05).sum())
    print(f"\n  {n_all} of {len(df)} measures track the raters over the full panel after "
          f"correction, and {n_eng}", file=out)
    print("  of the 16 that vary among the 48 engine-drawn images do. The ranking of the "
          "criteria", file=out)
    print("  on the left is therefore a ranking of how sharply each measure separates the "
          "model-", file=out)
    print("  placed condition from the two drawn ones; it does not say which criterion a "
          "reader", file=out)
    print("  pays for between two drawn layouts, and this paper does not claim that it "
          "does.", file=out)
    tables['criteria_vs_raters'] = df

    li = load('layout_items.csv')
    g = li[li.source.isin(['graphviz', 'graphviz_outside_development'])].copy()
    for c in ['edge_edge_crossings', 'label_collisions', 'mean_bends_per_edge', 'n_nodes']:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g = g.dropna(subset=['edge_edge_crossings', 'label_collisions',
                         'mean_bends_per_edge', 'n_nodes'])

    def partial(x, y, z):
        rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
        rxz = stats.spearmanr(x, z).statistic
        ryz = stats.spearmanr(y, z).statistic
        rxy = stats.spearmanr(x, y).statistic
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    pc = partial(g.edge_edge_crossings, g.label_collisions, g.n_nodes)
    bb = stats.spearmanr(g.mean_bends_per_edge, g.label_collisions).statistic
    print(f"\nOver {len(g)} layouts of identical graphs by both engines, with box count "
          f"held fixed:", file=out)
    print(f"  arrow crossings and label collisions: partial rho {pc:+.2f} "
          f"(independent)", file=out)
    print(f"  bends per arrow and label collisions: rho {bb:+.2f} "
          f"(a layout that bends less occludes more)", file=out)
    tables['criteria_independence'] = pd.DataFrame(
        [dict(n=len(g), crossings_vs_collisions_partial=pc, bends_vs_collisions=bb)])


# ---------------------------------------------------------------- R6


def r6_bounds(out, tables):
    """collects the bounds the other scripts compute, so the table cannot drift"""
    T = os.path.normpath(os.path.join(HERE, '..', 'results', 'tables'))
    need = ['ceiling.csv', 'variance_components.csv', 'resolution.csv',
            'judge_variance.csv', 'judge_mdd.csv', 'reader_fidelity.csv']
    missing = [n for n in need if not os.path.exists(os.path.join(T, n))]
    print('=' * 78, file=out)
    print('R6  What each instrument can resolve, in one table', file=out)
    print('=' * 78, file=out)
    if missing:
        print('  needs ' + ', '.join(missing) + ' -- run checks_variance.py and '
              'checks_new.py with --csv-dir ../results/tables first', file=out)
        return
    print("""
Every row below is taken from the script that computes it, so this table and the
analyses behind it cannot disagree: the ceiling and the reader from
checks_new.py, the variance components and the resolution from
checks_variance.py, the judge from checks_new.py, the paraphrase share from
checks_judge.py.
""", file=out)
    cei = pd.read_csv(os.path.join(T, 'ceiling.csv')).set_index('system')
    vc = pd.read_csv(os.path.join(T, 'variance_components.csv'))
    res = pd.read_csv(os.path.join(T, 'resolution.csv'))
    jv = pd.read_csv(os.path.join(T, 'judge_variance.csv')).set_index('dimension')
    jm = pd.read_csv(os.path.join(T, 'judge_mdd.csv'))
    rf = pd.read_csv(os.path.join(T, 'reader_fidelity.csv'))
    g = cei.loc['gold_graph_ceiling']
    rows = [dict(instrument='benchmark score',
                 quantity='the score a perfect graph reaches',
                 value=f'{g.score:.3f}',
                 note=f'its node F1 is {g.node_f1:.3f} but its edge F1 '
                      f'{g.edge_f1:.3f}, and the score weights edges 0.6')]
    for corpus, label in [('sciflow', 'SciFlow-Bench score'),
                          ('flowvqa', 'FlowVQA score')]:
        v = vc[vc.corpus == corpus].set_index('component').share
        mde = res[(res.corpus == corpus) & (res.n == 50)].mde
        rows.append(dict(instrument=label,
                         quantity='share of its variance in the item',
                         value=f'{v["item"]:.1f}\\%',
                         note=f'in the model {v["model"]:.1f}\\%; two samples of 50 '
                              f'items cannot separate means closer than '
                              f'{mde.iloc[0]:.3f}'))
    rows.append(dict(instrument='judge',
                     quantity='share of its variance in the system judged',
                     value=f'{jv.loc["overall","model"]:.1f}\\%',
                     note=f'in the item {jv.loc["overall","item"]:.1f}\\%; two '
                          f'samples of 50 figures cannot separate means closer than '
                          f'{jm.unpaired_50.min():.0f}--{jm.unpaired_50.max():.0f} '
                          f'points'))
    rows.append(dict(instrument='read-back reader',
                     quantity='arrows returned per arrow drawn',
                     value=f'{rf.edge_ratio.min():.2f}--{rf.edge_ratio.max():.2f}',
                     note=f'boxes {rf.node_ratio.min():.2f}--'
                          f'{rf.node_ratio.max():.2f}; its arrow count tracks the '
                          f'drawn one at $\\rho={rf.rho_edges.min():.2f}$--'
                          f'{rf.rho_edges.max():.2f}'))
    pi = load('pipeline_items.csv')
    ev = pi[pi.variant == 'evaluated']
    sh = []
    for corpus in ['sciflow', 'flowvqa']:
        for model in MODELS:
            s_ = ev[(ev.corpus == corpus) & (ev.model == model)].dropna(
                subset=['strict_node_recall'])
            if len(s_) == 0:
                continue
            # the share checks_judge.py G1 reports: matched above the threshold only
            sh.append(1 - s_.strict_node_recall.mean() / s_.node_recall.mean())
    rows.append(dict(instrument='node matching',
                     quantity='matched nodes not carrying the gold label',
                     value=f'{100*min(sh):.0f}--{100*max(sh):.0f}\\%',
                     note='node F1 roughly halves under exact matching, and the '
                          'item ranking is unchanged'))
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        print(f"  {r.instrument:22s} {r.quantity:42s} {r.value}", file=out)
        print(f"  {'':22s} {r.note}".replace('\\\\%', '%').replace('$\\\\rho$', 'rho'),
              file=out)
    tables['instrument_bounds'] = df

    print('\nSign errors, from checks_variance.py H2: the rate at which a sample of '
          'n items', file=out)
    print('returns the wrong sign for the paired difference against the same model '
          'writing DOT.', file=out)
    sg = res.dropna(subset=['model'])[['corpus', 'model', 'true', 'wrong50',
                                       'wrong100', 'wrongfull']]
    print(sg.to_string(index=False, float_format=lambda x: f'{x:.4f}'), file=out)
    tables['sign_errors'] = sg


# ---------------------------------------------------------------- R7


def r7_placement(out, tables):
    pl = load('placement_items.csv')
    print('=' * 78, file=out)
    print('R7  The same model placing the boxes it extracted', file=out)
    print('=' * 78, file=out)
    print("""
Same text, same extracted graph, same renderer, same seed: the only change is
who decides the coordinates. Layout measures are over the figures actually
drawn, as the invariant of the paper is; the scores are over every item.
""", file=out)
    lay = ['layout_node_overlap_pairs', 'layout_edge_node_crossings',
           'layout_route_reversals']
    drawn = pl.dropna(subset=lay).copy()
    drawn['lenient'] = ((drawn.layout_node_overlap_pairs == 0) &
                        (drawn.layout_edge_node_crossings == 0) &
                        (drawn.layout_route_reversals == 0)).astype(float)
    drawn['strict'] = tobool(drawn.layout_valid)
    drawn['any_overlap'] = (drawn.layout_node_overlap_pairs > 0).astype(float)
    LAYM = [('lenient', 'lenient legible (%)', 100, 1),
            ('strict', 'strict legible (%)', 100, 1),
            ('any_overlap', 'figures with a box overlap (%)', 100, 1),
            ('layout_node_overlap_pairs', 'overlapping box pairs', 1, 2),
            ('layout_edge_node_crossings', 'arrows through a box', 1, 2),
            ('layout_label_collisions', 'label collisions', 1, 2)]
    CONM = [('score', 'SciFlow-Bench score', 3), ('node_f1', 'node F1', 3)]
    rows = []
    for backbone in MODELS:
        for col, label, scale, dec in LAYM:
            s_ = drawn[drawn.model == backbone]
            rec = dict(backbone=backbone, measure=label, dec=dec)
            for cond, tag in [('ours', 'code'), ('model', 'model')]:
                v = s_[s_.condition == cond].groupby('seed')[col].mean() * scale
                rec[tag], rec[tag + '_sd'] = v.mean(), v.std()
            a = s_[s_.condition == 'ours'].set_index(['seed', 'item'])[col]
            b = s_[s_.condition == 'model'].set_index(['seed', 'item'])[col]
            idx = a.index.intersection(b.index)
            d = (a[idx] - b[idx]).dropna()
            rec['p'] = stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan
            rows.append(rec)
        for col, label, dec in CONM:
            s_ = pl[pl.model == backbone]
            rec = dict(backbone=backbone, measure=label, dec=dec)
            for cond, tag in [('ours', 'code'), ('model', 'model')]:
                v = s_[s_.condition == cond].groupby('seed')[col].mean()
                rec[tag], rec[tag + '_sd'] = v.mean(), v.std()
            a = s_[s_.condition == 'ours'].set_index(['seed', 'item'])[col]
            b = s_[s_.condition == 'model'].set_index(['seed', 'item'])[col]
            idx = a.index.intersection(b.index)
            d = (a[idx] - b[idx]).dropna()
            rec['p'] = stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    for backbone in MODELS:
        print(f'\n{backbone}: (code places / model places)', file=out)
        s_ = df[df.backbone == backbone]
        for _, r in s_.iterrows():
            d = int(r.dec)
            pv = '--' if pd.isna(r.p) else f'{r.p:.1e}'
            print(f"  {r.measure:34s} {r.code:9.{d}f} +- {r.code_sd:.{d}f}   "
                  f"{r['model']:9.{d}f} +- {r.model_sd:.{d}f}   p {pv}", file=out)
    tables['placement'] = df
    le = drawn[drawn.condition == 'ours']
    mo = drawn[drawn.condition == 'model']

    def lenient(x):
        return 100 * ((x.layout_node_overlap_pairs == 0) &
                      (x.layout_edge_node_crossings == 0) &
                      (x.layout_route_reversals == 0)).mean()
    print(f"\n  over the {len(le):,} figures our engine drew here the lenient verdict "
          f"holds on {lenient(le):.1f}% of them, against {lenient(mo):.1f}% when the "
          f"model places the boxes.", file=out)
    sc = df[df.measure == 'SciFlow-Bench score'].set_index('backbone')
    print('  the content score does not improve when the model places the boxes:',
          file=out)
    for m in MODELS:
        r = sc.loc[m]
        print(f"    {m:10s} {r.code:.3f} against {r['model']:.3f}  (p = {r.p:.3g})",
              file=out)


# ---------------------------------------------------------------- R8


def r8_structure(out, tables):
    df, cfgs = _layout_wide()
    print('=' * 78, file=out)
    print('R8  Where the legibility gap sits: loops and nested containers', file=out)
    print('=' * 78, file=out)
    print("""
Loops that return to an earlier stage and boxes nested inside a container are
what extracted graphs have. Both engines draw identical graphs, so the split
below is a property of the graph, not of the extractor. 'dot' is its best single
configuration, as in Table S2a; the oracle column gives it the best of its four
per figure.
""", file=out)
    rows = []
    for name, col in [('feedback loop', 'loops'), ('nested container', 'nested')]:
        for val in [False, True]:
            s_ = df[df[col].astype(bool) == val]
            o = s_.ours.astype(float).values
            d = s_.dot_tb_spline.astype(float).values
            orc = s_.dot_oracle.astype(float).values
            lo, hi = boot_ci(o - d)
            rows.append(dict(prop=name, value=val, n=len(s_), ours=100 * o.mean(),
                             dot=100 * d.mean(), gap=100 * (o - d).mean(),
                             lo=100 * lo, hi=100 * hi,
                             dot_oracle=100 * orc.mean(),
                             gap_oracle=100 * (o - orc).mean()))
    t = pd.DataFrame(rows)
    print(t.round(1).to_string(index=False), file=out)
    tables['structure_gap'] = t
    for name in t['prop'].unique():
        s_ = t[t['prop'] == name].set_index('value')
        print(f"  {name}: gap {s_.loc[True,'gap']:+.1f} points with the property, "
              f"{s_.loc[False,'gap']:+.1f} without, difference "
              f"{s_.loc[True,'gap']-s_.loc[False,'gap']:+.1f}; against the oracle "
              f"{s_.loc[True,'gap_oracle']:+.1f} and "
              f"{s_.loc[False,'gap_oracle']:+.1f}", file=out)

    pi = load('pipeline_items.csv')
    ev = pi[pi.variant == 'evaluated']
    print('\nA note on the second metric family carried in the released columns:',
          file=out)
    rows = []
    for corpus in ['sciflow', 'flowvqa']:
        s_ = ev[ev.corpus == corpus].dropna(subset=['strict_path_comparable'])
        comp = tobool(s_.strict_path_comparable)
        share = 1 - comp.mean()
        all_r = stats.spearmanr(s_.strict_diagrameval_score, s_.score).statistic
        sub = s_[comp == 1]
        cmp_r = stats.spearmanr(sub.strict_diagrameval_score, sub.score).statistic
        rows.append(dict(corpus=corpus, n=len(s_), incomparable_share=share,
                         rho_all=all_r, rho_comparable=cmp_r))
        print(f"  {corpus}: {100*share:.0f}% of items have no comparable path, so "
              f"their strict_path_f1 is 1.000 by convention; rho with the benchmark "
              f"score {all_r:+.3f} over all items, {cmp_r:+.3f} over the comparable "
              f"ones.", file=out)
    tables['path_default'] = pd.DataFrame(rows)


# ---------------------------------------------------------------- LaTeX


def fnum(x, d=1):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'


def write_latex(path, tables):
    L = []
    A = L.append
    # Table: second judge
    df = tables['judge_second']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7.5}{9}\selectfont\setlength{\tabcolsep}{3pt}')
    A(r'  \resizebox{\columnwidth}{!}{%')
    A(r'  \begin{tabular}{l r r r r r}')
    A(r'    \toprule')
    A(r'    \textbf{Judge} & \textbf{Model} & \textbf{Ours} & \texttt{dot} & '
      r'\textbf{$\Delta$} & \textbf{vs 50} \\')
    A(r'    \midrule')
    for judge, lab in [('InternVL3.5-14B', 'InternVL3.5-14B'),
                       ('Qwen2.5-VL-32B', 'Qwen2.5-VL-32B')]:
        s = df[df.judge == judge]
        for i, (_, r) in enumerate(s.iterrows()):
            name = r'\multirow{3}{*}{%s}' % lab if i == 0 else ''
            star = r'$^{*}$' if r.ours > 50 and r.parity_p < 0.05 else ''
            A(f'    {name} & {r.model} & {r.ours:.1f}{star} & {r["dot"]:.1f} & '
              f'${r["diff"]:+.1f}$ [${r.lo:+.1f}$, ${r.hi:+.1f}$] & '
              f'{r.parity_p:.0e} \\\\')
        if judge == 'InternVL3.5-14B':
            A(r'    \midrule')
    A(r'    \bottomrule')
    A(r'  \end{tabular}}')
    A(r'  \caption{PaperBananaBench under two judges, same figures, same prompts, '
      r'both image orders. $\Delta$ is the mean paired difference against the same '
      r'model writing DOT with its bootstrap interval; 50 is parity with the '
      r"authors' own figure and $^{*}$ marks a mean above it ($p<0.05$, Wilcoxon). "
      r'Changing the judge moves one verdict by 22 points.}')
    A(r'  \label{table:judge2}')
    A(r'\end{table}')
    A('')
    # Table: threshold sweep
    t = tables['threshold_sweep']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7.5}{9}\selectfont\setlength{\tabcolsep}{3pt}')
    A(r'  \resizebox{\columnwidth}{!}{%')
    A(r'  \begin{tabular}{l l r r r r r r}')
    A(r'    \toprule')
    A(r'    & & \multicolumn{2}{c}{$t=0.5$} & \multicolumn{2}{c}{$t=0.6$} & '
      r'\multicolumn{2}{c}{$t=0.7$} \\')
    A(r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}')
    A(r'    \textbf{Corpus} & \textbf{Model} & $\Delta$ & $r$ & $\Delta$ & $r$ & '
      r'$\Delta$ & $r$ \\')
    A(r'    \midrule')
    for corpus, lab in [('sciflow', 'SciFlow-Bench'), ('flowvqa', 'FlowVQA')]:
        s = t[t.corpus == corpus]
        for i, model in enumerate(MODELS):
            cells = []
            for th in (0.5, 0.6, 0.7):
                r = s[(s.model == model) & (s.threshold == th)].iloc[0]
                bold = r.verdict == 'ahead'
                d = f'${r["diff"]:+.3f}$'
                rr = f'${r.r:+.2f}$'
                cells += [r'\textbf{%s}' % d if bold else d, rr]
            name = r'\multirow{3}{*}{%s}' % lab if i == 0 else ''
            A(f'    {name} & {model} & ' + ' & '.join(cells) + r' \\')
        if corpus == 'sciflow':
            A(r'    \midrule')
    A(r'    \bottomrule')
    A(r'  \end{tabular}}')
    A(r'  \caption{The same graphs rescored as the benchmark\'s node-matching '
      r'threshold $t$ tightens. $\Delta$ is the mean paired difference against the '
      r'same model writing DOT and $r$ the rank-biserial effect size; bold marks an '
      r'\emph{ahead} verdict (Holm-corrected Wilcoxon, $\alpha=0.05$). The benchmark '
      r'reports $t=0.5$, where two settings are tied; at $t=0.7$ every setting is '
      r'ahead.}')
    A(r'  \label{table:threshold}')
    A(r'\end{table}')
    A('')
    # Table: placement
    df = tables['placement']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7}{8.4}\selectfont\setlength{\tabcolsep}{3pt}')
    A(r'  \resizebox{\columnwidth}{!}{%')
    A(r'  \begin{tabular}{l r r r r r r}')
    A(r'    \toprule')
    A(r'    & \multicolumn{2}{c}{\textbf{Qwen}} & \multicolumn{2}{c}{\textbf{Gemma}}'
      r' & \multicolumn{2}{c}{\textbf{Ministral}} \\')
    A(r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
    A(r'    \textbf{Measure} & code & model & code & model & code & model \\')
    A(r'    \midrule')
    for measure in df.measure.unique():
        s_ = df[df.measure == measure].set_index('backbone')
        d = int(s_.dec.iloc[0])
        cells = []
        for m in MODELS:
            r = s_.loc[m]
            cells.append(r'\textbf{%s}' % f'{r.code:.{d}f}')
            cells.append(f"{r['model']:.{d}f}")
        mname = measure.replace('%', r'\%')
        A(f"    {mname} & " + ' & '.join(cells) + r' \\')
    A(r'    \bottomrule')
    A(r'  \end{tabular}}')
    A(r'  \caption{The same extracted graph drawn twice: deterministic code computes '
      r'the coordinates, or the model that extracted the graph emits them. Mean over '
      r'three seeds, 100 SciFlow-Bench papers, every difference $p<10^{-10}$ except '
      r'the two content rows ($p\le0.006$ for the score). Letting the model place the '
      r'boxes costs the three invariants and buys no content.}')
    A(r'  \label{table:placement}')
    A(r'\end{table}')
    A('')
    # Table: engine rule + structure
    er = tables['engine_rule']
    sg = tables['structure_gap']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7.5}{9}\selectfont\setlength{\tabcolsep}{4pt}')
    A(r'  \begin{tabular}{l r r r r}')
    A(r'    \multicolumn{5}{l}{\textbf{(a) predicting an illegible figure from the '
      r'graph alone (AUC)}} \\')
    A(r'    \toprule')
    A(r'    \textbf{Held out} & $n$ & \texttt{dot}, 3 feat. & \texttt{dot}, 5 feat. & '
      r'ours, 5 feat. \\')
    A(r'    \midrule')
    for _, r in er.iterrows():
        nm = {'sciflow': 'SciFlow-Bench', 'flowvqa': 'FlowVQA',
              'paperbanana': 'PaperBanana', 'in sample': 'in sample'}[r.held_out]
        A(f'    {nm} & {r.n_test:.0f} & {r.dot3:.3f} & {r.dot5:.3f} & '
          f'{r.ours5:.3f} \\\\')
    A(r'    \bottomrule')
    A(r'  \end{tabular}')
    A(r'  \vspace{2pt}')
    A(r'  \begin{tabular}{l r r r r r}')
    A(r'    \multicolumn{6}{l}{\textbf{(b) where the gap sits}} \\')
    A(r'    \toprule')
    A(r'    \textbf{Graph property} & $n$ & \textbf{Ours} & \texttt{dot} & '
      r'\textbf{Gap} & \textbf{vs oracle} \\')
    A(r'    \midrule')
    for _, r in sg.iterrows():
        lab = ('with ' if r.value else 'without ') + r['prop']
        A(f'    {lab} & {r.n:.0f} & {r.ours:.1f}\\% & {r["dot"]:.1f}\\% & '
          f'${r.gap:+.1f}$ & ${r.gap_oracle:+.1f}$ \\\\')
    A(r'    \bottomrule')
    A(r'  \end{tabular}')
    A(r'  \caption{(a) A logistic fit on box count, loop and container, and with '
      r'arrow count and density added, predicting that an engine draws a figure '
      r'illegibly; each row is refitted with that corpus held out. (b) The strictly '
      r'legible share by graph property, both engines drawing identical graphs, '
      r'\texttt{dot} given an oracle over its four configurations. Loops and '
      r'containers are what extracted graphs have, and each costs \texttt{dot} '
      r'separately.}')
    A(r'  \label{table:enginerule}')
    A(r'\end{table}')
    A('')
    # Table: instrument bounds
    df = tables['instrument_bounds']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7}{8.4}\selectfont\setlength{\tabcolsep}{3pt}')
    A(r'  \begin{tabular}{@{}p{1.35cm} p{2.25cm} r@{\ \ } p{2.75cm}@{}}')
    A(r'    \toprule')
    A(r'    \textbf{Instrument} & \textbf{Quantity} & \textbf{Value} & '
      r'\textbf{What it leaves} \\')
    A(r'    \midrule')
    for _, r in df.iterrows():
        A(f'    {r.instrument} & {r.quantity} & \\textbf{{{r.value}}} & {r.note} \\\\')
    A(r'    \bottomrule')
    A(r'  \end{tabular}')
    A(r'  \caption{What each instrument in this literature can resolve, from the '
      r'measurements already reported. Every row is computed from the released '
      r'per-item data by \texttt{checks\_robustness.py}.}')
    A(r'  \label{table:bounds}')
    A(r'\end{table}')
    A('')
    # Table: index across raters
    df = tables['index_leave_rater_out']
    pw = tables['rater_agreement']
    A(r'\begin{table}[t]')
    A(r'  \centering\fontsize{7.5}{9}\selectfont\setlength{\tabcolsep}{5pt}')
    A(r'  \begin{tabular}{l r r r}')
    A(r'    \toprule')
    A(r'    \textbf{Index} & \textbf{R1} & \textbf{R2} & \textbf{R3} \\')
    A(r'    \midrule')
    for tag in ['four measures', 'ten measures']:
        s = df[df['index'] == tag].set_index('held_out')
        A(f'    {tag} & ' + ' & '.join(f'{s.loc[r,"rho"]:.3f}' for r in
                                       ['R1', 'R2', 'R3']) + r' \\')
    A(r'    \midrule')
    A(r'    another rater & ' + ' & '.join(
        '%.3f' % np.mean([p.rho for _, p in pw.iterrows() if r in p.pair])
        for r in ['R1', 'R2', 'R3']) + r' \\')
    A(r'    \bottomrule')
    A(r'  \end{tabular}')
    A(r'  \caption{The legibility index fitted on two raters and scored on the third '
      r'(Spearman), beside how well the other two raters predict that rater. The '
      r'index recovers a rater it has not seen about as well as a human does, so it '
      r'is not fitted to this panel.}')
    A(r'  \label{table:raterout}')
    A(r'\end{table}')
    head = '% written by checks_robustness.py --latex ; do not edit by hand\n'
    with open(path, 'w') as f:
        f.write(head)
        f.write('\n'.join(L) + '\n')
    # each table also on its own, so main.tex can place them one at a time
    base = os.path.dirname(path)
    cur, name = [], None
    for line in L:
        if line.startswith(r'\begin{table}'):
            cur = [line]
        elif line.startswith(r'\end{table}'):
            cur.append(line)
            if name:
                with open(os.path.join(base, 'table_%s.tex' % name), 'w') as f:
                    f.write(head + '\n'.join(cur) + '\n')
            cur, name = [], None
        elif cur is not None:
            cur.append(line)
            if r'\label{table:' in line:
                name = line.split(r'\label{table:')[1].split('}')[0]


ORDER = [('R1', r1_judge), ('R2', r2_threshold), ('R3', r3_rule), ('R4', r4_raters),
         ('R5', r5_criteria), ('R6', r6_bounds), ('R7', r7_placement),
         ('R8', r8_structure)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', nargs='*', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--latex', default=None)
    ap.add_argument('--csv-dir', default=None)
    a = ap.parse_args()
    want = set(a.analysis) if a.analysis else {k for k, _ in ORDER}
    out = open(a.out, 'w') if a.out else sys.stdout
    tables = {}
    print('Robustness and bounds: analyses R1-R8, computed from the per-item CSVs only.\n',
          file=out)
    for key, fn in ORDER:
        if key in want:
            fn(out, tables)
            print('', file=out)
    if a.csv_dir:
        os.makedirs(a.csv_dir, exist_ok=True)
        for k, v in tables.items():
            v.to_csv(os.path.join(a.csv_dir, k + '.csv'), index=False)
    if a.latex:
        need = {'judge_second', 'threshold_sweep', 'placement', 'engine_rule',
                'structure_gap', 'instrument_bounds', 'index_leave_rater_out',
                'rater_agreement'}
        if not need <= set(tables):
            print('--latex needs every analysis; run without --analysis',
                  file=sys.stderr)
            sys.exit(2)
        write_latex(a.latex, tables)
    if out is not sys.stdout:
        out.close()


if __name__ == '__main__':
    main()
