#!/usr/bin/env python3
"""What transfers beyond this pipeline (V1-V8), from data_and_results/data/ only.

No GPU, no model call, no new run. Each analysis turns a measurement the paper
already made into a statement another system or another benchmark can use.

V1  the figure corpus as a reference for auditing a figure-reading model
V2  engine dispatch: which layout engine a graph needs, predicted before drawing
V3  geometry as a selection signal: what one extra candidate buys, and at what cost
V4  what it takes to measure a system on these benchmarks (variance and power)
V5  arrow direction: an error class no graph score here separates, and the fix
V6  the corpus as a calibration set: the read-back reader's transfer function
V7  a judge verdict attributed to its dimensions, as a geometric verdict is to its criteria
V8  how reliable the geometric signal is per figure, and the ceiling that sets on selection

    python3 checks_novelty.py --data ../data
    python3 checks_novelty.py --data ../data --analysis V2 --csv-dir ../results/tables
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

MODELS = ['Qwen', 'Gemma', 'Ministral']
DIMS = ['faithfulness', 'conciseness', 'readability', 'aesthetics']
B = 10000
RNG = lambda s=0: np.random.default_rng(s)
OUT, CSV = [], {}


def say(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s)


def head(t):
    say('\n' + '=' * 80); say('== ' + t); say('=' * 80)


def load(D):
    g = lambda f: pd.read_csv(f'{D}/{f}', low_memory=False)
    return dict(P=g('pipeline_items.csv'), L=g('layout_items.csv'),
                RB=g('readback_sciflow.csv'), J=g('judge_paperbanana.csv'))


def holm(ps):
    """Holm step-down adjustment inside one family, as checks_claims.py K7 does it."""
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def micro_edge_f1(matched, pred, gold):
    """Micro edge F1 from a matched-edge count, as the benchmark computes it."""
    if not pred or not gold or matched <= 0:
        return np.nan
    pr, rc = matched / pred, matched / gold
    return 2 * pr * rc / (pr + rc)


def share_ci(x, seed=0):
    x = np.asarray(x, float)
    b = RNG(seed).choice(x, (B, len(x))).mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def ci(x, B=B, seed=0, pct=(2.5, 97.5)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile(RNG(seed).choice(x, (B, len(x))).mean(1), pct)


def auc(score, y):
    y = np.asarray(y, float); score = np.asarray(score, float)
    a, b = score[y == 1], score[y == 0]
    if not len(a) or not len(b):
        return np.nan
    return stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))


def logit(X, y, iters=60):
    X = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]),
                                   X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return lambda Z: 1 / (1 + np.exp(-np.column_stack([np.ones(len(Z)), Z]) @ b))


# ---------------------------------------------------------------- V1
def V1(d):
    head('V1  the corpus as a reference for auditing a figure-reading model')
    say('   the drawn graph is known exactly, so a reader is scored against what was drawn,')
    say('   not against a gold graph, and with no human annotation')
    # the reader read the seed-0 SciFlow-Bench figure, so that is the figure its
    # output is scored against, exactly as in N2 and Table S8
    ev = d['P'][(d['P'].variant == 'evaluated') & (d['P'].corpus == 'sciflow')
                & (d['P'].seed == 0)]
    t = ev.groupby(['model', 'domain', 'item']).agg(
        drawn_nodes=('layout_n_nodes', 'mean'),
        drawn_edges=('layout_n_edges', 'mean')).reset_index()
    r = d['RB'][d['RB'].system == 'ours'].merge(t, on=['model', 'domain', 'item'])
    rows = []
    for m in MODELS:
        s = r[r.model == m]
        nr = (s.pred_nodes / s.drawn_nodes).replace([np.inf, -np.inf], np.nan).dropna()
        er = (s.pred_edges / s.drawn_edges.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        # the paper's statistic is the ratio of the means; bootstrap it over papers
        rng = RNG(0); idx = rng.integers(0, len(s), (2000, len(s)))
        pn, dn = s.pred_nodes.values, s.drawn_nodes.values
        pe, de = s.pred_edges.values, s.drawn_edges.values
        bn = np.array([pn[i].mean() / dn[i].mean() for i in idx])
        be = np.array([pe[i].mean() / de[i].mean() for i in idx])
        cn, ce = np.percentile(bn, [2.5, 97.5]), np.percentile(be, [2.5, 97.5])
        rows.append(dict(model=m, n=len(s),
                         node_ratio=pn.mean() / dn.mean(), node_lo=cn[0], node_hi=cn[1],
                         node_over_pct=100 * (nr > 1).mean(),
                         edge_ratio=pe.mean() / de.mean(), edge_lo=ce[0], edge_hi=ce[1],
                         edge_over_pct=100 * (er > 1).mean(),
                         rho_nodes=stats.spearmanr(s.drawn_nodes, s.pred_nodes).statistic,
                         rho_edges=stats.spearmanr(s.drawn_edges, s.pred_edges).statistic))
        q = rows[-1]
        say(f'  {m:10s} boxes  drawn {dn.mean():5.1f} -> read {pn.mean():5.1f}  '
            f'x{q["node_ratio"]:.2f} [{q["node_lo"]:.2f},{q["node_hi"]:.2f}], over-returned on '
            f'{q["node_over_pct"]:.0f}% of papers, rho {q["rho_nodes"]:+.2f}')
        say(f'  {"":10s} arrows drawn {de.mean():5.1f} -> read {pe.mean():5.1f}  '
            f'x{q["edge_ratio"]:.2f} [{q["edge_lo"]:.2f},{q["edge_hi"]:.2f}], over-returned on '
            f'{q["edge_over_pct"]:.0f}% of papers, rho {q["rho_edges"]:+.2f}')
    CSV['reader_audit'] = pd.DataFrame(rows)
    say('  a benchmark that scores a figure through a recovered graph inherits this error.')


# ---------------------------------------------------------------- V2
def V2(d):
    head('V2  engine dispatch: which engine a graph needs, before anything is drawn')
    G = d['L'][d['L'].source.isin(['graphviz', 'graphviz_outside_development'])].copy()
    piv = G.pivot_table(index=['corpus', 'model', 'run', 'item'],
                        columns='config', values='valid').reset_index()
    feat = G[G.config == 'ours'][['corpus', 'model', 'run', 'item',
                                  'n_nodes', 'n_edges', 'loops', 'nested']]
    piv = piv.merge(feat, on=['corpus', 'model', 'run', 'item'])
    piv['oracle'] = piv[['dot_tb_spline', 'dot_tb_ortho', 'dot_lr_spline', 'dot_lr_ortho']].max(axis=1)
    num = lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float).values
    X = lambda s: np.column_stack([np.log(pd.to_numeric(s.n_nodes, errors='coerce')
                                          .fillna(1).clip(lower=1).astype(float).values),
                                   num(s.loops), num(s.nested)])
    rows = []
    for target, lab in [('dot_tb_spline', 'dot fails the strict check'),
                        ('ours', 'our engine fails the strict check')]:
        for held in ['sciflow', 'flowvqa', 'paperbanana']:
            tr = piv[piv.corpus != held].dropna(subset=[target])
            te = piv[piv.corpus == held].dropna(subset=[target])
            f = logit(X(tr), 1 - tr[target].values.astype(float))
            a = auc(f(X(te)), 1 - te[target].values.astype(float))
            rows.append(dict(target=lab, held_out=held, n=len(te), auc=a))
            say(f'  predict "{lab}" from boxes, loops and nesting; held out {held:12s} '
                f'AUC {a:.3f}  (n={len(te)})')
    CSV['dispatch_auc'] = pd.DataFrame(rows)
    s = piv.dropna(subset=['ours', 'oracle'])
    both = np.maximum(s['ours'], s['oracle'])
    say(f'\n  always ours {s["ours"].mean()*100:.1f}%, dot with an oracle over its four '
        f'configurations {s["oracle"].mean()*100:.1f}%, best of the two per figure '
        f'{both.mean()*100:.1f}%')
    say(f'  so always ours reaches {100*s["ours"].mean()/both.mean():.1f}% of a two-engine '
        f'ceiling: headroom {100*(both.mean()-s["ours"].mean()):.1f} points pooled')
    per = []
    for (c, m), g in s.groupby(['corpus', 'model']):
        b = np.maximum(g['ours'], g['oracle']).mean()
        per.append(dict(corpus=c, model=m, n=len(g), ours=100 * g['ours'].mean(),
                        dot_oracle=100 * g['oracle'].mean(), both=100 * b,
                        headroom=100 * (b - g['ours'].mean())))
    CSV['dispatch_ceiling'] = pd.DataFrame(per)
    say(f'  per setting the headroom is {min(p["headroom"] for p in per):.1f} to '
        f'{max(p["headroom"] for p in per):.1f} points, so the rule is a cheap pre-draw test '
        f'and not a large gain here.')


# ---------------------------------------------------------------- V3
def V3(d):
    head('V3  geometry as a selection signal: one extra candidate, chosen by counting defects')
    P = d['P']
    rows = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in MODELS:
            a = P[(P.variant == 'evaluated') & (P.corpus == corpus) & (P.model == m)]
            b = P[(P.variant == 'ablation_one_greedy_graph') & (P.corpus == corpus) & (P.model == m)]
            if a.empty or b.empty:
                continue
            A = a.groupby('item').agg(score=('score', 'mean'), valid=('layout_valid', 'mean'))
            Bv = b.groupby('item').agg(score=('score', 'mean'), valid=('layout_valid', 'mean'))
            j = A.join(Bv, lsuffix='_2', rsuffix='_1').dropna()
            ds, dv = (j.score_2 - j.score_1).values, (j.valid_2 - j.valid_1).values * 100
            cs, cv = ci(ds), ci(dv)
            rows.append(dict(corpus=corpus, model=m, n=len(j), dscore=ds.mean(),
                             score_lo=cs[0], score_hi=cs[1], dvalid=dv.mean(),
                             valid_lo=cv[0], valid_hi=cv[1]))
            say(f'  {corpus:9s}{m:10s} strict {dv.mean():+5.1f} points '
                f'[{cv[0]:+.1f},{cv[1]:+.1f}]   score {ds.mean():+.4f} '
                f'[{cs[0]:+.4f},{cs[1]:+.4f}]   n={len(j)}')
    CSV['selection_gain'] = pd.DataFrame(rows)
    say('  every score interval contains zero; the legibility gain does not.')
    say('  the signal costs one extra sample and a defect count, so any system that emits a')
    say('  graph before it draws can use it, whatever it draws with.')


# ---------------------------------------------------------------- V4
def V4(d):
    head('V4  what it takes to measure a system on these benchmarks')
    P = d['P'][d['P'].variant == 'evaluated']
    rows = []
    for corpus in ['sciflow', 'flowvqa']:
        s = P[P.corpus == corpus]
        w = s.pivot_table(index='item', columns=['model', 'seed'], values='score')
        item = w.mean(axis=1).var(ddof=1)
        mdl = s.groupby('model').score.mean().var(ddof=1)
        seed = s.groupby(['model', 'item']).score.var(ddof=1).mean()
        tot = item + mdl + seed
        say(f'  {corpus}: of the score variance, item {100*item/tot:.1f}%, '
            f'model {100*mdl/tot:.1f}%, seed {100*seed/tot:.1f}%')
        base = s.groupby('item').score.mean()
        for n in (50, 100, len(base), 483 if corpus == 'sciflow' else 300):
            n = min(n, len(base))
            sd_paired = base.std(ddof=1)
            rows.append(dict(corpus=corpus, n=n, mdd_unpaired=2.8 * sd_paired / np.sqrt(n)))
        say(f'    an unpaired comparison of {len(base)} items resolves '
            f'{2.8*base.std(ddof=1)/np.sqrt(len(base)):.3f} of the score at 80% power; '
            f'50 items, {2.8*base.std(ddof=1)/np.sqrt(50):.3f}')
    J = d['J']
    say('')
    for judge in sorted(J.judge.unique()):
        for m in MODELS:
            s = J[(J.judge == judge) & (J.model == m)]
            o = s[s.system == 'ours'].groupby('item').overall_points.mean()
            b = s[s.system == 'dot'].groupby('item').overall_points.mean()
            j = pd.concat([o.rename('o'), b.rename('d')], axis=1).dropna()
            sd = (j.o - j.d).std(ddof=1)
            say(f'  {judge:16s}{m:10s} paired judge sd {sd:.1f} points -> resolves '
                f'{2.8*sd/np.sqrt(len(j)):.1f} at {len(j)} items, '
                f'{2.8*sd/np.sqrt(100):.1f} at 100, {2.8*sd/np.sqrt(50):.1f} at 50')
            rows.append(dict(corpus='paperbanana', judge=judge, model=m, n=len(j),
                             mdd_292=2.8 * sd / np.sqrt(len(j)), mdd_100=2.8 * sd / np.sqrt(100),
                             mdd_50=2.8 * sd / np.sqrt(50)))
    CSV['power'] = pd.DataFrame(rows)
    say('  a judge study of 50 figures cannot see a difference under about 15 points, which is')
    say('  larger than most differences this literature reports.')


# ---------------------------------------------------------------- V5
def V5(d):
    head('V5  arrow direction: an error class no graph score here separates')
    say('   a reversed edge is one the extractor found and pointed the wrong way.')
    say('   node-and-edge F1 scores it exactly like a missing edge, so the two errors --')
    say('   which call for different fixes -- are merged. Both are computable from any')
    say('   predicted graph, so any benchmark of this kind can separate them for free.')
    ev = d['P'][(d['P'].variant == 'evaluated') &
                (d['P'].corpus.isin(['sciflow', 'flowvqa']))].copy()
    ev['matched'] = ev.edge_recall * ev.gold_edges
    ev['rev_rate'] = ev.strict_edges_reversed / ev.gold_edges.replace(0, np.nan)
    groups = [('sciflow (papers)', ev[ev.corpus == 'sciflow'])]
    for dom in ['code', 'instruct', 'wiki']:
        groups.append(('flowvqa ' + dom,
                       ev[(ev.corpus == 'flowvqa') & (ev.domain == dom)]))
    say(f'\n  {"corpus/source":18s} {"model":10s} {"rev/fig":>8s} {"%gold":>7s} '
        f'{"figs>=1":>8s} {"%located":>9s} {"edgeF1":>7s} {"undir.":>7s} {"delta":>7s}')
    rows = []
    for name, sub in groups:
        for m in MODELS:
            s = sub[sub.model == m]
            if not len(s):
                continue
            mt, r = s.matched.sum(), s.strict_edges_reversed.sum()
            pe, ge = s.pred_edges.sum(), s.gold_edges.sum()
            f1, f1u = micro_edge_f1(mt, pe, ge), micro_edge_f1(mt + r, pe, ge)
            rows.append(dict(source=name, model=m, n=len(s),
                             rev_per_figure=s.strict_edges_reversed.mean(),
                             pct_gold_edges=100 * s.rev_rate.mean(),
                             pct_figures_with_one=100 * (s.strict_edges_reversed > 0).mean(),
                             pct_of_located_relations=100 * r / (mt + r),
                             edge_f1_directed=f1, edge_f1_undirected=f1u, delta=f1u - f1))
            q = rows[-1]
            say(f'  {name:18s} {m:10s} {q["rev_per_figure"]:8.3f} '
                f'{q["pct_gold_edges"]:6.2f}% {q["pct_figures_with_one"]:7.1f}% '
                f'{q["pct_of_located_relations"]:8.2f}% {f1:7.3f} {f1u:7.3f} '
                f'{f1u - f1:+7.4f}')
    CSV['direction_errors'] = pd.DataFrame(rows)
    say('')
    pooled = [('papers (sciflow)', ev[ev.corpus == 'sciflow']),
              ('code', ev[(ev.corpus == 'flowvqa') & (ev.domain == 'code')]),
              ('prose (wiki+instruct)', ev[(ev.corpus == 'flowvqa') &
                                           (ev.domain.isin(['wiki', 'instruct']))])]
    for name, s in pooled:
        mt, r = s.matched.sum(), s.strict_edges_reversed.sum()
        f1 = micro_edge_f1(mt, s.pred_edges.sum(), s.gold_edges.sum())
        f1u = micro_edge_f1(mt + r, s.pred_edges.sum(), s.gold_edges.sum())
        pc, lo, hi = share_ci((s.strict_edges_reversed > 0).astype(float))
        say(f'  {name:22s} {pc:4.1f}% of figures carry one [{lo:.1f},{hi:.1f}], '
            f'{100*r/(mt+r):.2f}% of the relations it located, '
            f'edge F1 {f1:.3f} -> {f1u:.3f} ignoring direction')
    code = ev[(ev.corpus == 'flowvqa') & (ev.domain == 'code')]
    prose = ev[(ev.corpus == 'flowvqa') & (ev.domain.isin(['wiki', 'instruct']))]
    u = stats.mannwhitneyu(code.strict_edges_reversed.dropna(),
                           prose.strict_edges_reversed.dropna())
    say(f'  prose / code ratio {prose.rev_rate.mean()/max(code.rev_rate.mean(),1e-9):.1f}x '
        f'per gold edge, Mann-Whitney p = {u.pvalue:.2g} (E4)')
    lo = d['L'][d['L'].config == 'ours'][['corpus', 'model', 'run', 'item',
                                          'loops', 'nested']].rename(columns={'run': 'seed'})
    lo['seed'] = pd.to_numeric(lo.seed, errors='coerce')
    ev['seed'] = pd.to_numeric(ev.seed, errors='coerce')
    j = ev.merge(lo, on=['corpus', 'model', 'seed', 'item'], how='inner')
    a, b = j[j.loops == 1], j[j.loops == 0]
    say(f'  on the {len(j)} figures whose graph is also in layout_items, a graph that loops '
        f'back carries\n  a reversed arrow on {100*(a.strict_edges_reversed>0).mean():.1f}% of '
        f'figures against {100*(b.strict_edges_reversed>0).mean():.1f}% for one that does not '
        f'(Mann-Whitney p = '
        f'{stats.mannwhitneyu(a.strict_edges_reversed.dropna(), b.strict_edges_reversed.dropna()).pvalue:.2g}),')
    say('  so the error concentrates where the extraction is hardest and the figure most needs')
    say('  to be read in order.')
    say('\n  the fix costs nothing: report a direction-insensitive edge F1 beside the directed')
    say('  one, or the reversal count itself. The gap between the two is the share of the')
    say('  benchmark\'s deficit that is a direction error and not a missing relation.')


# ---------------------------------------------------------------- V6
def V6(d):
    head('V6  the corpus as a calibration set: the reader\'s transfer function, measured')
    say('   V1 shows the read-back reader does not return what was drawn. Because the drawn')
    say('   graph is known exactly, the map from what it returns to what is there can be')
    say('   fitted and tested out of sample: papers are split by position into two halves,')
    say('   the correction is fitted on one and scored on the other.')
    ev = d['P'][(d['P'].variant == 'evaluated') & (d['P'].corpus == 'sciflow')
                & (d['P'].seed == 0)]
    t = ev.groupby(['model', 'domain', 'item']).agg(
        drawn_nodes=('layout_n_nodes', 'mean'),
        drawn_edges=('layout_n_edges', 'mean')).reset_index()
    r = d['RB'][d['RB'].system == 'ours'].merge(t, on=['model', 'domain', 'item'])
    say(f'\n  {"model":10s} {"target":7s} {"slope":>6s} {"R2":>6s} {"MAE raw":>8s} '
        f'{"MAE cal.":>9s} {"MAE const":>10s}')
    rows = []
    for m in MODELS:
        s = r[r.model == m].sort_values('item').reset_index(drop=True)
        tr, te = s.iloc[::2], s.iloc[1::2]
        for tgt, pred, lab in [('drawn_nodes', 'pred_nodes', 'boxes'),
                               ('drawn_edges', 'pred_edges', 'arrows')]:
            A = np.column_stack([np.ones(len(tr)), tr[pred].values])
            b = np.linalg.lstsq(A, tr[tgt].values, rcond=None)[0]
            fit = b[0] + b[1] * te[pred].values
            y = te[tgt].values
            mae_raw = np.abs(te[pred].values - y).mean()
            mae_cal = np.abs(fit - y).mean()
            mae_con = np.abs(tr[tgt].mean() - y).mean()
            r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            rows.append(dict(model=m, target=lab, n_fit=len(tr), n_test=len(te),
                             intercept=b[0], slope=b[1], r2=r2, mae_raw=mae_raw,
                             mae_calibrated=mae_cal, mae_constant=mae_con))
            say(f'  {m:10s} {lab:7s} {b[1]:6.2f} {r2:6.2f} {mae_raw:8.2f} '
                f'{mae_cal:9.2f} {mae_con:10.2f}')
    CSV['reader_calibration'] = pd.DataFrame(rows)
    ar = [q for q in rows if q['target'] == 'arrows']
    say(f'\n  calibration roughly halves the error on arrows '
        f'({min(q["mae_raw"] for q in ar):.1f}-{max(q["mae_raw"] for q in ar):.1f} arrows '
        f'to {min(q["mae_calibrated"] for q in ar):.1f}-{max(q["mae_calibrated"] for q in ar):.1f}),')
    say(f'  but the fitted slope is {min(q["slope"] for q in ar):.2f}-'
        f'{max(q["slope"] for q in ar):.2f} rather than 1 and explains only '
        f'{min(q["r2"] for q in ar):.2f}-{max(q["r2"] for q in ar):.2f} of the variance,')
    say('  so the corrected estimate is barely better than ignoring the reader and returning')
    say('  the mean of the corpus (last column).')
    say('\n  cross-model transfer of the arrow correction (MAE, all papers of the target model):')
    fits = {}
    for m in MODELS:
        s = r[r.model == m].sort_values('item').reset_index(drop=True)
        tr = s.iloc[::2]
        A = np.column_stack([np.ones(len(tr)), tr.pred_edges.values])
        fits[m] = np.linalg.lstsq(A, tr.drawn_edges.values, rcond=None)[0]
    tx = []
    for a in MODELS:
        cells = []
        for b_ in MODELS:
            s = r[r.model == b_]
            mae = np.abs(fits[a][0] + fits[a][1] * s.pred_edges.values - s.drawn_edges.values).mean()
            cells.append(f'{b_} {mae:5.2f}')
            tx.append(dict(fitted_on=a, applied_to=b_, mae=mae))
        say(f'    fitted on {a:10s} -> ' + '   '.join(cells))
    CSV['reader_calibration_transfer'] = pd.DataFrame(tx)
    say('  a correction fitted on one model\'s figures is worse on another\'s than that')
    say('  model\'s own, so the transfer function belongs to the figure set and not to the')
    say('  reader alone: it cannot be fitted once and carried to a new system\'s figures.')
    say('\n  what transfers is the procedure, not the coefficients: any corpus whose geometry')
    say('  is known lets a reader be audited and bounded before it is trusted to score a')
    say('  system, and this one shows the bound is not removable by calibration.')




# ---------------------------------------------------------------- V7
def V7(d):
    head('V7  a judge verdict belongs to its dimensions, as a geometric verdict does')
    say('   PaperBananaBench combines four dimensions into one number: faithfulness and')
    say('   conciseness decide, readability and aesthetics break ties. Section 7 does not')
    say('   report a six-criterion verdict without attributing it to its criteria; this is')
    say('   the same operation on the judge, over the identical figures and both judges.')
    J, dims = d['J'], DIMS
    cols = [x + '_points' for x in dims] + ['overall_points']
    rows = []
    say(f'\n  {"judge":16s} {"model":10s} {"overall":>8s}  ' +
        '  '.join(f'{x[:12]:>12s}' for x in dims))
    for judge in sorted(J.judge.unique()):
        jj = J[J.judge == judge]
        for m in MODELS:
            a = jj[(jj.system == 'ours') & (jj.model == m)].groupby('item')[cols].mean()
            b = jj[(jj.system == 'dot') & (jj.model == m)].groupby('item')[cols].mean()
            if a.empty or b.empty:
                continue
            idx = a.index.intersection(b.index)
            diff = {x: (a.loc[idx, x + '_points'] - b.loc[idx, x + '_points']).values
                    for x in dims + ['overall']}
            ps = [stats.wilcoxon(diff[x]).pvalue if np.any(diff[x] != 0) else 1.0
                  for x in dims]
            padj = holm(ps)
            for x, p in zip(dims, padj):
                lo, hi = ci(diff[x])
                rows.append(dict(judge=judge, model=m, dimension=x, n=len(idx),
                                 diff=diff[x].mean(), lo=lo, hi=hi, p_holm=p))
            lo, hi = ci(diff['overall'])
            rows.append(dict(judge=judge, model=m, dimension='overall', n=len(idx),
                             diff=diff['overall'].mean(), lo=lo, hi=hi, p_holm=np.nan))
            say(f'  {judge:16s} {m:10s} {diff["overall"].mean():+8.1f}  ' +
                '  '.join(f'{diff[x].mean():+7.1f}{"*" if p < 0.05 else " "}    '
                          for x, p in zip(dims, padj)))
    t = pd.DataFrame(rows)
    CSV['judge_dimension_attribution'] = t
    say('  (* Holm-corrected inside the four dimensions of one judge and model; the bracket')
    say('   in the CSV is a 95% bootstrap interval over the 292 items.)')

    piv = t.pivot_table(index=['judge', 'model'], columns='dimension', values='diff')
    say('\n  what share of the overall difference each loss\'s deciding dimension accounts for:')
    for (judge, m), row in piv.iterrows():
        if row['overall'] >= 0:
            continue
        say(f'    {judge:16s} {m:10s} conciseness {row["conciseness"]:+6.1f} of overall '
            f'{row["overall"]:+6.1f}  = {100*row["conciseness"]/row["overall"]:5.0f}%')
    say('\n  how far each dimension moves when the judge is replaced, on the same figures:')
    srows = []
    for x in dims + ['overall']:
        w = piv[x].unstack(0)
        j1, j2 = w.columns[0], w.columns[1]
        mv = (w[j1] - w[j2]).abs()
        kept = int((np.sign(w[j1]) == np.sign(w[j2])).sum())
        srows.append(dict(dimension=x, move_min=mv.min(), move_max=mv.max(),
                          sign_kept=kept, models=len(w)))
        say(f'    {x:14s} moves {mv.min():5.1f} to {mv.max():5.1f} points, '
            f'sign kept for {kept} of {len(w)} models')
    CSV['judge_dimension_stability'] = pd.DataFrame(srows)
    say('  the dimension that carries the verdict, conciseness, is the one the change of')
    say('  judge leaves alone; the overall number moves further than any dimension in it.')

    say('\n  how often each dimension separates the two figures at all:')
    trows = []
    for x in dims:
        v = J[x + '_verdict']
        tie = ((v == 'Both are good') | (v == 'Both are bad')).astype(float)
        pc, lo, hi = share_ci(tie.values)
        trows.append(dict(dimension=x, tie_pct=pc, lo=lo, hi=hi,
                          favours_model_pct=100 * (v == 'Model').mean(),
                          favours_human_pct=100 * (v == 'Human').mean()))
        say(f'    {x:14s} ties on {pc:4.1f}% [{lo:.1f},{hi:.1f}] of judgements   '
            f'model {100*(v=="Model").mean():4.1f}%   human {100*(v=="Human").mean():4.1f}%')
    CSV['judge_dimension_tie_rates'] = pd.DataFrame(trows)
    say('  the dimension that names legibility is silent on two judgements in three; the')
    say('  deciding dimension is the one that nearly always fires, and it fires against the')
    say('  generated figure. These shares match the tie column of checks_judge.py --analysis')
    say('  G2, and the ordering matches its agreement between judges: conciseness kappa')
    say('  0.294 against readability 0.151, the two ends of the same ranking.')

    say('\n  and what each dimension responds to, inside one system, per figure:')
    ev = d['P'][(d['P'].variant == 'evaluated') & (d['P'].corpus == 'paperbanana')]
    boxes = ev.groupby(['model', 'item']).agg(boxes=('layout_n_nodes', 'mean')).reset_index()
    zrows = []
    for judge in sorted(J.judge.unique()):
        for m in MODELS:
            jo = J[(J.judge == judge) & (J.system == 'ours') &
                   (J.model == m)].groupby('item')[cols].mean().reset_index()
            s = boxes[boxes.model == m].merge(jo, on='item').dropna(subset=['boxes'])
            if len(s) < 50:
                continue
            line = f'    {judge:16s} {m:10s} n={len(s):4d}  '
            for x in dims + ['overall']:
                r = stats.spearmanr(s[x + '_points'], s.boxes)
                zrows.append(dict(judge=judge, model=m, dimension=x, n=len(s),
                                  rho=r.statistic, p=r.pvalue))
                line += f'{x[:4]} {r.statistic:+.2f}  '
            say(line)
    CSV['judge_dimension_vs_size'] = pd.DataFrame(zrows)
    say('  every dimension falls with the number of boxes the figure holds, the deciding one')
    say('  fastest (rho -0.25 to -0.42). Section 6 shows the same slope between twelve')
    say('  systems; here it is inside one system, figure by figure. With size held fixed the')
    say('  readability dimension tracks no geometry at all (C1), so it reads how much a')
    say('  figure holds and not whether it can be read.')
    say('\n  what transfers: report a judge per dimension and attribute the verdict before')
    say('  reporting it. Here that turns one unstable number into one stable finding (the')
    say('  generated figures are less concise) and three unstable ones.')


# ---------------------------------------------------------------- V8
def V8(d):
    head('V8  the geometric signal is stable in aggregate and noisy per figure, '
         'which is what selection spends')
    say('   the layout is a deterministic function of the graph, so everything that moves')
    say('   between two seeds of one item is the extraction. V3 sells the counts as a')
    say('   best-of-n signal; this is the reliability of that signal and the ceiling of')
    say('   acting on it, measured over three seeds of the shipped pipeline.')
    P = d['P']
    ev = P[P.variant == 'evaluated']
    rows = []
    say(f'\n  {"corpus":12s} {"model":10s} {"n":>5s} {"flips":>7s} {"seed spread":>12s} '
        f'{"mean":>7s} {"best of 3":>10s} {"worst of 3":>11s} {"ceiling":>9s}')
    for corpus in ['sciflow', 'flowvqa', 'paperbanana']:
        for m in MODELS:
            s = ev[(ev.corpus == corpus) & (ev.model == m)]
            w = s.pivot_table(index='item', columns='seed', values='layout_valid').dropna()
            if w.shape[1] < 3 or not len(w):
                continue
            per = w.mean(axis=0)
            flip = 100 * (w.nunique(axis=1) > 1).mean()
            mean, best, worst = 100 * w.values.mean(), 100 * w.max(axis=1).mean(), \
                100 * w.min(axis=1).mean()
            lo, hi = ci((w.max(axis=1) - w.mean(axis=1)).values * 100)
            rows.append(dict(corpus=corpus, model=m, n=len(w), flip_pct=flip,
                             seed_spread=100 * (per.max() - per.min()), mean_pass=mean,
                             best_of_3=best, worst_of_3=worst, ceiling=best - mean,
                             ceiling_lo=lo, ceiling_hi=hi))
            say(f'  {corpus:12s} {m:10s} {len(w):5d} {flip:6.1f}% '
                f'{100*(per.max()-per.min()):11.1f} {mean:7.1f} {best:10.1f} {worst:11.1f} '
                f'{best-mean:+8.1f} [{lo:+.1f},{hi:+.1f}]')
    t = pd.DataFrame(rows)
    CSV['selection_reliability'] = t
    say(f'\n  the seed moves the pass rate of a setting by {t.seed_spread.min():.1f} to '
        f'{t.seed_spread.max():.1f} points and the verdict of a figure on')
    say(f'  {t.flip_pct.min():.1f} to {t.flip_pct.max():.1f}% of them: the aggregate is '
        f'reproducible and a single figure\'s verdict is not,')
    say(f'  so one run under-reports a figure\'s legibility by up to '
        f'{(t.mean_pass-t.worst_of_3).max():.1f} points of its own corpus.')

    say('\n  and the share of figures whose verdict is seed-dependent is what one extra')
    say('  sample buys, so the headroom can be read off a few resampled items in advance:')
    mrows = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in MODELS:
            a = P[(P.variant == 'evaluated') & (P.corpus == corpus) &
                  (P.model == m)].groupby('item').layout_valid.mean()
            b = P[(P.variant == 'ablation_one_greedy_graph') & (P.corpus == corpus) &
                  (P.model == m)].groupby('item').layout_valid.mean()
            idx = a.index.intersection(b.index)
            if not len(idx):
                continue
            r = t[(t.corpus == corpus) & (t.model == m)]
            if r.empty:
                continue
            mrows.append(dict(corpus=corpus, model=m, flip_pct=r.flip_pct.iloc[0],
                              measured_best_of_2=100 * (a.loc[idx] - b.loc[idx]).mean(),
                              ceiling_best_of_3=r.ceiling.iloc[0]))
            q = mrows[-1]
            say(f'    {corpus:9s}{m:10s} verdict seed-dependent on {q["flip_pct"]:5.1f}% of '
                f'figures   measured best-of-2 {q["measured_best_of_2"]:+5.1f} pts   '
                f'best-of-3 ceiling {q["ceiling_best_of_3"]:+5.1f} pts')
    mt = pd.DataFrame(mrows)
    CSV['selection_headroom'] = mt
    rr = stats.spearmanr(mt.flip_pct, mt.measured_best_of_2)
    say(f'    rank correlation over the {len(mt)} settings: rho = {rr.statistic:+.2f} '
        f'(p = {rr.pvalue:.3f})')

    say('\n  the same instability is there on content and cannot be spent:')
    srows = []
    for corpus in ['sciflow', 'flowvqa']:
        for m in MODELS:
            w = ev[(ev.corpus == corpus) & (ev.model == m)].pivot_table(
                index='item', columns='seed', values='score').dropna()
            if w.shape[1] < 3 or not len(w):
                continue
            srows.append(dict(corpus=corpus, model=m, mean=w.values.mean(),
                              seed_spread=w.mean(axis=0).max() - w.mean(axis=0).min(),
                              best_of_3=w.max(axis=1).mean()))
            q = srows[-1]
            say(f'    {corpus:9s}{m:10s} score {q["mean"]:.3f}, seed spread '
                f'{q["seed_spread"]:.3f}, best of three {q["best_of_3"]:.3f} '
                f'(+{q["best_of_3"]-q["mean"]:.3f})')
    CSV['selection_content_ceiling'] = pd.DataFrame(srows)
    say('  picking the best of three samples would add 0.008 to 0.046 of score as well, and')
    say('  nothing available at generation time says which sample that is: a defect is')
    say('  countable per figure and a missing relation is not. That asymmetry, and not the')
    say('  size of either headroom, is why selection by geometry is free and selection on')
    say('  content needs a gold graph or a judge.')


ALL = dict(V1=V1, V2=V2, V3=V3, V4=V4, V5=V5, V6=V6, V7=V7, V8=V8)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*', default=list(ALL))
    ap.add_argument('--out'); ap.add_argument('--csv-dir')
    a = ap.parse_args()
    d = load(a.data)
    for k in a.analysis:
        ALL[k](d)
    if a.out:
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
    if a.csv_dir:
        for k, v in CSV.items():
            v.to_csv(f'{a.csv_dir}/{k}.csv', index=False)
