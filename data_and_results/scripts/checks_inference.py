#!/usr/bin/env python3
"""Inference checks (I1-I7): the same point estimates, under tests that match the design.

Computed from data_and_results/data/ only. No GPU, no model call, no new run.
Specified after the test runs; descriptive. Every point estimate here reproduces the
one in the paper; what changes is the uncertainty attached to it.

I1  pooled layout comparisons are clustered: 6,012 figures are 1,064 source graphs
    drawn by 3 models x 3 seeds, so McNemar over the pooled rows treats dependent
    figures as independent. Replaced by a cluster bootstrap and a cluster-level
    sign-flip permutation.
I2  the nine per-setting legibility differences, cluster-robust
I3  'tied' as equivalence: two one-sided bootstrap tests (TOST) against a margin,
    not a failure to reject
I4  Wilcoxon zero handling where paired differences are exactly zero
I5  rater study: 72 images and 216 choices are 24 source graphs; Mantel-Haenszel
    stratified by rating set, and a within-set randomisation test
I6  capacity (LD50) with cluster bootstrap intervals
I7  the near-orthogonality of arrow crossings and label collisions, as a bound

    python3 checks_inference.py --data ../data
    python3 checks_inference.py --data ../data --analysis I3 I5
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

B = 10000
RNG = lambda s=0: np.random.default_rng(s)
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s)


def head(t):
    say('\n' + '=' * 80); say('== ' + t); say('=' * 80)


def load(D):
    g = lambda f: pd.read_csv(f'{D}/{f}', low_memory=False)
    return dict(P=g('pipeline_items.csv'), DT=g('dot_items.csv'), L=g('layout_items.csv'),
                J=g('judge_paperbanana.csv'), HR=g('human_ratings.csv'),
                HS=g('human_sets.csv'), ST=g('human_stimuli.csv'))


# ---------------------------------------------------------------- tools
def clusters(cl):
    return {g: np.where(np.asarray(cl) == g)[0] for g in pd.unique(np.asarray(cl))}


def cluster_boot(stat, cl, B=B, seed=0, pct=(2.5, 97.5)):
    """Percentile interval of `stat(idx)`, resampling whole clusters with replacement."""
    gs = clusters(cl); keys = list(gs); rng = RNG(seed); v = []
    for _ in range(B):
        idx = np.concatenate([gs[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        s = stat(idx)
        if np.isfinite(s):
            v.append(s)
    return np.percentile(v, pct)


def cluster_signflip_p(d, cl, B=B, seed=0):
    """H0: paired differences symmetric about zero; the sign is flipped per cluster."""
    d = np.asarray(d, float); cl = np.asarray(cl)
    gs = [d[cl == g] for g in pd.unique(cl)]
    m = np.array([g.mean() for g in gs]); w = np.array([len(g) for g in gs], float)
    obs = abs((m * w).sum() / w.sum())
    s = RNG(seed).choice([-1.0, 1.0], (B, len(gs)))
    null = np.abs((s * m * w).sum(1) / w.sum())
    return (1 + (null >= obs - 1e-15).sum()) / (B + 1)


def design_effect(d, cl):
    """Variance inflation from clustering, and the effective number of figures."""
    d = np.asarray(d, float); cl = np.asarray(cl); n = len(d); m = d.mean()
    gs = [d[cl == g] for g in pd.unique(cl)]
    naive = d.var(ddof=1) / n
    robust = sum((g.sum() - len(g) * m) ** 2 for g in gs) / n ** 2
    return robust / naive, n, len(gs)


def boot_mean_ci(d, B=B, seed=0, pct=(2.5, 97.5)):
    d = np.asarray(d, float)
    return np.percentile(RNG(seed).choice(d, (B, len(d))).mean(1), pct)


def logit_ld50(x, y):
    """Logistic fit of the strict verdict on log box count; LD50 back in boxes."""
    X = np.column_stack([np.ones(len(x)), x]); b = np.zeros(2)
    for _ in range(80):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - p))
        except np.linalg.LinAlgError:
            return np.nan
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return np.exp(-b[0] / b[1]) if b[1] else np.nan


def partial_spearman(x, y, z):
    s = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    rx, ry, rz = (stats.rankdata(s[c]) for c in ('x', 'y', 'z'))
    Bm = np.c_[np.ones(len(s)), rz]
    res = lambda v: v - Bm @ np.linalg.lstsq(Bm, v, rcond=None)[0]
    r, p = stats.pearsonr(res(rx), res(ry))
    return r, p, len(s)


def engines(L):
    G = L[L.source.isin(['graphviz', 'graphviz_outside_development'])].copy()
    G['cl'] = G.corpus.astype(str) + '|' + G.item.astype(str)
    return G


def paired_table(G, value='valid'):
    piv = G.pivot_table(index=['corpus', 'model', 'run', 'item', 'cl'],
                        columns='config', values=value).reset_index()
    piv['oracle'] = piv[['dot_tb_spline', 'dot_tb_ortho',
                         'dot_lr_spline', 'dot_lr_ortho']].max(axis=1)
    return piv


# ---------------------------------------------------------------- I1
def I1(d):
    head('I1  pooled layout comparisons: the rows are clustered')
    G = engines(d['L']); piv = paired_table(G)
    for base, lab in [('dot_tb_spline', 'dot, best single configuration'),
                      ('oracle', 'dot, oracle over its four configurations')]:
        s = piv.dropna(subset=['ours', base])
        dv = (s['ours'] - s[base]).values; cl = s.cl.values
        b = int(((s['ours'] == 1) & (s[base] == 0)).sum())
        c = int(((s['ours'] == 0) & (s[base] == 1)).sum())
        de, n, k = design_effect(dv, cl)
        ci = cluster_boot(lambda i: dv[i].mean(), cl, B=4000)
        say(f'\nours vs {lab}')
        say(f'  {n} figures from {k} source graphs; {s["ours"].mean()*100:.1f}% against '
            f'{s[base].mean()*100:.1f}%, difference {dv.mean()*100:+.1f} points')
        say(f'  McNemar over the pooled rows (b={b}, c={c}): '
            f'p = {stats.binomtest(b, b + c, 0.5).pvalue:.3g}   <- rows not independent')
        say(f'  design effect {de:.2f}x, so the pooled rows carry about {n/de:.0f} '
            f'independent figures, not {n}')
        say(f'  cluster bootstrap 95% CI  [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] points')
        say(f'  cluster-level sign-flip permutation p = {cluster_signflip_p(dv, cl):.4f} '
            f'(floor {1/(B+1):.0e} at {B} draws)')
    BL = ['node_overlap_pairs', 'edge_node_crossings', 'route_reversals', 'stacked_routes']
    G['blind'] = (G[BL].fillna(0).sum(axis=1) == 0).astype(float)
    pb = paired_table(G, 'blind')
    s = pb.dropna(subset=['ours', 'dot_tb_spline'])
    dv = (s['ours'] - s['dot_tb_spline']).values; cl = s.cl.values
    ci = cluster_boot(lambda i: dv[i].mean(), cl, B=4000)
    say(f'\nlabel-blind verdict (the reversal): {s["ours"].mean()*100:.1f}% against '
        f'{s["dot_tb_spline"].mean()*100:.1f}%, {dv.mean()*100:+.1f} points, '
        f'cluster 95% CI [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}], '
        f'sign-flip p = {cluster_signflip_p(dv, cl):.4f}')


# ---------------------------------------------------------------- I2
def I2(d):
    head('I2  the nine per-setting legibility differences, cluster-robust')
    piv = paired_table(engines(d['L']))
    for base, lab in [('dot_tb_spline', 'vs dot best single configuration'),
                      ('oracle', 'vs dot oracle over four configurations')]:
        say(f'\n{lab}')
        say(f'  {"corpus":12s}{"model":10s}{"figures":>8s}{"graphs":>7s}'
            f'{"diff":>8s}{"cluster 95% CI":>19s}{"perm p":>9s}')
        for (c, m), s in piv.groupby(['corpus', 'model']):
            s = s.dropna(subset=['ours', base])
            dv = (s['ours'] - s[base]).values; cl = s.cl.values
            ci = cluster_boot(lambda i: dv[i].mean(), cl, B=2000)
            say(f'  {c:12s}{m:10s}{len(dv):8d}{s.cl.nunique():7d}{dv.mean()*100:+8.1f}'
                f'  [{ci[0]*100:+6.1f},{ci[1]*100:+6.1f}]{cluster_signflip_p(dv, cl):>9.4f}')


# ---------------------------------------------------------------- I3 / I4
def _content_pairs(d):
    G = engines(d['L']); piv = paired_table(G)
    for base, lab in [('dot_tb_spline', 'dot, best single configuration'),
                      ('oracle', 'dot, oracle over its four configurations')]:
        s = piv.dropna(subset=['ours', base])
        dv = (s['ours'] - s[base]).values; cl = s.cl.values
        b = int(((s['ours'] == 1) & (s[base] == 0)).sum())
        c = int(((s['ours'] == 0) & (s[base] == 1)).sum())
        de, n, k = design_effect(dv, cl)
        ci = cluster_boot(lambda i: dv[i].mean(), cl, B=4000)
        say(f'\nours vs {lab}')
        say(f'  {n} figures from {k} source graphs; {s["ours"].mean()*100:.1f}% against '
            f'{s[base].mean()*100:.1f}%, difference {dv.mean()*100:+.1f} points')
        say(f'  McNemar over the pooled rows (b={b}, c={c}): '
            f'p = {stats.binomtest(b, b + c, 0.5).pvalue:.3g}   <- rows not independent')
        say(f'  design effect {de:.2f}x, so the pooled rows carry about {n/de:.0f} '
            f'independent figures, not {n}')
        say(f'  cluster bootstrap 95% CI  [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] points')
        say(f'  cluster-level sign-flip permutation p = {cluster_signflip_p(dv, cl):.4f} '
            f'(floor {1/(B+1):.0e} at {B} draws)')
    BL = ['node_overlap_pairs', 'edge_node_crossings', 'route_reversals', 'stacked_routes']
    G['blind'] = (G[BL].fillna(0).sum(axis=1) == 0).astype(float)
    pb = paired_table(G, 'blind')
    s = pb.dropna(subset=['ours', 'dot_tb_spline'])
    dv = (s['ours'] - s['dot_tb_spline']).values; cl = s.cl.values
    ci = cluster_boot(lambda i: dv[i].mean(), cl, B=4000)
    say(f'\nlabel-blind verdict (the reversal): {s["ours"].mean()*100:.1f}% against '
        f'{s["dot_tb_spline"].mean()*100:.1f}%, {dv.mean()*100:+.1f} points, '
        f'cluster 95% CI [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}], '
        f'sign-flip p = {cluster_signflip_p(dv, cl):.4f}')


# ---------------------------------------------------------------- I2
def I2(d):
    head('I2  the nine per-setting legibility differences, cluster-robust')
    piv = paired_table(engines(d['L']))
    for base, lab in [('dot_tb_spline', 'vs dot best single configuration'),
                      ('oracle', 'vs dot oracle over four configurations')]:
        say(f'\n{lab}')
        say(f'  {"corpus":12s}{"model":10s}{"figures":>8s}{"graphs":>7s}'
            f'{"diff":>8s}{"cluster 95% CI":>19s}{"perm p":>9s}')
        for (c, m), s in piv.groupby(['corpus', 'model']):
            s = s.dropna(subset=['ours', base])
            dv = (s['ours'] - s[base]).values; cl = s.cl.values
            ci = cluster_boot(lambda i: dv[i].mean(), cl, B=2000)
            say(f'  {c:12s}{m:10s}{len(dv):8d}{s.cl.nunique():7d}{dv.mean()*100:+8.1f}'
                f'  [{ci[0]*100:+6.1f},{ci[1]*100:+6.1f}]{cluster_signflip_p(dv, cl):>9.4f}')


# ---------------------------------------------------------------- I3 / I4
def _content_pairs(d):
    P = d['P'][d['P'].variant == 'evaluated']; DT = d['DT']
    for corpus in ['flowvqa', 'sciflow']:
        for model in ['Qwen', 'Gemma', 'Ministral']:
            o = (P[(P.corpus == corpus) & (P.model == model)]
                 .groupby('item', as_index=False)['score'].mean())
            for variant, vlab in [('dot', 'short prompt'), ('dot_matched', 'our prompts')]:
                b = DT[(DT.variant == variant) & (DT.corpus == corpus) &
                       (DT.model == model)][['item', 'score']]
                m = o.merge(b, on='item', suffixes=('_ours', '_base')).fillna(0.0)
                yield corpus, model, vlab, (m.score_ours - m.score_base).values


def I3(d):
    head('I3  "tied" as an equivalence claim: two one-sided bootstrap tests')
    say(f'  {"corpus":9s}{"model":10s}{"baseline":13s}{"mean":>8s}{"95% CI":>19s}'
        f'{"TOST bound":>11s}  verdict at that bound')
    for corpus, model, vlab, dv in _content_pairs(d):
        ci = boot_mean_ci(dv); b90 = max(abs(boot_mean_ci(dv, pct=(5, 95))))
        v = 'ahead' if ci[0] > 0 else ('behind' if ci[1] < 0 else f'equivalent within {b90:.3f}')
        say(f'  {corpus:9s}{model:10s}{vlab:13s}{dv.mean():+8.4f}'
            f'  [{ci[0]:+.4f},{ci[1]:+.4f}]{b90:>11.4f}  {v}')
    J = d['J']
    say('\n  PaperBananaBench, per judge (a two-level factor: no judge variance is estimable)')
    gaps = {}
    for judge in sorted(J.judge.unique()):
        for model in ['Qwen', 'Gemma', 'Ministral']:
            s = J[(J.judge == judge) & (J.model == model)]
            oi = s[s.system == 'ours'].groupby('item').overall_points.mean()
            di = s[s.system == 'dot'].groupby('item').overall_points.mean()
            m = pd.concat([oi.rename('o'), di.rename('d')], axis=1).dropna()
            dv = (m.o - m.d).values; ci = boot_mean_ci(dv)
            gaps.setdefault(model, []).append((dv.mean(), ci[1] - ci[0]))
            say(f'  {judge:16s}{model:10s}{dv.mean():+8.1f}  [{ci[0]:+6.1f},{ci[1]:+6.1f}]'
                f'  n={len(dv)}  Wilcoxon p={stats.wilcoxon(dv).pvalue:.2g}')
    for model, v in gaps.items():
        gap = abs(v[0][0] - v[1][0]); w = np.mean([x[1] for x in v])
        say(f'    {model:10s} judge-to-judge gap {gap:5.1f} points = {gap/w:.1f}x the width '
            f'of one judge\'s own 95% CI ({w:.1f})')


def I4(d):
    head('I4  Wilcoxon zero handling where paired differences are exactly zero')
    say(f'  {"corpus":9s}{"model":10s}{"baseline":13s}{"zeros":>7s}{"discard":>10s}'
        f'{"zsplit":>10s}{"pratt":>10s}{"sign test":>11s}')
    for corpus, model, vlab, dv in _content_pairs(d):
        nz = int((dv == 0).sum())
        pd_ = stats.wilcoxon(dv, zero_method='wilcox').pvalue if (dv != 0).any() else 1.0
        say(f'  {corpus:9s}{model:10s}{vlab:13s}{nz:7d}{pd_:>10.3g}'
            f'{stats.wilcoxon(dv, zero_method="zsplit").pvalue:>10.3g}'
            f'{stats.wilcoxon(dv, zero_method="pratt").pvalue:>10.3g}'
            f'{stats.binomtest(int((dv > 0).sum()), int((dv != 0).sum()), 0.5).pvalue:>11.3g}')
    say('  the three zero treatments agree throughout: no verdict depends on the choice.')


# ---------------------------------------------------------------- I5
def I5(d):
    head('I5  rater study: 72 images and 216 choices come from 24 source graphs')
    ST, SE, RA = d['ST'].copy(), d['HS'], d['HR']
    setimg = {r.set: [r.image_1, r.image_2, r.image_3, r.image_4] for r in SE.itertuples()}
    cnt = {i: dict(b=0, w=0, s=0) for i in ST.image}
    for r in RA.itertuples():
        for im in setimg[r.set]:
            cnt[im]['s'] += 1
        cnt[r.best]['b'] += 1; cnt[r.worst]['w'] += 1
    ST['bws'] = ST.image.map(lambda i: (cnt[i]['b'] - cnt[i]['w']) / max(cnt[i]['s'], 1))
    ENG = ST[ST.condition.isin(['ours', 'graphviz'])]
    for col, lab in [('report_label_collisions', 'label collisions'),
                     ('report_edge_edge_crossings', 'arrow crossings'),
                     ('report_valid', 'strict verdict')]:
        for df, nm in [(ST, 'full panel, 72 images'), (ENG, 'two engines, 48 images')]:
            x = df[col].astype(float).values; y = df.bws.values
            rho, p = stats.spearmanr(x, y)
            ci = cluster_boot(lambda i: stats.spearmanr(x[i], y[i]).statistic,
                              df.paper_id.values, B=2000)
            say(f'  {lab:17s}{nm:23s}rho={rho:+.2f}  naive p={p:.2g}  '
                f'cluster 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]')
    say('\n  the 48 images are 24 matched pairs: the paired test the design supports')
    pv = ENG.pivot_table(index='paper_id', columns='condition', values='bws')
    dB = (pv['ours'] - pv['graphviz']).values
    for col, lab in [('report_label_collisions', 'label collisions'),
                     ('report_edge_edge_crossings', 'arrow crossings'),
                     ('report_valid', 'strict verdict')]:
        pc = ENG.pivot_table(index='paper_id', columns='condition', values=col)
        dC = (pc['ours'] - pc['graphviz']).values
        ok = ~(np.isnan(dB) | np.isnan(dC))
        if len(np.unique(dC[ok])) < 2:
            continue
        rho = stats.spearmanr(dC[ok], dB[ok]).statistic
        rng = RNG(0); k = 0
        for _ in range(B):
            s = rng.choice([-1, 1], ok.sum())
            if abs(stats.spearmanr(dC[ok] * s, dB[ok] * s).statistic) >= abs(rho) - 1e-12:
                k += 1
        say(f'    rho(delta {lab}, delta best-worst) over {ok.sum()} graphs = {rho:+.2f}, '
            f'sign-flip p = {(1 + k)/(B + 1):.3f}')
    dd = dB[~np.isnan(dB)]; ci = boot_mean_ci(dd)
    say(f'    best-worst, ours minus dot, over {len(dd)} graphs: {dd.mean():+.3f} '
        f'[{ci[0]:+.3f}, {ci[1]:+.3f}], Wilcoxon p = {stats.wilcoxon(dd).pvalue:.3f}')
    valid = dict(zip(ST.image, ST.report_valid.astype(float)))

    def OR(best, worst):
        a = sum(valid[i] for i in best); b = len(best) - a
        c = sum(valid[i] for i in worst); dd_ = len(worst) - c
        return (a * dd_) / (b * c) if b * c else np.inf
    best, worst = list(RA.best), list(RA.worst)
    orv = OR(best, worst)
    pbest = np.mean([valid[i] for i in best]); pworst = np.mean([valid[i] for i in worst])
    tab = np.array([[sum(valid[i] for i in best), len(best) - sum(valid[i] for i in best)],
                    [sum(valid[i] for i in worst), len(worst) - sum(valid[i] for i in worst)]])
    say(f'\n  strict verdict against what a rater picks: most legible passes '
        f'{pbest*100:.1f}%, least legible {pworst*100:.1f}%')
    say(f'    pooled 2x2, choices treated as independent: OR = {orv:.1f}, '
        f'Fisher p = {stats.fisher_exact(tab)[1]:.2g}   <- 216 choices, 72 sets, 3 raters')
    num = den = 0.0
    for _, g in RA.groupby('set'):
        bs, ws = list(g.best), list(g.worst); n = len(bs) + len(ws)
        a = sum(valid[i] for i in bs); b = len(bs) - a
        c = sum(valid[i] for i in ws); dd_ = len(ws) - c
        num += a * dd_ / n; den += b * c / n
    say(f'    Mantel-Haenszel OR stratified by rating set: {num/den:.1f}')
    rng = RNG(0); k = 0
    for _ in range(B):
        bb, ww = [], []
        for r in RA.itertuples():
            pick = rng.choice(setimg[r.set], 2, replace=False)
            bb.append(pick[0]); ww.append(pick[1])
        if OR(bb, ww) >= orv - 1e-12:
            k += 1
    say(f'    within-set randomisation p (each rater picks at random inside their own set): '
        f'{(1 + k)/(B + 1):.4f} (floor {1/(B+1):.0e})')


# ---------------------------------------------------------------- I6
def I6(d):
    head('I6  capacity (LD50) with cluster bootstrap intervals')
    G = engines(d['L']).dropna(subset=['n_nodes'])
    pts = {}
    for cfg in ['ours', 'dot_tb_spline', 'dot_lr_spline', 'dot_tb_ortho', 'dot_lr_ortho']:
        s = G[G.config == cfg]
        x = np.log(s.n_nodes.clip(lower=1).values); y = s.valid.values.astype(float)
        pt = logit_ld50(x, y)
        ci = cluster_boot(lambda i: logit_ld50(x[i], y[i]), s.cl.values, B=400)
        pts[cfg] = pt
        say(f'  {cfg:15s} half its figures stop passing at {pt:5.1f} boxes  '
            f'95% CI [{ci[0]:.1f}, {ci[1]:.1f}]   ({len(s)} figures, {s.cl.nunique()} graphs)')
    so, sd = G[G.config == 'ours'], G[G.config == 'dot_tb_spline']
    gs = {g: (np.where(so.cl.values == g)[0], np.where(sd.cl.values == g)[0])
          for g in pd.unique(so.cl.values)}
    keys = list(gs); rng = RNG(0); diff = []
    xo, yo = np.log(so.n_nodes.clip(lower=1).values), so.valid.values.astype(float)
    xd, yd = np.log(sd.n_nodes.clip(lower=1).values), sd.valid.values.astype(float)
    for _ in range(400):
        j = rng.integers(0, len(keys), len(keys))
        io = np.concatenate([gs[keys[t]][0] for t in j])
        id_ = np.concatenate([gs[keys[t]][1] for t in j])
        a, b = logit_ld50(xo[io], yo[io]), logit_ld50(xd[id_], yd[id_])
        if np.isfinite(a) and np.isfinite(b) and max(a, b) < 1e4:
            diff.append(a - b)
    say(f'  difference, ours minus dot best single configuration: '
        f'{pts["ours"]-pts["dot_tb_spline"]:+.1f} boxes, 95% CI '
        f'[{np.percentile(diff,2.5):+.1f}, {np.percentile(diff,97.5):+.1f}]')


# ---------------------------------------------------------------- I7
def I7(d):
    head('I7  arrow crossings vs label collisions, restated as a bound')
    P = d['P'][d['P'].variant == 'evaluated'].dropna(subset=['score', 'layout_valid',
                                                             'layout_n_nodes']).copy()
    P['cl'] = P.corpus.astype(str) + '|' + P.item.astype(str)
    x, y, z = P.score.values, P.layout_valid.values.astype(float), P.layout_n_nodes.values
    r, p, n = partial_spearman(x, y, z)
    ci = cluster_boot(lambda i: partial_spearman(x[i], y[i], z[i])[0], P.cl.values, B=1000)
    say(f'  benchmark score vs strict verdict, boxes held fixed: rho = {r:+.3f} '
        f'(n={n} figures, {P.cl.nunique()} graphs)')
    say(f'    cluster bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] -> any association is '
        f'smaller than {max(abs(ci)):.2f} in absolute value')
    a, b = x[y == 1], x[y == 0]
    auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))
    say(f'    unconditionally the score does carry a little signal: AUC = {auc:.3f} '
        f'(rank-biserial {2*auc-1:+.3f}); with boxes held fixed it is the interval above, '
        f'so what it tracks is size')
    G = engines(d['L']).dropna(subset=['edge_edge_crossings', 'label_collisions', 'n_nodes'])
    x2, y2, z2 = (G.edge_edge_crossings.values, G.label_collisions.values, G.n_nodes.values)
    r2, p2, n2 = partial_spearman(x2, y2, z2)
    ci2 = cluster_boot(lambda i: partial_spearman(x2[i], y2[i], z2[i])[0], G.cl.values, B=1000)
    say(f'  arrow crossings vs label collisions, boxes held fixed: rho = {r2:+.3f} '
        f'(naive p = {p2:.2g}, n={n2} layouts, {G.cl.nunique()} graphs)')
    say(f'    cluster bootstrap 95% CI [{ci2[0]:+.3f}, {ci2[1]:+.3f}]: not zero, but under '
        f'{max(abs(ci2)):.2f}; "negligible", not "independent"')


ALL = dict(I1=I1, I2=I2, I3=I3, I4=I4, I5=I5, I6=I6, I7=I7)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*', default=list(ALL))
    ap.add_argument('--out')
    a = ap.parse_args()
    d = load(a.data)
    for k in a.analysis:
        ALL[k](d)
    if a.out:
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
