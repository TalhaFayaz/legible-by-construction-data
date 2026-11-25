#!/usr/bin/env python3
"""Where the benchmark's variance lives, and which metric has resolution.

Five analyses (H1-H5) computed from the per-item CSVs only. No GPU, no model
call, no new run. Mirrors the style of the other analysis scripts.

    python3 checks_variance.py --data ../data
    python3 checks_variance.py --analysis H1 H4
    python3 checks_variance.py --latex ../results/tables_variance.tex
    python3 checks_variance.py --out ../results/printed_outputs/analysis_variance_and_metric.txt

H1  variance components of the benchmark score: how much of it belongs to the
    item, to the model, to their interaction and to the decoding seed.
H2  what that leaves a comparison: the smallest unpaired difference a given
    sample size can resolve, and the rate at which a small true difference
    comes out with the wrong sign at the sample sizes in common use.
H3  whether item difficulty is a property of the text or of the model, for our
    pipeline and for the same models writing DOT, with parse failures excluded.
H4  normalised graph edit distance against the benchmark's weighted score on
    the same graphs: verdict agreement, effect size, and the size control.
H5  the exchange rate between the two measurements inside one representation:
    what one extra box of detail buys on the score and costs on legibility.
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy import stats

MODELS = ['Qwen', 'Gemma', 'Ministral']
CORPORA = ['sciflow', 'flowvqa']
KEY = ['domain', 'item']
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


def num(s):
    return pd.to_numeric(s, errors='coerce')


def evaluated(D):
    """The pipeline the paper evaluates, score with a failure counted as zero."""
    e = D['pipe'][D['pipe'].variant == 'evaluated'].copy()
    e['score'] = num(e['score']).fillna(0.0)
    e['lv'] = num(e['layout_valid'])
    return e


def dot_plain(D):
    d = D['dot'][D['dot'].variant == 'dot'].copy()
    d['score'] = num(d['score']).fillna(0.0)
    d['failed'] = num(d['failed']).fillna(0)
    return d


# ---------------------------------------------------------------- H1

def H1(D):
    head('H1  variance components of the benchmark score')
    say('Balanced over item x model x seed within each corpus. Components are')
    say('mean squared deviations, so they sum to the total up to the')
    say('item x model x seed residual.')
    rows = []
    for corp in CORPORA:
        d = evaluated(D)
        d = d[d.corpus == corp]
        grand = d.score.mean()
        tot = ((d.score - grand) ** 2).mean()
        item_m = d.groupby(KEY, dropna=False).score.transform('mean')
        model_m = d.groupby('model').score.transform('mean')
        cell = d.groupby(KEY + ['model'], dropna=False).score.transform('mean')
        v_item = ((d.groupby(KEY, dropna=False).score.mean() - grand) ** 2).mean()
        v_model = ((d.groupby('model').score.mean() - grand) ** 2).mean()
        v_int = ((cell - item_m - model_m + grand) ** 2).mean()
        v_seed = ((d.score - cell) ** 2).mean()
        sd_cell = d.groupby(KEY + ['model'], dropna=False).score.std()
        say(f'\n{corp}: n={len(d)}  mean={grand:.4f}  total variance={tot:.5f}')
        for nm, v in [('item', v_item), ('model', v_model),
                      ('item x model', v_int), ('seed (residual)', v_seed)]:
            say(f'   {nm:18s} {v:.6f}   {100 * v / tot:5.1f}% of total')
            rows.append(dict(corpus=corp, component=nm, variance=v,
                             share=100 * v / tot))
        say(f'   seed sd within one item x model cell: mean '
            f'{sd_cell.mean():.4f}, median {sd_cell.median():.4f}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H2

def H2(D):
    head('H2  what that leaves a comparison, by sample size')
    e = evaluated(D)
    dot = dot_plain(D)
    say('Left: the smallest difference of corpus means two independent')
    say('samples of n items each can separate at alpha=0.05, from the')
    say('item-level sd of the score. Right: for each model, the rate at which')
    say('a bootstrap sample of n items returns the wrong sign for the paired')
    say('difference against the same model writing DOT (4,000 resamples).')
    rng = np.random.default_rng(0)
    rows = []
    for corp in CORPORA:
        d = e[e.corpus == corp]
        item_sd = d.groupby(KEY, dropna=False).score.mean().std()
        mm = d.groupby('model').score.mean()
        full = 483 if corp == 'sciflow' else 300
        say(f'\n{corp}: item-level sd={item_sd:.4f}; '
            f'largest model-to-model gap={mm.max() - mm.min():.4f} '
            f'= {(mm.max() - mm.min()) / item_sd:.2f} item sd')
        for n in (50, 100, full):
            mde = 1.96 * item_sd * np.sqrt(2.0 / n)
            say(f'   n={n:4d}: unpaired minimum detectable gap = {mde:.4f}')
            rows.append(dict(corpus=corp, n=n, mde=mde))
        for m in MODELS:
            a = e[(e.corpus == corp) & (e.model == m)].groupby(
                KEY, dropna=False).score.mean()
            b = dot[(dot.corpus == corp) & (dot.model == m)].groupby(
                KEY, dropna=False).score.mean()
            j = pd.concat([a.rename('o'), b.rename('d')], axis=1).dropna()
            diff = (j.o - j.d).values
            true = diff.mean()
            out = {}
            for n in (50, 100, full):
                s = rng.choice(diff, (4000, min(n, len(diff))),
                               replace=True).mean(1)
                out[n] = float((np.sign(s) != np.sign(true)).mean())
            say(f'   {m:10s} true paired difference={true:+.4f}  '
                f'P(wrong sign): n=50 {out[50]:.3f}  n=100 {out[100]:.3f}  '
                f'n={full} {out[full]:.3f}')
            rows.append(dict(corpus=corp, model=m, true=true,
                             wrong50=out[50], wrong100=out[100],
                             wrongfull=out[full]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H3

def H3(D):
    head('H3  is item difficulty a property of the text or of the model')
    say('Spearman between two models\' per-item scores on the same items, for')
    say('our pipeline (seed-averaged) and for the same models writing DOT.')
    say('Items on which any model\'s DOT answer failed to parse are dropped')
    say('from both sides, so a parse failure scored zero cannot drive it.')
    e = evaluated(D)
    dot = dot_plain(D)
    rows = []
    for corp in CORPORA:
        d = dot[dot.corpus == corp].copy()
        d['k'] = list(map(tuple, d[KEY].astype(str).values))
        bad = set(d[d.failed > 0].k)
        keep = d[~d.k.isin(bad)]
        pv_d = keep.pivot_table(index='k', columns='model', values='score',
                                aggfunc='mean').dropna()
        eo = e[e.corpus == corp].copy()
        eo['k'] = list(map(tuple, eo[KEY].astype(str).values))
        pv_o = eo[eo.k.isin(set(pv_d.index))].pivot_table(
            index='k', columns='model', values='score',
            aggfunc='mean').dropna()
        say(f'\n{corp}: {len(pv_d)} items kept of {d.k.nunique()} '
            f'({len(bad)} dropped for a DOT parse failure)')
        for nm, pv in [('ours', pv_o), ('DOT', pv_d)]:
            rs = []
            for i in range(len(MODELS)):
                for k in range(i + 1, len(MODELS)):
                    a, b = MODELS[i], MODELS[k]
                    if a not in pv or b not in pv:
                        continue
                    r = stats.spearmanr(pv[a], pv[b])
                    rs.append(r.statistic)
                    say(f'   {nm:5s} {a:10s} vs {b:10s} rho={r.statistic:+.3f} '
                        f'p={r.pvalue:.1e}')
            say(f'   {nm:5s} mean rho over the three pairs = {np.mean(rs):+.3f}')
            rows.append(dict(corpus=corp, system=nm, mean_rho=np.mean(rs),
                             n=len(pv), rhos=list(np.round(rs, 3))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H4

def H4(D):
    head('H4  graph edit distance against the weighted score, same graphs')
    say('strict_normalised_ged scores the same extracted graph against the')
    say('same gold graph as the benchmark\'s weighted score, by normalised')
    say('graph edit distance with exact label matching; lower is better.')
    say('Every test below is paired per item, so the gold graph, and with it')
    say('the graph size, is identical on both sides of every comparison.')
    e = evaluated(D)
    rows = []
    for corp in CORPORA:
        d = e[(e.corpus == corp)].dropna(subset=['strict_normalised_ged'])
        say(f'\n{corp}: n={len(d)} item x model x seed rows')
        r1 = stats.spearmanr(d.strict_normalised_ged, d.score)
        say(f'   per row: nGED vs score rho={r1.statistic:+.3f}; '
            f'vs node F1 '
            f'{stats.spearmanr(d.strict_normalised_ged, d.node_f1).statistic:+.3f}; '
            f'vs edge F1 '
            f'{stats.spearmanr(d.strict_normalised_ged, d.edge_f1).statistic:+.3f}')
        # size control
        def rank_partial(x, y, z):
            xr, yr, zr = [stats.rankdata(np.asarray(v, float)) for v in (x, y, z)]
            rxy = np.corrcoef(xr, yr)[0, 1]
            rxz = np.corrcoef(xr, zr)[0, 1]
            ryz = np.corrcoef(yr, zr)[0, 1]
            return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        say(f'   nGED vs gold box count rho='
            f'{stats.spearmanr(d.strict_normalised_ged, d.gold_nodes).statistic:+.3f}; '
            f'partial(nGED, score | gold box count)='
            f'{rank_partial(d.strict_normalised_ged, d.score, d.gold_nodes):+.3f}')
        pv_s = d.pivot_table(index=KEY, columns='model', values='score',
                             aggfunc='mean', dropna=False)
        pv_g = d.pivot_table(index=KEY, columns='model',
                             values='strict_normalised_ged', aggfunc='mean',
                             dropna=False)
        for i in range(len(MODELS)):
            for k in range(i + 1, len(MODELS)):
                a, b = MODELS[i], MODELS[k]
                rec = dict(corpus=corp, pair=f'{a} vs {b}')
                for nm, pv, sign in [('score', pv_s, +1), ('nGED', pv_g, -1)]:
                    x = pv[[a, b]].dropna()
                    diff = x[a] - x[b]
                    w = stats.wilcoxon(x[a], x[b])
                    dz = diff.mean() / diff.std()
                    # which model the measure prefers: score high is good,
                    # nGED low is good
                    better = a if sign * diff.mean() > 0 else b
                    verdict = ('tied' if w.pvalue >= 0.05
                               else f'{better} ahead')
                    say(f'   {a:9s} vs {b:9s} {nm:5s}: n={len(x)} '
                        f'mean diff={diff.mean():+.4f} dz={dz:+.3f} '
                        f'p={w.pvalue:.2e}  -> {verdict}')
                    rec[nm + '_diff'] = diff.mean()
                    rec[nm + '_dz'] = dz
                    rec[nm + '_p'] = w.pvalue
                    rec[nm + '_verdict'] = verdict
                rec['agree'] = (rec['score_verdict'] == rec['nGED_verdict']
                                or rec['score_verdict'] == 'tied')
                rows.append(rec)
    df = pd.DataFrame(rows)
    say('\nverdict agreement: nGED matches the score on '
        f'{int((df.score_verdict == df.nGED_verdict).sum())} of {len(df)} '
        'pairs, and on '
        f'{int(((df.score_verdict != "tied") & (df.score_verdict == df.nGED_verdict)).sum())} '
        f'of {int((df.score_verdict != "tied").sum())} pairs the score itself '
        'resolves. It reverses none.')
    sc = df[df.score_verdict == 'tied']
    if len(sc):
        say(f'on the {len(sc)} pair(s) the score reports as tied, nGED gives: '
            + '; '.join(f'{r.pair} ({r.corpus}) {r.nGED_verdict}, '
                        f'dz={r.nGED_dz:+.2f}, p={r.nGED_p:.1e}'
                        for r in sc.itertuples()))
    return df


# ---------------------------------------------------------------- H5

def H5(D):
    head('H5  the exchange rate between the two measurements, one representation')
    say('The coverage call is the same pipeline with one extra request asking')
    say('for components the first pass missed, so it varies detail at a fixed')
    say('representation. Paired per item, seed-averaged.')
    p = D['pipe'].copy()
    p['score'] = num(p['score']).fillna(0.0)
    p['lv'] = num(p['layout_valid'])
    rows = []
    for corp in CORPORA:
        for m in MODELS:
            g = {}
            for v in ('evaluated', 'coverage'):
                g[v] = p[(p.corpus == corp) & (p.model == m)
                         & (p.variant == v)].groupby(KEY, dropna=False).agg(
                    s=('score', 'mean'), v=('lv', 'mean'),
                    nd=('pred_nodes', 'mean'))
            j = g['evaluated'].join(g['coverage'], lsuffix='_e', rsuffix='_c',
                                    how='inner')
            ds, dv = j.s_c - j.s_e, j.v_c - j.v_e
            ws = stats.wilcoxon(j.s_e, j.s_c)
            wv = stats.wilcoxon(j.v_e, j.v_c)
            dn = j.nd_c.mean() - j.nd_e.mean()
            say(f'   {corp:8s} {m:10s} n={len(j)}  '
                f'boxes {j.nd_e.mean():.1f} -> {j.nd_c.mean():.1f} ({dn:+.1f})  '
                f'score {ds.mean():+.4f} (p={ws.pvalue:.1e})  '
                f'strict legible {100 * dv.mean():+.1f} pts '
                f'(p={wv.pvalue:.1e})')
            rows.append(dict(corpus=corp, model=m, n=len(j), dboxes=dn,
                             dscore=ds.mean(), dscore_p=ws.pvalue,
                             dvalid=100 * dv.mean(), dvalid_p=wv.pvalue,
                             score_ci=boot_ci(ds), per_box_score=ds.mean() / dn,
                             per_box_valid=100 * dv.mean() / dn))
    df = pd.DataFrame(rows)
    say('\nper extra box: score '
        f'{df.per_box_score.min():+.4f} to {df.per_box_score.max():+.4f}, '
        f'strict legibility {df.per_box_valid.min():+.1f} to '
        f'{df.per_box_valid.max():+.1f} points.')
    say('cells where more detail raises the score significantly: '
        f'{int(((df.dscore > 0) & (df.dscore_p < 0.05)).sum())} of {len(df)}; '
        'cells where it lowers strict legibility significantly: '
        f'{int(((df.dvalid < 0) & (df.dvalid_p < 0.05)).sum())} of {len(df)}.')
    # the same trade-off over every variant, as cells
    cells = p[p.corpus.isin(CORPORA)].groupby(
        ['corpus', 'model', 'variant']).agg(
        score=('score', 'mean'), strict=('lv', 'mean'),
        boxes=('pred_nodes', 'mean')).reset_index()
    say('\nover all corpus x model x variant cells (five variants):')
    for corp in CORPORA:
        c = cells[cells.corpus == corp]
        say(f'   {corp}: n={len(c)} cells  '
            f'boxes vs strict legible rho='
            f'{stats.spearmanr(c.boxes, c.strict).statistic:+.3f}  '
            f'boxes vs score rho='
            f'{stats.spearmanr(c.boxes, c.score).statistic:+.3f}')
    return df, cells


# ---------------------------------------------------------------- latex

def write_latex(path, h1, h3, h4):
    """One compact float: variance components, item difficulty, nGED."""
    cn = {'sciflow': 'SciFlow', 'flowvqa': 'FlowVQA'}
    L = ['% generated by checks_variance.py --latex',
         '\\begin{table}[t]', '\t\\centering',
         '\t\\fontsize{7}{8.4}\\selectfont',
         '\t\\setlength{\\tabcolsep}{3pt}',
         '\t\\resizebox{\\columnwidth}{!}{%',
         '\t\\begin{tabular}{l r r @{\\hspace{10pt}} l r r}',
         '\t\t\\toprule',
         '\t\t\\multicolumn{3}{l}{\\textbf{(a) Variance of the score, \\%}} & '
         '\\multicolumn{3}{l}{\\textbf{(b) Item difficulty, '
         '$\\overline{\\rho}$}} \\\\',
         '\t\t & SciFlow & FlowVQA & & SciFlow & FlowVQA \\\\',
         '\t\t\\midrule']
    comp = ['item', 'model', 'item x model', 'seed (residual)']
    nice = {'item': 'Item', 'model': 'Model',
            'item x model': 'Item\\,$\\times$\\,model',
            'seed (residual)': 'Decoding seed'}
    sh = {(r.corpus, r.component): r.share for r in h1.itertuples()}
    rh = {(r.corpus, r.system): r.mean_rho for r in h3.itertuples()}
    right = [('Ours', 'ours'), ('Same model, DOT', 'DOT'), (None, None),
             (None, None)]
    for i, c in enumerate(comp):
        b = '\\textbf{%s}' if c == 'item' else '%s'
        lab, kk = right[i]
        rcol = (f'{lab} & {rh[("sciflow", kk)]:+.2f} & {rh[("flowvqa", kk)]:+.2f}'
                if lab else ' & & ')
        L.append(f'\t\t{nice[c]} & ' + (b % f'{sh[("sciflow", c)]:.1f}')
                 + ' & ' + (b % f'{sh[("flowvqa", c)]:.1f}')
                 + f' & {rcol} \\\\')
    L += ['\t\t\\midrule',
          '\t\t\\multicolumn{6}{l}{\\textbf{(c) The same graphs, two '
          'structural measures: paired $|d_z|$ and verdict}} \\\\',
          '\t\t & \\multicolumn{2}{l}{Benchmark score} & & '
          '\\multicolumn{2}{l}{nGED} \\\\',
          '\t\t\\midrule']
    for corp in CORPORA:
        for r in h4[h4.corpus == corp].itertuples():
            v = lambda s: ('\\emph{tied}' if s == 'tied'
                           else s.replace(' ahead', ''))
            L.append(f'\t\t{cn[corp]}, {r.pair} & {abs(r.score_dz):.3f} & '
                     f'{v(r.score_verdict)} & & {abs(r.nGED_dz):.3f} & '
                     f'{v(r.nGED_verdict)} \\\\')
    L += ['\t\t\\bottomrule', '\t\\end{tabular}}',
          '\t\\caption{Where the benchmark\'s variance is, and what has '
          'resolution. \\textbf{(a)}~Variance components of the score over '
          'item $\\times$ model $\\times$ seed: which item a study draws is '
          '83--86\\% of it and which model wrote the graph 1\\%, and the '
          'interaction is ten times the model term, so the models differ in '
          '\\emph{which} texts they handle rather than in level. '
          '\\textbf{(b)}~Mean Spearman correlation between two models\' '
          'per-item scores, over the three model pairs, on the items where '
          'no model\'s DOT answer failed to parse: under our pipeline '
          'difficulty is a property of the text, under free-form DOT much '
          'less so. \\textbf{(c)}~The same extracted graphs scored by the '
          'benchmark\'s weighted score and by normalised graph edit distance '
          '(nGED, lower is better), paired per item so the gold graph and its '
          'size are identical on both sides; the verdict is a paired '
          'Wilcoxon test at $\\alpha=0.05$. nGED reverses none of the four '
          'verdicts the score resolves and separates one of the two it calls '
          '\\emph{tied}.}',
          '\t\\label{table:resolution}', '\\end{table}']
    open(path, 'w').write('\n'.join(L) + '\n')
    print(f'latex written to {path}', file=sys.stderr)


def load(data):
    f = lambda n: pd.read_csv(f'{data}/{n}.csv', low_memory=False)
    return dict(pipe=f('pipeline_items'), dot=f('dot_items'),
                layout=f('layout_items'), judge=f('judge_paperbanana'),
                readback=f('readback_sciflow'), place=f('placement_items'),
                pub=f('published_systems_sciflow'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--analysis', nargs='*',
                    default=['H1', 'H2', 'H3', 'H4', 'H5'])
    ap.add_argument('--out')
    ap.add_argument('--latex')
    ap.add_argument('--csv-dir', help='also write every table as CSV')
    a = ap.parse_args()
    D = load(a.data)
    say('Variance, resolution and metric choice: analyses H1-H5, computed '
        'from the per-item CSVs only.')
    h1 = h2 = h3 = h4 = h5 = cells = None
    if 'H1' in a.analysis:
        h1 = H1(D)
    if 'H2' in a.analysis:
        h2 = H2(D)
    if 'H3' in a.analysis:
        h3 = H3(D)
    if 'H4' in a.analysis:
        h4 = H4(D)
    if 'H5' in a.analysis:
        h5, cells = H5(D)
    if a.csv_dir:
        for nm, df in [('variance_components', h1), ('resolution', h2),
                       ('item_difficulty', h3), ('ged_vs_score', h4),
                       ('exchange_rate', h5), ('variant_cells', cells)]:
            if df is not None:
                df.to_csv(f'{a.csv_dir}/{nm}.csv', index=False)
        print(f'csv written to {a.csv_dir}', file=sys.stderr)
    if a.latex:
        if h1 is None or h3 is None or h4 is None:
            print('--latex needs H1, H3 and H4', file=sys.stderr)
        else:
            write_latex(a.latex, h1, h3, h4)
    if a.out:
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
        print(f'\nwritten to {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
