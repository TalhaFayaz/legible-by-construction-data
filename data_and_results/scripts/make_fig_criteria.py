#!/usr/bin/env python3
"""Which graph-drawing criterion a rater follows, from
data/human_{stimuli,sets,ratings}.csv only. Produces img/criteria_ranking.pdf
and .png and the CSV behind the figure. No GPU, no model call, no new run.

Two correlations per measure: over all 72 images, whose three conditions include
the model placing its own boxes, and over the 48 images the two real engines
drew. The first ranks how well a measure detects a figure nobody can read; only
the second could say which criterion a reader pays for between two drawn
layouts, and there nothing survives correction (checks_claims.py K6, K7).

    python3 make_fig_criteria.py --data ../data --out ../../img
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the defect counts and the geometric measures, in the paper's wording
CRIT = [
    ('report_label_collisions',     'Label collisions',      'defect'),
    ('report_node_overlap_pairs',   'Box overlaps',          'defect'),
    ('report_edge_node_crossings',  'Arrows through a box',  'defect'),
    ('report_edge_label_crossings', 'Arrows through a label','defect'),
    ('report_stacked_routes',       'Stacked routes',        'defect'),
    ('report_edge_edge_crossings',  'Arrow crossings',       'defect'),
    ('report_mean_bends_per_edge',  'Bends per arrow',       'other'),
    ('report_min_separation',       'Box separation',        'other'),
    ('report_total_edge_length',    'Total arrow length',    'other'),
    ('report_area',                 'Area',                  'other'),
    ('report_aspect_deviation',     'Page shape',            'other'),
    ('report_alignment_ratio',      'Node alignment',        'other'),
]
RIDGE_F = [c for c, _, _ in CRIT[:6]] + ['report_total_edge_length',
           'report_aspect_deviation', 'report_mean_bends_per_edge',
           'report_min_separation']


def bws(data):
    st = pd.read_csv(f'{data}/human_stimuli.csv')
    se = pd.read_csv(f'{data}/human_sets.csv')
    ra = pd.read_csv(f'{data}/human_ratings.csv')
    c = {i: [0, 0, 0] for i in st.image}
    for _, r in ra.iterrows():
        for im in se[se.set == r.set].iloc[0][
                ['image_1', 'image_2', 'image_3', 'image_4']]:
            c[im][2] += 1
        c[r.best][0] += 1
        c[r.worst][1] += 1
    st['bws'] = [(c[i][0] - c[i][1]) / max(c[i][2], 1) for i in st.image]
    return st, len(ra)


def cv_index(st):
    """Ridge over all ten measures, scored leaving one source graph out."""
    X = np.log1p(np.clip(st[RIDGE_F].astype(float).values, 0, None))
    y, pid = st.bws.values, st.paper_id.values
    best = -1
    for lam in (1, 5, 20, 50):
        pred = np.zeros(len(y))
        for q in np.unique(pid):
            tr = pid != q
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Z = (X[tr] - mu) / sd
            w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]),
                                Z.T @ (y[tr] - y[tr].mean()))
            pred[~tr] = ((X[~tr] - mu) / sd) @ w + y[tr].mean()
        best = max(best, stats.spearmanr(pred, y)[0])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data')
    ap.add_argument('--out', default='../../img')
    a = ap.parse_args()
    st, n_j = bws(a.data)
    eng = st[st.condition.isin(['ours', 'graphviz'])]

    rows = []
    for col, lab, kind in CRIT:
        r, p = stats.spearmanr(st[col].astype(float), st.bws)
        if eng[col].nunique() > 1:
            r2, p2 = stats.spearmanr(eng[col].astype(float), eng.bws)
            const = False
        else:
            r2, p2, const = np.nan, np.nan, True
        rows.append(dict(measure=lab, column=col, kind=kind, rho=r, agreement=abs(r), p=p,
                         rho_engines48=r2, agreement_engines48=abs(r2), p_engines48=p2,
                         constant_within_engines=const))
    d = pd.DataFrame(rows).sort_values('agreement', ascending=True)
    verdict = abs(stats.spearmanr(st.report_valid.astype(float), st.bws)[0])
    verdict_eng = abs(stats.spearmanr(eng.report_valid.astype(float), eng.bws)[0])
    index = cv_index(st)

    fig, ax = plt.subplots(figsize=(3.3, 3.45), dpi=200)
    y = np.arange(len(d))
    h = 0.36
    colors = ['#1f4e79' if k == 'defect' else '#c3cedb' for k in d.kind]
    ax.barh(y + h / 2, d.agreement, color=colors, height=h,
            edgecolor='white', linewidth=0.4)
    second = d.agreement_engines48.fillna(0.0).values
    ax.barh(y - h / 2, second, height=h, color='white', edgecolor='#555555',
            linewidth=0.6, hatch='////')
    for i, (v, w, c) in enumerate(zip(d.agreement, second, d.constant_within_engines)):
        ax.text(v + 0.012, i + h / 2, f'{v:.2f}', va='center', fontsize=6.0,
                color='#555555')
        ax.text(w + 0.012, i - h / 2, 'constant' if c else f'{w:.2f}', va='center',
                fontsize=6.0, color='#888888')
    ax.axvline(verdict, color='#b03030', linestyle='--', linewidth=1.0)
    ax.axvline(index, color='#777777', linestyle=':', linewidth=1.0)
    ax.text(verdict, len(d) - 0.55, 'strict\nverdict', fontsize=6.3, ha='center',
            color='#b03030', va='bottom')
    ax.text(index, len(d) - 0.55, 'fitted\nindex', fontsize=6.3, ha='center',
            color='#777777', va='bottom')
    ax.set_yticks(y)
    ax.set_yticklabels(d.measure, fontsize=7)
    ax.set_xlabel(r'Agreement with the raters ($|\rho|$)', fontsize=7.5)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(-0.75, len(d) + 0.55)
    ax.tick_params(labelsize=7)
    ax.grid(axis='x', color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    hh = [plt.Rectangle((0, 0), 1, 1, color='#1f4e79'),
          plt.Rectangle((0, 0), 1, 1, color='#c3cedb'),
          plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#555555', hatch='////')]
    ax.legend(hh, ['Defect counts, all 72 images', 'Other geometry, all 72 images',
                   'The 48 images the two engines drew'], fontsize=6.2,
              frameon=False, loc='upper center', bbox_to_anchor=(0.46, -0.16), ncol=1,
              handlelength=1.4, labelspacing=0.35)
    fig.tight_layout(pad=0.3)
    os.makedirs(a.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'criteria_ranking.{ext}'),
                    bbox_inches='tight')
    d = d.sort_values('agreement', ascending=False)
    d.to_csv(os.path.join(a.out, 'criteria_ranking.csv'), index=False)
    print(d.round(3).to_string(index=False))
    print(f"\nstrict verdict (binary) |rho| = {verdict:.3f} over all 72 images, "
          f"{verdict_eng:.3f} over the 48")
    print(f"ridge index, leave-one-graph-out |rho| = {index:.3f}")
    print(f"{n_j} judgements over {st.image.nunique()} images")
    print("The solid bars rank the measures by how sharply they separate the three")
    print("conditions, one of which is the model placing its own boxes. The hatched bars")
    nconst = int(d.constant_within_engines.sum())
    print(f"are the same correlation over the 48 images drawn by the two real engines;")
    print(f"{nconst} of the {len(d)} measures are identically zero there and none of the rest")
    print("survives correction within that family (checks_claims.py --analysis K6 K7).")
    print("wrote criteria_ranking.{pdf,png,csv} to", a.out)


if __name__ == '__main__':
    main()
