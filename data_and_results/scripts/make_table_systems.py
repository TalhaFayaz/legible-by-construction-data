#!/usr/bin/env python3
"""The published-systems comparison of the main paper, rebuilt from the data.

Two tables of the main paper rest on the same two files, `published_systems_
sciflow.csv` (the graphs the benchmark recovered from each shipped system's
image) and `readback_sciflow.csv` (every figure, ours included, drawn and read
back by one reader):

  * the shipped-systems table -- each system's edge F1, weighted score and node
    F1 under the benchmark's own protocol, its score under the symmetric
    read-back, and that score as a share of the ceiling the gold graphs reach
    under the same protocol,
  * the paired read-back table -- the 21 paired differences between our three
    models and the seven shipped systems, each with the verdict the paper's
    protocol gives it (paired Wilcoxon, Holm-corrected within a model, verdict
    from the sign of the rank-biserial r at alpha = 0.05) and with the smallest
    difference that sample can resolve at 80% power, so that a verdict and the
    resolution behind it are never printed apart.

Both are printed, written as CSV beside the other released tables, and emitted
as the LaTeX the paper carries, so the paper and the data cannot drift apart.
The statistics are imported from `make_tables.py` rather than reimplemented,
and the resolution is the paired minimum detectable difference
`2.80 * sd(diff) / sqrt(n)` that `checks_claims.py --analysis K8` reports.

    python3 make_table_systems.py                # both tables, printed
    python3 make_table_systems.py --latex        # and write the LaTeX
    python3 make_table_systems.py --out ../results/printed_outputs/make_table_systems.txt

Needs pandas, numpy and scipy. Reads `../data/`; writes
`published_systems_summary.csv` and `published_systems_readback.csv` into
`../results/tables/`, and with `--latex` one `table_<label>.tex` per table
beside them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:          # so that -I and any working directory work
    sys.path.insert(0, str(HERE))

from make_tables import (MODELS, SHIPPED, across_runs, compare, holm, keyed,  # noqa: E402
                         load, runs_of, scores, verdict)

TABLES = HERE.parent / "results" / "tables"

# the order the paper prints the shipped systems in: by the read-back score
ORDER = ("gemini3", "gemini2.5", "seedream", "graphviz", "qwen", "pixart", "sdxl")
SHORT = {"gemini3": "Gemini 3 Pro Image", "gemini2.5": "Gemini 2.5 Flash Image",
         "seedream": "Seedream 4.0", "graphviz": "Graphviz (GPT-4o$\\rightarrow$DOT)",
         "qwen": "Qwen-Image", "pixart": "PixArt-$\\Sigma$", "sdxl": "SDXL"}


def paired_mde(diffs):
    """The smallest paired difference this sample resolves at 80% power."""
    values = np.asarray(diffs, dtype=float)
    return 2.80 * values.std(ddof=1) / np.sqrt(len(values))


def ceiling(rb):
    """What the gold graphs reach when drawn by our engine and read back."""
    gold = rb[rb.system == "gold_graph_ceiling"]
    return float(scores(gold, "score").mean())


# ------------------------------------------------------------------- table one


def summary(pipe, pub, rb):
    """Every system, both halves of the protocol, and the share of the ceiling."""
    ceil = ceiling(rb)
    rows = []
    for model in MODELS:
        runs = runs_of(pipe, "sciflow", model)
        edge, score, node = (across_runs(runs, c)[0]
                             for c in ("edge_f1", "score", "node_f1"))
        read = rb[(rb.system == "ours") & (rb.model == model)]
        rows.append(dict(system="Ours + " + model, key="ours+" + model, ours=True,
                         n=len(runs[0]),
                         edge_f1=edge, score=score, node_f1=node,
                         readback_n=len(read),
                         readback=float(scores(read, "score").mean())))
    for key in ORDER:
        own = pub[pub.system == key]
        read = rb[rb.system == key]
        rows.append(dict(system=SHIPPED[key], key=key, ours=False, n=len(own),
                         edge_f1=float(scores(own, "edge_f1").mean()),
                         score=float(scores(own, "score").mean()),
                         node_f1=float(scores(own, "node_f1").mean()),
                         readback_n=len(read),
                         readback=float(scores(read, "score").mean())))
    table = pd.DataFrame(rows)
    table["readback_pct_ceiling"] = 100 * table["readback"] / ceil
    table["ceiling"] = ceil
    return table


# ------------------------------------------------------------------- table two


def pairs(rb):
    """The 21 paired read-back differences, each beside what its sample resolves."""
    rows = []
    for model in MODELS:
        ours = keyed(rb[(rb.system == "ours") & (rb.model == model)], "score")
        got = {}
        for key in ORDER:
            result = compare(ours, keyed(rb[rb.system == key], "score"))
            if result:
                got[key] = result
        adjusted = dict(zip(got, holm([v["p"] for v in got.values()])))
        for key, value in got.items():
            theirs = keyed(rb[rb.system == key], "score")
            shared = sorted(set(ours) & set(theirs))
            diffs = [ours[k] - theirs[k] for k in shared]
            mde = paired_mde(diffs)
            rows.append(dict(
                model=model, against=key, system=SHIPPED[key], n=value["n"],
                diff=value["diff"], lo=value["ci"][0], hi=value["ci"][1],
                r=value["r"], p=value["p"], p_holm=adjusted[key],
                verdict=verdict(value["r"], adjusted[key]),
                paired_mde=mde,
                clears_its_resolution=bool(abs(value["diff"]) > mde)))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- printing


def report(table, pair):
    out = ["Shipped systems on SciFlow-Bench, one metric implementation for all of them",
           "left: each system's own graph or the graph the benchmark recovered from its image;",
           "right: every figure, ours included, drawn and read back by one reader",
           "",
           f"{'system':<26}{'n':>5}{'edge F1':>9}{'score':>8}{'node F1':>9}"
           f"{'read back':>11}{'n':>5}{'% ceil.':>9}"]
    for row in table.itertuples():
        out.append(f"{row.system:<26}{row.n:>5}{row.edge_f1:>9.3f}{row.score:>8.3f}"
                   f"{row.node_f1:>9.3f}{row.readback:>11.3f}{row.readback_n:>5}"
                   f"{row.readback_pct_ceiling:>9.1f}")
    out.append("")
    out.append(f"the gold graphs reach {table.ceiling.iloc[0]:.3f} under the read-back protocol,")
    out.append("so every number in that column lives in the bottom fifth of the metric's range")

    out.append("\nThe 21 paired read-back differences (ours minus the system),")
    out.append("verdict: paired Wilcoxon, Holm-corrected over the seven systems within a model,")
    out.append("from the sign of r at alpha = 0.05; resolution: the paired difference this")
    out.append("sample detects at 80% power (2.80 sd / sqrt(n)), as in checks_claims.py --analysis K8")
    out.append("")
    out.append(f"{'model':<10}{'against':<24}{'n':>5}{'diff':>9}{'r':>7}{'p Holm':>9}"
               f"{'verdict':>9}{'resolves':>10}{'clears':>8}")
    for row in pair.itertuples():
        out.append(f"{row.model:<10}{row.system.replace('$\\rightarrow$', '->').replace('$\\Sigma$', 'Sigma'):<24}"
                   f"{row.n:>5}{row.diff:>+9.4f}{row.r:>+7.2f}{row.p_holm:>9.2g}"
                   f"{row.verdict:>9}{row.paired_mde:>10.4f}"
                   f"{('yes' if row.clears_its_resolution else 'no'):>8}")

    for model in MODELS:
        got = pair[pair.model == model]
        ahead = (got.verdict == "ahead").sum()
        tied = (got.verdict == "tied").sum()
        behind = (got.verdict == "behind").sum()
        out.append(f"\n  ours + {model}: ahead of {ahead} of the seven shipped systems, "
                   f"tied with {tied}, behind {behind}")
        for row in got[got.verdict != "tied"].itertuples():
            if not row.clears_its_resolution:
                out.append(f"      {row.verdict} of {row.system} by {row.diff:+.4f}, "
                           f"below the {row.paired_mde:.4f} this sample resolves")
    out.append(f"\n  what these samples resolve at 80% power: {pair.paired_mde.min():.3f}"
               f" to {pair.paired_mde.max():.3f} score points"
               f" (mean {pair.paired_mde.mean():.3f})")
    clear = int((~pair.clears_its_resolution).sum())
    out.append(f"\n  {clear} of the {len(pair)} paired differences do not clear what this sample")
    out.append("  resolves at 80% power, so the ordering inside the middle of the table is tight.")
    out.append("  The column is reported as a measurement of the protocol and as evidence that")
    out.append("  coordinate-free generation is competitive with the systems these benchmarks")
    out.append("  ship, not as a general ranking of systems.")
    return "\n".join(out)


# ----------------------------------------------------------------------- LaTeX


def access(tables=TABLES):
    """how each system is distributed: `open` weights, or a provider API only.

    Kept as data rather than in the code so that the open-weight reading of the
    comparison in S5.4 is auditable from the released files like every other
    number in the paper.
    """
    rows = pd.read_csv(Path(tables) / "system_access.csv")
    return dict(zip(rows["key"], rows["weights"]))




def latex_summary(table):
    head = ("% generated by make_table_systems.py --latex; do not edit by hand\n"
            "\\begin{table}[t]\n\t\\centering\n\t\\fontsize{6.6}{7.9}\\selectfont\n"
            "\t\\setlength{\\tabcolsep}{3pt}\n\t\\resizebox{\\columnwidth}{!}{%\n"
            "\t\\begin{tabular}{l l r r r r r r}\n\t\t\\toprule\n"
            "\t\t & & & \\multicolumn{3}{c}{\\textbf{Own graph or recovered}}"
            " & \\multicolumn{2}{c}{\\textbf{Read back}} \\\\\n"
            "\t\t\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
            "\t\t\\textbf{System} & \\textbf{Wts.} & $n$ & \\textbf{Edge F1} & \\textbf{Score}"
            " & \\textbf{Node F1} & \\textbf{Score} & \\textbf{\\% ceil.} \\\\\n"
            "\t\t\\midrule\n")
    # the best cell of a column is the best of the values as the table prints them,
    # so a tie at three decimals is shown as a tie and not broken invisibly
    columns = ("edge_f1", "score", "node_f1", "readback")
    best = {c: max(round(v, 3) for v in table[c]) for c in columns}
    weights, body, shipped = access(), "", False
    for row in table.itertuples():
        if not row.ours and not shipped:
            body += "\t\t\\midrule\n"   # ours above, the systems the benchmark ships below
            shipped = True
        cells = []
        for column in columns:
            value = round(getattr(row, column), 3)
            printed = f"{value:.3f}"
            cells.append(f"\\textbf{{{printed}}}" if value == best[column] else printed)
        name = (f"\\rowcolor{{cvprblue!10}}{row.system}" if row.ours else SHORT[row.key])
        body += (f"\t\t{name} & {weights[row.key]} & {row.n} & {cells[0]} & {cells[1]}"
                 f" & {cells[2]} & {cells[3]} & {row.readback_pct_ceiling:.1f} \\\\\n")
    return head + body + "\t\t\\bottomrule\n\t\\end{tabular}}\n"


def latex_pairs(pair):
    head = ("% generated by make_table_systems.py --latex; do not edit by hand\n"
            "\\begin{table}[t]\n\t\\centering\n\t\\fontsize{6.6}{7.9}\\selectfont\n"
            "\t\\setlength{\\tabcolsep}{3pt}\n\t\\resizebox{\\columnwidth}{!}{%\n"
            "\t\\begin{tabular}{l l r l r l r l}\n\t\t\\toprule\n"
            "\t\t & & \\multicolumn{2}{c}{\\textbf{Qwen}} & \\multicolumn{2}{c}{\\textbf{Gemma}}"
            " & \\multicolumn{2}{c}{\\textbf{Ministral}} \\\\\n"
            "\t\t\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "\t\t\\textbf{Shipped system} & \\textbf{Wts.} & $\\Delta$ & & $\\Delta$ & & $\\Delta$ & \\\\\n"
            "\t\t\\midrule\n")
    weights, body, seen = access(), "", None
    for key in sorted(ORDER, key=lambda k: (access()[k] != "closed", ORDER.index(k))):
        if seen is not None and weights[key] != seen:
            body += "\t\t\\midrule\n"   # closed-weight systems above, open below
        seen = weights[key]
        cells = []
        for model in MODELS:
            row = pair[(pair.model == model) & (pair.against == key)].iloc[0]
            mark = "" if row.clears_its_resolution else "$^{\\dagger}$"
            cells.append(f"${row['diff']:+.3f}$ & {row['verdict']}{mark}")
        body += f"\t\t{SHORT[key]} & {weights[key]} & " + " & ".join(cells) + " \\\\\n"
    return head + body + "\t\t\\bottomrule\n\t\\end{tabular}}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true",
                        help="write the LaTeX of both tables beside the CSVs")
    parser.add_argument("--out", help="write the printed report here as well")
    parser.add_argument("--no-csv", action="store_true", help="print only")
    args = parser.parse_args()

    pipe = load("pipeline_items.csv")
    pub = load("published_systems_sciflow.csv")
    rb = load("readback_sciflow.csv")

    table = summary(pipe, pub, rb)
    pair = pairs(rb)
    text = report(table, pair)
    print(text)

    if not args.no_csv:
        TABLES.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLES / "published_systems_summary.csv", index=False)
        pair.to_csv(TABLES / "published_systems_readback.csv", index=False)
        print(f"\nwrote {TABLES / 'published_systems_summary.csv'}")
        print(f"wrote {TABLES / 'published_systems_readback.csv'}")
    if args.latex:
        (TABLES / "table_sciflow.tex").write_text(latex_summary(table), encoding="utf8")
        (TABLES / "table_readback.tex").write_text(latex_pairs(pair), encoding="utf8")
        print(f"wrote {TABLES / 'table_sciflow.tex'}")
        print(f"wrote {TABLES / 'table_readback.tex'}")
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
