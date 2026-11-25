# Data and results behind the paper

Everything measured for the paper: per-item measurements as CSV, a script that
rebuilds every data table from them, the printed outputs, all run logs, and the
complete raw runs. The scripts that read the paper expect its two LaTeX files,
`main.tex` and its appendix `appendix.tex`, beside this folder. Pointers of the
form "§n" and "Table S…" in the analysis tables below follow the numbering of
an earlier, longer version of the paper and its supplement.

## Quick start

    cd scripts
    python3 make_tables.py              # every table, in paper order
    python3 legibility.py --selfcheck   # the measure, scoring the rating study
    python3 make_tables.py --table 2 8  # only some tables
    python3 checks_variance.py          # where the score's variance is
    python3 checks_new.py               # the seven analyses that need no new run
    python3 checks_claims.py            # what the data say about each claim (K1-K8)
    python3 checks_attribution.py       # where the legibility difference comes from (A1-A7)
    python3 checks_robustness.py          # the eight analyses R1-R8 (below)
    python3 checks_inference.py         # I1-I7: the tests the design supports
    python3 checks_novelty.py           # V1-V8: what transfers beyond this pipeline
    python3 checks_validity.py          # what each instrument supports
    python3 make_table_audit.py --latex # the audit at a glance (Table 2)
    python3 make_table_systems.py --latex # the shipped-systems comparison (Table 2, Table S7)
    python3 check_text_numbers.py       # trace every number in both LaTeX files
    python3 make_fig_panels.py --data ../data --out ../../img      # img/appendix_panels.pdf
    python3 make_fig_resolution.py --data ../data --out ../results/figures
    python3 make_fig_capacity.py --data ../data --out ../../img    # img/capacity_frontier.pdf

Needs Python 3.9+ with pandas, numpy, scipy and matplotlib. No GPU and no other
code. The output is identical to `results/tables_from_data.txt` and to the files
in `results/printed_outputs/`.

## Folders

| Folder | What |
|---|---|
| `data/` | one CSV per kind of measurement, one row per item (below) |
| `scripts/make_tables.py` | rebuilds the fourteen data tables behind the paper from `data/`; they are numbered as this script numbers them, not as the paper numbers its own tables |
| `scripts/legibility.py` | the measure on its own: all three verdicts, the fitted index and the engine-choice rule, in one file with no dependencies (below) |
| `scripts/make_norms.py` | writes the two normalisation tables `legibility.py` reads |
| `scripts/checks_*.py` | the fourteen analysis groups (below); each takes `--analysis` and `--out`, and most also `--latex`, `--csv-dir` or both |
| `scripts/make_fig_*.py` | the figures of the paper and the released panels, each with the CSVs behind them |
| `scripts/make_table_audit.py` | builds Table 2, *the audit at a glance*, and `results/tables/audit_summary.csv` |
| `scripts/check_text_numbers.py` | traces every numeric token of `main.tex` and `appendix.tex` back to `data/`, `results/tables/` and `results/printed_outputs/` |
| `scripts/export_csv_from_raw.py` | rebuilds `data/` from the raw zips |
| `results/tables_from_data.txt` | the output of `make_tables.py` |
| LaTeX tables | not shipped: `checks_*.py --latex` writes them on demand, and `main.tex` and `appendix.tex` carry their tables inline |
| `results/tables/` | every table of the analysis groups as CSV, with `index_norms.csv` and `engine_rule_norms.csv` for `legibility.py` |
| `results/figures/` | the released panels and the CSVs behind them |
| `results/printed_outputs/` | the printed output of every command behind the paper |
| `logs/logs.zip` | the log of every run (175 logs) |
| `raw/` | the complete runs: every extracted graph, layout and model answer, 5 zips, 118{,}511 files |

## The CSVs

All scores are copied unchanged from the run files. A failed item has an empty
score; the paper counts it as zero, and `make_tables.py` does the same.

| File | One row per | Key columns | Used in |
|---|---|---|---|
| `pipeline_items.csv` | item × run of our pipeline | `variant`, `corpus`, `model`, `seed`, `item`, `domain`; scores `node_f1`, `edge_f1`, `score`; structural distances `strict_*`; layout measures `layout_*`; `layout_valid` = strict legible | Tables 1, 4, S6, S7 |
| `dot_items.csv` | item × DOT baseline run | `variant` (`dot` = short prompt, `dot_matched` = our prompts), `corpus`, `model`, `item`, scores | Table 1 |
| `published_systems_sciflow.csv` | SciFlow-Bench paper × shipped system | `system`, `item`, scores (graphs recovered by the benchmark) | Table S6 |
| `readback_sciflow.csv` | SciFlow-Bench paper × system, every figure read back by one reader | `system` (`ours` = seed 0; `gold_graph_ceiling` = gold graphs drawn by our engine), `model`, scores | Table S6, ceiling |
| `judge_paperbanana.csv` | PaperBananaBench case × system × seed × image order × judge | `judge`, `system`, `model`, `seed`, `order`; per dimension the verdict and its points (100 win, 0 loss, 50 tie) | Table 1; §7 |
| `layout_items.csv` | figure × layout engine or configuration | `source` (`graphviz`, `graphviz_outside_development`, `layout_ablation`), `config`, `model`, `run`; layout measures, `valid` = strict legible | Tables S2, S3; §5.2 |
| `placement_items.csv` | paper × seed × condition | `condition` (`ours` = our engine places the boxes, `model` = the model places them), layout measures, scores | §5.1 |
| `human_stimuli.csv` | rated image | `image`, `paper_id`, `condition` (`ours`, `graphviz`, `model`), its layout measures `report_*` | Table S5 |
| `human_sets.csv` | set of four images shown together | `set`, `image_1`–`image_4` | Table S5 |
| `human_ratings.csv` | rater × set | `rater` (R1–R3), `set`, `best`, `worst`, `seconds` | Table S5 |
| `threshold_sweep.csv` | corpus × model × matching threshold | `threshold` (0.5, 0.6, 0.7), `ours`, `dot`, `diff`, `r`, `p_holm`, `verdict` | Appendix S2 |

`threshold_sweep.csv` is the one file here not derived from the per-item CSVs:
rescoring at a different cosine threshold needs the sentence embeddings, so the
numbers are copied from the run that produced
`results/printed_outputs/analysis_a6_threshold_errors.txt`, which is the output
of that run and is released beside it.

### Variants and names

- `evaluated`: the pipeline the paper evaluates (no coverage call; control-flow
  prompt for program text). `coverage`: the same with the coverage call.
  `coverage_no_control_flow` (FlowVQA only): coverage call, without the
  control-flow prompt. `ablation_one_greedy_graph`, `ablation_unconstrained`:
  seed 0 with best-of-2 or constrained decoding removed.
- Models: Qwen = Qwen2.5-VL-32B-Instruct, Gemma = Gemma 4 26B-A4B-it,
  Ministral = Ministral 3 14B Instruct. Corpora: `sciflow` (483 papers),
  `flowvqa` (300 test items; `domain` = wiki, instruct, code), `paperbanana`
  (292 diagram test cases).
- `score` is SciFlow-Bench's graph score, 0.4 × node F1 + 0.6 × edge F1.
  `strict_*` columns score the same graphs with exact label matching;
  `strict_normalised_ged` is normalised graph edit distance, lower is better.
- Layout measures: `edge_edge_crossings` = arrow crossings,
  `edge_node_crossings` = arrows through a box, `node_overlap_pairs` =
  overlapping box pairs, `label_collisions`, `edge_label_crossings` = arrows
  through a label, `area`, `aspect_deviation` = distance from the target page
  shape, `valid` = strict legible.
- Layout `config`: `ours`; `dot_tb_spline`, `dot_lr_spline`, `dot_tb_ortho`,
  `dot_lr_ortho` = Graphviz top-down / left-right with curved / orthogonal
  edges; `full`, `no_*`, `none` = the layout ablation (`none` = all six off).

## Inference: the tests the design supports (`checks_inference.py`, I1-I7)

Added with the current version of the paper. **No point estimate in the paper
changes.** Each analysis replaces a test whose assumptions this design does not
meet with one that matches it, and the paper now reports the second:

| | What was wrong | What replaces it |
|---|---|---|
| `I1` | the 6,012 figures of §5.2 are 1,064 source graphs drawn by up to 3 models × 3 seeds, so a McNemar p over the pooled rows treats dependent figures as independent (variance inflation 1.7-1.8×, about 3,500 independent figures) | bootstrap resampling whole source graphs, plus a cluster-level sign-flip permutation p (floor 1e-4 at 10,000 draws): +21.0 points, 95% CI [+19.5, +22.6] |
| `I2` | the nine per-setting legibility differences carried one pooled p | a cluster-robust interval per setting (narrowest +13.3 to +22.7) |
| `I3` | a *tied* verdict read as a failure to reject | two one-sided tests at α=0.05; the margin the 90% interval excludes (0.010 score on SciFlow-Bench, 0.020-0.025 on FlowVQA, 6 judge points), and both judges' own intervals on PaperBananaBench |
| `I4` | Wilcoxon's treatment of zero paired differences was never checked | discard, zsplit and Pratt agree throughout; no verdict depends on it |
| `I5` | 72 rated images and 216 choices are 24 source graphs; a Spearman over 48 images and a Fisher p over 216 choices assume independence | cluster intervals over source graphs; Mantel-Haenszel OR stratified by rating set (10.4); a within-set randomisation test (p < 1e-4); and the paired within-graph test for the 48 images two engines drew (+0.32, -0.08, -0.16; sign-flip p = 0.26, 0.98, 0.81) |
| `I6` | the capacity (LD50) had no interval | cluster bootstrap: 52.7 [49.0, 57.0] boxes against 23.3 [22.3, 24.3], difference +29.4 [+25.6, +33.4] |
| `I7` | "independent" asserted from a small rho at large n | equivalence bounds. Score vs strict verdict: rho +0.003 at fixed size, CI [-0.025, +0.033] (unconditionally +0.01). Crossings vs label collisions: rho +0.02 is *not* zero at 30,051 layouts (p = 9e-5), CI [+0.01, +0.04], so negligible rather than independent |

## What transfers (`checks_novelty.py`, V1-V8)

The seven results of §8, each written as a statement another system can use.
`--csv-dir ../results/tables` writes the tables below. `V7` and `V8` are new in
this version and, like the rest of this file, add no run: both read the per-item
CSVs only.

| Analysis | Result | Table |
|---|---|---|
| `V1` | the released figures calibrate any model that reads diagrams, because their geometry is known exactly: the reader returns 1.21-1.30 boxes and 1.66-1.85 arrows per one drawn, every ratio's interval clear of 1.0 | `reader_audit.csv` |
| `V2` | which engine a graph needs is predictable before anything is drawn, and the two-engine ceiling bounds what acting on it is worth (90.0% against 86.9%, headroom 3.1 points). `dispatch_auc.csv` holds the three-feature form of the rule (boxes, loops, nesting: AUC 0.75-0.79 for `dot`, 0.75-0.85 for ours); the five-feature rule the paper reports and `legibility.py` ships (AUC 0.75-0.81 and 0.82-0.88) is `checks_robustness.py --analysis R3` | `dispatch_auc.csv`, `dispatch_ceiling.csv` |
| `V3` | defect counting is a free best-of-n signal: +3 to +19 points of strict legibility, every content interval containing zero | `selection_gain.csv` |
| `V4` | what these benchmarks can resolve: the score's variance is 83-86% item and about 1% model, and the judge resolves 6.3 points over 292 items, 10.8 over 100 and 15.3 over 50 | `power.csv` |
| `V5` | a reversed arrow — a relation found and pointed the wrong way — is an error node-and-edge F1 cannot separate from a missing relation: 2.1% of SciFlow-Bench papers carry one [1.7, 2.5] against 13.5% [11.9, 15.1] of FlowVQA prose items, 1.8x as often per gold edge on prose as on code (p = 2.5e-14) and 5.0% against 2.8% on graphs that loop back; reversed edges are 0.5-3.4% of the relations the extractor located, so scoring direction-insensitively moves edge F1 by +0.001 to +0.010. That gap is the share of the deficit which is a direction error, and reporting it costs nothing. Only predicted graphs carry the column, so it is measured for our own extractor and not for the image baselines (see also `checks_bounds.py --analysis E4`) | `direction_errors.csv` |
| `V7` | a judged verdict belongs to its dimensions, as a geometric verdict belongs to its criteria. Attributed, the PaperBananaBench comparison is a *conciseness* comparison: that dimension carries -24.8 to -34.8 of Gemma's -39.2 to -17.1 points and more than the whole of Ministral's, it is the only one of the four whose sign survives the change of judge for all three models, and the only one that change barely moves (0.1-10.0 points, against 15.1-22.0 for the number built from all four). The dimension that names legibility calls the two figures equally good on 65.1% of judgements, reverses for every model between judges, and responds to how much a figure holds (rho = -0.14 to -0.45 against the box count we drew, inside one system) rather than to its geometry (C1). Holm-corrected inside the four dimensions of one judge and model | `judge_dimension_attribution.csv`, `judge_dimension_stability.csv`, `judge_dimension_tie_rates.csv`, `judge_dimension_vs_size.csv` |
| `V8` | the geometric signal is reproducible in the aggregate and not per figure, and that is what selection spends. Over three seeds a setting's strictly legible share moves 0.7-4.7 points while an individual figure's verdict changes on 2.7-53.7% of them; the worst of three seeds sits up to 27.0 points below the mean of its own setting and the best 1.0-26.7 above it. Across the six settings where the paper measures what one extra sample buys, the share of figures whose verdict a seed changes ranks the settings as the gain does (rho = +0.83, p = 0.042), so best-of-n can be priced by resampling a few items before paying for it. The same headroom is there on content (+0.008 to +0.046 of score) and cannot be spent, because a defect is countable per figure without a reference and a missing relation is not | `selection_reliability.csv`, `selection_headroom.csv`, `selection_content_ceiling.csv` |
| `V6` | the corpus as a calibration set, tested rather than asserted: the map from what the read-back reader returns to what was drawn, fitted on half the papers and scored on the other half, halves the error on arrows (MAE 11.8-18.4 to 4.1-6.3) but has slope 0.26-0.45 and R² 0.26-0.62, against 5.5-7.4 for predicting the corpus mean; a correction fitted on one model's figures is worse on another's (8.6 arrows against 6.6) than that model's own. The reader can be bounded on such a corpus and not repaired by it, and the transfer function belongs to the figure set, not to the reader | `reader_calibration.csv`, `reader_calibration_transfer.csv` |

## The measure on its own

`scripts/legibility.py` is the part of this repository meant to be reused
elsewhere. It needs no packages, no GPU and nothing from this pipeline except
the numbers a system already has if it knows where its boxes, routes and labels
are, so it applies to TikZ, SVG, mxGraph or any other vector output:

    python3 legibility.py --selfcheck                     # reproduces the rating study
    python3 legibility.py --score measures.csv --out scored.csv
    python3 legibility.py --engine --boxes 24 --arrows 31 --loops 1 --nested 0

It provides three things, with weights read from `results/tables/` so that it
cannot disagree with the paper:

- **The three verdicts.** *Lenient*: no overlapping box pair, no arrow through a
  box, no route that doubles back. *Strict*: in addition no label collision, no
  arrow through a label and no stacked routes. *Label-blind*: strict without
  those two label criteria. The strict formula reproduces the released `valid`
  column on 72 of 72 rated images. Report label-blind beside strict: it is what
  separates a layout difference from a label-placement difference, and in this
  paper it is the whole difference between the two engines (§5.2, A2).
- **The fitted index** (`index_weights.csv`, `index_norms.csv`), on `log1p`,
  z-scored inputs, in a four-measure and a ten-measure form. `--selfcheck`
  reports its rank correlation with the raters, and the single counts for
  comparison: label collisions +0.75, arrow crossings +0.26. The paper's 0.76
  and 0.77 are the leave-one-source-graph-out values, computed by
  `checks_new.py --analysis N5`. Every one of those numbers is over a panel
  whose three conditions include the model placing its own boxes, which is what
  validates these counts as a detector of a figure nobody can read. Over the 48
  images the two engines drew none of them tracks the raters, so `--selfcheck`
  prints that column beside each correlation: the weights are not a ranking of
  two drawn layouts (`checks_claims.py --analysis K6 K7`).
- **The engine-choice rule** (`engine_rule_weights.csv`,
  `engine_rule_norms.csv`): the probability that Graphviz `dot` draws a graph
  illegibly, from box count, arrow count, density, whether the graph loops back
  and whether it has nested containers — all known before anything is drawn
  (AUC 0.75–0.81 on a held-out corpus, `checks_robustness.py --analysis R3`).

Beside scoring a drawing, the counts work as a reward or as an acceptance test:
best-of-*n* by defect count raised the strictly legible share by 3.0 to 18.8
points here at the cost of one extra sample, and the index is a single number to
optimise against.

## The raw runs

| Zip | Contents |
|---|---|
| `raw_runs_sciflow.zip` | our pipeline on SciFlow-Bench, every model and seed, with and without the coverage call |
| `raw_runs_flowvqa_paperbanana.zip` | the same on FlowVQA and PaperBananaBench |
| `raw_baselines_judge_readback.zip` | DOT baselines (short prompt and our prompts), shipped systems' scores, read-back, both judges |
| `raw_layout_ablations_human.zip` | Graphviz layouts, layout, placement and pipeline ablations, robustness analyses, human study, development studies, and `README.md`, which describes every run folder |
| `raw_development_runs.zip` | development runs and an earlier pipeline ablation that is not in the paper |

In each run folder, `full.jsonl` has one record per item and
`artifacts/<item>/` holds the extracted graph (`figure.json`) and its layout
(`layout.json`). Server paths in logs are replaced by `<home>`, `<server>`
and `<user>`. The corpora themselves are not included; they come from their
authors.

---

## The content comparison in detail, and the post-hoc register

The paper's §5.2 states what the content measurements support and points here
for the rest. The content comparison does two things: it bounds the layout claim
of §5.2 -- the legibility difference was not bought with content -- and it states
the pipeline's standing among the systems the benchmarks ship, which is the
generation half of the paper's title. The per-setting detail is maintained here,
once, with the tables it is computed from. No number below is new; each is
produced by the script named beside it.

**Against the seven systems SciFlow-Bench ships.** Table 2 of the paper and Table S7 of the supplement
are regenerated from `published_systems_sciflow.csv` and `readback_sciflow.csv`
by `make_table_systems.py`, which writes `published_systems_summary.csv` and
`published_systems_readback.csv` beside the other tables. Under the symmetric
read-back, where every figure including ours is drawn and recovered by one
reader, ours + Ministral is ahead of six of the seven and behind Gemini 3 Pro
Image, ours + Gemma is ahead of five and tied with Gemini 2.5 Flash Image, and
ours + Qwen is ahead of three; the verdicts are this paper's protocol throughout
(paired Wilcoxon, Holm-corrected over the seven systems within a model, verdict
from the sign of r at alpha = 0.05). Both tables carry a `Wts.` column, read
from `system_access.csv`, recording whether each system's weights are public or
it is reachable only through a provider API; the Graphviz row is an open
renderer driven by a closed-weight model. Grouped that way, all three of our
models are ahead of all three open-weight systems the benchmark ships (eight of
those nine differences clear their sample's resolution; the exception is ours +
Qwen over Qwen-Image), and the only system ahead of ours is closed-weight. Two
of the three open-weight baselines recover under 3% of the gold-graph ceiling,
so S5.4 reports the open-weight lead as resting on Qwen-Image. Three bounds are
printed with the ordering. Six of
the 21 paired differences do not clear the 0.009-0.017 their own samples detect
at 80% power, among them two of the six margins that put Ministral ahead
(`checks_claims.py --analysis K8`). The column occupies the bottom fifth of the
metric's range: ours + Ministral reaches 17.4% of the 0.611 the gold graphs
reach under the same protocol. And the reader behind every number in it returns
1.21-1.30 boxes and 1.66-1.85 arrows per one drawn (`checks_new.py --analysis
N2`). The left half of Table 2, the benchmark's own protocol, scores our graph
directly and every other system's from a graph recovered from its pixels; that
is a difference in protocol and not in quality, so the paper reads no result
from it. Everything in both tables comes from our own metric implementation and
our own open reader, so neither is comparable to a published figure.

**FlowVQA is where the content metric has room, so the paper reads it first.**
Its edge half, carrying 0.6 of the score's weight, is exactly zero on at most 3%
of items against 60-76% on SciFlow-Bench. On the 100 code items ours is ahead
for every model, by +0.09 to +0.23 (Holm, p <= 1.4e-6), reaching 0.81-0.84 where
the baseline as the benchmark runs it reaches 0.61-0.72; on prose it is ahead in
two of six settings and no system passes 0.56, so what a text states about its
own control flow sets the level both can reach. The two Holm-corrected wins,
+0.058 and +0.117, clear that corpus's own resolution; Gemma's +0.005 does not
and is reported as tied (`checks_variance.py`, `power.csv`).

**SciFlow-Bench runs at the floor of the same metric, and the paper reads
nothing from its level.** Every system compared scores 0.10-0.13 there, its edge
half is exactly zero on most items, and no system reaches a fifth of the 0.611
the gold graphs themselves reach (`checks_claims.py --analysis K3`). A
difference of +0.023 inside the bottom fifth of an instrument's range is not
evidence of a content advantage, and two of the six score comparisons return the
wrong sign on 17.8% and 31.0% of bootstrap resamples even at the full sample,
which is why Table 1 of the paper marks the three SciFlow-Bench settings as
pre-specified, reported as run, and carrying no content claim. What the corpus shows
independently of the metric's level is that Ministral's DOT fails to parse on 74
papers even after repair, while ours returns a figure for every one.

**Neither gain is a matter of drawing more boxes.** Node and edge precision is
higher on SciFlow-Bench for all three models and recall higher on FlowVQA for
all three, and where the two drew within two boxes of each other ours is ahead
in seven of twelve settings and behind in none (`make_tables.py --table 3`).
Taking the per-item score as a difficulty measure, two of our models agree on
which texts are hard at mean rho = 0.87 and 0.82 against 0.75 and 0.57 for the
same models writing DOT (`item_difficulty.csv`).

**The prompt-matched rule** was fixed before the runs: it credits the
representation if ours is ahead in at least five of nine settings. On
SciFlow-Bench our prompt does not help DOT and ours stays ahead with every model
even on the papers where matched DOT parses (r = 0.29-0.37); on FlowVQA the
prompt carries the gain, and where matched DOT parses the two are tied for all
three models, so what remains there is that ours never fails to parse
(`attribution_matched.csv`, `attribution_matched_cells.csv`).

**The shipped-systems table has two halves and they mean different things.** One
caveat governs all of it: every number comes from our own metric implementation
and our own open reader, so it is internally comparable and *not* comparable to
any published figure. Under the benchmark's own protocol our score comes from
the graph we drew and every other system's from one recovered from its pixels --
a difference in protocol, not in quality -- so the paper reads no result from
that half at all. The symmetric half, where every figure including ours is drawn
and read back by one reader, is the one §5 reports, and even there the
ordering is tight: six of its twenty-one paired differences do not clear what
the sample resolves, ours among them (`checks_claims.py --analysis K8`,
`published_systems_sciflow.csv`, `readback_sciflow.csv`).

**The register of post-hoc analyses** lives in this file and not in the paper's
appendix, which now points here. Seventy-five analyses were specified after the
test runs; every one is named and computed by a `checks_*.py` script below, all
read the per-item CSVs and nothing else, and all are reported as descriptive.
`checks_claims.py --analysis K5 K7` regenerates Table S4 of the supplement from the
paper's own source, so the register and the paper cannot drift apart.

---

# The analysis register

Every analysis below was specified **after** the test runs. All of them read the
per-item CSVs above and nothing else: none adds a run, a condition or a model
call, and none decides a reported verdict. All are reported as descriptive.
Where one of them is a family of tests read as a single finding -- which of six
criteria carries a difference, which of four judged dimensions does, in which of
five size bins it lives, which of twenty-one measures a rater follows --
`checks_claims.py --analysis K7` corrects it inside that family and reports
which members survive; V7 and A7, new in this version, are corrected inside
their own families the same way. The confirmatory comparisons —
the nine content verdicts, the nine legibility verdicts and the ablations — were
fixed before the test runs and are Holm-corrected within each model's
comparisons on one metric; they are the only tests a verdict in the paper rests
on. Table S4 of the supplement (`table:claims`) marks which of the paper's headline
numbers come from the confirmatory group and which are descriptive.

Each script accepts `--analysis` to run a subset, `--out` to write its printed
output, and `--latex` to write its tables. `make_fig_dose.py` takes its two
paths positionally, the second being the output PDF:

    python3 make_fig_dose.py ../data ../../img/dose_response.pdf

## `checks_validity.py` — what each instrument supports

    python3 checks_validity.py --analysis W3
    python3 checks_validity.py --latex ../results/tables_validity.tex

| Check | Question | Paper |
|---|---|---|
| W1 headroom | how high the SciFlow-Bench score can go at all, and how far a gain is from the seed-to-seed spread | §7 |
| W2 equivalence | for each verdict reported as *tied*, the largest difference the data still allow (bootstrap TOST bound) | §6 |
| W3 judge audit | how far the PaperBananaBench judge moves with image order, how far the two judges agree, and whether the second judge favours the graphs its own checkpoint extracted (it does not) | §7 |
| W4 study precision | what the best-worst study with 24 graphs and three raters could and could not detect | §7 |

The ceiling of W1 is the `gold_graph_ceiling` system in `readback_sciflow.csv`:
the gold graphs drawn by our engine and read back by the same reader, that is
the score this protocol returns for a perfect graph.

## `checks_criteria.py` — the graph-drawing criteria

| Check | Question | Paper |
|---|---|---|
| B1 angle | at what angle each engine's arrows cross, and what the crossings weigh once the angle is counted | §5.2 |
| B2 criteria | the criteria of §4 that Table S3 does not list: bends, separation, container titles, arrow length, ink | §5.2 |
| B3 loops, nesting | is the legibility gap wider on graphs that loop back or nest | §5.2 |
| B4 sources | FlowVQA by source: where the pipeline is ahead of DOT and by how much | §5.2 |
| B5 precision | precision and recall against the same model writing DOT | the content detail below |
| B6 size | the judge against figure size over twelve systems, and the fit that separates size from representation | §6, `results/figures/judge_panels.pdf` |
| B7 raters | which geometric difference predicts which layout the raters preferred | §7, Table S5 |
| B8 second metric | why the `strict_*` DiagramEval-style columns are **not** reported | released only |

B8 records a negative result worth knowing before reusing those columns:
`strict_path_f1` is 1.0 by convention when a pair of graphs has no comparable
path, which is 31% of SciFlow-Bench items, and `strict_diagrameval_score` is the
mean of node and path F1, so it inherits that default. The apparent disagreement
between that family and the benchmark's score (per-item Spearman −0.22 on
SciFlow-Bench) disappears when incomparable items are dropped (+0.21). The same
columns do support one check that holds: scoring our own graphs with exact label
matching instead of the benchmark's 0.5 cosine threshold ranks items the same
way (Spearman 0.75–0.83). `strict_normalised_ged` is not affected by the path
convention and is reported in the paper (H4).

## `checks_instruments.py` — what the instruments respond to

| Check | Question | Paper |
|---|---|---|
| C1 judge | does the PaperBananaBench judge's own readability dimension respond to measurable readability, once figure size is held fixed | §6, Table S5 |
| C2 outside | is the layout gap as large on the two corpora no layout choice was made on | §5.2 |
| C3 size | how legibility of both engines falls with the number of boxes, and where the two separate | §5.2, `img/legibility_size.pdf` |
| C4 reader | does the read-back reader recover legible figures better than illegible ones | §6, Table S8 |
| C5 independence | do the benchmark score and the strict verdict rank the same figures | §6, Table 2 |
| C6 areas | is the SciFlow-Bench gain carried by one of the five areas | §5 |

C1, C4 and C5 are correlations within our own figures: they bound what each
instrument can detect, and they are not a comparison between systems. C6 is
printed by the script but has no table in the paper, for space.

## `checks_invariants.py` — the guarantee, and the raters

| Check | Question | Paper |
|---|---|---|
| D1 invariant | over every figure the engine drew, does the lenient verdict ever fail | abstract, §3.1, §5.1 |
| D2 replication | does SciFlow-Bench's inverse-parsing step reproduce with a different reader | §6 |
| D3 power | what difference this benchmark can resolve, and at what sample size | §4, Appendix S3 |
| D4 decision time | do the raters' response times carry the same signal as their choices | §7, Appendix S2 |
| D5 prediction | does the strict verdict predict which image a rater picks | §7, §6 |

D1 counts the three *lenient* columns over every row of `pipeline_items.csv` and
`placement_items.csv` where the layout was measured. That is 28,388 figures, of
which our layout code drew 27,493: the other 895 are the half of the placement
ablation in which the model places the boxes itself, and 765 of those overlap a
box. The invariant is stated over the 27,493, and `--csv-dir` writes both
denominators to `results/tables/invariant_denominator.csv`. D1 also reads
`layout_items.csv` for the comparison with Graphviz on identical graphs. D2
pairs `published_systems_sciflow.csv` against `readback_sciflow.csv` on paper and
area together. D3 uses the per-seed rows of `pipeline_items.csv` against
`dot_items.csv`. D4 uses the `seconds` column of `human_ratings.csv`. D5
cross-tabulates each `best` and `worst` choice against `report_valid` in
`human_stimuli.csv`.

## `checks_bounds.py` — bounds on each measurement

| Check | Question | Paper |
|---|---|---|
| E1 bound | what bounds the SciFlow-Bench score, and why every system bunches | §7 |
| E2 criterion | which graph-drawing criterion predicts human judgement | §7, Table S5 |
| E3 tail | what unconstrained generation costs in the tail of wall clock | §5.2 |
| E4 direction | arrow direction as an error the benchmark's metric does not separate | released only |
| E5 both | content and legibility together, over the nine settings | Table 1 |
| E6 selection | what choosing between two samples by geometry buys | §8 |
| E7 one order | what a single image order would have changed for the judge | §7 |

## `checks_engine.py` — choosing an engine

| Check | Question | Paper |
|---|---|---|
| F1 size-matched | is the content gain a size effect, as the PaperBananaBench loss is | §5.2, abstract |
| F2 criteria | do arrow crossings carry any information about label occlusion once graph size is held fixed | §7, `img/criteria_ranking.pdf` |
| F3 oracle | how far Graphviz is behind when given the best of its four configurations per figure | §5.2 |
| F4 loop/container | how the legibility gap splits by whether the graph loops back and whether it nests | §5.2 |

F1 pairs `pipeline_items.csv` (mean over seeds) against `dot_items.csv` on
`pred_nodes` and keeps the items where the two differ by at most two boxes; the
tolerance sweep from 0 to 5 boxes is printed with it. F2 uses the `graphviz` and
`graphviz_outside_development` rows of `layout_items.csv`, pooled over both
engines. F3 and F4 pivot the same rows over the four `dot` configurations, so
they use the 6,007 figures for which all four are present.

## `checks_judge.py` — the judge, and selection

| Check | Question | Paper |
|---|---|---|
| G1 paraphrase | how much of the node matching the benchmark reports comes from its 0.5 cosine threshold rather than from the label | Appendix S2, abstract |
| G2 judge dimensions | does the judge's position bias have one sign across its four dimensions, and which dimension do two judges agree on | §6, §7 |
| G3 selector ceiling | what any policy choosing between our engine and `dot` per figure could reach | §8, abstract |
| G4 engine rule | does F4's engine-choice rule hold on a corpus held out of the fit | §8 |

G1 pairs the `node_*` columns of `pipeline_items.csv` against their `strict_*`
counterparts. G2 pivots `judge_paperbanana.csv` over the `order` column per
dimension and cross-tabulates the two judges on seed 0. G3 and G4 pivot the
`graphviz` rows of `layout_items.csv` over the four `dot` configurations;
G4's logistic fit is refitted three times, each time holding out one corpus.
`--latex` writes `results/tables_judge.tex`, which the paper does **not**
include — the appendix is at its page limit, and the body states the ranges.

`make_fig_judge.py` draws four released panels from the same CSVs and writes
`results/figures/judge_panels.{pdf,png}` with the four CSVs behind them.

## `checks_variance.py` — variance, resolution, metric choice

    python3 checks_variance.py --analysis H1 H4
    python3 checks_variance.py --latex ../results/tables_resolution.tex
    python3 checks_variance.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| H1 variance | how much of the benchmark score's variance belongs to the item, the model, their interaction and the decoding seed | §7, Table S7a |
| H2 resolution | the smallest unpaired difference a given sample size can resolve, and how often a small true difference comes out with the wrong sign | §7, abstract |
| H3 difficulty | is item difficulty a property of the text or of the model, for our pipeline and for the same models writing DOT | §7, Table S7b |
| H4 edit distance | normalised graph edit distance against the weighted score on the same graphs: verdict agreement, effect size, size control | §7, Table S7c |
| H5 exchange rate | what one extra box of detail buys on the score and costs on legibility, inside one representation | §5.2, abstract |

H1 decomposes the score over item × model × seed within each corpus; the
components are mean squared deviations. H2 derives the unpaired minimum
detectable difference from the item-level sd and bootstraps the sign-error rate
of each paired comparison against `dot_items.csv`. H3 drops every item on which
any model's DOT answer failed to parse, so a parse failure scored zero cannot
drive the correlation. H4 is paired per item throughout, so the gold graph and
its size are identical on both sides of every comparison; it also reports the
rank-partial correlation with gold box count held fixed. H5 pairs the
`coverage` variant against `evaluated` per item, and repeats the comparison over
all 27 corpus × model × variant cells.

`make_fig_resolution.py` draws the three released panels from the same CSVs:

    python3 make_fig_resolution.py --data ../data --out ../results/figures

It writes `results/figures/resolution_panels.{pdf,png}` and the three CSVs
behind them, so each panel can be checked without rerunning anything. The
figure is a released artifact, not a float in the paper.

## `checks_robustness.py` — robustness and bounds

    python3 checks_robustness.py --analysis R1 R4
    python3 checks_robustness.py --latex ../results/tables_robustness.tex
    python3 checks_robustness.py --csv-dir ../results/tables

`--latex` writes `results/tables_robustness.tex` and, beside it, one
`results/table_<label>.tex` per table.

| Check | Question | Paper |
|---|---|---|
| R1 judge | how far the PaperBananaBench comparison moves when the judge is changed, and whether our figures reach parity with the authors' own | §7, Table 1 |
| R2 threshold | what the content comparison looks like as the benchmark's node-matching threshold tightens | Appendix S2 |
| R3 rule | which engine a figure needs, predicted from the graph alone and validated on a held-out corpus; and the ceiling of any two-engine policy | §8 |
| R4 raters | the legibility index fitted on two raters and scored on the third | §7, Table S5 |
| R5 criteria | which graph-drawing criterion tracks the raters -- over all 72 images and over the 48 the two engines drew, Holm within each -- and whether crossings and occlusion are substitutes | §7 |
| R6 bounds | what each instrument can resolve, collected in one table | §6, Table S4 |
| R7 placement | the same model placing the boxes of the graph it extracted | §5.1, Table S3 |
| R8 structure | where the legibility gap sits, by loop and by container; and the path-comparability default in the strict columns | §5.2, Table S2b; §8 |

R1 reads `judge_paperbanana.csv` and pairs each system per item, averaging the
two image orders. R3 and R8 build their wide table with `checks_judge.py`'s own
`_layout_wide`, so they cannot disagree with G3 and G4. R6 does not recompute
anything: it reads `results/tables/*.csv` written by `checks_variance.py` and
`checks_new.py`, so the summary table and the analyses behind it cannot drift.
R7 reports layout measures over the figures actually drawn, as D1 does, and the
scores over every item.

## `checks_claims.py` — what the data say about each claim

    python3 checks_claims.py --analysis K1 K2
    python3 checks_claims.py --latex ../results/tables_claims.tex
    python3 checks_claims.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| K1 guarantee | what the lenient invariant is worth beside Graphviz rather than beside the model | §5.1, Table S3 |
| K2 raters | how much of the rater result is the model-placed condition | §7, Table S5 |
| K3 floor | where the SciFlow-Bench score sits between its floor and its ceiling | §5.1, Table S6 |
| K4 resolve | every reported difference against what its instrument can resolve | §6, §7 |
| K5 register | every number in the abstract: instrument, pre-specified or not, bound | §6, Table S4 |
| K6 raters, by condition | which criterion tracks the raters once the model-placed condition is dropped | §7, Table S5 |
| K7 families | every descriptive family of tests, Holm-corrected inside its family | §4, §6 |
| K8 read-back | every read-back difference beside what that sample can resolve | Appendix S2, §6 |

These eight re-express measurements the other groups already compute, over the
comparison set that decides what a claim is worth: a guarantee is worth what
the alternative anyone would actually use does not give, a rater result is
worth what survives dropping the condition nobody would ship, and a score is
worth its distance from both ends of its own range.

- **K1** counts the three *lenient* criteria for Graphviz as well as for our
  engine on the identical graphs. `dot` with spline edges never overlaps a box
  either and satisfies all three on 99.1% of figures, which is why the paper
  reports them as a property of the construction and locates the advantage in
  the *strict* verdict (86.9% against 54.0%). The 6.5% of the placement
  ablation is a measurement of the model, not a layout baseline.
- **K2** repeats D5's odds ratio over the 48 images the two engines drew. The
  10.9 of the full panel is 0.52 (p=0.18) there, and 171 of the 216 *least
  legible* choices are the model-placed image; the raters put `dot` slightly
  ahead of our engine (115 *most legible* picks against 90). The study
  therefore validates the counts as a detector of unusable figures and settles
  nothing about either engine.
- **K3** reports both halves of the content metric and the share of its range
  in use. On SciFlow-Bench the edge half, 0.6 of the weight, is exactly zero on
  60–76% of items for every system compared, and the symmetric read-back puts
  our strongest model 1.5% of the ceiling ahead of the DOT convention, our
  middle model tied and our weakest 2.2% behind. The paper claims no content
  advantage.
- **K6** repeats the rater correlations over the 48 images the two engines drew,
  which is the sample a claim about two drawn layouts needs. Over all 72 images
  label collisions track the raters at rho=-0.75 and arrow crossings at -0.26;
  over the 48 the sign of label collisions reverses (+0.23, p=0.11), the strict
  verdict goes to -0.26 and the cross-validated index to 0.11. The ranking of
  the criteria is a ranking of how sharply each number separates the
  model-placed condition from the two drawn ones, and nothing more.
- **K7** corrects each descriptive family inside itself: four of the six strict
  criteria survive, five of five size bins, fourteen of twenty-one rater
  correlations over the full panel and none of the sixteen over the 48.
- **K8** puts each of the 21 paired read-back differences beside the smallest
  difference that sample can resolve at 80% power: six do not clear it, ours
  against the DOT convention among them, and the reader behind every number in
  that column returns 1.2-1.3 boxes and 1.7-1.9 arrows per one drawn.
- **K4** and **K5** are bookkeeping: every reported difference beside the
  resolution of its instrument, and every number in the abstract beside its
  instrument, its status (confirmatory or descriptive) and what bounds it.
  `--latex` writes `results/tables_claims.tex` and one `table_<label>.tex` per
  table; of those the paper carries `table_guarantee` and `table_claims`
  inline, and states the numbers of the other two in the text.

## `checks_attribution.py` — where the legibility difference comes from

    python3 checks_attribution.py --analysis A1 A2
    python3 checks_attribution.py --latex ../results/tables_attribution.tex
    python3 checks_attribution.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| A1 criteria | which of the six strict criteria carries the difference to `dot` | §5.2, Table S2a |
| A2 blind | the strict verdict with the two label criteria removed | §5.2, Table S2a |
| A3 engine | how much of the difference is the layout stages, not the schema | §5.2, Table S2b |
| A4 select | how much is choosing a candidate by the counts the verdict reports | §5.2, §8 |
| A5 size | the graph sizes the rater study covers, and where the difference lives | §5.2, §7 |
| A6 matched | the same comparison with the choice between candidates matched on both sides | §5.2, Table S2a |
| A7 residual | what is left to fail for each engine, and at which graph sizes | §5.2, §8, §9 |

These six answer the question a reader asks of any geometric comparison between
two engines — which criterion is doing the work, and whether the engine
reporting the counts is also the engine selecting on them. They read
`layout_items.csv`, `pipeline_items.csv` and `human_stimuli.csv` and nothing
else.

- **A1, A2** split the strict verdict into its six criteria over the 6,007
  figures for which our layout and all four `dot` configurations exist. The two
  engines are within 2.6 points on four of the six. The whole difference is the
  two criteria a label-placement stage addresses, which `dot` does not have, and
  on the remaining four — the *label-blind* verdict — `dot` passes 98.9% against
  our 96.7% (McNemar p=2e-17). The sign reverses, which is why the paper reports
  the result as one defect class rather than as a layout advantage.
- **A3** walks our own layout ablation down to `none`, all six stages off, on the
  same axis as the `dot` configurations: 17.4% strictly legible, below every
  spline configuration of `dot`. Two of the three *lenient* criteria hold in
  every row, so they follow from laying out on a grid and marking back edges;
  the third needs the obstacle-aware router.
- **A4** prices the selection: keeping the first layout instead of the best of
  the variants costs 12.5 points, and one greedy graph instead of the better of
  two a further 7.2. `dot`'s oracle row in A2 is the corresponding allowance on
  the other side.
- **A6** spends that allowance. Our shipped layout is the best of several
  variants chosen by the counts the verdict reports and `dot`'s best
  configuration is one layout chosen by nobody, so the two are compared again
  with the choice matched: both engines choosing (86.9% against `dot`'s oracle
  at 65.9%) or neither (our first layout at 73.8% against `dot` top-down curved
  at 52.9%). The strict difference is 21 points either way rather than 33, the
  label-blind reversal widens to 2.8 and 3.9 points, and per corpus and model
  the matched difference runs +5.3 to +36.6 points. Both engines draw the
  identical graph in every row, so the graph-level best-of-two is no part of
  it.
- **A7** asks what is left once the comparison is attributed, and is new in this
  version. Of our 784 strict failures, exactly one of the six criteria fails on
  71.7% [67.9, 75.4] resampling source graphs; an arrow crossing a label is
  present in 90.6% [88.1, 92.7] of them and is the only defect in 62.4%; label
  collisions survive on 2.3% of all our figures and stacked routes on 3.3%.
  `dot`'s residue is the same class (present in 76.3% of its failures, sole
  defect in 46.5%, label collision in 53.2%), so the defect class that separates
  the engines also bounds the better of them, and it grows with the graph: our
  strict failures run from 0.0% at eight boxes or fewer to 30.9% above thirty,
  27.3 of those points an arrow across a label. What that names is a missing
  stage rather than a tuning target, for either engine: a label placer free to
  leave its own route. Tables: `residual_profile.csv`,
  `residual_profile_by_size.csv`.
- **A5** puts the 24 rated graphs on the size axis of A1. At 8-20 boxes `dot`
  collides a label on 0.04 figures on average, against 0.90 above thirty, so the
  rater study of §7 and the engine comparison of §5.2 are measuring different
  parts of the same curve, and neither licenses a ranking of the engines in the
  other's range.

## `checks_new.py` — seven analyses that need no new run

    python3 checks_new.py --analysis N2 N3
    python3 checks_new.py --latex ../results/tables_new.tex
    python3 checks_new.py --csv-dir ../results/tables

| Check | Question | Paper |
|---|---|---|
| N1 judge power | what a vision–language judge comparison can resolve, and where its variance is | §7 |
| N2 reader | how faithfully a read-back reader recovers a graph whose geometry we know | §6, Table S8 |
| N3 capacity | the graph size at which half an engine's figures stop passing the strict check | §5.2, `img/capacity_frontier.pdf` (a) |
| N4 layout variance | how much of legibility's variance belongs to the engine and how much to the graph | §5.2 |
| N5 index | an index of the ten layout measures fitted to the human ratings | §7, Table S5 |
| N6 ceiling | every SciFlow-Bench score as a share of what the benchmark can return | §7, Table S6 |
| N7 frontier | content against legibility over every corpus × model × variant cell | §8, `img/capacity_frontier.pdf` (b) |

N1 decomposes the judge's points over item × model × seed × image order inside
one system and derives the paired and unpaired minimum detectable difference at
80% power. N2 scores the read-back reader against the graph the pipeline drew
rather than against the gold graph; that ground truth exists only because the
geometry is explicit, and no image baseline supports the same measurement. N3
fits the strict verdict on log box count by IRLS and reports the box count at
which half the figures stop passing. N5 log1p-transforms and standardises the
ten layout measures and cross-validates leaving one source graph out, so no
rated image is predicted by a fit that has seen another layout of the same
graph; it writes both the ten-measure and the four-measure weights. N7 marks a
cell dominated only by cells of the same corpus and model, so a configuration
is never compared against a different extractor.

`--latex` writes `results/tables_new.tex` and one `table_<label>.tex` beside it;
the paper includes none of them, the body stating the numbers and citing the
analysis that computes them (the appendix is at its page limit).
`--csv-dir` writes `judge_variance.csv`, `judge_mdd.csv`,
`judge_order_instability.csv`, `reader_fidelity.csv`, `capacity.csv`,
`capacity_by_size.csv`, `legibility_variance.csv`, `index_weights.csv`,
`index_cv.csv`, `ceiling.csv` and `frontier.csv`.

`make_fig_capacity.py` draws the capacity panels from the same CSVs:

    python3 make_fig_capacity.py --data ../data --out ../../img

It writes `img/capacity_frontier.{pdf,png}` and the two CSVs behind the panels.
Panel (a) is the fitted version of the legibility-against-size panel that
`make_fig_panels.py` used to draw, which is why that script now draws two
panels and still writes `legibility_size.csv`.

## The audit at a glance (`make_table_audit.py`)

Table 2 of the paper puts the three instruments and the counts of this bench
side by side, each shown the same stimulus: the +21.0-point legibility
difference of §5.2, attributed to one defect class and sized to a point. The
script builds that table from the per-item measurements and from tables the
other scripts already wrote, and it reuses `partial_spearman`, `cluster_boot`
and `engines` from `checks_inference.py` and `kappa` from `checks_judge.py` (its `G2`), so
nothing in it can drift from what those scripts print:

    python3 make_table_audit.py                               # the readings
    python3 make_table_audit.py --latex                       # the LaTeX table
    python3 make_table_audit.py --csv-dir ../results/tables    # audit_summary.csv

The four readings it reproduces are the graph score against the strict verdict
at fixed figure size (rho = +0.003, 95% CI [-0.025, +0.033] over source graphs,
7,047 figures from 772 graphs; AUC 0.593 unconditionally), the judge's own
*readability* dimension (65.1% ties, kappa = 0.151 across judges against 0.294
for *conciseness*, its sign kept for 0 of 3 models), the read-back reader
(1.21-1.30 boxes and 1.66-1.85 arrows per one drawn, correction slope
0.26-0.45 over the six fits), and the geometric counts against the raters
(label collisions at rho = 0.75 against 0.26 for arrow crossings, the two
near-orthogonal at fixed size, rho = +0.023 [+0.008, +0.036] over 30,051
layouts). The printed output is `results/printed_outputs/make_table_audit.txt`.

## Tracing the paper's numbers (`check_text_numbers.py`)

The paper carries its tables inline, so a number in the text can drift from the
measurement behind it without anything failing. This script pulls every numeric
token out of `main.tex` and `appendix.tex` and looks for each one in
`data/`, `results/tables/` and `results/printed_outputs/`, reporting the ones it
cannot locate:

    python3 check_text_numbers.py
    python3 check_text_numbers.py --verbose          # the line of every miss
    python3 check_text_numbers.py --out ../results/printed_outputs/check_text_numbers.txt

It is a tracing aid and not a test: a token can be missing because it is a page
measurement or a rounding of a value stored at full precision. What matters is
that the unmatched list stays short enough to read and holds no empirical claim
nobody can locate. On the released version it is empty: all 392 numeric tokens
of both files appear in the released results.
