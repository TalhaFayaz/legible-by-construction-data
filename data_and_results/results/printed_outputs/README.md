# Results behind the paper

The printed output of each command in [`docs/reproduce.md`](../../docs/reproduce.md),
run on the evaluated pipeline (no coverage pass, control-flow mode for program
text): three models (Qwen2.5-VL-32B-Instruct, Gemma 4 26B-A4B-it, Ministral 3 14B
Instruct), three seeds each. Every file is recomputed without a GPU from the
released runs.

| File | What | Command (`docs/reproduce.md`) |
|---|---|---|
| `sciflow.txt` | SciFlow-Bench, 483 papers: every model against the 7 shipped systems and the same model writing DOT | `ModelComparison` (§3) |
| `flowvqa.txt` | FlowVQA, 300 test items: every model against the same model writing DOT | `ModelComparison --no-shipped` (§3) |
| `sciflow_symmetric.txt` | SciFlow-Bench with every figure, ours included, read back from its image by one reader | `ModelComparison` on `runs/inverse` (§4) |
| `paperbanana_vs_dot.txt` | PaperBananaBench, 292 cases, judge in both image orders: ours against the same model writing DOT | `PaperBananaJudge compare` (§5) |
| `paperbanana_vs_coverage.txt` | the same, ours against the pipeline with the coverage pass | `PaperBananaJudge compare` (§5) |
| `paperbanana_judge2.txt` | the same comparison, seed 0, with a second judge (Qwen2.5-VL-32B-Instruct) | `PaperBananaJudge compare` on `runs/judge2` (§5) |
| `sciflow_vs_matched_dot.txt`, `flowvqa_vs_matched_dot.txt` | ours against the same model writing DOT from our own prompts | `ModelComparison --no-shipped` with the matched runs (§2) |
| `paperbanana_vs_matched_dot.txt` | the same on PaperBananaBench, judge in both orders | `PaperBananaJudge compare` (§2, §5) |
| `sciflow_clean441.txt` | the SciFlow-Bench table without the 40 development papers and their two cross-listed copies | `ModelComparison` on the filtered runs |
| `placement_ablation_<model>.txt` | the model placing the boxes itself against our engine, 100 papers, seeds 0–2 | `PlacementAblation --report` (§9) |
| `pipeline_ablation.txt` | best-of-2 and constrained decoding removed, coverage pass added; seed 0, SciFlow and FlowVQA | `PipelineAblation` (§6) |
| `layout_ablation_<model>.txt` | layout-engine components switched off on the saved graphs | `LayoutAblation` (§7) |
| `graphviz_<model>.txt` | Graphviz `dot` on the same graphs, same measures | `GraphvizBaseline` (§7) |
| `compile_check.txt` | every SciFlow figure compiles | `CompileCheck` (§8) |
| `human_study.txt` | best-worst ratings of 24 graphs in three layouts | `RatingPage analyse` (§11) |

Statistics everywhere: paired Wilcoxon signed-rank test per item, matched-pairs
rank-biserial r, 95 % percentile-bootstrap CI of the mean difference (10,000
resamples, seeded), Holm's correction within each family. A verdict follows the
direction of the rank test (the sign of r); where a few large differences pull
the mean the other way, the CI of the mean difference shows it.

The `checks_*.txt` files are the printed output of the fourteen analysis groups in
[`../../README.md`](../../README.md); each is reproduced by its script with
`--out`, and needs no GPU and no run. `checks_attribution.txt` is the group that
attributes the legibility difference to its criteria (A1-A7), and
`legibility_selfcheck.txt` is `legibility.py --selfcheck`. `make_table_systems.txt` is the shipped-systems comparison of Table 2 of the paper and Table S7 of the supplement, with every paired read-back difference beside what its own sample resolves.

`docs/reproduce.md`, referenced above, and the raw run folders it names are part
of the full release and not of this archive; every file here is reproduced from
the CSVs in `../../data/` by the scripts in `../../scripts/`.
