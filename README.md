# Legible by Construction — data and results

Data, measurements, scripts and run logs behind the paper
**"Legible by Construction: Coordinate-Free Diagram Generation with Vision–Language Models"**.

Every GitHub release of this repository is archived on Zenodo with its own DOI.

## Contents

Everything is in [`data_and_results/`](data_and_results/); start with its
[README](data_and_results/README.md).

| Folder | What |
|---|---|
| `data/` | per-item measurements as CSV for SciFlow-Bench, FlowVQA and PaperBananaBench, and the human rating study (raters anonymized as R1–R3) |
| `scripts/` | the scripts that rebuild every table and figure of the paper from `data/` |
| `results/` | the analysis tables, the released panels, and the printed output of every command |
| `logs/` | the logs of all runs |
| `raw/` | the complete raw runs: every extracted graph, its layout, and every model answer |

    cd data_and_results/scripts
    python3 make_tables.py          # every table, in paper order

Needs Python 3.9+ with pandas, numpy, scipy and matplotlib; no GPU. The corpora
themselves are not included; they are available from their authors.

## Licence

Data, measurements and documentation: [CC BY 4.0](LICENSE-CC-BY-4.0).
Scripts (`data_and_results/scripts/`): [MIT](LICENSE-MIT).
