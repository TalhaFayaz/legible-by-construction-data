# Legible by Construction — data and results

Data, measurements, scripts and run logs behind the paper
**"Legible by Construction: Coordinate-Free Diagram Generation with Vision–Language Models"**.

Talha Fayaz, Zeeshan Khalil, Dmitry Ignatov, Radu Timofte — CAIDAS, University of Würzburg, Germany.

Every GitHub release of this repository is archived on Zenodo with its own DOI.

## Contents

Everything is in [`data_and_results/`](data_and_results/); start with its
[README](data_and_results/README.md).

| Folder | What |
|---|---|
| `data/` | per-item measurements as CSV for SciFlow-Bench, FlowVQA and PaperBananaBench, and the human rating study (raters anonymized as R1–R3) |
| `scripts/` | the scripts that rebuild every table and figure of the paper from `data/`; `run_all.sh` runs them all |
| `results/` | the analysis tables, the released panels, and the printed output of every command |
| `logs/` | the logs of all runs |
| `raw/` | the complete raw runs: every extracted graph, its layout, and every model answer |

    cd data_and_results
    bash run_all.sh

`run_all.sh` rebuilds every table, analysis output and data figure of the paper
from the CSV files and checks that each rebuilt file matches the shipped one. It
needs Python 3.10 or newer and sets up the pinned packages of `requirements.txt`
in a local `.venv/` if they are missing (internet access needed once); no GPU,
about 7 minutes. The corpora themselves are not included; they are available
from their authors.

## Licence

Data, measurements and documentation: [CC BY 4.0](LICENSE-CC-BY-4.0).
Scripts (`data_and_results/scripts/`): [MIT](LICENSE-MIT).
