"""Flatten the released runs into one CSV per kind of measurement.

Unzip every file in ../raw into one folder (it then holds ``runs/`` and
``human_study/``), pass that folder as ``--src``, and the CSVs in ../data are
written again. Every value is copied from the run files unchanged; nothing is
recomputed here. Standard library only.

    python3 export_csv_from_raw.py --src <unzipped raw> --out ../data
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MODELS = {"qwen": "Qwen", "gemma4": "Gemma", "gemma": "Gemma", "ministral": "Ministral"}
POINTS = {"Model": 100.0, "Human": 0.0, "Tie": 50.0, "Both are good": 50.0, "Both are bad": 50.0}
DIMS = ("faithfulness", "conciseness", "readability", "aesthetics", "overall")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flat(d: dict | None, prefix: str) -> dict:
    """Scalar fields of a nested dict as prefix_a_b columns."""
    out = {}
    for k, v in (d or {}).items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, name + "_"))
        elif not isinstance(v, list):
            out[name] = v
    return out


def write(path: Path, rows: list[dict], first: list[str]) -> None:
    cols = list(first)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name}: {len(rows)} rows, {len(cols)} columns")


def item_row(r: dict) -> dict:
    """The fields every pipeline or baseline record shares."""
    row = {"item": r["paper_id"], "domain": r.get("domain", ""), "failed": r.get("failed", False),
           "n_candidates": r.get("n_candidates"), "gold_nodes": r.get("gold_nodes"),
           "gold_edges": r.get("gold_edges"), "pred_nodes": r.get("pred_nodes"),
           "pred_edges": r.get("pred_edges"), "seconds": r.get("seconds")}
    native = r.get("native") or {}
    row.update({"node_f1": native.get("node_f1"), "edge_f1": native.get("edge_f1"),
                "score": native.get("final_score"),
                "node_precision": native.get("node_precision"), "node_recall": native.get("node_recall"),
                "edge_precision": native.get("edge_precision"), "edge_recall": native.get("edge_recall")})
    row.update(flat(r.get("strict"), "strict_"))
    row.update(flat(r.get("layout"), "layout_"))
    return row


def pipeline(runs: Path) -> list[dict]:
    rows = []
    pattern = re.compile(r"^(nocov|final)_(sciflow|flowvqa|paperbanana)_(qwen|gemma4|ministral)(_cf)?_seed(\d)$")
    for d in sorted(runs.iterdir()):
        m = pattern.match(d.name)
        if not m:
            continue
        kind, corpus, model, cf, seed = m.groups()
        if kind == "nocov":
            variant = "evaluated"
        elif corpus == "flowvqa" and not cf:
            variant = "coverage_no_control_flow"
        else:
            variant = "coverage"
        for r in jsonl(d / "full.jsonl"):
            rows.append({"variant": variant, "corpus": corpus, "model": MODELS[model],
                         "seed": int(seed), "run": d.name, **item_row(r)})
    for model_dir in sorted((runs / "pipeline_ablation_final").iterdir()):
        if not model_dir.is_dir():
            continue
        for d in sorted(model_dir.iterdir()):
            corpus, change = d.name.split("_", 1)
            variant = {"n1": "ablation_one_greedy_graph", "unconstrained": "ablation_unconstrained"}[change]
            for r in jsonl(d / "full.jsonl"):
                rows.append({"variant": variant, "corpus": corpus, "model": MODELS[model_dir.name],
                             "seed": 0, "run": f"pipeline_ablation_final/{model_dir.name}/{d.name}",
                             **item_row(r)})
    return rows


def dot(runs: Path) -> list[dict]:
    rows = []
    sources = [(d, "dot") for d in sorted(runs.glob("dot_*_*")) if d.is_dir()]
    sources += [(d, "dot_matched") for d in sorted((runs / "dotmatched").iterdir()) if d.is_dir()]
    for d, variant in sources:
        name = d.name.removeprefix("dot_")
        corpus, model = name.split("_")
        scored = d / "repaired.jsonl" if (d / "repaired.jsonl").exists() else d / "full.jsonl"
        for r in jsonl(scored):
            rows.append({"variant": variant, "corpus": corpus, "model": MODELS[model],
                         "scored_file": f"{d.relative_to(runs)}/{scored.name}", **item_row(r)})
    return rows


def published(runs: Path) -> list[dict]:
    data = json.loads((runs / "baseline_scores.json").read_text())
    return [{"system": system, **item_row(r)} for system, recs in data.items() for r in recs]


def readback(runs: Path) -> list[dict]:
    rows = []
    for p in sorted((runs / "inverse").glob("*.jsonl")):
        stem = p.stem
        if stem.startswith("nocov_ours_"):
            system, model = "ours", MODELS[stem.removeprefix("nocov_ours_")]
        elif stem.startswith("ours_"):
            system, model = "ours_coverage", MODELS[stem.removeprefix("ours_")]
        elif stem == "ceiling_gold":
            system, model = "gold_graph_ceiling", ""
        else:
            system, model = stem, ""
        for r in jsonl(p):
            row = item_row(r)
            rows.append({"system": system, "model": model, **row})
    return rows


def judge(runs: Path) -> list[dict]:
    rows = []
    pattern = re.compile(r"^(swap_)?(nocov|ours|dot|dotmatched)_(qwen|gemma4|ministral)(?:_seed(\d))?$")
    for folder, judge_name in (("judge", "InternVL3.5-14B"), ("judge2", "Qwen2.5-VL-32B")):
        for p in sorted((runs / folder).glob("*.jsonl")):
            m = pattern.match(p.stem)
            if not m:
                continue
            swap, kind, model, seed = m.groups()
            system = {"nocov": "ours", "ours": "ours_coverage", "dot": "dot",
                      "dotmatched": "dot_matched"}[kind]
            if seed is None:
                seed = "0" if kind in ("ours", "nocov") else ""
            for r in jsonl(p):
                row = {"judge": judge_name, "system": system, "model": MODELS[model], "seed": seed,
                       "order": "swapped" if swap else "normal", "item": r["id"]}
                for dim in DIMS:
                    row[f"{dim}_verdict"] = r.get(dim)
                    row[f"{dim}_points"] = POINTS.get(r.get(dim), 50.0)
                row["seconds"] = r.get("seconds")
                rows.append(row)
    return rows


def layout(runs: Path) -> list[dict]:
    rows = []
    def add(source, corpus, model, run, recs, configs):
        for rec in recs:
            for config in configs(rec):
                m = rec[config] if isinstance(rec.get(config), dict) else rec
                row = {"source": source, "corpus": corpus, "model": MODELS[model], "run": run,
                       "item": rec["paper"], "config": config,
                       "loops": rec.get("loops"), "nested": rec.get("nested"), "arrows": rec.get("arrows")}
                row.update({k: v for k, v in m.items() if not isinstance(v, (dict, list)) and k != "paper"})
                rows.append(row)
    engines = lambda rec: [k for k in rec if isinstance(rec[k], dict)]
    for model_dir in sorted((runs / "graphviz_nocov").iterdir()):
        for p in sorted(model_dir.glob("run*.json")):
            add("graphviz", "sciflow", model_dir.name, int(p.stem[3:]), json.loads(p.read_text()), engines)
    for d in sorted((runs / "graphviz_outdev").iterdir()):
        if d.is_dir():
            corpus, model = d.name.split("_")
            for p in sorted(d.glob("run*.json")):
                add("graphviz_outside_development", corpus, model, 0, json.loads(p.read_text()), engines)
    for model_dir in sorted((runs / "layout_ablation_nocov").iterdir()):
        for p in sorted(model_dir.glob("run*_*.json")):
            run, config = p.stem.split("_", 1)
            recs = json.loads(p.read_text())
            for rec in recs:
                rec["_config"] = config
            add("layout_ablation", "sciflow", model_dir.name, int(run[3:]),
                recs, lambda rec: [rec["_config"]])
    for r in rows:
        r.pop("_config", None)
    return rows


def placement(runs: Path) -> list[dict]:
    rows = []
    for d in sorted((runs / "placement_final").iterdir()):
        if not d.is_dir():
            continue
        model, seed = d.name.rsplit("_seed", 1)
        for r in jsonl(d / "results.jsonl"):
            row = {"model": MODELS[model], "seed": int(seed), "condition": r["condition"]}
            row.update(item_row(r))
            row.update(flat({k: v for k, v in r.items() if k not in
                             ("paper_id", "domain", "failed", "layout", "strict", "native", "condition")}, ""))
            rows.append(row)
    return rows


def human(src: Path, out: Path) -> None:
    study = json.loads((src / "human_study" / "stimuli.json").read_text())
    stim = study["stimuli"]
    items = stim.items() if isinstance(stim, dict) else ((s.get("id"), s) for s in stim)
    write(out / "human_stimuli.csv",
          [{"image": i, **flat(s, "")} for i, s in items], ["image"])
    write(out / "human_sets.csv",
          [{"set": i, **{f"image_{j + 1}": x for j, x in enumerate(t)}}
           for i, t in enumerate(study["tuples"])], ["set"])
    ratings = []
    for p in sorted((src / "human_study").glob("ratings_R*.json")):
        ratings += [{"rater": r["rater"], "set": r["tuple_id"], "best": r["best"],
                     "worst": r["worst"], "seconds": r["seconds"]} for r in json.loads(p.read_text())]
    write(out / "human_ratings.csv", ratings, ["rater", "set"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs, out = args.src / "runs", args.out
    out.mkdir(parents=True, exist_ok=True)
    write(out / "pipeline_items.csv", pipeline(runs), ["variant", "corpus", "model", "seed", "run", "item"])
    write(out / "dot_items.csv", dot(runs), ["variant", "corpus", "model", "scored_file", "item"])
    write(out / "published_systems_sciflow.csv", published(runs), ["system", "item"])
    write(out / "readback_sciflow.csv", readback(runs), ["system", "model", "item"])
    write(out / "judge_paperbanana.csv", judge(runs), ["judge", "system", "model", "seed", "order", "item"])
    write(out / "layout_items.csv", layout(runs), ["source", "corpus", "model", "run", "item", "config"])
    write(out / "placement_items.csv", placement(runs), ["model", "seed", "condition", "item"])
    human(args.src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
