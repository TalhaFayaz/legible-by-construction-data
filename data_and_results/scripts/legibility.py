#!/usr/bin/env python3
"""The released legibility measure, in one file and with no dependencies.

Three things any diagram system can reuse, whatever it draws with, as long as it
knows where its boxes, routes and labels are:

  1. the three verdicts of the paper (lenient, label-blind and strict legibility),
  2. the index fitted to the best--worst-scaling ratings (four or ten measures),
     which is validated as a detector of figures a reader cannot follow and not
     as a ranking of two drawn layouts (see --selfcheck),
  3. the rule that says, from the graph alone, whether `dot` will draw a figure
     illegibly -- i.e. whether an engine like ours is needed.

Weights and normalisation constants are read from ../results/tables/, so this
file cannot drift away from the numbers in the paper.

  python3 legibility.py --selfcheck            # reproduce the rating study
  python3 legibility.py --score FILE.csv       # score a table of measures
  python3 legibility.py --engine --boxes 24 --arrows 31 --loops 1 --nested 0

A row to score needs the defect counts below; absent columns count as zero, and
a `report_` prefix (as in the released data) is accepted.
"""
import argparse, csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "..", "results", "tables")
DATA = os.path.join(HERE, "..", "data")

# the six counts that decide a verdict (all must be zero)
LENIENT = ["node_overlap_pairs", "edge_node_crossings", "route_reversals"]
LABEL = ["label_collisions", "edge_label_crossings"]
BLIND = LENIENT + ["stacked_routes"]          # strict, minus the two label criteria
STRICT = BLIND + LABEL
# the ten measures the index is fitted on
TEN = ["label_collisions", "node_overlap_pairs", "edge_node_crossings",
       "stacked_routes", "edge_label_crossings", "edge_edge_crossings",
       "total_edge_length", "mean_bends_per_edge", "aspect_deviation", "area"]
ENGINE_FEATS = ["lbox", "larr", "density", "loops", "nested"]


def _rows(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def _num(row, key, default=0.0):
    v = row.get(key, row.get("report_" + key, default))
    if v in (None, "", "NA", "nan"):
        return default
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return 1.0 if v.strip().lower() == "true" else 0.0
    return float(v)


def load_index(which="four"):
    """weights and log1p/z normalisation of the fitted index"""
    key = {"four": "four_measure", "ten": "ten_measure"}[which]
    w = {r["measure"]: float(r["weight"]) for r in _rows(os.path.join(TABLES, "index_weights.csv"))
         if r["index"] == key}
    n = {r["measure"]: (float(r["log1p_mean"]), float(r["log1p_sd"]))
         for r in _rows(os.path.join(TABLES, "index_norms.csv"))}
    return w, n


def load_engine_rule():
    w = _rows(os.path.join(TABLES, "engine_rule_weights.csv"))[0]
    n = {r["feature"]: (float(r["mean"]), float(r["sd"]))
         for r in _rows(os.path.join(TABLES, "engine_rule_norms.csv"))}
    return {k: float(v) for k, v in w.items()}, n


def verdicts(row):
    """(lenient, label_blind, strict) as defined in the paper's Metrics section.

    The label-blind verdict is strict without its two label criteria. Reporting
    it beside the strict one is what separates a layout difference from a
    label-placement difference: an engine with no label stage can pass
    label-blind and fail strict on the same figure (5.3).
    """
    len_ok = all(_num(row, c) == 0 for c in LENIENT)
    blind_ok = len_ok and all(_num(row, c) == 0 for c in BLIND)
    return len_ok, blind_ok, blind_ok and all(_num(row, c) == 0 for c in LABEL)


def index(row, which="four", weights=None):
    """the fitted legibility index; higher is more legible, 0 is the study mean"""
    w, n = weights or load_index(which)
    out = 0.0
    for m, weight in w.items():
        mean, sd = n[m]
        z = (math.log1p(max(_num(row, m), 0.0)) - mean) / (sd or 1.0)
        out += weight * z
    return out


def engine_risk(n_boxes, n_arrows, loops, nested, rule=None):
    """P(Graphviz dot draws this graph illegibly), from the graph alone"""
    w, n = rule or load_engine_rule()
    f = {"lbox": math.log(n_boxes + 1), "larr": math.log(n_arrows + 1),
         "density": n_arrows / max(n_boxes, 1),
         "loops": 1.0 if loops else 0.0, "nested": 1.0 if nested else 0.0}
    z = w["intercept"]
    for k in ENGINE_FEATS:
        mean, sd = n[k]
        z += w[k] * (f[k] - mean) / (sd or 1.0)
    return 1.0 / (1.0 + math.exp(-z))


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else float("nan")


def selfcheck():
    """recompute the rating study from the released ratings, with this file only"""
    st = _rows(os.path.join(DATA, "human_stimuli.csv"))
    sets = {r["set"]: r for r in _rows(os.path.join(DATA, "human_sets.csv"))}
    cnt = {r["image"]: [0, 0, 0] for r in st}          # best, worst, shown
    for r in _rows(os.path.join(DATA, "human_ratings.csv")):
        s = sets[r["set"]]
        for k in ("image_1", "image_2", "image_3", "image_4"):
            cnt[s[k]][2] += 1
        cnt[r["best"]][0] += 1
        cnt[r["worst"]][1] += 1
    bws = [(cnt[r["image"]][0] - cnt[r["image"]][1]) / max(cnt[r["image"]][2], 1) for r in st]
    w4, w10 = load_index("four"), load_index("ten")
    print(f"{len(st)} rated images, {sum(c[2] for c in cnt.values()) // 4} choices")
    eng = [i for i, r in enumerate(st) if r.get("condition") in ("ours", "graphviz")]
    for name, vals in (
            ("four-measure index", [index(r, weights=w4) for r in st]),
            ("ten-measure index", [index(r, weights=w10) for r in st]),
            ("strict verdict alone", [1.0 if verdicts(r)[2] else 0.0 for r in st]),
            ("arrow crossings alone", [-_num(r, "edge_edge_crossings") for r in st]),
            ("label collisions alone", [-_num(r, "label_collisions") for r in st])):
        sub = spearman([vals[i] for i in eng], [bws[i] for i in eng])
        tag = ", in sample" if name.endswith("index") else ""
        print(f"  rho with the raters  {spearman(vals, bws):+.3f}   (on the {len(eng)} images "
              f"the two engines drew: {sub:+.3f}{tag})   {name}")
    print("  (the paper reports the two indices under leave-one-source-graph-out")
    print("   cross-validation: 0.76 and 0.77; the rows above are in sample.)")
    print("  The first column is over all three conditions, one of which is the model")
    print("  placing its own boxes: it measures how well each number detects a figure")
    print("  nobody can read, which is what these counts are for. The bracket is the same")
    print("  correlation between two layouts a real engine drew; nothing in it survives")
    print("  correction, so do not read these weights as a ranking of drawn layouts.")
    print("  The two index rows are in sample on that subset; cross-validated the index")
    print("  does not predict those 48 images at all (rho 0.11, checks_claims.py K6).")
    strict = [r for r in st if verdicts(r)[2]]
    print(f"  strict-legible stimuli {len(strict)}/{len(st)}; verdict agrees with the "
          f"released column on {sum(verdicts(r)[2] == (r['report_valid'].lower() == 'true') for r in st)}/{len(st)}")
    blind = [r for r in st if verdicts(r)[1]]
    print(f"  label-blind-legible stimuli {len(blind)}/{len(st)}: on these images the strict")
    print("  verdict differs from the label-blind one only in the label criteria (5.3)")
    print("\n  engine rule, P(dot draws it illegibly):")
    for b, a, lo, ne in ((6, 6, 0, 0), (24, 31, 1, 0), (24, 31, 1, 1), (53, 70, 1, 1)):
        print(f"    {b:3d} boxes, {a:3d} arrows, loop={lo}, nested={ne}  ->  "
              f"{engine_risk(b, a, lo, ne):.2f}")


def score(path, out):
    rows = _rows(path)
    w4, w10 = load_index("four"), load_index("ten")
    cols = list(rows[0].keys()) + ["lenient_legible", "label_blind_legible",
                                   "strict_legible", "index_four", "index_ten"]
    sink = open(out, "w", newline="", encoding="utf8") if out else sys.stdout
    wr = csv.DictWriter(sink, fieldnames=cols)
    wr.writeheader()
    for r in rows:
        le, bl, st_ = verdicts(r)
        r = dict(r, lenient_legible=le, label_blind_legible=bl, strict_legible=st_,
                 index_four=round(index(r, weights=w4), 4),
                 index_ten=round(index(r, weights=w10), 4))
        wr.writerow(r)
    if out:
        sink.close()
        print(f"wrote {out}  ({len(rows)} figures)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--score", metavar="FILE.csv")
    p.add_argument("--out", metavar="FILE.csv")
    p.add_argument("--engine", action="store_true")
    p.add_argument("--boxes", type=int, default=0)
    p.add_argument("--arrows", type=int, default=0)
    p.add_argument("--loops", type=int, default=0)
    p.add_argument("--nested", type=int, default=0)
    a = p.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.score:
        score(a.score, a.out)
    elif a.engine:
        print(f"{engine_risk(a.boxes, a.arrows, a.loops, a.nested):.4f}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
