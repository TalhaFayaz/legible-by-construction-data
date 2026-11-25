#!/usr/bin/env python3
"""Compare the outputs rebuilt by run_all.sh with the shipped results.

Text outputs are compared line by line after dropping lines that only report
where a file was written (they contain machine-specific paths). Figures are
compared through the CSV files behind them and, for PNG images, pixel by pixel.

    python3 compare_outputs.py --shipped ../results --rebuilt ../../reproduced/results
"""
import argparse
import sys
from pathlib import Path


def text_lines(path):
    return [line.rstrip() for line in path.read_text(errors="replace").splitlines()
            if line.strip() and not line.strip().startswith("wrote ")]


def same_csv(a, b):
    """Equal cell by cell; numbers may differ by floating-point rounding only."""
    import numpy as np
    import pandas as pd
    x, y = pd.read_csv(a), pd.read_csv(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v) \
                and not pd.api.types.is_bool_dtype(u):
            if not np.allclose(u.astype(float), v.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not (u.astype(str) == v.astype(str)).all():
            return False
    return True


def same_png(a, b):
    import matplotlib.image as mpimg
    import numpy as np
    x, y = mpimg.imread(a), mpimg.imread(b)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def compare(shipped, rebuilt, sub, patterns, since=0.0):
    rows = []
    for pattern in patterns:
        for new in sorted((rebuilt / sub).glob(pattern)):
            if new.stat().st_mtime <= since:
                continue  # an input copied in, not rebuilt
            old = shipped / sub / new.name
            if not old.exists():
                rows.append(("NEW", f"{sub}/{new.name}"))
                continue
            if new.suffix == ".png":
                ok = same_png(old, new)
            elif new.suffix == ".csv":
                ok = same_csv(old, new)
            elif new.suffix in (".txt", ".tex"):
                ok = text_lines(old) == text_lines(new)
            else:
                continue
            rows.append(("SAME" if ok else "DIFF", f"{sub}/{new.name}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True, type=Path)
    ap.add_argument("--rebuilt", required=True, type=Path)
    ap.add_argument("--since", type=Path, help="compare only files written after this marker file")
    a = ap.parse_args()
    t = a.since.stat().st_mtime if a.since else 0.0
    rows = []
    rows += compare(a.shipped, a.rebuilt, ".", ["tables_from_data.txt"], t)
    rows += compare(a.shipped, a.rebuilt, "printed_outputs", ["*.txt"], t)
    rows += compare(a.shipped, a.rebuilt, "tables", ["*.csv"], t)
    rows += compare(a.shipped, a.rebuilt, "figures", ["*.csv", "*.png"], t)
    for status, name in rows:
        if status != "SAME":
            print(f"  {status:4}  {name}")
    n = {s: sum(1 for r in rows if r[0] == s) for s in ("SAME", "DIFF", "NEW")}
    print(f"\n  {n['SAME']} identical, {n['DIFF']} different, {n['NEW']} without a shipped counterpart "
          f"({len(rows)} files compared)")
    return 1 if n["DIFF"] else 0


if __name__ == "__main__":
    sys.exit(main())
