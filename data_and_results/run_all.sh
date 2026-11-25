#!/usr/bin/env bash
# Rebuild every table, analysis output and data figure of the paper from data/,
# then compare everything with the shipped results/.
#
#     bash run_all.sh
#
# Needs Python 3.10 or newer. If the packages in requirements.txt are not
# installed at exactly these versions, a local environment .venv/ is created
# and they are installed there (internet access needed once). Nothing outside
# this folder is changed. The rebuilt files go to reproduced/; the shipped
# results/ are left untouched and serve as the reference. No GPU, about 7 min.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/reproduced"
LOG="$OUT/logs"

say() { printf '%s\n' "$*"; }
die() { say ""; say "STOPPED: $*"; exit 1; }

# ---------------------------------------------------------------- python ---
say "== 1/4  Python environment"
PY_SYS="$(command -v python3 || true)"
[ -n "$PY_SYS" ] || die "python3 not found; install Python 3.10 or newer."
"$PY_SYS" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "Python 3.10 or newer is needed (found $("$PY_SYS" --version 2>&1))."

has_reqs() {  # exit 0 if the interpreter has every pinned package at its pinned version
    "$1" - "$ROOT/requirements.txt" <<'EOF' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, want = line.split("==")
    try:
        if version(name) != want:
            sys.exit(1)
    except PackageNotFoundError:
        sys.exit(1)
EOF
}

if has_reqs "$PY_SYS"; then
    PY="$PY_SYS"
    say "   using $("$PY" --version 2>&1) with the pinned packages already installed"
else
    PY="$ROOT/.venv/bin/python"
    if [ ! -x "$PY" ] || ! has_reqs "$PY"; then
        say "   creating .venv/ and installing requirements.txt (one time, needs internet)"
        "$PY_SYS" -m venv "$ROOT/.venv" || die "could not create .venv (on Debian/Ubuntu: apt install python3-venv)."
        "$PY" -m pip install --quiet --upgrade pip || die "pip upgrade failed."
        "$PY" -m pip install --quiet -r "$ROOT/requirements.txt" || die "installing requirements.txt failed."
        has_reqs "$PY" || die "packages in .venv do not match requirements.txt."
    fi
    say "   using .venv ($("$PY" --version 2>&1))"
fi
export MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------- working copy ---
say "== 2/4  Working copy in reproduced/"
rm -rf "$OUT"
mkdir -p "$OUT/results/tables" "$OUT/results/printed_outputs" "$OUT/results/figures" "$LOG"
cp -r "$ROOT/scripts" "$OUT/scripts"
ln -s "$ROOT/data" "$OUT/data"
# inputs some analyses read: tables written by other analyses and the printed
# output of the runs (made with the pipeline itself, not rebuilt here)
cp "$ROOT"/results/tables/* "$OUT/results/tables/"
cp "$ROOT"/results/printed_outputs/*.txt "$OUT/results/printed_outputs/"
touch "$OUT/.start"
sleep 1

# ---------------------------------------------------------------- run ------
say "== 3/4  Rebuilding (each step: seconds, status)"
cd "$OUT/scripts"
FAILED=0
step() {  # step <name> <stdout file, relative to scripts/> <script and arguments...>
    local name="$1" dest="$2"; shift 2
    local t0=$SECONDS
    if "$PY" "$@" > "$dest" 2> "$LOG/$name.err"; then
        printf '   %-34s %4ss  ok\n' "$name" $((SECONDS - t0))
    else
        printf '   %-34s %4ss  FAILED (see reproduced/logs/%s.err)\n' "$name" $((SECONDS - t0)) "$name"
        tail -n 3 "$LOG/$name.err" | sed 's/^/        /'
        FAILED=$((FAILED + 1))
    fi
}
P=../results/printed_outputs
T=../results/tables
F=../results/figures

step "tables of the paper"          ../results/tables_from_data.txt make_tables.py
step "checks_standing"              $P/checks_standing.txt          checks_standing.py
step "checks_decomposition"         $P/checks_decomposition.txt     checks_decomposition.py
step "checks_human_instrument"      $P/checks_human_instrument.txt  checks_human_instrument.py
step "checks_variance"              $P/checks_variance.txt          checks_variance.py --csv-dir $T
step "checks_attribution"           $P/checks_attribution.txt       checks_attribution.py --csv-dir $T
step "checks_bounds"                $P/checks_bounds.txt            checks_bounds.py
step "checks_criteria"              $P/checks_criteria.txt          checks_criteria.py
step "checks_engine"                $P/checks_engine.txt            checks_engine.py
step "checks_inference"             $P/checks_inference.txt         checks_inference.py
step "checks_instruments"           $P/checks_instruments.txt       checks_instruments.py
step "checks_invariants"            $P/checks_invariants.txt        checks_invariants.py --csv-dir $T
step "checks_judge"                 $P/checks_judge.txt             checks_judge.py
step "checks_new"                   $P/checks_new.txt               checks_new.py --csv-dir $T
step "checks_novelty"               $P/checks_novelty.txt           checks_novelty.py --csv-dir $T
step "checks_robustness"            $P/checks_robustness.txt        checks_robustness.py --csv-dir $T
step "checks_validity"              $P/checks_validity.txt          checks_validity.py
step "checks_claims"                $P/checks_claims.txt            checks_claims.py --csv-dir $T
step "make_norms"                   $P/make_norms.txt               make_norms.py
step "legibility selfcheck"         $P/legibility_selfcheck.txt     legibility.py --selfcheck
step "make_table_systems"           $P/make_table_systems.txt       make_table_systems.py
step "make_table_audit"             $P/make_table_audit.txt         make_table_audit.py --csv-dir $T
step "figure: standing (paper Fig. 3)"     $LOG/fig_standing.txt  make_fig_standing.py --tables $T --out $F
step "figure: criteria ranking"     $LOG/fig_criteria.txt           make_fig_criteria.py --out $F
step "figure: rating study (paper Fig. 8)" $LOG/fig_human.txt     make_fig_human.py --out $F
step "figure: capacity frontier"    $LOG/fig_capacity.txt           make_fig_capacity.py --out $F
step "figure: dose response"        $LOG/fig_dose.txt               make_fig_dose.py ../data $F/dose_response.pdf
step "figure: judge panels"         $LOG/fig_judge.txt              make_fig_judge.py --out $F
step "figure: legibility by size"   $LOG/fig_legibility_size.txt    make_fig_legibility_size.py --out $F
step "figure: appendix panels"      $LOG/fig_panels.txt             make_fig_panels.py --out $F
step "figure: resolution"           $LOG/fig_resolution.txt         make_fig_resolution.py --out $F

# ------------------------------------------------------------ compare ------
say "== 4/4  Comparing the rebuilt files with the shipped results/"
"$PY" compare_outputs.py --shipped "$ROOT/results" --rebuilt "$OUT/results" --since "$OUT/.start"
CMP=$?

say ""
if [ "$FAILED" -eq 0 ] && [ "$CMP" -eq 0 ]; then
    say "DONE: every step ran, and every rebuilt file matches the shipped results."
    say "Rebuilt files: reproduced/results/ (figures in reproduced/results/figures/)."
    exit 0
fi
[ "$FAILED" -gt 0 ] && say "$FAILED step(s) failed; their error output is in reproduced/logs/."
[ "$CMP" -ne 0 ] && say "Some rebuilt files differ from the shipped ones (listed above)."
exit 1
