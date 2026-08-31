#!/bin/bash
# TileOP-guard batch runner: compile each case then gfrun it, classify result.
# Usage: bash run_guard.sh <subdir>        # e.g. vec / sfu / tlsu / cube / fixp / misc
#        bash run_guard.sh <subdir> <case> # single case (no .cpp suffix)
#
# Requires: COMPILER_DIR (linx toolchain bin), GFRUN (path to gfrun binary).
set -u

SUB="${1:?usage: run_guard.sh <subdir> [case]}"
ONLY="${2:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
GUARD_ROOT="$HERE"
SUBDIR="$GUARD_ROOT/$SUB"
: "${COMPILER_DIR:?export COMPILER_DIR to linx toolchain bin}"
: "${GFRUN:?export GFRUN to the gfrun binary path}"

# ROOT = repo root (…/microbenchmark/..)
ROOT="$(echo "$GUARD_ROOT" | sed -E 's@(.*)/microbenchmark/.*@\1@')"
ELF_DIR="$ROOT/output/microbenchmark/tileop-guard/$SUB/elf/$SUB"

pass=0; cfail=0; rfail=0
echo "=== tileop-guard: $SUB ==="
printf '%-28s %-8s %-8s\n' "case" "compile" "gfrun"
printf '%-28s %-8s %-8s\n' "----" "-------" "-----"

for src in "$SUBDIR"/src/*.cpp; do
    [ -e "$src" ] || { echo "(no cases)"; break; }
    name="$(basename "$src" .cpp)"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue

    clog="$(cd "$SUBDIR" && make TESTCASE="$name" 2>&1)"
    if [ $? -ne 0 ]; then
        printf '%-28s %-8s %-8s\n' "$name" "FAIL" "-"
        echo "$clog" | grep -E 'error:|assert' | head -3 | sed 's/^/    /'
        cfail=$((cfail+1)); continue
    fi
    elf="$ELF_DIR/$name.elf"
    rlog="$("$GFRUN" -f "$elf" 2>&1)"
    if echo "$rlog" | grep -q "Reach the End of Benchmark"; then
        printf '%-28s %-8s %-8s\n' "$name" "ok" "ok"
        pass=$((pass+1))
    else
        printf '%-28s %-8s %-8s\n' "$name" "ok" "FAIL"
        echo "$rlog" | grep -iE 'assert|error|abort|fault|fatal' | head -3 | sed 's/^/    /'
        rfail=$((rfail+1))
    fi
done

echo "----"
echo "pass=$pass compile-fail=$cfail run-fail=$rfail"
