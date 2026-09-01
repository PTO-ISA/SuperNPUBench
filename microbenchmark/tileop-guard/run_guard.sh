#!/bin/bash
# TileOP-guard batch runner (res_check precision path).
# Per case: compile -> golden.py gen (host inputs) -> gfrun (reads inputs, dumps
# out.bin) -> golden.py check (independent numpy compare). Classify:
#   compile-fail | run-fail(crash, no end marker) | precision-fail | pass | run-only
# A case is golden-checked iff registered in golden/golden.py (gen rc!=3).
#
# Usage: bash run_guard.sh <subdir> [case]
# Requires: COMPILER_DIR (linx toolchain bin), GFRUN (gfrun binary).
set -u

SUB="${1:?usage: run_guard.sh <subdir> [case]}"
ONLY="${2:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
GUARD_ROOT="$HERE"
SUBDIR="$GUARD_ROOT/$SUB"
GOLDEN="$GUARD_ROOT/golden/golden.py"
: "${COMPILER_DIR:?export COMPILER_DIR to linx toolchain bin}"
: "${GFRUN:?export GFRUN to the gfrun binary path}"

ROOT="$(echo "$GUARD_ROOT" | sed -E 's@(.*)/microbenchmark/.*@\1@')"
ELF_DIR="$ROOT/output/microbenchmark/tileop-guard/$SUB/elf/$SUB"

pass=0; cfail=0; rfail=0; pfail=0; ronly=0
echo "=== tileop-guard: $SUB (res_check) ==="
printf '%-28s %-8s %-8s %-10s\n' "case" "compile" "gfrun" "precision"
printf '%-28s %-8s %-8s %-10s\n' "----" "-------" "-----" "---------"

for src in "$SUBDIR"/src/*.cpp; do
    [ -e "$src" ] || { echo "(no cases)"; break; }
    name="$(basename "$src" .cpp)"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue

    clog="$(cd "$SUBDIR" && make TESTCASE="$name" 2>&1)"
    if [ $? -ne 0 ]; then
        printf '%-28s %-8s %-8s %-10s\n' "$name" "FAIL" "-" "-"
        echo "$clog" | grep -E 'error:|assert' | head -3 | sed 's/^/    /'
        cfail=$((cfail+1)); continue
    fi

    chk="$GUARD_ROOT/compare/$SUB/$name"
    mkdir -p "$chk"
    python3 "$GOLDEN" gen "$name" "$chk" >/dev/null 2>&1
    genrc=$?   # 0 = registered golden; 3 = not registered (run-only)

    elf="$ELF_DIR/$name.elf"
    rlog="$(timeout 120 "$GFRUN" -f "$elf" 2>&1)"
    if ! echo "$rlog" | grep -q "Reach the End of Benchmark"; then
        printf '%-28s %-8s %-8s %-10s\n' "$name" "ok" "FAIL" "-"
        echo "$rlog" | grep -iE 'assert|error|abort|fault|fatal' | head -3 | sed 's/^/    /'
        rfail=$((rfail+1)); continue
    fi

    if [ "$genrc" -eq 3 ]; then
        printf '%-28s %-8s %-8s %-10s\n' "$name" "ok" "ok" "run-only"
        ronly=$((ronly+1)); continue
    fi

    cmsg="$(python3 "$GOLDEN" check "$name" "$chk" 2>&1)"
    if [ $? -eq 0 ]; then
        printf '%-28s %-8s %-8s %-10s\n' "$name" "ok" "ok" "PASS"
        pass=$((pass+1))
    else
        printf '%-28s %-8s %-8s %-10s\n' "$name" "ok" "ok" "MISMATCH"
        echo "$cmsg" | head -2 | sed 's/^/    /'
        pfail=$((pfail+1))
    fi
done

echo "----"
echo "pass=$pass  compile-fail=$cfail  run-fail=$rfail  precision-fail=$pfail  run-only=$ronly"
