#!/bin/bash
# Performance comparison for mxquant: full (TLSU) vs pure Vector.
#
# Builds the kernel-only ELFs (no res_check) for:
#   mxquant               full kernel: strided TLOAD/TSTORE on [512,256]
#   mxquant_compute       same Vector work, TLOAD/TSTORE removed and the input
#                         materialised by TEXPANDS; |x| = max(x, -x)
#   mxquant_compute_tabs  as compute, but |x| folded with a single TABS.
#                         TIMING ONLY: gfrun's TABS CUBE_M32 writeback is
#                         permuted (issue #678), so never res_check this
#   mxquant_texpands      only the input TEXPANDS, nothing else
# then runs gfsim on each and prints the counters plus the TABS saving and the
# pure-algorithm cost with the input TEXPANDS subtracted.
#
# The kernel has no Shared binder, so gfsim auto-detects single-PE and would
# only run 1/4 of the work; the runs below force 4 PEs with ``--conf fourpe``.
#
# Usage:
#   export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
#   GFSIM=/path/to/SuperScalarModel/bin/gfsim ./perf_compare.sh
#
# Scheduling mode can be overridden, e.g. tile mode:
#   GFSIM_EXTRA_ARGS="-s bctrl.vec_cell_sched_enable=false" ./perf_compare.sh
#
# Artifacts (logs + stdout/stderr) land under perf_runs/<timestamp>/.

set -euo pipefail

: "${COMPILER_DIR:?Set COMPILER_DIR to the Linx compiler bin directory}"
: "${GFSIM:?Set GFSIM to the gfsim binary path}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# mxquant -> kernel -> test -> one-level-arch
ONE="$(cd "$HERE/../../.." && pwd)"
ELF_DIR="$ONE/output/kernel/mxquant/elf"
OBJ_DIR="$ONE/output/kernel/mxquant/src"
# gfsim resolves configs/ (e.g. --conf fourpe) relative to the CWD, so it must
# be launched from the model root (the parent of bin/).
MODEL_DIR="$(cd "$(dirname "$GFSIM")/.." && pwd)"

CASES=(mxquant mxquant_compute mxquant_compute_tabs mxquant_texpands)
SHORT=(big compute compute_tabs texpands)
GFSIM_EXTRA_ARGS="${GFSIM_EXTRA_ARGS:-}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="$HERE/perf_runs/$TS"
mkdir -p "$RUN"
echo "gfsim extra args: '${GFSIM_EXTRA_ARGS}'" | tee "$RUN/cmdline.txt"

for tc in "${CASES[@]}"; do
    echo "== build $tc =="
    rm -f "$OBJ_DIR/$tc.o"
    make -C "$HERE" TESTCASE="$tc" COMPILER_DIR="$COMPILER_DIR" \
        >"$RUN/build_$tc.log" 2>&1
    elf="$ELF_DIR/kernel_mxquant_${tc}_PE4.elf"
    echo "== gfsim $tc =="
    (cd "$MODEL_DIR" && "$GFSIM" -f "$elf" --conf fourpe $GFSIM_EXTRA_ARGS) \
        >"$RUN/$tc.stdout" 2>"$RUN/$tc.stderr"
done

echo
echo "artifacts: $RUN"
echo

extract() {  # file, pattern
    grep -E "$2" "$1" | head -1 | grep -oE '[0-9]+' | tail -1
}

hdr="$(printf '%-30s' "counter")"
for s in "${SHORT[@]}"; do hdr+="$(printf ' %12s' "$s")"; done
echo "$hdr"
printf '%-30s' "------------------------------"
for s in "${SHORT[@]}"; do printf ' %12s' "------------"; done
echo

row() {  # display, pattern
    local line
    line="$(printf '%-30s' "$1")"
    for tc in "${CASES[@]}"; do
        line+="$(printf ' %12s' "$(extract "$RUN/$tc.stdout" "$2")")"
    done
    echo "$line"
}

row "Total Cycles"         "^Total Cycles"
row "Vector Active Wall"   "Vector Active Wall Cycles \(union\)"
row "TLSU Active Wall"     "TLSU Active Wall Cycles \(union\)"
row "Vector Busy (PE-sum)" "Vector Busy Resource-Cycles"
row "Vector Tileops"       "Vector Execute Tileop Counter"
row "TLSU Tileops"         "TLSU Execute Tileop Counter"

echo
c_tot="$(extract "$RUN/mxquant_compute.stdout" "^Total Cycles")"
k_tot="$(extract "$RUN/mxquant_compute_tabs.stdout" "^Total Cycles")"
t_tot="$(extract "$RUN/mxquant_texpands.stdout" "^Total Cycles")"
c_va="$(extract "$RUN/mxquant_compute.stdout" "Vector Active Wall Cycles \(union\)")"
k_va="$(extract "$RUN/mxquant_compute_tabs.stdout" "Vector Active Wall Cycles \(union\)")"
t_va="$(extract "$RUN/mxquant_texpands.stdout" "Vector Active Wall Cycles \(union\)")"

printf '%-42s %10s\n' "compute      - texpands (Total)"          "$((c_tot - t_tot))"
printf '%-42s %10s\n' "compute_tabs - texpands (Total)"          "$((k_tot - t_tot))"
printf '%-42s %10s\n' "compute      - texpands (Vector Active)"  "$((c_va - t_va))"
printf '%-42s %10s\n' "compute_tabs - texpands (Vector Active)"  "$((k_va - t_va))"
echo
printf '%-42s %10s\n' "TABS saving in compute (Total)"           "$((c_tot - k_tot))"
