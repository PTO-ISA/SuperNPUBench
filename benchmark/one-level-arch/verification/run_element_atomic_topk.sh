#!/usr/bin/env bash
set -euo pipefail

: "${COMPILER_DIR:?set COMPILER_DIR to the fresh Linx compiler bin directory}"
: "${SSM:?set SSM to the SuperScalarModel checkout or build root}"
: "${LINX_RUNTIME_ROOT:?set LINX_RUNTIME_ROOT to a complete Linx musl toolchain root}"
: "${API_INCLUDE:?set API_INCLUDE to the current Linx-TileOP-API include directory}"

TARGET_TRIPLE=${TARGET_TRIPLE:-linx64v5-unknown-linux-musl}
SYSROOT=${SYSROOT:-"$LINX_RUNTIME_ROOT/sysroot"}
RESOURCE_DIR=${RESOURCE_DIR:-"$LINX_RUNTIME_ROOT/lib/clang/15.0.4"}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd "$script_dir/.." && pwd)
case_dir="$bench_root/test/kernel/sort"
built_elf="$bench_root/output/kernel/sort/elf/element_atomic_topk.elf"
if [[ -n ${ARTIFACT_DIR:-} ]]; then
    artifact_dir=$ARTIFACT_DIR
else
    run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
    artifact_dir="$bench_root/output/verification/element_atomic_topk/$run_id"
fi
gfrun=${GFRUN:-"$SSM/bin/gfrun"}
gfsim=${GFSIM:-"$SSM/bin/gfsim"}

mkdir -p "$artifact_dir"
mkdir -p "$bench_root/output"

for required in \
    "$COMPILER_DIR/clang++" \
    "$COMPILER_DIR/llvm-objdump" \
    "$COMPILER_DIR/ld.lld" \
    "$SYSROOT/usr/lib/crt1.o" \
    "$SYSROOT/usr/lib/libc.a" \
    "$SYSROOT/usr/lib/libc++.a" \
    "$SYSROOT/usr/lib/libc++abi.a" \
    "$SYSROOT/usr/lib/libunwind.a" \
    "$RESOURCE_DIR/include" \
    "$API_INCLUDE/common/pto_tileop.hpp"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required compiler/runtime input: $required" >&2
        exit 1
    fi
done

set +e
"$case_dir/element_atomic_topk/check_reference.sh" \
    > >(tee "$artifact_dir/reference.log") 2>&1
reference_status=$?
set -e
printf '%s\n' "$reference_status" > "$artifact_dir/reference.exit"
if (( reference_status != 0 )); then
    exit "$reference_status"
fi
python3 "$case_dir/element_atomic_topk/generate_case.py" \
    --out "$artifact_dir/golden"
make -C "$case_dir" TESTCASE=element_atomic_topk \
    COMPILER_DIR="$COMPILER_DIR" \
    TARGET_TRIPLE="$TARGET_TRIPLE" \
    SYSROOT="$SYSROOT" \
    RESOURCE_DIR="$RESOURCE_DIR" \
    API_INCLUDE="$API_INCLUDE" \
    clean all 2>&1 | tee "$artifact_dir/build.log"
elf="$artifact_dir/element_atomic_topk.elf"
cp "$built_elf" "$elf"
"$COMPILER_DIR/llvm-objdump" -dl "$elf" > "$artifact_dir/element_atomic_topk.diss"
"$COMPILER_DIR/llvm-objdump" -t "$elf" > "$artifact_dir/element_atomic_topk.symbols"
dump_range=$(python3 "$case_dir/element_atomic_topk/compare_memory.py" \
    --symbols "$artifact_dir/element_atomic_topk.symbols" \
    --golden "$artifact_dir/golden" --print-range)
dump_base=${dump_range%%:*}

for step in disassembly gfrun gfrun-memory gfsim gfsim-invariants gfsim-memory; do
    printf '%s\n' 125 > "$artifact_dir/$step.exit"
done

write_final_provenance() {
    local run_status=$?
    trap - EXIT
    set +e
    python3 "$case_dir/element_atomic_topk/write_provenance.py" \
        --out "$artifact_dir/provenance.json" \
        --elf "$elf" --compiler-bin "$COMPILER_DIR" \
        --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
        --api-include "$API_INCLUDE" --bench-root "$bench_root/../.." \
        --golden-dir "$artifact_dir/golden" --target-triple "$TARGET_TRIPLE" \
        --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
        --artifact-dir "$artifact_dir" --dump-range "$dump_range" \
        --gfrun "$gfrun" --gfsim "$gfsim" \
        --run-exit-status "$run_status" \
        > "$artifact_dir/provenance.log" 2>&1
    local provenance_status=$?
    set -e
    if (( run_status != 0 )); then
        exit "$run_status"
    fi
    exit "$provenance_status"
}
trap write_final_provenance EXIT

set +e
python3 "$case_dir/element_atomic_topk/check_disassembly.py" \
    --dis "$artifact_dir/element_atomic_topk.diss" --self-test \
    > >(tee "$artifact_dir/disassembly.log") 2>&1
disassembly_status=$?
set -e
printf '%s\n' "$disassembly_status" > "$artifact_dir/disassembly.exit"
if (( disassembly_status != 0 )); then
    exit "$disassembly_status"
fi

for symbol in \
    element_atomic_topk_input \
    element_atomic_topk_high_hist \
    element_atomic_topk_low_hist \
    element_atomic_topk_output \
    element_atomic_topk_status; do
    if ! grep -Eq "[[:space:]]${symbol}$" \
        "$artifact_dir/element_atomic_topk.symbols"; then
        echo "missing cross-model result symbol: $symbol" >&2
        exit 1
    fi
done

set +e
"$gfrun" --pto059-execution-mask-dev \
    -s softcore.multiThreadNum=1 \
    --dump-force \
    --dump-memory "$dump_range:$artifact_dir/gfrun.mem" \
    -f "$elf" \
    > "$artifact_dir/gfrun.log" 2>&1
gfrun_status=$?
set -e
printf '%s\n' "$gfrun_status" > "$artifact_dir/gfrun.exit"
if (( gfrun_status != 0 )); then
    exit "$gfrun_status"
fi
grep -Eq '^Suaccelss to Reach the End of Benchmark! R2 = 0$' \
    "$artifact_dir/gfrun.log"
set +e
python3 "$case_dir/element_atomic_topk/compare_memory.py" \
    --symbols "$artifact_dir/element_atomic_topk.symbols" \
    --golden "$artifact_dir/golden" \
    --dump "$artifact_dir/gfrun.mem" --dump-base "$dump_base" \
    > >(tee "$artifact_dir/gfrun-memory.log") 2>&1
gfrun_memory_status=$?
set -e
printf '%s\n' "$gfrun_memory_status" > "$artifact_dir/gfrun-memory.exit"
if (( gfrun_memory_status != 0 )); then
    exit "$gfrun_memory_status"
fi

set +e
"$gfsim" --pto059-execution-mask-dev \
    -s core.threadCount=1 core.stdPeCount=1 \
    --dump-force \
    --dump-memory "$dump_range:$artifact_dir/gfsim.mem" \
    -f "$elf" \
    > "$artifact_dir/gfsim.log" 2>&1
gfsim_status=$?
set -e
printf '%s\n' "$gfsim_status" > "$artifact_dir/gfsim.exit"
if (( gfsim_status != 0 )); then
    exit "$gfsim_status"
fi
grep -Eq 'Total Cycles|total cycles|cycle_count' "$artifact_dir/gfsim.log"
if grep -Eq 'invariants:.*stq_conservation=OK.*a3_tile_violation=0([[:space:]]|$)' \
    "$artifact_dir/gfsim.log" &&
    ! grep -Eq 'LOG_ERROR|ASSERTION FAILED|a3_tile_violation peId=' \
        "$artifact_dir/gfsim.log"; then
    printf '0\n' > "$artifact_dir/gfsim-invariants.exit"
else
    printf '1\n' > "$artifact_dir/gfsim-invariants.exit"
    echo "gfsim architectural invariant check failed" >&2
    exit 1
fi
printf '%s\n' \
    'gfsim does not emit an R2 register line; exit status and the independently checked status memory segment establish completion without --test-finisher 1.' \
    > "$artifact_dir/gfsim-r2-contract.txt"
set +e
python3 "$case_dir/element_atomic_topk/compare_memory.py" \
    --symbols "$artifact_dir/element_atomic_topk.symbols" \
    --golden "$artifact_dir/golden" \
    --dump "$artifact_dir/gfsim.mem" --dump-base "$dump_base" \
    > >(tee "$artifact_dir/gfsim-memory.log") 2>&1
gfsim_memory_status=$?
set -e
printf '%s\n' "$gfsim_memory_status" > "$artifact_dir/gfsim-memory.exit"
if (( gfsim_memory_status != 0 )); then
    exit "$gfsim_memory_status"
fi

echo "element_atomic_topk: PASS"
echo "ELF: $elf"
echo "artifacts: $artifact_dir"
