#!/usr/bin/env bash
set -euo pipefail

default_cases=(
    histogram_tile_element
    selected_radix_tile_element
    element_expression_chain
    signed_element_expression
    generic_predicated_cfg_i32
    generic_typed_tile_cfg_i32
    generic_atomic_cfg_i32
    elementwise_atomic_histogram
    concat_gather_s32_original
    concat_gather_s32_alias_probe
    topk_boundaries_0
    topk_boundaries_1
    topk_boundaries_2
    topk_boundaries
    indexed_gather_tile_element
)
if (( $# == 0 )); then
    cases=("${default_cases[@]}")
elif (( $# == 2 )) && [[ $1 == --case ]] && \
     [[ $2 == indexed_gather_tile_element || $2 == histogram_tile_element || \
        $2 == selected_radix_tile_element || $2 == generic_atomic_cfg_i32 || \
        $2 == elementwise_atomic_histogram || $2 == concat_gather_s32_original || \
        $2 == concat_gather_s32_alias_probe || \
        $2 == element_expression_chain || $2 == signed_element_expression || \
        $2 == generic_predicated_cfg_i32 || $2 == generic_typed_tile_cfg_i32 || \
        $2 == topk_boundaries* ]]; then
    cases=("$2")
else
    echo "usage: $0 [--case histogram_tile_element|selected_radix_tile_element|indexed_gather_tile_element|element_expression_chain|signed_element_expression|generic_predicated_cfg_i32|generic_typed_tile_cfg_i32|generic_atomic_cfg_i32|elementwise_atomic_histogram|concat_gather_s32_original|concat_gather_s32_alias_probe|topk_boundaries[_0|_1|_2]]" >&2
    exit 2
fi

: "${COMPILER_DIR:?set COMPILER_DIR to the fresh Linx compiler bin directory}"
: "${SSM:?set SSM to the SuperScalarModel checkout or build root}"
: "${LINX_RUNTIME_ROOT:?set LINX_RUNTIME_ROOT to a complete Linx musl toolchain root}"
: "${API_INCLUDE:?set API_INCLUDE to the current Linx-TileOP-API include directory}"

TARGET_TRIPLE=${TARGET_TRIPLE:-linx64v5-unknown-linux-musl}
SYSROOT=${SYSROOT:-"$LINX_RUNTIME_ROOT/sysroot"}
RESOURCE_DIR=${RESOURCE_DIR:-"$LINX_RUNTIME_ROOT/lib/clang/15.0.4"}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd "$script_dir/.." && pwd)
case_dir="$bench_root/test/kernel/element_wise/tile_element_suite"
built_dir="$bench_root/output/kernel/element_wise/tile_element_suite/elf"
api_source_include=$(cd "$API_INCLUDE" && pwd)
api_source_root=$(cd "$api_source_include/.." && pwd)
installed_api_include="$RESOURCE_DIR/include/tileop-api"
gfrun=${GFRUN:-"$SSM/bin/gfrun"}
gfsim=${GFSIM:-"$SSM/bin/gfsim"}
if [[ -n ${ARTIFACT_DIR:-} ]]; then
    artifact_dir=$ARTIFACT_DIR
else
    run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
    artifact_dir="$bench_root/output/verification/tile_element_suite/$run_id"
fi

if [[ -e "$artifact_dir" ]]; then
    echo "artifact directory already exists: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"

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
    "$api_source_include/common/pto_tileop.hpp" \
    "$api_source_root/Makefile" \
    "$gfrun" \
    "$gfsim"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required compiler/runtime/model input: $required" >&2
        exit 1
    fi
done

make -C "$api_source_root" install \
    INSTALL_DIR="$installed_api_include" \
    2>&1 | tee "$artifact_dir/api-install.log"
if [[ ! -e "$installed_api_include/common/pto_tileop.hpp" ]]; then
    echo "official TileOp API install did not produce the public header" >&2
    exit 1
fi

python3 "$case_dir/test_compare_memory.py" 2>&1 \
    | tee "$artifact_dir/compare-memory-self-test.log"

run_case() {
    local case_name=$1
    local case_artifact="$artifact_dir/$case_name"
    local golden="$case_artifact/golden"
    local elf="$case_artifact/$case_name.elf"
    local disassembly="$case_artifact/$case_name.diss"
    local symbols="$case_artifact/$case_name.symbols"
    local ir="$case_artifact/$case_name.ll"
    mkdir -p "$golden"

    python3 "$case_dir/generate_case.py" --case "$case_name" --out "$golden"
    make -B -C "$case_dir" TESTCASE="$case_name" \
        COMPILER_DIR="$COMPILER_DIR" \
        TARGET_TRIPLE="$TARGET_TRIPLE" \
        SYSROOT="$SYSROOT" \
        RESOURCE_DIR="$RESOURCE_DIR" \
        API_INCLUDE="$installed_api_include" \
        clean all 2>&1 | tee "$case_artifact/build.log"
    cp "$built_dir/$case_name.elf" "$elf"
    if [[ "$case_name" == signed_element_expression ||
          "$case_name" == generic_predicated_cfg_i32 ||
          "$case_name" == generic_typed_tile_cfg_i32 ||
          "$case_name" == generic_atomic_cfg_i32 ||
          "$case_name" == elementwise_atomic_histogram ||
          "$case_name" == concat_gather_s32_original ||
          "$case_name" == concat_gather_s32_alias_probe ||
          "$case_name" == histogram_tile_element ||
          "$case_name" == selected_radix_tile_element ||
          "$case_name" == element_expression_chain || \
          "$case_name" == indexed_gather_tile_element || \
          "$case_name" == topk_boundaries* ]]; then
        make -B -C "$case_dir" TESTCASE="$case_name" \
            COMPILER_DIR="$COMPILER_DIR" \
            TARGET_TRIPLE="$TARGET_TRIPLE" \
            SYSROOT="$SYSROOT" \
            RESOURCE_DIR="$RESOURCE_DIR" \
            API_INCLUDE="$installed_api_include" \
            IR_TARGET="$ir" ir 2>&1 | tee "$case_artifact/ir-build.log"
    fi
    "$COMPILER_DIR/llvm-objdump" -dl \
        --disassembler-options=no-tile-macros "$elf" > "$disassembly"
    "$COMPILER_DIR/llvm-objdump" -t "$elf" > "$symbols"

    if [[ "$case_name" == signed_element_expression ||
          "$case_name" == generic_predicated_cfg_i32 ||
          "$case_name" == generic_typed_tile_cfg_i32 ||
          "$case_name" == generic_atomic_cfg_i32 ||
          "$case_name" == elementwise_atomic_histogram ||
          "$case_name" == concat_gather_s32_original ||
          "$case_name" == concat_gather_s32_alias_probe ||
          "$case_name" == histogram_tile_element ||
          "$case_name" == selected_radix_tile_element ||
          "$case_name" == element_expression_chain || \
          "$case_name" == indexed_gather_tile_element || \
          "$case_name" == topk_boundaries* ]]; then
        python3 "$case_dir/check_disassembly.py" \
            --case "$case_name" --dis "$disassembly" --ir "$ir" \
            --self-test 2>&1 | tee "$case_artifact/disassembly.log"
    else
        python3 "$case_dir/check_disassembly.py" \
            --case "$case_name" --dis "$disassembly" --self-test \
            2>&1 | tee "$case_artifact/disassembly.log"
    fi

    local dump_range
    dump_range=$(python3 "$case_dir/compare_memory.py" \
        --symbols "$symbols" --golden "$golden" --print-range)
    local dump_base=${dump_range%%:*}

    printf '%q ' "$gfrun" --pto059-execution-mask-dev \
        -s softcore.multiThreadNum=1 --dump-force \
        --dump-memory "$dump_range:$case_artifact/gfrun.mem" -f "$elf" \
        > "$case_artifact/gfrun.argv"
    printf '\n' >> "$case_artifact/gfrun.argv"
    "$gfrun" --pto059-execution-mask-dev \
        -s softcore.multiThreadNum=1 \
        --dump-force \
        --dump-memory "$dump_range:$case_artifact/gfrun.mem" \
        -f "$elf" > "$case_artifact/gfrun.log" 2>&1
    printf '0\n' > "$case_artifact/gfrun.exit"
    grep -Eq '^Suaccelss to Reach the End of Benchmark! R2 = 0$' \
        "$case_artifact/gfrun.log"
    python3 "$case_dir/compare_memory.py" \
        --symbols "$symbols" --golden "$golden" \
        --dump "$case_artifact/gfrun.mem" --dump-base "$dump_base" \
        2>&1 | tee "$case_artifact/gfrun-memory.log"

    printf '%q ' "$gfsim" --pto059-execution-mask-dev \
        -s core.threadCount=1 core.stdPeCount=1 --dump-force \
        --dump-memory "$dump_range:$case_artifact/gfsim.mem" -f "$elf" \
        > "$case_artifact/gfsim.argv"
    printf '\n' >> "$case_artifact/gfsim.argv"
    "$gfsim" --pto059-execution-mask-dev \
        -s core.threadCount=1 core.stdPeCount=1 \
        --dump-force \
        --dump-memory "$dump_range:$case_artifact/gfsim.mem" \
        -f "$elf" > "$case_artifact/gfsim.log" 2>&1
    printf '0\n' > "$case_artifact/gfsim.exit"
    grep -Eq 'Total Cycles|total cycles|cycle_count' "$case_artifact/gfsim.log"
    local invariant_status=0
    if ! grep -Eq \
        'invariants:.*stq_conservation=OK \(created=[0-9]+ destroyed=[0-9]+\).*a3_tile_violation=0([[:space:]]|$)' \
        "$case_artifact/gfsim.log"; then
        invariant_status=1
    fi
    if grep -Eiq \
        'LOG_ERROR|ASSERTION|(^|[^[:alpha:]])assert([^[:alpha:]]|$)' \
        "$case_artifact/gfsim.log"; then
        invariant_status=1
    fi
    printf '%s\n' "$invariant_status" \
        > "$case_artifact/gfsim-invariants.exit"
    if (( invariant_status != 0 )); then
        echo "$case_name: gfsim invariant gate failed" >&2
        exit "$invariant_status"
    fi
    printf '%s\n' \
        'gfsim completion is established by exit status and checked status memory; it does not emit an R2 line.' \
        > "$case_artifact/gfsim-r2-contract.txt"
    python3 "$case_dir/compare_memory.py" \
        --symbols "$symbols" --golden "$golden" \
        --dump "$case_artifact/gfsim.mem" --dump-base "$dump_base" \
        2>&1 | tee "$case_artifact/gfsim-memory.log"

    python3 "$bench_root/test/kernel/sort/element_atomic_topk/write_provenance.py" \
        --out "$case_artifact/provenance.json" \
        --elf "$elf" --compiler-bin "$COMPILER_DIR" \
        --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
        --api-include "$api_source_include" --bench-root "$bench_root/../.." \
        --golden-dir "$golden" --target-triple "$TARGET_TRIPLE" \
        --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
        --artifact-dir "$case_artifact" --dump-range "$dump_range" \
        --gfrun "$gfrun" --gfsim "$gfsim" --run-exit-status 0 \
        > "$case_artifact/provenance.log" 2>&1
    echo "$case_name: PASS"
}

for case_name in "${cases[@]}"; do
    run_case "$case_name"
done

printf '%s\n' \
    "All ${#cases[@]} selected fresh TileOp ELFs passed disassembly, gfrun, gfrun memory, gfsim invariants, and gfsim memory checks." \
    > "$artifact_dir/PASS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$artifact_dir/FROZEN"
hash_manifest="$artifact_dir/.SHA256SUMS.tmp"
find "$artifact_dir" -type f ! -name '.SHA256SUMS.tmp' -print0 \
    | sort -z | xargs -0 shasum -a 256 > "$hash_manifest"
mv "$hash_manifest" "$artifact_dir/SHA256SUMS"
chmod -R a-w "$artifact_dir"

echo "tile_element_suite: PASS"
echo "artifacts: $artifact_dir"
