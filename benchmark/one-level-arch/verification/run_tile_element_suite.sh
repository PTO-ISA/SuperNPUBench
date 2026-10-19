#!/usr/bin/env bash
set -euo pipefail

default_cases=(
    gather_v2_f32_0_original
    gather_v2_s32_1_original
    gather_v2_s32_1_alias_probe
    gather_v2_f16_2_original
    view_copy_s32_0_original
    view_copy_s32_0_alias_probe
    view_copy_f16_1_original
    view_copy_f16_2_tail
    broadcast_inline_s32_original
    broadcast_inline_s32_alias_probe
    gather_f32_0_original
    gather_f32_0_alias_probe
    gather_f32_1_original
    gather_f32_1_alias_probe
    gather_f32_2_original
    gather_f32_2_alias_probe
    gather_f32_3_original
    gather_f32_3_alias_probe
    gather_f32_3_index_alias_probe
    transpose_f16_0_original
    transpose_f16_0_alias_probe
    transpose_f16_1_original
    transpose_f16_1_alias_probe
    transpose_f16_2_original
    transpose_f16_2_alias_probe
    transpose_f16_3_original
    transpose_f16_3_alias_probe
    broadcast_f16_2d_original
    broadcast_f16_2d_alias_probe
    broadcast_f16_3d_original
    broadcast_f16_3d_alias_probe
    broadcast_f16_5d_original
    broadcast_f16_5d_alias_probe
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
    concat_gather_f16_original
    concat_gather_f16_alias_probe
    broadcast_fourpe
    gather_fourpe
    concat_gather_fourpe
    concat_scatter_fourpe
    concat_scatter_s32_original
    concat_scatter_s32_alias_probe
    concat_scatter_f16_original
    concat_scatter_f16_alias_probe
    group_count_s32_original
    group_count_s32_strong
    elementwise_if_for
    elementwise_if_for_strong
    elementwise_masked_gather_scatter
    elementwise_masked_gather_scatter_strong
    topk_boundaries_0
    topk_boundaries_1
    topk_boundaries_2
    topk_boundaries
    indexed_gather_tile_element
)
if (( $# == 0 )); then
    cases=("${default_cases[@]}")
elif (( $# == 2 )) && [[ $1 == --case ]] && \
     [[ $2 == gather_v2_* || $2 == view_copy_* || $2 == broadcast_inline_s32_* || $2 == gather_f32_* || $2 == transpose_f16_* || $2 == broadcast_f16_* || $2 == indexed_gather_tile_element || $2 == histogram_tile_element || \
        $2 == selected_radix_tile_element || $2 == generic_atomic_cfg_i32 || \
        $2 == elementwise_atomic_histogram || $2 == concat_gather_s32_original || \
        $2 == concat_gather_s32_alias_probe || \
        $2 == concat_gather_f16_original || $2 == concat_gather_f16_alias_probe || \
        $2 == broadcast_fourpe || $2 == gather_fourpe || $2 == concat_gather_fourpe || $2 == concat_scatter_fourpe || $2 == concat_scatter_s32_original || $2 == concat_scatter_s32_alias_probe || \
        $2 == concat_scatter_f16_original || $2 == concat_scatter_f16_alias_probe || \
        $2 == group_count_s32_original || $2 == group_count_s32_strong || \
        $2 == elementwise_if_for || $2 == elementwise_if_for_strong || \
        $2 == elementwise_masked_gather_scatter || \
        $2 == elementwise_masked_gather_scatter_strong || \
        $2 == element_expression_chain || $2 == signed_element_expression || \
        $2 == generic_predicated_cfg_i32 || $2 == generic_typed_tile_cfg_i32 || \
        $2 == topk_boundaries* ]]; then
    cases=("$2")
else
    echo "usage: $0 [--case histogram_tile_element|selected_radix_tile_element|indexed_gather_tile_element|element_expression_chain|signed_element_expression|generic_predicated_cfg_i32|generic_typed_tile_cfg_i32|generic_atomic_cfg_i32|elementwise_atomic_histogram|concat_gather_s32_original|concat_gather_s32_alias_probe|concat_gather_f16_original|concat_gather_f16_alias_probe|group_count_s32_original|group_count_s32_strong|elementwise_if_for|elementwise_if_for_strong|elementwise_masked_gather_scatter|elementwise_masked_gather_scatter_strong|topk_boundaries[_0|_1|_2]]" >&2
    exit 2
fi

: "${COMPILER_DIR:?set COMPILER_DIR to the fresh Linx compiler bin directory}"
: "${SSM:?set SSM to the SuperScalarModel checkout or build root}"
: "${LINX_RUNTIME_ROOT:?set LINX_RUNTIME_ROOT to a complete Linx musl toolchain root}"
: "${API_INCLUDE:?set API_INCLUDE to the current Linx-TileOP-API include directory}"

# Functional acceptance covers every default case. Timing runs remain an
# explicit selection for cases supported by the separate timing work.
MODELS=${MODELS:-gfrun}
case "$MODELS" in
    gfrun|gfrun,gfsim) ;;
    *) echo "MODELS must be gfrun or gfrun,gfsim" >&2; exit 2 ;;
esac
for case_name in "${cases[@]}"; do
    if [[ ( "$case_name" == gather_v2_* || "$case_name" == view_copy_* || "$case_name" == broadcast_inline_s32_* || "$case_name" == gather_f32_* || "$case_name" == transpose_f16_* || "$case_name" == broadcast_f16_* || "$case_name" == elementwise_masked_gather_scatter* ||
            "$case_name" == broadcast_fourpe || "$case_name" == gather_fourpe || "$case_name" == concat_gather_fourpe || "$case_name" == concat_gather_f16_* || "$case_name" == concat_scatter_* ) &&
          "$MODELS" != gfrun ]]; then
        echo "$case_name is a functional-only case and requires MODELS=gfrun" >&2
        exit 2
    fi
done

TARGET_TRIPLE=${TARGET_TRIPLE:-linx64v5-unknown-linux-musl}
SYSROOT=${SYSROOT:-"$LINX_RUNTIME_ROOT/sysroot"}
RESOURCE_DIR=${RESOURCE_DIR:-"$LINX_RUNTIME_ROOT/lib/clang/15.0.4"}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench_root=$(cd "$script_dir/.." && pwd)
case_dir="$bench_root/test/kernel/element_wise/tile_element_suite"
built_dir="$bench_root/output/kernel/element_wise/tile_element_suite/elf"
api_source_include=$API_INCLUDE
api_source_root=$(dirname "$API_INCLUDE")
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

# These globals survive function unwinding when set -e aborts a case.
active_artifact=$artifact_dir
active_elf="$artifact_dir/not-built.elf"
active_golden="$artifact_dir/golden"
active_dump_range=""
active_pe_count=1
provenance_source_args=()
[[ -z ${COMPILER_SOURCE_ROOT:-} ]] || provenance_source_args+=(--compiler-source "$COMPILER_SOURCE_ROOT")
[[ -z ${MODEL_SOURCE_ROOT:-} ]] || provenance_source_args+=(--model-source "$MODEL_SOURCE_ROOT")
[[ -z ${API_SOURCE_ROOT:-} ]] || provenance_source_args+=(--api-source "$API_SOURCE_ROOT")

write_active_provenance() {
    local run_status=$1
    local verify_args=()
    if (( run_status == 0 )) || [[ -f "$active_artifact/input-provenance.json" ]]; then
        verify_args=(--verify-inputs-from "$active_artifact/input-provenance.json")
    fi
    python3 "$bench_root/test/kernel/sort/element_atomic_topk/write_provenance.py" \
        --out "$active_artifact/provenance.json" \
        --elf "$active_elf" --compiler-bin "$COMPILER_DIR" \
        --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
        --api-include "$api_source_include" --bench-root "$bench_root/../.." \
        --golden-dir "$active_golden" --target-triple "$TARGET_TRIPLE" \
        --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
        --artifact-dir "$active_artifact" --dump-range "$active_dump_range" \
        --models "$MODELS" --pe-count "$active_pe_count" --gfrun "$gfrun" \
        --gfsim "$gfsim" --run-exit-status "$run_status" --allow-incomplete \
        ${provenance_source_args[@]+"${provenance_source_args[@]}"} \
        ${verify_args[@]+"${verify_args[@]}"} \
        > "$active_artifact/provenance.log" 2>&1
}

record_failure() {
    local run_status=$?
    trap - EXIT
    if (( run_status != 0 )); then
        set +e
        write_active_provenance "$run_status"
        exit "$run_status"
    fi
}
trap record_failure EXIT
api_source_include=$(cd "$API_INCLUDE" && pwd)
api_source_root=$(cd "$api_source_include/.." && pwd)

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
    "$gfrun"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required compiler/runtime/model input: $required" >&2
        exit 1
    fi
done

if [[ "$MODELS" == gfrun,gfsim && ! -e "$gfsim" ]]; then
    echo "missing selected timing model: $gfsim" >&2
    exit 1
fi

make -C "$api_source_root" install \
    INSTALL_DIR="$installed_api_include" \
    2>&1 | tee "$artifact_dir/api-install.log"
if [[ ! -e "$installed_api_include/common/pto_tileop.hpp" ]]; then
    echo "official TileOp API install did not produce the public header" >&2
    exit 1
fi

python3 "$case_dir/test_compare_memory.py" 2>&1 \
    | tee "$artifact_dir/compare-memory-self-test.log"
python3 "$case_dir/test_check_disassembly_scope.py" 2>&1 \
    | tee "$artifact_dir/disassembly-scope-self-test.log"
python3 "$case_dir/test_concat_scatter_partition.py" 2>&1 \
    | tee "$artifact_dir/concat-scatter-partition-self-test.log"
python3 "$case_dir/test_concat_gather_f16_seed.py" 2>&1 \
    | tee "$artifact_dir/concat-f16-seed-self-test.log"
python3 "$script_dir/test_gfrun_dump.py" 2>&1 \
    | tee "$artifact_dir/gfrun-dump-self-test.log"
python3 "$bench_root/test/kernel/sort/element_atomic_topk/test_write_provenance.py" 2>&1 \
    | tee "$artifact_dir/provenance-self-test.log"
python3 "$case_dir/test_broadcast_golden.py" 2>&1 \
    | tee "$artifact_dir/broadcast-golden-self-test.log"
python3 "$case_dir/test_transpose_golden.py" 2>&1 \
    | tee "$artifact_dir/transpose-golden-self-test.log"
python3 "$case_dir/test_gather_golden.py" 2>&1 \
    | tee "$artifact_dir/gather-golden-self-test.log"
python3 "$case_dir/test_gather_v2_golden.py" 2>&1 \
    | tee "$artifact_dir/gather-v2-golden-self-test.log"
python3 "$case_dir/test_view_copy_golden.py" 2>&1 \
    | tee "$artifact_dir/view-copy-golden-self-test.log"
python3 "$script_dir/test_gfrun_dump.py" 2>&1 \
    | tee "$artifact_dir/dump-self-test.log"

run_case() {
    local case_name=$1
    local pe_count=1
    if [[ "$case_name" == *_fourpe || "$case_name" == gather_v2_* || "$case_name" == view_copy_* ]]; then pe_count=4; fi
    local case_artifact="$artifact_dir/$case_name"
    local golden="$case_artifact/golden"
    local elf="$case_artifact/$case_name.elf"
    local disassembly="$case_artifact/$case_name.diss"
    local symbols="$case_artifact/$case_name.symbols"
    local ir="$case_artifact/$case_name.ll"
    active_artifact=$case_artifact
    active_elf=$elf
    active_golden=$golden
    active_dump_range=""
    active_pe_count=$pe_count
    mkdir -p "$golden"

    python3 "$case_dir/generate_case.py" --case "$case_name" --out "$golden"
    make -B -C "$case_dir" TESTCASE="$case_name" \
        COMPILER_DIR="$COMPILER_DIR" \
        TARGET_TRIPLE="$TARGET_TRIPLE" \
        SYSROOT="$SYSROOT" \
        RESOURCE_DIR="$RESOURCE_DIR" \
        API_INCLUDE="$installed_api_include" \
        ELEMENT_FIXTURE_DIR="$golden" \
        clean all 2>&1 | tee "$case_artifact/build.log"
    cp "$built_dir/$case_name.elf" "$elf"
    if [[ "$case_name" == gather_v2_* || "$case_name" == view_copy_* || "$case_name" == broadcast_inline_s32_* || "$case_name" == gather_f32_* || "$case_name" == transpose_f16_* || "$case_name" == broadcast_f16_* || "$case_name" == signed_element_expression ||
          "$case_name" == generic_predicated_cfg_i32 ||
          "$case_name" == generic_typed_tile_cfg_i32 ||
          "$case_name" == generic_atomic_cfg_i32 ||
          "$case_name" == elementwise_atomic_histogram ||
          "$case_name" == concat_gather_s32_original ||
          "$case_name" == concat_gather_s32_alias_probe ||
          "$case_name" == concat_gather_f16_original ||
          "$case_name" == concat_gather_f16_alias_probe ||
          "$case_name" == broadcast_fourpe || "$case_name" == gather_fourpe || "$case_name" == concat_gather_fourpe || "$case_name" == concat_scatter_fourpe ||
          "$case_name" == concat_scatter_s32_original ||
          "$case_name" == concat_scatter_s32_alias_probe ||
          "$case_name" == concat_scatter_f16_original ||
          "$case_name" == concat_scatter_f16_alias_probe ||
          "$case_name" == group_count_s32_original ||
          "$case_name" == group_count_s32_strong ||
          "$case_name" == elementwise_if_for ||
          "$case_name" == elementwise_if_for_strong ||
          "$case_name" == elementwise_masked_gather_scatter ||
          "$case_name" == elementwise_masked_gather_scatter_strong ||
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
            ELEMENT_FIXTURE_DIR="$golden" \
            IR_TARGET="$ir" ir 2>&1 | tee "$case_artifact/ir-build.log"
    fi
    "$COMPILER_DIR/llvm-objdump" -dl \
        --disassembler-options=no-tile-macros "$elf" > "$disassembly"
    "$COMPILER_DIR/llvm-objdump" -t "$elf" > "$symbols"

    if [[ "$case_name" == gather_v2_* || "$case_name" == view_copy_* || "$case_name" == broadcast_inline_s32_* || "$case_name" == gather_f32_* || "$case_name" == transpose_f16_* || "$case_name" == broadcast_f16_* || "$case_name" == signed_element_expression ||
          "$case_name" == generic_predicated_cfg_i32 ||
          "$case_name" == generic_typed_tile_cfg_i32 ||
          "$case_name" == generic_atomic_cfg_i32 ||
          "$case_name" == elementwise_atomic_histogram ||
          "$case_name" == concat_gather_s32_original ||
          "$case_name" == concat_gather_s32_alias_probe ||
          "$case_name" == concat_gather_f16_original ||
          "$case_name" == concat_gather_f16_alias_probe ||
          "$case_name" == broadcast_fourpe || "$case_name" == gather_fourpe || "$case_name" == concat_gather_fourpe || "$case_name" == concat_scatter_fourpe ||
          "$case_name" == concat_scatter_s32_original ||
          "$case_name" == concat_scatter_s32_alias_probe ||
          "$case_name" == concat_scatter_f16_original ||
          "$case_name" == concat_scatter_f16_alias_probe ||
          "$case_name" == group_count_s32_original ||
          "$case_name" == group_count_s32_strong ||
          "$case_name" == elementwise_if_for ||
          "$case_name" == elementwise_if_for_strong ||
          "$case_name" == elementwise_masked_gather_scatter ||
          "$case_name" == elementwise_masked_gather_scatter_strong ||
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
    active_dump_range=$dump_range

    python3 "$bench_root/test/kernel/sort/element_atomic_topk/write_provenance.py" \
        --out "$case_artifact/input-provenance.json" \
        --elf "$elf" --compiler-bin "$COMPILER_DIR" \
        --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
        --api-include "$api_source_include" --bench-root "$bench_root/../.." \
        --golden-dir "$golden" --target-triple "$TARGET_TRIPLE" \
        --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
        --artifact-dir "$case_artifact" --dump-range "$dump_range" \
        --models "$MODELS" --pe-count "$pe_count" --gfrun "$gfrun" --gfsim "$gfsim" --run-exit-status 125 \
        ${provenance_source_args[@]+"${provenance_source_args[@]}"} \
        > "$case_artifact/input-provenance.log" 2>&1

    python3 "$script_dir/gfrun_dump.py" \
        --provenance "$case_artifact/input-provenance.json"
    grep -Eq '^(Success|Suaccelss) to Reach the End of Benchmark! R2 = 0$' \
        "$case_artifact/gfrun.log"
    python3 "$case_dir/compare_memory.py" \
        --symbols "$symbols" --golden "$golden" \
        --dump "$case_artifact/gfrun.mem" --dump-base "$dump_base" \
        2>&1 | tee "$case_artifact/gfrun-memory.log"

    if [[ "$MODELS" == gfrun,gfsim ]]; then
        printf '%q ' "$gfsim" --pto059-execution-mask-dev \
            -s core.threadCount=1 core.stdPeCount=1 bctrl.assemble_enable=true --dump-force \
            --dump-memory "$dump_range:$case_artifact/gfsim.mem" -f "$elf" \
            > "$case_artifact/gfsim.argv"
        printf '\n' >> "$case_artifact/gfsim.argv"
        "$gfsim" --pto059-execution-mask-dev \
            -s core.threadCount=1 core.stdPeCount=1 bctrl.assemble_enable=true \
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

    fi

    write_active_provenance 0
    active_artifact=$artifact_dir
    active_elf="$artifact_dir/not-built.elf"
    active_golden="$artifact_dir/golden"
    active_dump_range=""
    active_pe_count=1
    echo "$case_name: PASS"
}

for case_name in "${cases[@]}"; do
    run_case "$case_name"
done

printf '%s\n' \
    "All ${#cases[@]} selected fresh TileOp ELFs passed compile/disassembly and independent memory checks for models=$MODELS; timing invariants checked when gfsim is selected." \
    > "$artifact_dir/PASS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$artifact_dir/FROZEN"
hash_manifest="$artifact_dir/.SHA256SUMS.tmp"
find "$artifact_dir" -type f ! -name '.SHA256SUMS.tmp' -print0 \
    | sort -z | xargs -0 shasum -a 256 > "$hash_manifest"
mv "$hash_manifest" "$artifact_dir/SHA256SUMS"
chmod -R a-w "$artifact_dir"

echo "tile_element_suite: PASS"
echo "artifacts: $artifact_dir"
