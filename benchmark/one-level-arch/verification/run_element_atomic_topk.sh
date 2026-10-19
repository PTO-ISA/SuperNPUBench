#!/usr/bin/env bash
set -euo pipefail

validate_models() {
    case "$1" in
        gfrun|gfrun,gfsim) return 0 ;;
        *) echo "MODELS must be gfrun or gfrun,gfsim" >&2; return 2 ;;
    esac
}

require_selected_models() {
    local selected=$1
    local selected_gfrun=$2
    local selected_gfsim=$3
    if [[ ! -e "$selected_gfrun" ]]; then
        echo "missing selected functional model: $selected_gfrun" >&2
        return 1
    fi
    if [[ "$selected" == gfrun,gfsim && ! -e "$selected_gfsim" ]]; then
        echo "missing selected timing model: $selected_gfsim" >&2
        return 1
    fi
}

if [[ ${1:-} == --self-test-model-selection ]]; then
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT
    : > "$tmp_dir/gfrun"
    validate_models gfrun
    require_selected_models gfrun "$tmp_dir/gfrun" "$tmp_dir/missing-gfsim"
    ! require_selected_models gfrun "$tmp_dir/missing-gfrun" "$tmp_dir/missing-gfsim"
    ! require_selected_models gfrun,gfsim "$tmp_dir/gfrun" "$tmp_dir/missing-gfsim"
    ! validate_models gfsim
    echo "element_atomic_topk model selection self-test: PASS"
    exit 0
fi

: "${COMPILER_DIR:?set COMPILER_DIR to the fresh Linx compiler bin directory}"
: "${SSM:?set SSM to the SuperScalarModel checkout or build root}"
: "${LINX_RUNTIME_ROOT:?set LINX_RUNTIME_ROOT to a complete Linx musl toolchain root}"
: "${API_INCLUDE:?set API_INCLUDE to the current Linx-TileOP-API include directory}"

TARGET_TRIPLE=${TARGET_TRIPLE:-linx64v5-unknown-linux-musl}
SYSROOT=${SYSROOT:-"$LINX_RUNTIME_ROOT/sysroot"}
RESOURCE_DIR=${RESOURCE_DIR:-"$LINX_RUNTIME_ROOT/lib/clang/15.0.4"}
MODELS=${MODELS:-gfrun}
validate_models "$MODELS"

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

if [[ -e "$artifact_dir" ]]; then
    echo "artifact directory already exists: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"
elf="$artifact_dir/element_atomic_topk.elf"
ir="$artifact_dir/element_atomic_topk.ll"
dump_range=""
provenance_source_args=()
[[ -z ${COMPILER_SOURCE_ROOT:-} ]] || provenance_source_args+=(--compiler-source "$COMPILER_SOURCE_ROOT")
[[ -z ${MODEL_SOURCE_ROOT:-} ]] || provenance_source_args+=(--model-source "$MODEL_SOURCE_ROOT")
[[ -z ${API_SOURCE_ROOT:-} ]] || provenance_source_args+=(--api-source "$API_SOURCE_ROOT")
provenance_model_args=(--models "$MODELS" --gfrun "$gfrun")
if [[ "$MODELS" == gfrun,gfsim ]]; then
    provenance_model_args+=(--gfsim "$gfsim")
fi

write_final_provenance() {
    local run_status=$?
    trap - EXIT
    set +e
    local verify_args=()
    if (( run_status == 0 )) || [[ -f "$artifact_dir/input-provenance.json" ]]; then
        verify_args=(--verify-inputs-from "$artifact_dir/input-provenance.json")
    fi
    python3 "$case_dir/element_atomic_topk/write_provenance.py" \
        --out "$artifact_dir/provenance.json" \
        --elf "$elf" --compiler-bin "$COMPILER_DIR" \
        --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
        --api-include "$API_INCLUDE" --bench-root "$bench_root/../.." \
        --golden-dir "$artifact_dir/golden" --target-triple "$TARGET_TRIPLE" \
        --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
        --artifact-dir "$artifact_dir" --dump-range "$dump_range" \
        "${provenance_model_args[@]}" \
        --run-exit-status "$run_status" --allow-incomplete \
        ${provenance_source_args[@]+"${provenance_source_args[@]}"} ${verify_args[@]+"${verify_args[@]}"} \
        > "$artifact_dir/provenance.log" 2>&1
    local provenance_status=$?
    set -e
    if (( run_status != 0 )); then
        exit "$run_status"
    fi
    if (( provenance_status == 0 )); then
        echo "element_atomic_topk: PASS models=$MODELS"
        echo "ELF: $elf"
        echo "artifacts: $artifact_dir"
    fi
    exit "$provenance_status"
}
trap write_final_provenance EXIT

require_selected_models "$MODELS" "$gfrun" "$gfsim"
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

# Keep the selected resource tree intact and install the requested API into
# this run's resource overlay before -mlxbc can preinclude stale headers.
resource_overlay="$artifact_dir/clang-resource"
mkdir -p "$resource_overlay"
cp -R "$RESOURCE_DIR/." "$resource_overlay/"
RESOURCE_DIR=$resource_overlay
rm -rf "$RESOURCE_DIR/include/tileop-api"
api_source_root=$(cd "$API_INCLUDE/.." && pwd)
if [[ -f "$api_source_root/Makefile" && -d "$api_source_root/include" ]]; then
    make -C "$api_source_root" install \
        INSTALL_DIR="$RESOURCE_DIR/include/tileop-api" \
        2>&1 | tee "$artifact_dir/api-install.log"
else
    cp -R "$API_INCLUDE" "$RESOURCE_DIR/include/tileop-api"
fi

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
cp "$built_elf" "$elf"
make -B -C "$case_dir" TESTCASE=element_atomic_topk \
    COMPILER_DIR="$COMPILER_DIR" \
    TARGET_TRIPLE="$TARGET_TRIPLE" \
    SYSROOT="$SYSROOT" \
    RESOURCE_DIR="$RESOURCE_DIR" \
    API_INCLUDE="$API_INCLUDE" \
    IR_TARGET="$ir" ir 2>&1 | tee "$artifact_dir/ir-build.log"
"$COMPILER_DIR/llvm-objdump" -dl "$elf" > "$artifact_dir/element_atomic_topk.diss"
"$COMPILER_DIR/llvm-objdump" -t "$elf" > "$artifact_dir/element_atomic_topk.symbols"
dump_range=$(python3 "$case_dir/element_atomic_topk/compare_memory.py" \
    --symbols "$artifact_dir/element_atomic_topk.symbols" \
    --golden "$artifact_dir/golden" --print-range)
dump_base=${dump_range%%:*}

steps=(disassembly gfrun gfrun-memory)
if [[ "$MODELS" == gfrun,gfsim ]]; then
    steps+=(gfsim gfsim-invariants gfsim-memory)
fi
for step in "${steps[@]}"; do
    printf '%s\n' 125 > "$artifact_dir/$step.exit"
done



set +e
python3 "$case_dir/element_atomic_topk/check_disassembly.py" \
    --dis "$artifact_dir/element_atomic_topk.diss" --ir "$ir" --self-test \
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

python3 "$case_dir/element_atomic_topk/write_provenance.py" \
    --out "$artifact_dir/input-provenance.json" \
    --elf "$elf" --compiler-bin "$COMPILER_DIR" \
    --model-root "$SSM" --runtime-root "$LINX_RUNTIME_ROOT" \
    --api-include "$API_INCLUDE" --bench-root "$bench_root/../.." \
    --golden-dir "$artifact_dir/golden" --target-triple "$TARGET_TRIPLE" \
    --sysroot "$SYSROOT" --resource-dir "$RESOURCE_DIR" \
    --artifact-dir "$artifact_dir" --dump-range "$dump_range" \
    "${provenance_model_args[@]}" ${provenance_source_args[@]+"${provenance_source_args[@]}"} --run-exit-status 125 \
    > "$artifact_dir/input-provenance.log" 2>&1

set +e
python3 "$script_dir/gfrun_dump.py" \
    --provenance "$artifact_dir/input-provenance.json"
gfrun_status=$?
set -e
if (( gfrun_status != 0 )); then
    exit "$gfrun_status"
fi
grep -Eq '^(Success|Suaccelss) to Reach the End of Benchmark! R2 = 0$' \
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

if [[ "$MODELS" == gfrun,gfsim ]]; then
    set +e
    "$gfsim" --pto059-execution-mask-dev \
        -s core.threadCount=1 core.stdPeCount=1 bctrl.assemble_enable=true \
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
        ! grep -Eiq 'LOG_ERROR|ASSERTION|(^|[^[:alpha:]])assert([^[:alpha:]]|$)|a3_tile_violation peId=' \
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
fi
