#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERPROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
if [[ -z "$SUPERPROJECT_ROOT" ]]; then
    SUPERPROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi
COMPILER_DIR="${COMPILER_DIR:-${SUPERPROJECT_ROOT}/compiler/llvm/build-linxisa-clang/bin}"
DATA_OBJ_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

build_one() {
    local name="$1"
    local data_file="${DATA_OBJ_DIR}/${name}.data"
    local obj_file="${OUTPUT_DIR}/${name}.o"

    echo "Building $obj_file from $data_file"

    local asm_file="${OUTPUT_DIR}/${name}.s"

    cat > "$asm_file" << EOF
.section .data
.global _binary_${name}_data_start
_binary_${name}_data_start:
.incbin "${data_file}"
.global _binary_${name}_data_end
_binary_${name}_data_end:
.global _binary_${name}_data_size
.equ _binary_${name}_data_size, .-_binary_${name}_data_start
EOF

    $COMPILER_DIR/clang++ -target linx64-linx-none-elf -c "$asm_file" -o "$obj_file"
}

build_one "input_131072"
build_one "top_2048_out"

echo "Done building data object files"
