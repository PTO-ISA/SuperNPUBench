#!/bin/bash
set -euo pipefail

COMPILER_DIR="${COMPILER_DIR:-/usr/bin}"
LINX_TARGET="${LINX_TARGET:-linx64-linx-none-elf}"
DATA_OBJ_DIR="${1:?data object directory required}"
OUTPUT_DIR="${2:?output directory required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "${DATA_OBJ_DIR}/inserted_slot.data" ||
      ! -f "${DATA_OBJ_DIR}/lookup_keys.data" ||
      ! -f "${DATA_OBJ_DIR}/lookup_values.data" ]]; then
    (cd "$CASE_DIR" && python3 gen_data_simple.py)
fi

build_one() {
    local name="$1"
    local data_file="${DATA_OBJ_DIR}/${name}.data"
    local obj_file="${OUTPUT_DIR}/${name}.o"

    echo "Building $obj_file from $data_file"

    # Create assembly file in output directory
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

    "${COMPILER_DIR}/clang++" -target "$LINX_TARGET" -c "$asm_file" -o "$obj_file"
}

build_one "inserted_slot"
build_one "lookup_keys"
build_one "lookup_values"

echo "Done building data object files"
