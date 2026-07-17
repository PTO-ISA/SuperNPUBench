#!/usr/bin/env bash
set -euo pipefail
: "${COMPILER_DIR:?set COMPILER_DIR to compiler/llvm/build-linxisa-clang/bin}"
DATA_OBJ_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

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

    $COMPILER_DIR/clang++ -target linx64-linx-none-elf -c "$asm_file" -o "$obj_file"
}

build_one "inserted_slot"
build_one "lookup_keys"
build_one "lookup_values"

echo "Done building data object files"
