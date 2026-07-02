#!/bin/bash
COMPILER_DIR="${COMPILER_DIR:-/remote/lms01/j00827727/jcore/compilers/linx_blockisa_llvm_musl0.56.16/bin}"
DATA_OBJ_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

build_one() {
    local name="$1"
    local data_file="${DATA_OBJ_DIR}/${name}"
    local obj_file="${OUTPUT_DIR}/${name}.o"

    # Derive symbol name from filename (replace dots and hyphens with underscores)
    local sym_name=$(echo "$name" | sed 's/[.-]/_/g')

    echo "Building $obj_file from $data_file (symbol: _binary_${sym_name}_start)"

    local asm_file="${OUTPUT_DIR}/${name}.s"

    cat > "$asm_file" << EOF
.section .data
.global _binary_${sym_name}_start
_binary_${sym_name}_start:
.incbin "${data_file}"
.global _binary_${sym_name}_end
_binary_${sym_name}_end:
.global _binary_${sym_name}_size
.equ _binary_${sym_name}_size, .-_binary_${sym_name}_start
EOF

    $COMPILER_DIR/clang++ -target linx64v5 -c "$asm_file" -o "$obj_file"
}

build_one "buckets.bin"
build_one "buckets_size.bin"
build_one "lookup_keys.bin"
build_one "lookedup_values.bin"
build_one "key_score_digest.bin"

echo "Done building hkv data object files"
