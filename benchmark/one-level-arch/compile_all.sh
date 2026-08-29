#!/bin/bash
# PTO ISA compilation script for all kernel operators

# Don't use set -e as some operators may fail to compile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COMPILER_DIR:?Set COMPILER_DIR to the in-repo Linx compiler bin directory}"
export COMPILER_DIR
REPO_ROOT=${REPO_ROOT:-$SCRIPT_DIR}

echo "=========================================="
echo "[PTO ISA] Starting full compilation"
echo "REPO_ROOT: $REPO_ROOT"
echo "=========================================="

# Function to compile an operator
compile_operator() {
    local operator_path=$1
    local operator_name=$2

    echo ""
    echo "------------------------------------------"
    echo "Compiling: $operator_name"
    echo "Path: $operator_path"
    echo "------------------------------------------"

    if [ ! -d "$operator_path" ]; then
        echo "Warning: Directory not found: $operator_path"
        return 1
    fi

    cd "$operator_path"

    if [ -f "compile.all" ]; then
        echo "Running compile.all with baremetal=${baremetal:-off}..."
        export baremetal=${baremetal:-off}
        if bash compile.all 2>&1; then
            echo "✓ $operator_name compilation completed"
        else
            echo "✗ $operator_name compilation failed"
        fi
    else
        echo "Warning: No compile.all found in $operator_path"
        return 1
    fi
}

# Compile all operators
compile_operator "$REPO_ROOT/test/kernel/matmul" "matmul"
compile_operator "$REPO_ROOT/test/kernel/broadcast" "broadcast"
compile_operator "$REPO_ROOT/test/kernel/concat" "concat"
compile_operator "$REPO_ROOT/test/kernel/gather" "gather"
compile_operator "$REPO_ROOT/test/kernel/transpose" "transpose"
compile_operator "$REPO_ROOT/test/kernel/element_wise/gelu" "gelu"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducemax_col" "reducemax_col"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducemax_row" "reducemax_row"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducesum_col" "reducesum_col"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducesum_row" "reducesum_row"
compile_operator "$REPO_ROOT/test/kernel/control" "control"
compile_operator "$REPO_ROOT/test/kernel/fa" "fa"
compile_operator "$REPO_ROOT/test/kernel/sort" "sort"
compile_operator "$REPO_ROOT/test/kernel/deepseek" "deepseek"
compile_operator "$REPO_ROOT/test/kernel/flashMLA" "flashMLA"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/vec" "multi_thread/vec"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/broadcast" "multi_thread/broadcast"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/concat" "multi_thread/concat"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/conv2d" "multi_thread/conv2d"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/element_wise/gelu" "multi_thread/element_wise/gelu"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/gather" "multi_thread/gather"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/normalization/rms_norm" "multi_thread/normalization/rms_norm"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/normalization/rms_norm_binary" "multi_thread/normalization/rms_norm_binary"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/reduction/cumsum_row" "multi_thread/reduction/cumsum_row"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/reduction/reducemax_row" "multi_thread/reduction/reducemax_row"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/reduction/reduceprod_row" "multi_thread/reduction/reduceprod_row"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/reduction/reducesum_row" "multi_thread/reduction/reducesum_row"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/transpose" "multi_thread/transpose"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/matmul" "multi_thread/matmul"
compile_operator "$REPO_ROOT/test/kernel/multi_thread/fa" "multi_thread/fa"

echo ""
echo "=========================================="
echo "Full compilation completed!"
echo "=========================================="
echo ""
echo "Generated ELF files:"
find "$REPO_ROOT/output" -name "*.elf" -type f | wc -l
echo "ELF files are located in: $REPO_ROOT/output/"
