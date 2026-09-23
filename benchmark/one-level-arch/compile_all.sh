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
compile_operator "$REPO_ROOT/test/solution/moe_dispatch" "solution/moe_dispatch"
compile_operator "$REPO_ROOT/test/solution/moe_combine" "solution/moe_combine"
compile_operator "$REPO_ROOT/test/solution/mega_moe" "solution/mega_moe"
compile_operator "$REPO_ROOT/test/solution/group_token_vec" "solution/group_token_vec"
compile_operator "$REPO_ROOT/test/solution/group_token_old" "solution/group_token_old"
compile_operator "$REPO_ROOT/test/solution/normalization/rms_norm" "solution/normalization/rms_norm"
compile_operator "$REPO_ROOT/test/solution/normalization/rms_norm_split_r" "solution/normalization/rms_norm_split_r"
compile_operator "$REPO_ROOT/test/solution/normalization/group_norm_grad" "solution/normalization/group_norm_grad"
compile_operator "$REPO_ROOT/test/solution/normalization/group_norm_grad_1d" "solution/normalization/group_norm_grad_1d"
compile_operator "$REPO_ROOT/test/solution/view_copy" "solution/view_copy"
compile_operator "$REPO_ROOT/test/solution/gather_v2" "solution/gather_v2"
compile_operator "$REPO_ROOT/test/solution/conv2d" "solution/conv2d"
compile_operator "$REPO_ROOT/test/kernel/vec" "vec"
compile_operator "$REPO_ROOT/test/kernel/broadcast" "broadcast"
compile_operator "$REPO_ROOT/test/kernel/concat" "concat"
compile_operator "$REPO_ROOT/test/kernel/conv2d" "conv2d"
compile_operator "$REPO_ROOT/test/kernel/element_wise/gelu" "element_wise/gelu"
compile_operator "$REPO_ROOT/test/kernel/gather" "gather"
compile_operator "$REPO_ROOT/test/kernel/reduction/cumsum_row" "reduction/cumsum_row"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducemax_row" "reduction/reducemax_row"
compile_operator "$REPO_ROOT/test/kernel/reduction/reduceprod_row" "reduction/reduceprod_row"
compile_operator "$REPO_ROOT/test/kernel/reduction/reducesum_row" "reduction/reducesum_row"
compile_operator "$REPO_ROOT/test/kernel/transpose" "transpose"
compile_operator "$REPO_ROOT/test/kernel/matmul" "matmul"
compile_operator "$REPO_ROOT/test/kernel/fa" "fa"

echo ""
echo "=========================================="
echo "Full compilation completed!"
echo "=========================================="
echo ""
echo "Generated ELF files:"
find "$REPO_ROOT/output" -name "*.elf" -type f | wc -l
echo "ELF files are located in: $REPO_ROOT/output/"
