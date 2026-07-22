#!/usr/bin/env bash
set -uo pipefail

# LinxISA compilation script for all kernel operators

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COMPILER_DIR:?Set COMPILER_DIR to the in-repo Linx compiler bin directory}"
export COMPILER_DIR
REPO_ROOT=${REPO_ROOT:-$SCRIPT_DIR}

echo "=========================================="
echo "[LinxISA] Starting full compilation"
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
            return 1
        fi
    else
        echo "Warning: No compile.all found in $operator_path"
        return 1
    fi
}

# Compile all operators
failures=()
compile_operator "$REPO_ROOT/test/tileop_api" "tileop_api" || failures+=("tileop_api")
compile_operator "$REPO_ROOT/test/kernel/matmul" "matmul" || failures+=("matmul")
compile_operator "$REPO_ROOT/test/kernel/broadcast" "broadcast" || failures+=("broadcast")
compile_operator "$REPO_ROOT/test/kernel/concat" "concat" || failures+=("concat")
compile_operator "$REPO_ROOT/test/kernel/gather" "gather" || failures+=("gather")
compile_operator "$REPO_ROOT/test/kernel/transpose" "transpose" || failures+=("transpose")
compile_operator "$REPO_ROOT/test/kernel/element_wise/gelu" "gelu" || failures+=("gelu")
compile_operator "$REPO_ROOT/test/kernel/reduction/reducemax_col" "reducemax_col" || failures+=("reducemax_col")
compile_operator "$REPO_ROOT/test/kernel/reduction/reducemax_row" "reducemax_row" || failures+=("reducemax_row")
compile_operator "$REPO_ROOT/test/kernel/reduction/reducesum_col" "reducesum_col" || failures+=("reducesum_col")
compile_operator "$REPO_ROOT/test/kernel/reduction/reducesum_row" "reducesum_row" || failures+=("reducesum_row")
compile_operator "$REPO_ROOT/test/kernel/control" "control" || failures+=("control")
compile_operator "$REPO_ROOT/test/kernel/fa" "fa" || failures+=("fa")
compile_operator "$REPO_ROOT/test/kernel/sort" "sort" || failures+=("sort")

echo ""
echo "=========================================="
echo "Full compilation completed!"
echo "=========================================="
echo ""
echo "Generated ELF files:"
find "$REPO_ROOT/output" -name "*.elf" -type f | wc -l
echo "ELF files are located in: $REPO_ROOT/output/"

if ((${#failures[@]})); then
    echo "Failed operators: ${failures[*]}" >&2
    exit 1
fi
