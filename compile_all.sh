#!/bin/bash
# Full compilation script for all kernel operators

# Don't use set -e as some operators may fail to compile

export COMPILER_DIR=${COMPILER_DIR:-/Users/liyi/Documents/GitHub/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin}
REPO_ROOT=${REPO_ROOT:-/home/river/projects/jcore/SuperNPUBench}

echo "=========================================="
echo "Starting full compilation of all operators"
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

echo ""
echo "=========================================="
echo "Full compilation completed!"
echo "=========================================="
echo ""
echo "Generated ELF files:"
find "$REPO_ROOT/output" -name "*.elf" -type f | wc -l
echo "ELF files are located in: $REPO_ROOT/output/"
