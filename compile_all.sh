#!/usr/bin/env bash
set -uo pipefail

# Top-level compilation script for all architecture backends
#   benchmark/two-level-arch  (was benchmark-linxisa, Linx two-level block ISA)
#   benchmark/one-level-arch  (was benchmark-ptoisa, PTO one-level tile ISA)

ARCH=${1:-all}  # Default: compile all

echo "=========================================="
echo "SuperNPUBench Build System"
echo "Architecture backend: $ARCH"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

compile_two_level() {
    echo ""
    echo ">>> Compiling two-level-arch backend..."
    if [ -f "$SCRIPT_DIR/benchmark/two-level-arch/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark/two-level-arch/compile_all.sh"
    else
        echo "Error: benchmark/two-level-arch/compile_all.sh not found" >&2
        return 1
    fi
}

compile_one_level() {
    echo ""
    echo ">>> Compiling one-level-arch backend..."
    if [ -f "$SCRIPT_DIR/benchmark/one-level-arch/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark/one-level-arch/compile_all.sh"
    else
        echo "Error: benchmark/one-level-arch/compile_all.sh not found" >&2
        return 1
    fi
}

case $ARCH in
    two-level|two-level-arch|linx)
        compile_two_level
        ;;
    one-level|one-level-arch|pto)
        compile_one_level
        ;;
    all)
        failures=()
        compile_two_level || failures+=("two-level")
        compile_one_level || failures+=("one-level")
        if ((${#failures[@]})); then
            echo "Failed architecture backends: ${failures[*]}" >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [two-level|one-level|all]"
        echo "  two-level  - Compile two-level-arch backend only (benchmark/two-level-arch)"
        echo "  one-level  - Compile one-level-arch backend only (benchmark/one-level-arch)"
        echo "  all        - Compile all backends (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Build completed for: $ARCH"
echo "=========================================="
