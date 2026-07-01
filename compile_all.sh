#!/bin/bash
# Top-level compilation script for all ISA backends

ISA=${1:-all}  # Default: compile all

echo "=========================================="
echo "SuperNPUBench Build System"
echo "ISA backend: $ISA"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILURES=0

compile_linx() {
    echo ""
    echo ">>> Compiling LinxISA backend..."
    if [ -f "$SCRIPT_DIR/benchmark-linxisa/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark-linxisa/compile_all.sh"
    else
        echo "Warning: benchmark-linxisa/compile_all.sh not found"
        return 1
    fi
}

compile_pto() {
    echo ""
    echo ">>> Compiling PTO ISA backend..."
    if [ -f "$SCRIPT_DIR/benchmark-ptoisa/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark-ptoisa/compile_all.sh"
    else
        echo "Warning: benchmark-ptoisa/compile_all.sh not found"
        return 1
    fi
}

case $ISA in
    linx|benchmark-linxisa)
        if ! compile_linx; then
            FAILURES=$((FAILURES + 1))
        fi
        ;;
    pto|benchmark-ptoisa)
        if ! compile_pto; then
            FAILURES=$((FAILURES + 1))
        fi
        ;;
    all)
        if ! compile_linx; then
            FAILURES=$((FAILURES + 1))
        fi
        if ! compile_pto; then
            FAILURES=$((FAILURES + 1))
        fi
        ;;
    *)
        echo "Usage: $0 [linx|pto|all]"
        echo "  linx  - Compile LinxISA backend only (benchmark-linxisa)"
        echo "  pto   - Compile PTO ISA backend only (benchmark-ptoisa)"
        echo "  all   - Compile all backends (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Build completed for: $ISA"
echo "=========================================="

if [ "$FAILURES" -ne 0 ]; then
    echo "Backend compilation failures: $FAILURES"
    exit 1
fi
