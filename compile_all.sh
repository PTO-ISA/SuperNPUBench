#!/bin/bash
# Top-level compilation script for all ISA backends

ISA=${1:-all}  # Default: compile all

echo "=========================================="
echo "SuperNPUBench Build System"
echo "ISA backend: $ISA"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

compile_linx() {
    echo ""
    echo ">>> Compiling LinxISA backend..."
    if [ -f "$SCRIPT_DIR/benchmark-linxisa/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark-linxisa/compile_all.sh"
    else
        echo "Warning: benchmark-linxisa/compile_all.sh not found"
    fi
}

compile_pto() {
    echo ""
    echo ">>> Compiling PTO ISA backend..."
    if [ -f "$SCRIPT_DIR/benchmark-ptoisa/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark-ptoisa/compile_all.sh"
    else
        echo "Warning: benchmark-ptoisa/compile_all.sh not found"
    fi
}

case $ISA in
    linx|benchmark-linxisa)
        compile_linx
        ;;
    pto|benchmark-ptoisa)
        compile_pto
        ;;
    all)
        compile_linx
        compile_pto
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
