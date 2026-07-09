#!/bin/bash
# Microbench top-level build: cube / vector / memory / scalar
# Each category has its own compile.all that builds every generated case.

CATEGORY=${1:-all}

compile_cube()   { echo ""; echo ">>> cube (CUBE)";       (cd cube   && bash compile.all); }
compile_vector() { echo ""; echo ">>> vector (TEPL)";     (cd vector && bash compile.all); }
compile_memory() { echo ""; echo ">>> memory (TLSU)";     (cd memory && bash compile.all); }
compile_scalar() { echo ""; echo ">>> scalar (GPR ALU)";  (cd scalar && bash compile.all); }

case "$CATEGORY" in
    cube)   compile_cube ;;
    vector) compile_vector ;;
    memory) compile_memory ;;
    scalar) compile_scalar ;;
    all)
        compile_cube
        compile_vector
        compile_memory
        compile_scalar
        ;;
    *)
        echo "Usage: $0 [cube|vector|memory|scalar|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Microbench build completed for: $CATEGORY"
echo "=========================================="
