#!/bin/bash
echo "=========================================="
echo "LinxISA Microbenchmarks Compilation"
echo "=========================================="

CATEGORY=${1:-all}

compile_elem()    { echo ""; echo ">>> Elem (Tile-Tile)";     cd elem;    bash compile.all; cd ..; }
compile_scalar()  { echo ""; echo ">>> Scalar (Tile-Scalar)"; cd scalar;  bash compile.all; cd ..; }
compile_memory()  { echo ""; echo ">>> Memory (Load/Store)";  cd memory;  bash compile.all; cd ..; }
compile_layout()  { echo ""; echo ">>> Layout (Transpose)";   cd layout;  bash compile.all; cd ..; }

case $CATEGORY in
    elem)    compile_elem ;;
    scalar)  compile_scalar ;;
    memory)  compile_memory ;;
    layout)  compile_layout ;;
    all)
        compile_elem; compile_scalar; compile_memory; compile_layout
        ;;
    *) echo "Usage: $0 [elem|scalar|memory|layout|all]"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo "Microbenchmarks compilation completed!"
echo "=========================================="
