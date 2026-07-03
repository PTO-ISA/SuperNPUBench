#!/bin/bash
# Top-level microbench compilation script

echo "=========================================="
echo "LinxISA Microbenchmarks Compilation"
echo "=========================================="

CATEGORY=${1:-all}

compile_cube() {
    echo ""
    echo ">>> Compiling Cube Microbenchmarks..."
    cd cube
    bash compile.all
    cd ..
}

compile_vector() {
    echo ""
    echo ">>> Compiling Vector Microbenchmarks..."
    cd vector
    bash compile.all
    cd ..
}

compile_memory() {
    echo ""
    echo ">>> Compiling Memory Microbenchmarks..."
    cd memory
    bash compile.all
    cd ..
}

case $CATEGORY in
    cube)
        compile_cube
        ;;
    vector)
        compile_vector
        ;;
    memory)
        compile_memory
        ;;
    all)
        compile_cube
        compile_vector
        compile_memory
        ;;
    *)
        echo "Usage: $0 [cube|vector|memory|all]"
        echo "  cube    - Compile cube microbenchmarks only"
        echo "  vector  - Compile vector microbenchmarks only"
        echo "  memory  - Compile memory microbenchmarks only"
        echo "  all     - Compile all microbenchmarks (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Microbenchmarks compilation completed!"
echo "=========================================="
