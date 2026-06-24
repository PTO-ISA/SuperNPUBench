#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/npu_compile/compile_matmul.all"
"$SCRIPT_DIR/npu_compile/compile_matmul_reuseA.all"
"$SCRIPT_DIR/npu_compile/compile_matmul_reuseB.all"
"$SCRIPT_DIR/npu_compile/compile_matmul_reuseAB.all"

"$SCRIPT_DIR/npu_compile/compile_matmul_dynamic.all"
"$SCRIPT_DIR/npu_compile/compile_matmul_dynamic_reuseA.all"
"$SCRIPT_DIR/npu_compile/compile_matmul_dynamic_reuseB.all"
