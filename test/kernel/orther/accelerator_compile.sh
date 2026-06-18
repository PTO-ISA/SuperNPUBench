#! /bin/bash

./accelerator_compile_new/compile_matmul.all
./accelerator_compile_new/compile_matmul_reuseA.all
./accelerator_compile_new/compile_matmul_reuseB.all
./accelerator_compile_new/compile_matmul_reuseAB.all

./accelerator_compile_new/compile_matmul_dynamic.all
./accelerator_compile_new/compile_matmul_dynamic_reuseA.all
./accelerator_compile_new/compile_matmul_dynamic_reuseB.all