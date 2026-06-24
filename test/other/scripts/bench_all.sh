#!/bin/bash
set -e
set -x
set -o pipefail

cd $(dirname $0)/../..

export CC_OPT=default

rm -rf output/matmul_compile_output
python3 test/ascpp/run_compile.py test/ascpp/matmul -o output/matmul_compile_output    -f test/ascpp/filter.conf | tee bench.log
Total=$(grep "Total test cases:" bench.log | awk '{print $4}')
PASS=$(grep "Suaccelssful:" bench.log | awk '{print $2}')
if [[ x"$Total" != x"$PASS" ]]; then
  exit 1
fi

rm -rf output/fa_normal_compile_output
python3 test/ascpp/run_compile.py test/ascpp/fa     -o output/fa_normal_compile_output -f test/ascpp/filter.conf | tee bench.log
Total=$(grep "Total test cases:" bench.log | awk '{print $4}')
PASS=$(grep "Suaccelssful:" bench.log | awk '{print $2}')
if [[ x"$Total" != x"$PASS" ]]; then
  exit 1
fi

python3 test/scripts/run_compile.py
ERRS=$(grep fail: cm_log/compile_summary.log | awk '{print $2}')
PASS=$(($ERRS <= 4))
if [[ x"$PASS" != x"1" ]]; then
  cat cm_log/compile_summary.log
  exit 1
fi

# ELF_LIST="output/tileop_test/elf/*.elf output/lmbench/elf/*.elf output/kernel/elf/*.elf output/deepseek/elf/*.elf"
# 
# realpath $ELF_LIST > tmp.list
# 
# if [[ -f $QEMU ]]; then
#   ARGS="$ARGS -m $QEMU"
# fi
# python3 test/scripts/run_qemu.py -i tmp.list -o cm_log/qemu_run.log $ARGS
# rm -f tmp.list
