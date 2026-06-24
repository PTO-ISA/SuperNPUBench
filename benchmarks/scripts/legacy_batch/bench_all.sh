#!/bin/bash
set -e
set -x
set -o pipefail

cd $(dirname $0)/../..

export CC_OPT=default

python3 benchmarks/scripts/legacy_batch/run_compile.py
ERRS=$(grep fail: benchmarks/cm_log/compile_summary.log | awk '{print $2}')
PASS=$(($ERRS <= 4))
if [[ x"$PASS" != x"1" ]]; then
  cat benchmarks/cm_log/compile_summary.log
  exit 1
fi

# ELF_LIST="output/api/tileop/elf/*.elf output/microbench/lmbench/elf/*.elf output/kernels/composite/elf/*.elf output/models/deepseekv3/elf/*.elf"
# 
# realpath $ELF_LIST > tmp.list
# 
# if [[ -f $QEMU ]]; then
#   ARGS="$ARGS -m $QEMU"
# fi
# python3 benchmarks/scripts/legacy_batch/run_qemu.py -i tmp.list -o benchmarks/cm_log/qemu_run.log $ARGS
# rm -f tmp.list
