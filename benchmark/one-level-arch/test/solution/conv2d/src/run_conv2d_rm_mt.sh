#!/bin/bash
# Compile, run, and verify conv2d_rm_mt (multi-thread batch-of-4) with random
# data via RES_CHECK file I/O (new SMT4 per-thread-stack model).
#
# Each PE dumps its own output slice to res_pe<tid>.bin; the host side
# verifies every PE slice against the corresponding golden slice.
#
# Usage: run_conv2d_rm_mt.sh TYPE IN_H IN_W IN_C OUT_C tilM tilN tilK
set -e

TYPE=$1; IN_H=$2; IN_W=$3; IN_C=$4; OUT_C=$5
tM=$6; tN=$7; tK=$8

export COMPILER_DIR=/mnt/workspace/v310/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
GFRUN=/mnt/workspace/v310/SuperScalarModel/bin/gfrun
ROOT=/mnt/workspace/v310/SuperNPUBench/benchmark/one-level-arch
LABEL=conv2d_rm_mt_rc_${TYPE}_${IN_H}x${IN_W}x${IN_C}_${OUT_C}
CMP_DIR=$ROOT/compare/$LABEL
ELF=$ROOT/output/kernel/conv2d/elf/kernel_conv2d/conv2d_rm_mt_${TYPE}_IN${IN_H}x${IN_W}x${IN_C}_OUT${OUT_C}_tM${tM}_tN${tN}_tK${tK}.elf

mkdir -p "$CMP_DIR"
python3 "$ROOT/test/kernel/conv2d/src/conv2d_rm_mt_gen.py" \
    --IN_H $IN_H --IN_W $IN_W --IN_C $IN_C --OUT_C $OUT_C --TYPE $TYPE \
    --out_dir "$CMP_DIR" > /dev/null
rm -f "$CMP_DIR"/res_pe*.bin
echo "#define CHK_DIR \"$CMP_DIR\"" > /tmp/conv2d_rm_mt_chk_dir.h

echo "--- Compiling $LABEL ---"
make -C "$ROOT/test/kernel/conv2d" \
     TYPE=$TYPE IN_H=$IN_H IN_W=$IN_W IN_C=$IN_C OUT_C=$OUT_C \
     tilM=$tM tilN=$tN tilK=$tK TESTCASE=conv2d_rm_mt PLAT=linx \
     CC_OPTS="-DRES_CHECK -DENABLE_BINARY_OUTPUT -include /tmp/conv2d_rm_mt_chk_dir.h" \
     2>&1 | grep -E "error" || true

echo "--- Running $LABEL (multiThreadNum=4) ---"
$GFRUN -s softcore.multiThreadNum=4 -f "$ELF" 2>&1 | grep -E "R2 =|faild" | tail -2

gM=$((IN_H * IN_W))
gN=$OUT_C
echo "--- Verifying $LABEL ---"
python3 - "$CMP_DIR" $gM $gN <<'EOF'
import sys
import numpy as np
cmp_dir, gM, gN = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
golden = np.fromfile(f'{cmp_dir}/golden.bin', dtype=np.float32).reshape(4, gM, gN)
all_ok = True
for pe in range(4):
    try:
        res = np.fromfile(f'{cmp_dir}/res_pe{pe}.bin', dtype=np.float32)
    except OSError:
        print(f'  PE{pe}: res file missing'); all_ok = False; continue
    if res.size != gM * gN:
        print(f'  PE{pe}: BAD SIZE {res.size}'); all_ok = False; continue
    d = np.abs(res.reshape(gM, gN) - golden[pe])
    nz = int(np.count_nonzero(res))
    mm = int((d > 1e-2).sum())
    print(f'  PE{pe}: nonzero={nz}/{gM*gN}, max_diff={d.max():.3e}, mismatch={mm}')
    all_ok = all_ok and nz == gM * gN and mm == 0
print(f'  Verdict: {"PASS" if all_ok else "FAIL"}')
EOF
