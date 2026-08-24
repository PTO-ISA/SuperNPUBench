#!/bin/bash
# Compile, run, and verify conv2d_rm kernel for a given configuration.
# Usage: run_test.sh TYPE IN_H IN_W IN_C OUT_C tilM tilN tilK DATA_DIR LABEL
set -e

TYPE=$1; IN_H=$2; IN_W=$3; IN_C=$4; OUT_C=$5
tM=$6; tN=$7; tK=$8; DATA_DIR=$9; LABEL=${10}

export COMPILER_DIR=/mnt/workspace/v310/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
GFRUN=/mnt/workspace/v310/SuperScalarModel/bin/gfrun
ROOT=/mnt/workspace/v310/SuperNPUBench/benchmark/one-level-arch
COMPARE_DIR=$ROOT/compare/${LABEL}
ELF_DIR=$ROOT/output/kernel/conv2d_rm/elf/kernel_conv2d_rm
ELF=${ELF_DIR}/conv2d_rm_${TYPE}_IN${IN_H}x${IN_W}x${IN_C}_OUT${OUT_C}_tM${tM}_tN${tN}_tK${tK}.elf

mkdir -p "$COMPARE_DIR"
cp "$DATA_DIR/src0.bin" "$COMPARE_DIR/src0.bin"
cp "$DATA_DIR/src1.bin" "$COMPARE_DIR/src1.bin"
cp "$DATA_DIR/golden.bin" "$COMPARE_DIR/golden.bin"
echo "#define CHK_DIR \"$COMPARE_DIR\"" > /tmp/conv2d_rm_chk_dir.h

echo "--- Compiling ${LABEL} ---"
make -C "$ROOT/test/kernel/conv2d_rm" \
     TYPE=$TYPE IN_H=$IN_H IN_W=$IN_W IN_C=$IN_C OUT_C=$OUT_C \
     tilM=$tM tilN=$tN tilK=$tK TESTCASE=conv2d_rm PLAT=linx \
     CC_OPTS="-DRES_CHECK -DENABLE_BINARY_OUTPUT -include /tmp/conv2d_rm_chk_dir.h" 2>&1 | grep -v "^Makefile:" | tail -1

echo "--- Running ${LABEL} ---"
$GFRUN -f "$ELF" 2>&1 | grep -E "data (read|write)|R2 ="

gM=$((IN_H * IN_W))
gN=$OUT_C
echo "--- Verifying ${LABEL} ---"
python3 -c "
import numpy as np
gM, gN = $gM, $gN
res = np.fromfile('$COMPARE_DIR/res.bin', dtype=np.float32)
golden = np.fromfile('$COMPARE_DIR/golden.bin', dtype=np.float32)
if res.size == 0:
    print('  ERROR: res.bin is empty!')
    exit(1)
res_2d = res.reshape(gM, gN)
golden_2d = golden.reshape(gM, gN)
diff = np.abs(res_2d - golden_2d)
nz = res_2d != 0
nz_diff = diff[nz]
eps = 1e-2
mismatch = int((nz_diff > eps).sum()) if len(nz_diff) > 0 else 0
has_inf = bool(np.isinf(res).any())
has_nan = bool(np.isnan(res).any())
max_diff = float(nz_diff.max()) if len(nz_diff) > 0 else 0.0
mean_diff = float(nz_diff.mean()) if len(nz_diff) > 0 else 0.0
print(f'  Non-zero: {int(nz.sum())}/{res.size} ({nz.sum()/res.size*100:.1f}%)')
print(f'  inf={has_inf} nan={has_nan}')
print(f'  Max diff: {max_diff:.8e}')
print(f'  Mean diff: {mean_diff:.8e}')
print(f'  Mismatch(eps={eps}): {mismatch}/{int(nz.sum())}')
verdict = 'PASS' if mismatch == 0 and int(nz.sum()) == res.size and not has_inf and not has_nan else 'FAIL'
print(f'  Verdict: {verdict}')
"
