#!/bin/bash
# Compile, run, and verify conv2d_rm_shared (multi-PE) kernel.
#
# The gfrun SIMT model cannot handle file I/O ecalls in multi-threaded mode
# (assertion "Block BARG target"). Verification uses --dump-memory instead:
#   1. Compile without RES_CHECK (volatile init: input=2.0, weight=1.0)
#   2. Run gfrun with -t 1 to find TSTORE B.IOR output address
#   3. Run gfrun with --dump-memory to extract the output
#   4. Compare with expected (input * weight * K = 2.0 * 1.0 * IN_C for all-ones)
#
# Usage: run_conv2d_rm_shared.sh TYPE IN_H IN_W IN_C OUT_C tilM tilN tilK [DATA_DIR] [LABEL]
#   DATA_DIR and LABEL are optional (for RES_CHECK mode, currently unused)
set -e

TYPE=$1; IN_H=$2; IN_W=$3; IN_C=$4; OUT_C=$5
tM=$6; tN=$7; tK=$8

export COMPILER_DIR=/mnt/workspace/v310/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
GFRUN=/mnt/workspace/v310/SuperScalarModel/bin/gfrun
ROOT=/mnt/workspace/v310/SuperNPUBench/benchmark/one-level-arch
ELF_DIR=$ROOT/output/kernel/conv2d/elf/kernel_conv2d
ELF=${ELF_DIR}/conv2d_rm_shared_${TYPE}_IN${IN_H}x${IN_W}x${IN_C}_OUT${OUT_C}_tM${tM}_tN${tN}_tK${tK}.elf

LABEL=${10:-conv2d_rm_shared_${TYPE}_${IN_H}x${IN_W}x${IN_C}}
TMPDIR=/tmp/conv2d_rm_shared_${LABEL}
mkdir -p "$TMPDIR"

echo "--- Compiling ${LABEL} ---"
make -C "$ROOT/test/kernel/conv2d" \
     TYPE=$TYPE IN_H=$IN_H IN_W=$IN_W IN_C=$IN_C OUT_C=$OUT_C \
     tilM=$tM tilN=$tN tilK=$tK TESTCASE=conv2d_rm_shared PLAT=linx 2>&1 | grep -v "^Makefile:" | tail -1

gM=$((IN_H * IN_W))
gN=$OUT_C
gK=$IN_C
OUT_SIZE=$((gM * gN * 4))  # float32
OUT_SIZE_HEX=$(printf '0x%x' $OUT_SIZE)

echo "--- Finding output address (trace run) ---"
# Extract the TSTORE B.IOR output base address from the trace.
# Search for B.IOR lines that appear after [TSTORE] block headers
# (TLOAD B.IOR lines have the same stride when gN == tK, so we must
# distinguish TSTORE B.IOR from TLOAD B.IOR).
TRACE_FILE="$TMPDIR/trace.log"
$GFRUN -s softcore.multiThreadNum=4 -f "$ELF" -t 1 > "$TRACE_FILE" 2>&1
OUT_ADDR=$(grep "\[TSTORE\]" -A10 "$TRACE_FILE" | grep "B.IOR" | head -1 \
    | grep -oP '0x[0-9a-fA-F]{7,}' | head -1)
if [ -z "$OUT_ADDR" ]; then
    echo "  ERROR: Could not find TSTORE output address"
    exit 1
fi
echo "  Output base address: $OUT_ADDR"

echo "--- Running ${LABEL} (4 PEs, dump-memory) ---"
$GFRUN -s softcore.multiThreadNum=4 -f "$ELF" \
    --dump-memory "${OUT_ADDR}:${OUT_SIZE_HEX}:${TMPDIR}/dump.bin" 2>&1 \
    | grep -E "Block number|R2 =|Suaccelss"

echo "--- Verifying ${LABEL} (all-ones: expect $(python3 -c "print(2.0 * $gK if '$TYPE' == 'FP32' else 1.0 * $gK)")) ---"
python3 -c "
import numpy as np
gM, gN, gK = $gM, $gN, $gK
# FP32: volatile init input=2.0 (0x40000000), weight=1.0 (0x3F800000)
# FP16: volatile init input=1.0 (0x3C00), weight=1.0 (0x3C00)
input_val = 2.0 if '$TYPE' == 'FP32' else 1.0
expected = input_val * 1.0 * gK
res = np.fromfile('$TMPDIR/dump.bin', dtype=np.float32)
if res.size == 0:
    print('  ERROR: dump.bin is empty!')
    exit(1)
diff = np.abs(res - expected)
nz = res != 0
eps = 1e-2
mismatch = int((diff > eps).sum())
has_inf = bool(np.isinf(res).any())
has_nan = bool(np.isnan(res).any())
max_diff = float(diff.max())
print(f'  Size: {res.size} (expected {gM*gN})')
print(f'  Non-zero: {int(nz.sum())}/{res.size} ({nz.sum()/res.size*100:.1f}%)')
print(f'  inf={has_inf} nan={has_nan}')
print(f'  Expected: {expected}, Got: min={res.min()}, max={res.max()}')
print(f'  Max diff: {max_diff:.8e}')
print(f'  Mismatch(eps={eps}): {mismatch}/{res.size}')
verdict = 'PASS' if mismatch == 0 and int(nz.sum()) == res.size and not has_inf and not has_nan else 'FAIL'
print(f'  Verdict: {verdict}')
"
