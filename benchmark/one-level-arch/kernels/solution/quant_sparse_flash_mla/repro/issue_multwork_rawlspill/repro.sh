#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Repro:  [LinxV5] "cannot prove a complete Local tile descriptor for raw spill"
#         (LinxV5TileFixup.cpp:674) when the kernel has kWorkCount > 1
#
# Minimal trigger: same kernel, same shape, only S1 changes (kWorkCount =
# B*S1*N2*GSliceCount).  s1=1 builds fine, s1>=2 dies in the backend.
#
# Compiler used: LinxISA/llvm-project dev-llvm15_56 @ b82a1798ff3e
#   (also reproduced on ccd4275f5263)
#
# Usage:
#   export COMPILER_DIR=<linx-toolchain-build>/output/linx_blockisa_llvm_musl/bin
#   ./repro.sh
# ---------------------------------------------------------------------------
set -u

: "${COMPILER_DIR:?set COMPILER_DIR to the linx_blockisa_llvm_musl/bin directory}"
: "${QSMLA_SPARSE_DATA_ROOT:=/tmp/qsmla-sparse-hif8}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# this repro dir lives at <repo>/benchmark/one-level-arch/kernels/solution/quant_sparse_flash_mla/repro/<name>
SC="$(cd "$HERE/../../../.." && pwd)"        # -> benchmark/one-level-arch
T="$SC/test/solution/quant_sparse_flash_mla"
cd "$T" || exit 1

# 1) case data (deterministic inputs + FP32 golden)
python3 run_qsmla_cpu_reference.py --case csa_small --dtype HIF8 \
        --output-root "$QSMLA_SPARSE_DATA_ROOT" >/dev/null

COMMON="TESTCASE=quant_sparse_flash_mla QSMLA_SPARSE_EMBED_INPUT=on B=1 s2=128 \
 N1=64 N2=1 D=512 wleft=127 wright=0 softmax_scale=0.04419417 \
 cmp_s2=64 ori_topk=40 cmp_topk=40 cmp_ratio=4 Tm=64 Tk=128 Td_block=512 \
 QSMLA_DTYPE=HIF8 QSMLA_SPARSE_DATA_ROOT=$QSMLA_SPARSE_DATA_ROOT \
 QSMLA_SPARSE_CASE=csa_small IMPL=csa_v3_1_1_4pe"

echo "=== PASS expected: s1=1  (kWorkCount = 1*1*1*1 = 1) ==="
touch src/quant_sparse_flash_mla.cpp
if make $COMMON s1=1 >/tmp/repro_s1_1.log 2>&1; then
    echo "  s1=1 : BUILD OK"
else
    echo "  s1=1 : BUILD FAILED (unexpected)"; tail -5 /tmp/repro_s1_1.log
fi

echo "=== FAIL expected: s1=4  (kWorkCount = 1*4*1*1 = 4) ==="
touch src/quant_sparse_flash_mla.cpp
if make $COMMON s1=4 >/tmp/repro_s1_4.log 2>&1; then
    echo "  s1=4 : BUILD OK (bug not reproduced!)"
else
    echo "  s1=4 : BUILD FAILED (expected)"; grep -m1 "fatal error" /tmp/repro_s1_4.log
fi

echo
echo "also fails: B=8 s1=1 / B=1 s1=2 / B=2 s1=2  (any kWorkCount > 1)"
echo "does not fail: v3.2 (O not resident) also fails -> not O-residency specific"
