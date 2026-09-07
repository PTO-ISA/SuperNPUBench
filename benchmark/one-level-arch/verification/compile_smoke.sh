#!/bin/bash
# SuperNPUBench — 精简测试编译脚本（芯片核心功能验证）
# 16 个用例覆盖：CUBE/TEPL/TLSU/GPR + 多线程 + 全部 ISA 族

: "${COMPILER_DIR:?Set COMPILER_DIR to the Linx compiler bin directory}"
export COMPILER_DIR
export baremetal=${baremetal:-off}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

PASS=0; FAIL=0

smoke() {
    local name=$1; shift
    echo ""
    echo "------------------------------------------"
    echo "  $name"
    echo "------------------------------------------"
    if "$@" 2>&1; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL+1))
    fi
}

echo "=========================================="
echo "  SuperNPUBench Smoke Test — 编译"
echo "  COMPILER_DIR=$COMPILER_DIR"
echo "=========================================="

# Ensure output directory exists (make clean fails if output/ is missing)
mkdir -p "$REPO_ROOT/output"

# --- 1. matmul (CUBE: 基础 GEMM) ---
smoke "matmul FP32" \
    make -C "$REPO_ROOT/test/kernel/matmul" TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=32 tN=32 tK=32 clean all

# --- 2. fa/sfa (CUBE+TEPL: Flash Attention) ---
smoke "sfa" \
    make -C "$REPO_ROOT/test/kernel/fa" TESTCASE=sfa Sq=256 Skv=512 Tm=16 Tk=32 clean all

# --- 3. flashMLA (CUBE+TEPL: MLA) ---
smoke "flashMLA" \
    make -C "$REPO_ROOT/test/kernel/flashMLA" TESTCASE=flashMLA Sq=64 QHeadPerHK=1 NumBlocks=2 MaxBlocksPerSeq=2 Dk=512 Dv=512 DChunk=128 VChunk=128 Tm=16 Tk=16 clean all

# --- 4. gelu (TEPL: 多项式激活) ---
smoke "gelu bf16" \
    make -C "$REPO_ROOT/test/kernel/element_wise/gelu" TESTCASE=gelu DTYPE=__bf16 tMs=2048 gMs=196608 SHAPE_NAME=24_8_1024 Approximate=false clean all

# --- 5. reducesum_col (TEPL: 列归约) ---
smoke "reducesum_col" \
    make -C "$REPO_ROOT/test/kernel/reduction/reducesum_col" TESTCASE=reducesum_col COMPILER_DIR="$COMPILER_DIR" DType=int32_t tM_s=32 tN_s=64 gM_s=2048 gN_s=64 clean all

# --- 6. reducemax_row (TEPL: 行归约) ---
smoke "reducemax_row" \
    make -C "$REPO_ROOT/test/kernel/reduction/reducemax_row" TESTCASE=reducemax_row COMPILER_DIR="$COMPILER_DIR" DType=int32_t tM_s=16 tN_s=128 gM_s=16 gN_s=8192 clean all

# --- 7. broadcast (TEPL: 行广播) ---
smoke "broadcast vec_07" \
    make -C "$REPO_ROOT/test/kernel/broadcast" TESTCASE=broadcast_vec_07 DType=__half tMs=16 MAX_DIMs=2 IN_DIMs=2 OUT_DIMs=2 gIMs=1443 gOMs=1443 kInner=129 kTileRows=16 clean all

# --- 8. concat (TLSU: N-D concat + MGATHER) ---
smoke "concat gather" \
    make -C "$REPO_ROOT/test/kernel/concat" TESTCASE=concat_gather DType=int32_t tM=512 MAX_DIM=8 DATA_DIM=2 CONCAT_DIM=1 IN_SHAPE=64,2 IN_SHAPE_NAME=64_2 OUT_SHAPE=64,2000 OUT_SHAPE_NAME=64_2000 gIM=128 gOM=128000 clean all

# --- 9. transpose (TEPL: 硬件转置) ---
smoke "transpose 2D" \
    make -C "$REPO_ROOT/test/kernel/transpose" TESTCASE=transpose COMPILER_DIR="$COMPILER_DIR" DType=__half MAX_DIM=8 tM=512 IN_DIM=2 OUT_DIM=2 TRANSPOSE_DIM1=0 TRANSPOSE_DIM0=1 gIM=1476 gOM=32 clean all

# --- 10. gather (TLSU: 行索引 gather) ---
smoke "gather" \
    make -C "$REPO_ROOT/test/kernel/gather" TESTCASE=gather COMPILER_DIR="$COMPILER_DIR" DType=__fp32 OType=uint32_t gKs=131072 gMs=32 gNs=256 tMs=32 tNs=64 clean all

# --- 11. control (TEPL+TLSU: 哈希查找) ---
smoke "hashtable_lookup" \
    make -C "$REPO_ROOT/test/kernel/control" TESTCASE=hashtable_lookup_simd SUFFIX=_kNum6144_kMaxProbe512_knum_col256_debug_off EXTRA_DEFINES="-DkNum=6144 -DMAX_PROBE=512 -DNUM_COL=256 -DFOR_GFSIM" clean all

# --- 12. sort (TEPL: radix 直方图) ---
smoke "topk" \
    make -C "$REPO_ROOT/test/kernel/sort" TESTCASE=topk clean all

# --- 13. multi_thread/matmul (CUBE: 共享 tile) ---
smoke "multi_thread/matmul" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/matmul" TESTCASE=matmul COMPILER_DIR="$COMPILER_DIR" B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32 clean all

# --- 14. multi_thread/vec (TEPL: 多线程元素加) ---
smoke "multi_thread/vec" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/vec" TESTCASE=tadd COMPILER_DIR="$COMPILER_DIR" TileRows=16 TileCols=16 clean all

# --- 15. multi_thread/fa (CUBE+TEPL: 多线程 FA) ---
smoke "multi_thread/fa" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/fa" TESTCASE=fa_2d_unroll_gmma COMPILER_DIR="$COMPILER_DIR" Sq=128 Skv=64 Tm=16 Tk=16 clean all

# --- 16. multi_thread/element_wise/gelu (SPMD: 连续分片) ---
smoke "multi_thread/element_wise/gelu" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/element_wise/gelu" TESTCASE=gelu COMPILER_DIR="$COMPILER_DIR" clean all

echo ""
echo "=========================================="
echo "  Smoke Test 编译完成"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "=========================================="
echo "ELF: $(find "$REPO_ROOT/output" -name '*.elf' -type f | wc -l)"
