#ifndef QUANT_BATCH_MATMUL_MXFP4_MT_KERNEL_HPP
#define QUANT_BATCH_MATMUL_MXFP4_MT_KERNEL_HPP

#include <common/pto_tileop.hpp>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// quant_batch_matmul_mxfp4_mt — 4-PE cooperative MX fp4 matmul（PTO 一层）
//
// 基于 matmul_shared_lowp / matmul_test_mt 的 4-PE cooperative 模式重写。
// 与单 PE 版本（quant_batch_matmul_mxfp4）的关键区别：
//   - A/B 通过 SharedTile 装入，4 个 PE 协作执行 TMATMUL_MX；
//   - 每 PE 只持有并写回 [kPeM, gN] 行切片（kPeM = tM / 4）；
//   - A 和 A_mx 是全局共享矩阵（非 PE-local 副本）。
//
// A/B 为 e2m1x2 fp4 打包（__fp4_e2m1x2，每字节 2 个元素），
// MX scaling (e8m0, smatrix_wfactor=32) 用 SharedTile 路径。
//
// tM 仅支持 64（kPeM=16）与 128（kPeM=32），符合 cooperative group_M 契约。
// 编译时尺寸：gM/gN/gKv 均需被 tM/tN/tKv 整除。
//
// 运行: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
// ============================================================================
template <int tM, int tN, int tKv, int gM, int gN, int gKv>
__attribute__((noinline)) void quant_batch_matmul_mxfp4_mt(
    float *dst, __fp4_e2m1x2 *src0, __fp4_e2m1x2 *src1,
    __fp8_e8m0 *src0_mx, __fp8_e8m0 *src1_mx)
{
    using dtypeA = __fp4_e2m1x2;
    using dtypeB = __fp4_e2m1x2;
    constexpr int kPeNum = 4;
    constexpr int smatrix_wfactor = 32;
    // PackedFactor=2: __fp4_e2m1x2 每字节 2 个元素
    constexpr int kPackedFactor = 2;

    static_assert(tM == 64 || tM == 128,
                  "4-PE cooperative requires tM == 64 (kPeM=16) or tM == 128 "
                  "(kPeM=32)");
    constexpr int kPeM = tM / kPeNum;

    static_assert(gM % tM == 0, "gM must be divisible by tM");
    static_assert(gN % tN == 0, "gN must be divisible by tN");
    static_assert(gKv % tKv == 0, "gKv must be divisible by tKv");

    // Scale group size is in carrier units (smatrix_wfactor=32 carriers per
    // scale block = 64 logical fp4 elements).
    constexpr int kScaleK = tKv / smatrix_wfactor;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;
    constexpr int kGScaleK = gKv / smatrix_wfactor;

    static_assert(tKv % smatrix_wfactor == 0,
                  "tKv must be divisible by smatrix_wfactor (32)");
    static_assert(kScaleK >= 1, "scale group K must be >= 1");

    const uint32_t tid = get_thread_idx();

    // --- Global tensor slices ---
    // B is stored as [N, Kv] row-major (PTO spec #257 TransB=0: [N, K]).
    using gmASlice = global_tensor<dtypeA, RowMajor<tM, tKv>>;
    using gmBSlice = global_tensor<dtypeB, RowMajor<tN, tKv>>;

    // --- Shared matrix tiles (cooperative) ---
    using tileAMatrix = SharedMatrixLeft<dtypeA, tM, tKv>;
    using tileBMatrix = SharedMatrixRight<dtypeB, tN, tKv>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;

    // --- MX scale shared tiles ---
    // TransB=0: ScaleB physical [N, KBlocks], valid [N, kScaleK].
    using tileAScaleMatrix =
        SharedMatrixLeft<__fp8_e8m0, tM, kPaddedScaleK, tM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<__fp8_e8m0, tN, kPaddedScaleK, tN, kScaleK>;
    using tileAScaleShared = SharedTile<tileAScaleMatrix>;
    using tileBScaleShared = SharedTile<tileBScaleMatrix>;

    // --- PE-private CUBE accumulator ---
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    // --- SharedTile capacity check ---
    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kSharedOperandBytes =
        tileAMatrix::LogicalTileBytes + tileBMatrix::LogicalTileBytes +
        tileAScaleMatrix::LogicalTileBytes + tileBScaleMatrix::LogicalTileBytes;
    static_assert(kSharedOperandBytes <= kSharedTRegBytes,
                  "A/B and MX scale tiles exceed the 256 KiB SharedTReg pool");

    // --- Global scale slice tensors (manual, not global_iterator which
    // mis-computes offsets for padded scale tiles) ---
    // B scale stored as [N, kGScaleK] row-major (matching B's [N, Kv] layout).
    using gmAScaleSlice = global_tensor<__fp8_e8m0, RowMajor<tM, kGScaleK>>;
    using gmBScaleSlice = global_tensor<__fp8_e8m0, RowMajor<tN, kGScaleK>>;

    // --- Global C slice (manual offset, not global_iterator) ---
    using gmCSlice = global_tensor<float, RowMajor<kPeM, gN>>;

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gKv / tKv;

    for (int b = 0; b < Batch; ++b) {
        size_t bOffA  = (size_t)b * gM * gKv;
        size_t bOffB  = (size_t)b * gN * gKv;   // B is [gN, gKv]
        size_t bOffC  = (size_t)b * gM * gN;
        size_t bOffAS = (size_t)b * gM * kGScaleK;
        size_t bOffBS = (size_t)b * gN * kGScaleK; // B scale is [gN, kGScaleK]

#pragma clang loop unroll(full)
        for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
            for (int j = 0; j < Nb; ++j) {
                tileC tC;

#pragma clang loop unroll(full)
                for (int k = 0; k < Kb; ++k) {
                    // GM slices
                    gmASlice gA(src0 + bOffA +
                                (size_t)i * tM * gKv +
                                (size_t)k * tKv);
                    // B [N, Kv]: slice [tN, tKv] at [j*tN, k*tKv]
                    gmBSlice gB(src1 + bOffB +
                                (size_t)j * tN * gKv +
                                (size_t)k * tKv);

                    // Shared tiles
                    tileAShared tA;
                    tileBShared tB;
                    TLOAD<tileAMatrix, 1>(tA, gA);
                    TLOAD<tileBMatrix, 1>(tB, gB);

                    // MX scale tiles
                    tileAScaleShared tAScale;
                    tileBScaleShared tBScale;
                    gmAScaleSlice gAScale(src0_mx + bOffAS +
                                          (size_t)i * tM * kGScaleK +
                                          (size_t)k * kScaleK);
                    // B scale [N, kGScaleK]: slice [tN, kScaleK] at [j*tN, k*kScaleK]
                    gmBScaleSlice gBScale(src1_mx + bOffBS +
                                          (size_t)j * tN * kGScaleK +
                                          (size_t)k * kScaleK);
                    TLOAD<tileAScaleMatrix, 1>(tAScale, gAScale);
                    TLOAD<tileBScaleMatrix, 1>(tBScale, gBScale);

                    auto mxOptions = fixp::keep_acc();
                    if (k == 0) {
                        TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale,
                                      mxOptions);
                    } else {
                        TMATMUL_MX_ACC<3>(tC, tC, tA, tAScale, tB, tBScale,
                                          mxOptions);
                    }
                }

                // Each PE writes its own [kPeM, tN] row slice
                gmCSlice gC(dst + bOffC +
                            ((size_t)i * tM + (size_t)tid * kPeM) * gN +
                            (size_t)j * tN);
                TSTORE_CUBE(gC, tC);
            }
        }
    }
}

#endif  // QUANT_BATCH_MATMUL_MXFP4_MT_KERNEL_HPP
