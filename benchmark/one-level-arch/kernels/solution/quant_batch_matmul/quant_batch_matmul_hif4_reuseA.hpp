#ifndef QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP
#define QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// quant_batch_matmul_hif4_reuseA — reuseA 调度（对标 matmul_lowp_reuseA）。
//
// 与 quant_batch_matmul_hif4 的区别：
//   1. 循环倒置：i(M)->j(N)->k(K) 改为 m(M)->k(K)->n(N)。
//      A(m,k) 在每个 K step 只 TLOAD 一次，内层 n 循环复用同一份 A。
//      A 加载次数从 Mb*Nb*Kb 降到 Mb*Kb（Nb 倍复用）。
//   2. internal-acc：C 的 partial sum 用 raw_acc() 保持 raw，跨 K 累加，
//      只在最后一个 K step acc_hint() 才 materialize 出来 TSTORE。
//      需要 cube.enable_internal_acc=true（默认开）。
//   3. B/scaleB 仍是 4-PE Shared（GMMA.LD），A/scaleA 仍是 Local CellReg。
//
// 前置条件（编译期）：gM%tM==0、gN%tN==0、gK%tK==0、tK%64==0。
// ============================================================================
template <int tM, int tN, int tK, int gM, int gN, int gK>
__attribute__((noinline)) void quant_batch_matmul_hif4_reuseA(
    float *dst, __fp4_hif4x2 *src0, __fp4_hif4x2 *src1,
    uint32_t *src0_mx, uint32_t *src1_mx)
{
    using dtype = __fp4_hif4x2;
    using scale_dtype = uint32_t;
    constexpr int kPeNum = 4;
    constexpr int kPackedFactor = 2;
    constexpr int kScaleGroup = 64;

    static_assert(tM == 64 || tM == 128, "4-PE cooperative requires tM == 64 or 128");
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;

    static_assert(gM % tM == 0, "gM must be divisible by tM");
    static_assert(gN % tN == 0, "gN must be divisible by tN");
    static_assert(gK % tK == 0, "gK must be divisible by tK");
    static_assert(gK % kScaleGroup == 0 && tK % kScaleGroup == 0,
                  "HiF4 requires K and tK divisible by scale group (64)");

    constexpr int kStoredGK = gK / kPackedFactor;
    constexpr int kStoredTK = tK / kPackedFactor;
    constexpr int kScaleK = (tK + kScaleGroup - 1) / kScaleGroup;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;
    constexpr int kGScaleK = gK / kScaleGroup;

    constexpr int Kb = gK / tK;
    constexpr int Mb = gM / kGroupM;
    constexpr int Nb = gN / tN;

    const uint32_t tid = get_thread_idx();

    using gmASlice = global_tensor<dtype, RowMajor<kPeM, kStoredGK>>;
    using gmBSlice = global_tensor<dtype, RowMajor<tN, kStoredGK>>;
    using tileALocalM16 = CubeTileM16<dtype, kPeM, tK>;
    using tileALocalM32 = CubeTileM32<dtype, kPeM, tK>;
    using tileALocal = std::conditional_t<(kPeM <= 16), tileALocalM16, tileALocalM32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileAScaleLocal =
        Tile<Location::Scaling, scale_dtype, kPeM, kPaddedScaleK,
             BLayout::RowMajor, kPeM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<scale_dtype, tN, kPaddedScaleK, tN, kScaleK>;
    using tileBScaleShared = SharedTile<tileBScaleMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    using gmAScaleSlice = global_tensor<scale_dtype, RowMajor<kPeM, kGScaleK>>;
    using gmBScaleSlice = global_tensor<scale_dtype, RowMajor<tN, kGScaleK>>;
    using gmCSlice = global_tensor<float, RowMajor<kPeM, gN>>;

    const constexpr auto mxBegin = fixp::keep_acc().raw_acc();
    const constexpr auto mxMiddle = fixp::keep_acc().raw_acc().acc_hint();
    const constexpr auto mxEnd = fixp::keep_acc().acc_hint();

    for (int b = 0; b < Batch; ++b) {
        size_t bOffA  = (size_t)b * gM * kStoredGK;
        size_t bOffB  = (size_t)b * gN * kStoredGK;
        size_t bOffC  = (size_t)b * gM * gN;
        size_t bOffAS = (size_t)b * gM * kGScaleK;
        size_t bOffBS = (size_t)b * gN * kGScaleK;

#pragma clang loop unroll(full)
        for (int m = 0; m < Mb; ++m) {
            tileC cTiles[Nb];

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                // A → Local CellReg（本 K step 加载一次，内层 n 循环复用）。
                gmASlice gA(src0 + bOffA +
                            ((size_t)m * kGroupM + (size_t)tid * kPeM) *
                                kStoredGK +
                            (size_t)k * kStoredTK);
                tileALocal aTile;
                TLOAD_CUBE(aTile, gA);
                gmAScaleSlice gAScale(
                    src0_mx + bOffAS +
                    ((size_t)m * kGroupM + (size_t)tid * kPeM) * kGScaleK +
                    (size_t)k * kScaleK);
                tileAScaleLocal aScaleTile;
                TLOAD(aScaleTile, gAScale);

#pragma clang loop unroll(full)
                for (int n = 0; n < Nb; ++n) {
                    // B → Shared（每个 (n,k) 加载一次）。
                    gmBSlice gB(src1 + bOffB +
                                (size_t)n * tN * kStoredGK +
                                (size_t)k * kStoredTK);
                    tileBShared bTile;
                    TLOAD<tileBMatrix, 1>(bTile, gB);
                    gmBScaleSlice gBScale(src1_mx + bOffBS +
                                          (size_t)n * tN * kGScaleK +
                                          (size_t)k * kScaleK);
                    tileBScaleShared bScaleTile;
                    TLOAD<tileBScaleMatrix, 1>(bScaleTile, gBScale);

                    // internal-acc K-chain：begin=raw_acc, middle=raw_acc|acc_hint,
                    // end=acc_hint。
                    if (k == 0) {
                        TMATMUL_MX<3>(cTiles[n], aTile, aScaleTile, bTile,
                                      bScaleTile, mxBegin, (size_t)kGroupM);
                    } else if (k == Kb - 1) {
                        TMATMUL_MX_ACC<3>(cTiles[n], cTiles[n], aTile, aScaleTile,
                                          bTile, bScaleTile, mxEnd,
                                          (size_t)kGroupM);
                    } else {
                        TMATMUL_MX_ACC<3>(cTiles[n], cTiles[n], aTile, aScaleTile,
                                          bTile, bScaleTile, mxMiddle,
                                          (size_t)kGroupM);
                    }
                }
            }

#pragma clang loop unroll(full)
            for (int n = 0; n < Nb; ++n) {
                gmCSlice gC(dst + bOffC +
                            ((size_t)m * kGroupM + (size_t)tid * kPeM) * gN +
                            (size_t)n * tN);
                TSTORE_CUBE(gC, cTiles[n]);
            }
        }
    }
}

#endif  // QUANT_BATCH_MATMUL_HIF4_REUSEA_KERNEL_HPP