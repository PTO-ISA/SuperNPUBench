#ifndef QUANT_BATCH_MATMUL_HIF4_MT_KERNEL_HPP
#define QUANT_BATCH_MATMUL_HIF4_MT_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

#ifndef Batch
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// quant_batch_matmul_hif4_mt — 4-PE cooperative HiF4X2 matmul（PTO 一层）
//
// 基于 matmul_shared_lowp 的 4-PE cooperative 模式，专用于 HiF4X2 (hifloat4)。
//   - 数据类型：__fp4_hif4x2（hifloat4）
//   - scale 类型：uint32_t（raw U32 scale word，即 "32B scale"）
//   - scale group：64 logical K elements per U32 word
//
// Tile K 维度使用逻辑 K（tK），GM 存储步长使用 carrier K（tK/PackedFactor）。
// tM 仅支持 64（kPeM=16）与 128（kPeM=32）。
// 编译时尺寸：gM/gN/gK 均需被 tM/tN/tK 整除，gK/tK 需被 64 整除。
//
// 运行: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
// ============================================================================
template <int tM, int tN, int tK, int gM, int gN, int gK>
__attribute__((noinline)) void quant_batch_matmul_hif4_mt(
    float *dst, __fp4_hif4x2 *src0, __fp4_hif4x2 *src1,
    uint32_t *src0_mx, uint32_t *src1_mx)
{
    using dtype = __fp4_hif4x2;
    using scale_dtype = uint32_t;
    constexpr int kPeNum = 4;
    constexpr int kPackedFactor = 2;
    // HiF4: 64 logical K elements per U32 scale word.
    constexpr int kScaleGroup = 64;

    static_assert(tM == 64 || tM == 128,
                  "4-PE cooperative requires tM == 64 (kPeM=16) or tM == 128 "
                  "(kPeM=32)");
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;

    static_assert(gM % tM == 0, "gM must be divisible by tM");
    static_assert(gN % tN == 0, "gN must be divisible by tN");
    static_assert(gK % tK == 0, "gK must be divisible by tK");
    static_assert(gK % kPackedFactor == 0 && tK % kPackedFactor == 0,
                  "K must be divisible by the packed element factor");
    static_assert(tM % kPeNum == 0, "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0 && gM % kGroupM == 0,
                  "M dimensions must be divisible by group_M");
    static_assert(gK % kScaleGroup == 0 && tK % kScaleGroup == 0,
                  "HiF4 requires K and tK divisible by scale group (64)");

    // Carrier K for GM storage (packed FP4x2: 2 logical per byte)
    constexpr int kStoredGK = gK / kPackedFactor;
    constexpr int kStoredTK = tK / kPackedFactor;
    constexpr int kScaleK = (tK + kScaleGroup - 1) / kScaleGroup;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;
    constexpr int kGScaleK = gK / kScaleGroup;

    const uint32_t tid = get_thread_idx();

    // GM slices use carrier-stride columns
    using gmASlice = global_tensor<dtype, RowMajor<kGroupM, kStoredGK>>;
    using gmBSlice = global_tensor<dtype, RowMajor<tN, kStoredGK>>;

    // Shared tiles use logical K (PTO ISA CUBE contract)
    using tileAMatrix = SharedMatrixLeft<dtype, kGroupM, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;

    // MX scale shared tiles (U32, logical group-64)
    using tileAScaleMatrix =
        SharedMatrixLeft<scale_dtype, kGroupM, kPaddedScaleK,
                         kGroupM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<scale_dtype, tN, kPaddedScaleK,
                         tN, kScaleK>;
    using tileAScaleShared = SharedTile<tileAScaleMatrix>;
    using tileBScaleShared = SharedTile<tileBScaleMatrix>;

    // PE-private CUBE accumulator
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    // SharedTile capacity check
    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kSharedOperandBytes =
        tileAMatrix::LogicalTileBytes + tileBMatrix::LogicalTileBytes +
        tileAScaleMatrix::LogicalTileBytes + tileBScaleMatrix::LogicalTileBytes;
    static_assert(kSharedOperandBytes <= kSharedTRegBytes,
                  "A/B and MX scale tiles exceed the 256 KiB SharedTReg pool");

    // Global scale slice tensors (carrier-stride for packed B)
    using gmAScaleSlice = global_tensor<scale_dtype, RowMajor<kGroupM, kGScaleK>>;
    using gmBScaleSlice = global_tensor<scale_dtype, RowMajor<tN, kGScaleK>>;

    // Global C slice
    using gmCSlice = global_tensor<float, RowMajor<kPeM, gN>>;

    constexpr int Mb = gM / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    for (int b = 0; b < Batch; ++b) {
        size_t bOffA  = (size_t)b * gM * kStoredGK;
        size_t bOffB  = (size_t)b * gN * kStoredGK;   // B is [gN, gK_stored]
        size_t bOffC  = (size_t)b * gM * gN;
        size_t bOffAS = (size_t)b * gM * kGScaleK;
        size_t bOffBS = (size_t)b * gN * kGScaleK;   // B scale is [gN, kGScaleK]

#pragma clang loop unroll(full)
        for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
            for (int j = 0; j < Nb; ++j) {
                tileC tC;

#pragma clang loop unroll(full)
                for (int k = 0; k < Kb; ++k) {
                    // GM slices (carrier-stride for packed FP4x2)
                    gmASlice gA(src0 + bOffA +
                                (size_t)i * kGroupM * kStoredGK +
                                (size_t)k * kStoredTK);
                    gmBSlice gB(src1 + bOffB +
                                (size_t)j * tN * kStoredGK +
                                (size_t)k * kStoredTK);

                    tileAShared tA;
                    tileBShared tB;
                    TLOAD<tileAMatrix, 1>(tA, gA);
                    TLOAD<tileBMatrix, 1>(tB, gB);

                    tileAScaleShared tAScale;
                    tileBScaleShared tBScale;
                    gmAScaleSlice gAScale(src0_mx + bOffAS +
                                          (size_t)i * kGroupM * kGScaleK +
                                          (size_t)k * kScaleK);
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

                gmCSlice gC(dst + bOffC +
                            ((size_t)i * kGroupM + (size_t)tid * kPeM) * gN +
                            (size_t)j * tN);
                TSTORE_CUBE(gC, tC);
            }
        }
    }
}

#endif  // QUANT_BATCH_MATMUL_HIF4_MT_KERNEL_HPP
