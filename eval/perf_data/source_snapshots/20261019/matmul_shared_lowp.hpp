#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

namespace matmul_shared_lowp_detail {

template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int ValidN>
PTO_SHARED_INLINE void run_block(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                 scale_dtype *a_scale_ptr,
                                 scale_dtype *b_scale_ptr, int m_begin,
                                 int n_begin) {
    constexpr int kTileRows = ValidM <= 64 ? 64 : 128;
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    // HiF4's U32 scale contract covers 64 K lanes. Padding its last physical
    // K tile with zero payload avoids an unsupported mixed-64/32 accumulator
    // chain while preserving the logical gK result.
    constexpr int kPhysicalGK =
        UseMx && ScaleGroup == 64 ? ((gK + tK - 1) / tK) * tK : gK;
    constexpr int kStoredGK = kPhysicalGK / PackedFactor;
    constexpr int kStoredTK = tK / PackedFactor;
    constexpr int kGlobalScaleK =
        (kPhysicalGK + ScaleGroup - 1) / ScaleGroup;
    constexpr int kScaleK = (tK + ScaleGroup - 1) / ScaleGroup;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;
    constexpr int kFullKBlocks = kPhysicalGK / tK;
    constexpr int kTailK = kPhysicalGK % tK;
    constexpr int kTotalKBlocks = kFullKBlocks + (kTailK != 0 ? 1 : 0);
    constexpr auto matmulOptions = fixp::keep_acc();

    using gmA = global_tensor<dtype, RowMajor<gM, kStoredGK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, kStoredGK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using gmAScale =
        global_tensor<scale_dtype, RowMajor<gM, kGlobalScaleK>>;
    using gmBScale =
        global_tensor<scale_dtype, RowMajor<gN, kGlobalScaleK>>;

    using tileAMatrix =
        SharedMatrixLeft<dtype, kTileRows, tK, ValidM, tK>;
    using tileBMatrix =
        SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileAScaleMatrix =
        SharedMatrixLeft<scale_dtype, kTileRows, kPaddedScaleK,
                         ValidM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<scale_dtype, tN, kPaddedScaleK,
                          ValidN, kScaleK>;
    using tileAScale = SharedTile<tileAScaleMatrix>;
    using tileBScale = SharedTile<tileBScaleMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN, kPeM, ValidN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN, kPeM, ValidN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    constexpr int kSharedOperandBytes =
        tileAMatrix::LogicalTileBytes + tileBMatrix::LogicalTileBytes +
        (UseMx ? tileAScaleMatrix::LogicalTileBytes +
                     tileBScaleMatrix::LogicalTileBytes
               : 0);
    static_assert(kSharedOperandBytes <= 256 * 1024,
                  "A/B and MX scale tiles exceed the 256 KiB SharedTReg pool");

    tileC tC;
#pragma clang loop unroll(full)
    for (int k = 0; k < kFullKBlocks; ++k) {
        gmA gA(a_ptr + m_begin * kStoredGK + k * kStoredTK);
        gmB gB(b_ptr + n_begin * kStoredGK + k * kStoredTK);
        tileAShared tA;
        tileBShared tB;
        TLOAD<tileAMatrix, 1>(tA, gA);
        TLOAD<tileBMatrix, 1>(tB, gB);

        if constexpr (UseMx) {
            const int scale_begin = (k * tK) / ScaleGroup;
            gmAScale gAScale(a_scale_ptr +
                             m_begin * kGlobalScaleK + scale_begin);
            gmBScale gBScale(b_scale_ptr +
                             n_begin * kGlobalScaleK + scale_begin);
            tileAScale tAScale;
            tileBScale tBScale;
            TLOAD<tileAScaleMatrix, 1>(tAScale, gAScale);
            TLOAD<tileBScaleMatrix, 1>(tBScale, gBScale);
            if constexpr (kTotalKBlocks == 1) {
                TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale,
                              matmulOptions);
            } else if (k == 0) {
                if constexpr (kTailK != 0) {
                    TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale,
                                  matmulOptions);
                } else {
                    TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale,
                                  matmulOptions.raw_acc());
                }
            } else if (k == kFullKBlocks - 1 && kTailK == 0) {
                TMATMUL_MX_ACC<3>(tC, tC, tA, tAScale, tB, tBScale,
                                  matmulOptions.acc_hint());
            } else {
                TMATMUL_MX_ACC<3>(
                    tC, tC, tA, tAScale, tB, tBScale,
                    matmulOptions.raw_acc().acc_hint());
            }
        } else {
            if constexpr (kTotalKBlocks == 1) {
                TMATMUL(tC, tA, tB, matmulOptions);
            } else if (k == 0) {
                if constexpr (kTailK != 0) {
                    TMATMUL(tC, tA, tB, matmulOptions);
                } else {
                    TMATMUL(tC, tA, tB, matmulOptions.raw_acc());
                }
            } else if (k == kFullKBlocks - 1 && kTailK == 0) {
                TMATMUL_ACC(tC, tC, tA, tB, matmulOptions.acc_hint());
            } else {
                TMATMUL_ACC(tC, tC, tA, tB,
                            matmulOptions.raw_acc().acc_hint());
            }
        }
    }

    if constexpr (kTailK != 0) {
        constexpr int kTailScaleK =
            (kTailK + ScaleGroup - 1) / ScaleGroup;
        constexpr int kPaddedTailScaleK =
            ((kTailScaleK + 31) / 32) * 32;
        using tailAMatrix =
            SharedMatrixLeft<dtype, kTileRows, kTailK, ValidM, kTailK>;
        using tailBMatrix =
            SharedMatrixRight<dtype, tN, kTailK, ValidN, kTailK>;
        using tailAShared = SharedTile<tailAMatrix>;
        using tailBShared = SharedTile<tailBMatrix>;

        gmA gA(a_ptr + m_begin * kStoredGK +
               kFullKBlocks * kStoredTK);
        gmB gB(b_ptr + n_begin * kStoredGK +
               kFullKBlocks * kStoredTK);
        tailAShared tA;
        tailBShared tB;
        TLOAD<tailAMatrix, 1>(tA, gA);
        TLOAD<tailBMatrix, 1>(tB, gB);

        if constexpr (UseMx) {
            using tailAScaleMatrix =
                SharedMatrixLeft<scale_dtype, kTileRows,
                                 kPaddedTailScaleK, ValidM,
                                 kTailScaleK>;
            using tailBScaleMatrix =
                SharedMatrixRight<scale_dtype, tN,
                                  kPaddedTailScaleK, ValidN,
                                  kTailScaleK>;
            using tailAScale = SharedTile<tailAScaleMatrix>;
            using tailBScale = SharedTile<tailBScaleMatrix>;
            constexpr int kScaleBegin =
                (kFullKBlocks * tK) / ScaleGroup;
            gmAScale gAScale(a_scale_ptr +
                             m_begin * kGlobalScaleK + kScaleBegin);
            gmBScale gBScale(b_scale_ptr +
                             n_begin * kGlobalScaleK + kScaleBegin);
            tailAScale tAScale;
            tailBScale tBScale;
            TLOAD<tailAScaleMatrix, 1>(tAScale, gAScale);
            TLOAD<tailBScaleMatrix, 1>(tBScale, gBScale);
            if constexpr (kFullKBlocks == 0) {
                TMATMUL_MX<3>(tC, tA, tAScale, tB, tBScale,
                              matmulOptions);
            } else {
                TMATMUL_MX_ACC<3>(tC, tC, tA, tAScale, tB, tBScale,
                                  matmulOptions);
            }
        } else {
            if constexpr (kFullKBlocks == 0) {
                TMATMUL(tC, tA, tB, matmulOptions);
            } else {
                TMATMUL_ACC(tC, tC, tA, tB, matmulOptions);
            }
        }
    }

    const int tid = static_cast<int>(get_thread_idx());
    if (tid * kPeM < ValidM) {
        gmC gC(c_ptr + (m_begin + tid * kPeM) * gN + n_begin);
        TSTORE_CUBE(gC, tC);
    }
}

constexpr int floor_power_of_two(int value) {
    int result = 1;
    while ((result << 1) <= value) {
        result <<= 1;
    }
    return result;
}

template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int RemainingN>
PTO_SHARED_INLINE void run_n_tail(
    float *c_ptr, dtype *a_ptr, dtype *b_ptr, scale_dtype *a_scale_ptr,
    scale_dtype *b_scale_ptr, int m_begin, int n_begin) {
    constexpr int kChunkN = floor_power_of_two(RemainingN);
    run_block<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
              gM, gN, gK, tN, tK, ValidM, kChunkN>(
        c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, m_begin, n_begin);
    if constexpr (RemainingN > kChunkN) {
        run_n_tail<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
                   gM, gN, gK, tN, tK, ValidM,
                   RemainingN - kChunkN>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, m_begin,
            n_begin + kChunkN);
    }
}

template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_m_strip(
    float *c_ptr, dtype *a_ptr, dtype *b_ptr, scale_dtype *a_scale_ptr,
    scale_dtype *b_scale_ptr, int m_begin) {
    constexpr int kFullNBlocks = gN / tN;
    constexpr int kTailN = gN % tN;
#pragma clang loop unroll(disable)
    for (int j = 0; j < kFullNBlocks; ++j) {
        run_block<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
                  gM, gN, gK, tN, tK, ValidM, tN>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, m_begin,
            j * tN);
    }
    if constexpr (kTailN != 0) {
        run_n_tail<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
                   gM, gN, gK, tN, tK, ValidM, kTailN>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, m_begin,
            kFullNBlocks * tN);
    }
}

template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup, int gM, int gN, int gK, int tN, int tK,
          int RemainingM>
PTO_SHARED_INLINE void run_m_tail(
    float *c_ptr, dtype *a_ptr, dtype *b_ptr, scale_dtype *a_scale_ptr,
    scale_dtype *b_scale_ptr, int m_begin) {
    constexpr int kChunkM = floor_power_of_two(RemainingM);
    run_m_strip<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
                gM, gN, gK, tN, tK, kChunkM>(
        c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, m_begin);
    if constexpr (RemainingM > kChunkM) {
        run_m_tail<dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
                   gM, gN, gK, tN, tK, RemainingM - kChunkM>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr,
            m_begin + kChunkM);
    }
}

}  // namespace matmul_shared_lowp_detail

// Four-PE low-precision matmul.
//
// A/B are Core-shared operands. Each PE owns one [tM / 4, tN] FP32 C tile.
// PackedFactor is 1 for 8-bit formats and 2 for packed FP4x2 formats.
// MXFP8/MXFP4 additionally load group-32 E8M0 scale data through Shared
// tiles. HiF4X2 is a Matrix-MX-only input and uses one raw U32 scale word per
// 64 logical K elements.
// HiF8 is a normal TMATMUL input in PTO v0.58 and does not take MX scales.
template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup,
          int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared_lowp(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                        scale_dtype *a_scale_ptr,
                        scale_dtype *b_scale_ptr) {
    constexpr int kPeNum = 4;
    // PTO v0.58 cooperative TMATMUL encodes at most group_M=128. Keep the
    // externally configured tM (and its ELF name), but split tM=256 into two
    // complete 128-row Shared A groups.
    constexpr int kGroupM = tM <= 128 ? tM : 128;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8) or 2 (FP4x2)");
    static_assert(gK % PackedFactor == 0 && tK % PackedFactor == 0,
                  "K must be divisible by the packed element factor");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");
    static_assert((kGroupM & (kGroupM - 1)) == 0,
                  "group_M must be a power of two");
    static_assert((tN & (tN - 1)) == 0,
                  "tN must be a power of two");

    constexpr int kFullMBlocks = gM / kGroupM;
    constexpr int kTailM = gM % kGroupM;
#pragma clang loop unroll(disable)
    for (int i = 0; i < kFullMBlocks; ++i) {
        matmul_shared_lowp_detail::run_m_strip<
            dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
            gM, gN, gK, tN, tK, kGroupM>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr,
            i * kGroupM);
    }

    if constexpr (kTailM != 0) {
        matmul_shared_lowp_detail::run_m_tail<
            dtype, scale_dtype, PackedFactor, UseMx, ScaleGroup,
            gM, gN, gK, tN, tK, kTailM>(
            c_ptr, a_ptr, b_ptr, a_scale_ptr, b_scale_ptr,
            kFullMBlocks * kGroupM);
    }
}
