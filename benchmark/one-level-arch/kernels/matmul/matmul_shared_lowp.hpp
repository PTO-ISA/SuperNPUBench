#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE low-precision matmul.
//
// A/B are Core-shared operands. Each PE owns one [tM / 4, tN] FP32 C tile.
// PackedFactor is 1 for 8-bit formats and 2 for packed FP4x2 formats.
// MXFP8/MXFP4 additionally load E8M0 scale data through Shared tiles.
// HiF8 is a normal TMATMUL input in PTO v0.58 and does not take MX scales.
// For packed FP4x2, the matrix contract counts packed storage elements:
//   AScale: [group_M, ceil(stored_K / 32)]
//   BScale: [ceil(stored_K / 32), N]
template <typename dtype, int PackedFactor, bool UseMx,
          int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared_lowp(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                        uint8_t *a_scale_ptr, uint8_t *b_scale_ptr) {
    constexpr int kPeNum = 4;
    // PTO v0.58 cooperative TMATMUL encodes at most group_M=128. Keep the
    // externally configured tM (and its ELF name), but split tM=256 into two
    // complete 128-row Shared A groups.
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;
    constexpr int kStoredGK = gK / PackedFactor;
    constexpr int kStoredTK = tK / PackedFactor;
    constexpr int kScaleGroup = 32;
    constexpr int kScaleK = (kStoredTK + kScaleGroup - 1) / kScaleGroup;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8) or 2 (FP4x2)");
    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(gK % PackedFactor == 0 && tK % PackedFactor == 0,
                  "K must be divisible by the packed element factor");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0 && gM % kGroupM == 0,
                  "M dimensions must be divisible by group_M");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");
    static_assert(!UseMx || (gK % kScaleGroup == 0 &&
                             tK % kScaleGroup == 0),
                  "MX formats require K and tK divisible by 32");

    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, kStoredGK>>;
    using gmB = global_tensor<dtype, RowMajor<kStoredGK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kGroupM, kStoredTK>;
    using tileBMatrix = SharedMatrixRight<dtype, kStoredTK, tN>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    using itA = global_iterator<gmA, tileAMatrix>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    // Scaling tiles contain the actual E8M0 payload: one byte per 32 matrix
    // storage elements. They must not inherit the matrix operand's full K
    // extent, otherwise a capacity-bound N=256 B-scale tile is needlessly
    // inflated from (K/32)*N bytes to K*N bytes.
    using scale_dtype = __fp8_e8m0;
    using gmAScale =
        global_tensor<scale_dtype, RowMajor<gM, gK / kScaleGroup>>;
    using gmBScale =
        global_tensor<scale_dtype, RowMajor<gK / kScaleGroup, gN>>;
    using tileAScaleMatrix =
        SharedMatrixLeft<scale_dtype, kGroupM, kPaddedScaleK,
                         kGroupM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<scale_dtype, kPaddedScaleK, tN,
                          kScaleK, tN>;
    using tileAScale = SharedTile<tileAScaleMatrix>;
    using tileBScale = SharedTile<tileBScaleMatrix>;
    using itAScale = global_iterator<gmAScale, tileAScaleMatrix>;
    using itBScale = global_iterator<gmBScale, tileBScaleMatrix>;

    itAScale gIterAScale(reinterpret_cast<scale_dtype *>(a_scale_ptr));
    itBScale gIterBScale(reinterpret_cast<scale_dtype *>(b_scale_ptr));

    constexpr int Mb = gM / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

#pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileAShared tA;
                tileBShared tB;
                auto gA = gIterA(i, k);
                auto gB = gIterB(k, j);
                TLOAD<tileAMatrix, 1>(tA, gA);
                TLOAD<tileBMatrix, 1>(tB, gB);

                if constexpr (UseMx) {
                    tileAScale tAScale;
                    tileBScale tBScale;
                    auto gAScale = gIterAScale(i, k);
                    auto gBScale = gIterBScale(k, j);
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
                } else {
                    if (k == 0) {
                        TMATMUL(tC, tA, tB);
                    } else {
                        TMATMUL_ACC(tC, tC, tA, tB);
                    }
                }
            }

            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}
