#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE multi-thread matmul with Shared TLOAD.
//
// Mathematical semantics:
//   C = A * B
//   A: [M, K], B: [K, N], C: [M, N]
//
// Host-visible storage:
//   - A is an array of four PE matrices, each with shape [gM, gK].
//   - C is an array of four PE matrices, each with shape [gM, gN].
//   - B is one shared matrix with shape [gK, gN].
//
// Difference from matmul_multithread:
//   - Both A and B are loaded directly into SharedTile via the new
//     GM->Shared TLOAD (PTO v0.58 reissue), eliminating the
//     TMOV_L2S_PUBLISH step entirely.
//   - A is a Shared Left tile, B is a Shared Right tile.
//   - C remains a PE-private local accumulator tile.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    // PTO v0.58 cooperative TMATMUL supports group_M up to 128. A configured
    // tM=256 is therefore materialized as two complete 128-row Shared groups.
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0 && gM % kGroupM == 0,
                  "M dimensions must be divisible by group_M");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    const uint32_t tid = get_thread_idx();

    // a_ptr += tid * gM * gK;
    // c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kGroupM, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tK, tN>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    // TMATMUL is issued cooperatively by four PEs.  The shared A tile covers
    // the complete [group_M, tK] block, while each PE keeps its CUBE row
    // slice of C in a private accumulator tile.
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    using itA = global_iterator<gmA, tileAMatrix>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = gM / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    #pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

            if constexpr (Kb == 1) {
                auto gA = gIterA(i, 0);
                auto gB = gIterB(0, j);
                tileAShared tAShared;
                tileBShared tBShared;
                TLOAD<tileAMatrix, 1>(tAShared, gA);
                TLOAD<tileBMatrix, 1>(tBShared, gB);
                TMATMUL(tC, tAShared, tBShared);
            } else {
                {
                    auto gA = gIterA(i, 0);
                    auto gB = gIterB(0, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileAMatrix, 1>(tAShared, gA);
                    TLOAD<tileBMatrix, 1>(tBShared, gB);
                    TMATMUL(tC, tAShared, tBShared);
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(k, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileAMatrix, 1>(tAShared, gA);
                    TLOAD<tileBMatrix, 1>(tBShared, gB);
                    TMATMUL_ACC(tC, tC, tAShared, tBShared);
                }
            }

            // itC advances by the per-PE CUBE row count. Map PE tid to its
            // row slice in the current [group_M, tN] output block.
            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}
