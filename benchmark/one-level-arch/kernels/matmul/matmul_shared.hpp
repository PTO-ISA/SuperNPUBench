#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

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
    constexpr int kTileByteLimit = 8 * 1024;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM * tK * sizeof(dtype) < kTileByteLimit,
                  "each PE A tile must be smaller than 8 KB");
    static_assert(tM * tN * sizeof(float) < kTileByteLimit,
                  "each PE C tile must be smaller than 8 KB");
    static_assert(tK * tN * sizeof(dtype) < kTileByteLimit,
                  "shared B tile must be smaller than 8 KB");

    const uint32_t tid = get_thread_idx();

    // a_ptr += tid * gM * gK;
    // c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileALocal = TileLeft<dtype, tM, tK>;
    using tileBLocal = TileRight<dtype, tK, tN>;
    using tileAShared = SharedTile<tileALocal>;
    using tileBShared = SharedTile<tileBLocal>;
    using tileC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;

    using itA = global_iterator<gmA, tileALocal>;
    using itB = global_iterator<gmB, tileBLocal>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = gM / tM;
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
                TLOAD<tileALocal, 1>(tAShared, gA);
                TLOAD<tileBLocal, 1>(tBShared, gB);
                TMATMUL(tC, tAShared, tBShared);
            } else {
                {
                    auto gA = gIterA(i, 0);
                    auto gB = gIterB(0, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileALocal, 1>(tAShared, gA);
                    TLOAD<tileBLocal, 1>(tBShared, gB);
                    TMATMUL(tC, tAShared, tBShared);
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(k, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileALocal, 1>(tAShared, gA);
                    TLOAD<tileBLocal, 1>(tBShared, gB);
                    TMATMUL_ACC(tC, tC, tAShared, tBShared);
                }
            }

            auto gC = gIterC(i, j);
            TSTORE(gC, tC);
        }
    }
}
