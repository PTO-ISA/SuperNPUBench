#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE multi-thread GMMA matmul.
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
// Thread/PE mapping:
//   - get_thread_idx() selects one complete A/C matrix from those arrays.
//   - The kernel's gM and tM are already PE-local dimensions; no further
//     row splitting occurs inside the kernel.
//   - B is not split. Each PE loads the complete [tK, tN] rhs operand into
//     TileRight.
//
// Tile mapping:
//   - Each PE holds A_pe [tM, tK] and C_pe [tM, tN].
//   - The four PE-local A cells collectively form A_big [4*tM, tK].
//   - Each PE presents one TileRight B operand with shape [tK, tN].
//   - TMATMUL collectively computes C_big [4*tM, tN], while each PE receives
//     only its own accumulator C_pe [tM, tN].
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_multithread(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kTileByteLimit = 8 * 1024;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM <= 32,
                  "local CUBE_M16/M32 supports at most 32 rows per PE");
    static_assert(tM * tK * sizeof(dtype) < kTileByteLimit,
                  "each PE A tile must be smaller than 8 KB");
    static_assert(tM * tN * sizeof(float) < kTileByteLimit,
                  "each PE C tile must be smaller than 8 KB");
    static_assert(tK * tN * sizeof(dtype) < kTileByteLimit,
                  "shared B tile must be smaller than 8 KB");

    const uint32_t tid = get_thread_idx();

    // A/C are arrays of PE matrices. Select one complete matrix before
    // constructing the PE-local global iterators. B keeps its shared base.
    a_ptr += tid * gM * gK;
    c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // PE-private lhs and output cells.
    using tileAM16 = CubeTileM16<dtype, tM, tK>;
    using tileAM32 = CubeTileM32<dtype, tM, tK>;
    using tileA = std::conditional_t<(tM <= 16), tileAM16, tileAM32>;
    using tileCM16 = CubeAccumulatorM16<float, tM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, tM, tN>;
    using tileC = std::conditional_t<(tM <= 16), tileCM16, tileCM32>;

    // TLOAD requires a local tile. Publish it to compiler-managed shared
    // storage before passing B to the shared-right TMATMUL overload.
    using tileBLocal = SharedMatrixRight<dtype, tK, tN>;
    using tileBShared = SharedTile<tileBLocal>;

    using itA = global_iterator<gmA, tileA>;
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
                tileA tA;
                tileBLocal tBLocal;

                auto gA = gIterA(i, 0);
                TLOAD_CUBE(tA, gA);
                auto gB = gIterB(0, j);
                TLOAD(tBLocal, gB);
                tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                TMATMUL(tC, tA, tBShared);
            } else {
                // Initialize the accumulator from the first K block.
                {
                    tileA tA;
                    tileBLocal tBLocal;
                    auto gA = gIterA(i, 0);
                    auto gB = gIterB(0, j);
                    TLOAD_CUBE(tA, gA);
                    TLOAD(tBLocal, gB);
                    tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                    TMATMUL(tC, tA, tBShared);
                }

                // Accumulate all remaining K blocks.
                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    tileA tA;
                    tileBLocal tBLocal;
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(k, j);
                    TLOAD_CUBE(tA, gA);
                    TLOAD(tBLocal, gB);
                    tileBShared tBShared = TMOV_L2S_PUBLISH(tBLocal);
                    TMATMUL_ACC(tC, tC, tA, tBShared);
                }
            }

            auto gC = gIterC(i, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}
