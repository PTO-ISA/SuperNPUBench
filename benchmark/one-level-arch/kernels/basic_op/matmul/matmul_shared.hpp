#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE multi-thread matmul with Shared TLOAD.
//
// Mathematical semantics:
//   C = A * B^T
//   A: [M, K], B: [N, K], C: [M, N]
//
// Host-visible storage:
//   - A is an array of four PE matrices, each with shape [gM, gK].
//   - C is an array of four PE matrices, each with shape [gM, gN].
//   - B is one shared B-major matrix with shape [gN, gK].
//
// Both A and B are loaded directly into SharedTile via GM->Shared TLOAD.
// A is a Shared Left tile, B is a Shared Right tile, and C remains a
// PE-private local accumulator tile.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    // PTO v0.58 cooperative TMATMUL supports group_M up to 128. A configured
    // tM=256 is therefore materialized as two complete 128-row Shared groups.
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    // SharedMatrixLeft physical Rows must be a supported size (64 or 128).
    // When gM < kTileRows, the tile still allocates the full physical rows
    // but ValidRow carries the actual row count so TLOAD only reads gM rows.
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kValidRowM = (gM < kTileRows) ? gM : kTileRows;
    // kPeM follows the effective group_M (kValidRowM), not the physical
    // kGroupM: when gM < 128 the TMATMUL sees group_M=kValidRowM, so per-PE
    // rows must match cooperative_group_m_rows_per_pe(kValidRowM).
    constexpr int kPeM = kValidRowM <= 64 ? 16 : 32;

    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert(gM % kGroupM == 0 || gM < kGroupM,
                  "gM must be a multiple of kGroupM or smaller than it");
    static_assert(gM % kPeM == 0 || gM < kGroupM,
                  "gM must be a multiple of kPeM, or a partial tile (gM < kGroupM)");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    // K chain: single=0, begin=raw_acc, middle=raw_acc|acc_hint, end=acc_hint.
    // C remains explicit; gfsim also needs cube.enable_internal_acc=true.
    constexpr auto matmulOptions = fixp::keep_acc();
    const uint32_t tid = get_thread_idx();

    // a_ptr += tid * gM * gK;
    // c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kTileRows, tK, kValidRowM, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
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

    constexpr int Mb = (gM + kGroupM - 1) / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr int kSharedTRegBytes = 256 * 1024;
    static_assert(tileAMatrix::LogicalTileBytes +
                          tileBMatrix::LogicalTileBytes <=
                      kSharedTRegBytes,
                  "A/B tiles exceed the 256 KiB SharedTReg pool");
    #pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

            if constexpr (Kb == 1) {
                auto gA = gIterA(i, 0);
                auto gB = gIterB(j, 0);
                tileAShared tAShared;
                tileBShared tBShared;
                TLOAD<tileAMatrix, 1>(tAShared, gA);
                TLOAD<tileBMatrix, 1>(tBShared, gB);
                TMATMUL(tC, tAShared, tBShared, matmulOptions);
            } else {
                {
                    auto gA = gIterA(i, 0);
                    auto gB = gIterB(j, 0);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileAMatrix, 1>(tAShared, gA);
                    TLOAD<tileBMatrix, 1>(tBShared, gB);
                    TMATMUL(tC, tAShared, tBShared, matmulOptions.raw_acc());
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(j, k);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileAMatrix, 1>(tAShared, gA);
                    TLOAD<tileBMatrix, 1>(tBShared, gB);
                    if (k == Kb - 1) {
                        TMATMUL_ACC(tC, tC, tAShared, tBShared,
                                    matmulOptions.acc_hint());
                    } else {
                        TMATMUL_ACC(tC, tC, tAShared, tBShared,
                                    matmulOptions.raw_acc().acc_hint());
                    }
                }
            }

            // itC advances by the per-PE CUBE row count. Map PE tid to its
            // row slice in the current [group_M, tN] output block.  When
            // gM < kGroupM only the first ceil(gM/kPeM) PEs have valid rows;
            // the rest are skipped to avoid out-of-bounds C writes.
            if constexpr (gM >= kGroupM) {
                auto gC = gIterC(i * kPeNum + tid, j);
                TSTORE_CUBE(gC, tC);
            } else {
                if ((i * kPeNum + static_cast<int>(tid)) * kPeM < gM) {
                    auto gC = gIterC(i * kPeNum + tid, j);
                    TSTORE_CUBE(gC, tC);
                }
            }
        }
    }
}
