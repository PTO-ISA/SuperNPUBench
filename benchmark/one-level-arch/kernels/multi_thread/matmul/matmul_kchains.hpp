#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE cooperative matmul with a tiled K reduction.
//
// Mathematical semantics:
//   C = A * B
//   A: [M, K], B: [K, N], C: [M, N]
//
// The K reduction uses direct compiler APIs: TMATMUL initializes C and
// TMATMUL_ACC accumulates subsequent pieces. keep_acc preserves FP32 output;
// raw_acc/acc_hint enable CCTRL; the ACC API still requires explicit Cin.
// B is stored as [K,N], so Shared-B TransB is enabled.
//
// Host-visible storage (same as matmul_shared):
//   - A is an array of four PE matrices, each with shape [gM, gK].
//   - C is an array of four PE matrices, each with shape [gM, gN].
//   - B is one shared matrix with shape [gK, gN].
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int ChainK>
void matmul_kchains(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    // PTO v0.58 cooperative TMATMUL supports group_M up to 128. A configured
    // tM=256 is therefore materialized as two complete 128-row Shared groups.
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;
    // SharedMatrixLeft physical Rows must be a supported size (64 or 128). When
    // kGroupM is already one of those, the tile is exact; otherwise allocate the
    // next supported size and let ValidRow carry the real cooperative row count.
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kKChains = gK / ChainK;

    static_assert(gM % tM == 0, "M must be divisible by tM");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % ChainK == 0, "K must be divisible by ChainK");
    static_assert(ChainK > 0, "ChainK must be positive");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0 && gM % kGroupM == 0,
                  "M dimensions must be divisible by group_M");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    // K chain: single=0, begin=raw_acc, middle=raw_acc|acc_hint, end=acc_hint.
    // C remains explicit; gfsim also needs cube.enable_internal_acc=true.
    constexpr auto matmulOptions = fixp::keep_acc().transpose_b();
    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // A chain piece: [group_M, ChainK] (Left, M x Kc). B chain piece:
    // [ChainK, tN] (Right, Kc x N). C: per-PE [kPeM, tN] local accumulator.
    using tileAMatrix =
        SharedMatrixLeft<dtype, kTileRows, ChainK, kGroupM, ChainK>;
    using tileBMatrix = SharedMatrixRight<dtype, ChainK, tN>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    // TMATMUL is issued cooperatively by four PEs. The shared A tile covers the
    // complete [group_M, ChainK] block, while each PE keeps its CUBE row slice
    // of C in a private accumulator tile.
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
    constexpr int kSharedTRegBytes = 256 * 1024;
    static_assert(tileAMatrix::LogicalTileBytes + tileBMatrix::LogicalTileBytes
                      <= kSharedTRegBytes,
                  "one A/B K-chain piece exceeds SharedTReg capacity");

#pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC;

            // K K-chain. A is [group_M, ChainK], B is [ChainK, tN]; each piece
            // reduces a ChainK-wide slice of the K dimension. The accumulator
            // is passed directly to TMATMUL_ACC; only the completed chain is stored.
#pragma clang loop unroll(full)
            for (int kc = 0; kc < kKChains; ++kc) {
                tileAShared tAShared;
                tileBShared tBShared;
                auto gA = gIterA(i, kc);
                auto gB = gIterB(kc, j);
                TLOAD<tileAMatrix, 1>(tAShared, gA);
                TLOAD<tileBMatrix, 1>(tBShared, gB);

                if constexpr (kKChains == 1) {
                    TMATMUL(tC, tAShared, tBShared, matmulOptions);
                } else if (kc == 0) {
                    TMATMUL(tC, tAShared, tBShared, matmulOptions.raw_acc());
                } else if (kc == kKChains - 1) {
                    TMATMUL_ACC(tC, tC, tAShared, tBShared, matmulOptions.acc_hint());
                } else {
                    TMATMUL_ACC(tC, tC, tAShared, tBShared, matmulOptions.raw_acc().acc_hint());
                }
            }

            // itC advances by the per-PE CUBE row count. Map PE tid to its
            // row slice in the current [group_M, tN] output block.
            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}
