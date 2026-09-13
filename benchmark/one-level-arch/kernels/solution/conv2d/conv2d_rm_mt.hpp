#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

template <typename E_, int R_, int C_, int VR_ = R_, int VC_ = C_>
using CubeTileA = std::conditional_t<
    (R_ <= 16), CubeTileM16<E_, R_, C_, VR_, VC_>,
    CubeTileM32<E_, R_, C_, VR_, VC_>>;

template <typename E_, int R_, int C_, int VR_ = R_, int VC_ = C_>
using CubeTileC = std::conditional_t<
    (R_ <= 16), CubeAccumulatorM16<E_, R_, C_, VR_, VC_>,
    CubeAccumulatorM32<E_, R_, C_, VR_, VC_>>;

// Four-PE multi-thread conv2d 1x1 with RowMajor global tensors, modeled on
// the upstream matmul_shared Shared-Right form (TileOP-API f94bc12, PTO ISA
// 0.58.4 CUBE cell layout contracts).
//
// Mathematical semantics (per PE, batch of 4 inputs):
//   C_pe = A_pe * B   (conv2d 1x1 = matrix multiply)
//   A_pe (input):  RowMajor<gM, gK>  (NHWC flattened, gM = in_h*in_w, gK = in_c)
//   B (weight):    RowMajor<gK, gN>  (W^T, shared across all PEs)
//   C_pe (output): RowMajor<gM, gN>  (NHWC flattened)
//
// Host-visible storage:
//   - input is an array of four PE matrices, each with shape [gM, gK].
//   - output is an array of four PE matrices, each with shape [gM, gN].
//   - weight is one shared matrix with shape [gK, gN].
//
// Thread/PE mapping (batch-level parallelism):
//   - get_thread_idx() selects one complete input/output matrix pair from
//     those arrays: 4 inputs convolved with the same weight.
//   - Per-PE base pointers are computed once at kernel entry, so the block
//     iterators stay pure block-index addressing (no per-lane address
//     computation inside the unrolled loops).
//
// Tile mapping (new CUBE cell contracts):
//   - Each PE holds a private CUBE_M16/M32 A tile [tM, tK] loaded via
//     TLOAD_CUBE (hardware ND->CUBE cell conversion).
//   - Each PE holds a private CUBE_N8 B tile [tK, tN] loaded via TLOAD_CUBE
//     from the shared weight. NOTE: the Shared-Right cooperative form
//     (private A + SharedTile B) is NOT usable for the batch pattern under
//     the PTO 0.58.4 ADR-0100 contract: the cooperative TMATMUL interprets
//     LB0 as the Core-total group_M and distributes rowsPerPe (16/32) rows
//     per PE, so a private A with ValidRow == tM (<= 16) yields output on
//     PE0 only. Sharing B for batch inputs would require publishing the four
//     PEs' inputs as quarters of a group_M Shared A via B.ASSEMBLE
//     (future work). Until then each PE issues its own local TMATMUL with a
//     private B copy (4x B GM traffic, correct results).
//   - Each PE accumulates a private CubeAccumulatorM16/M32 C tile [tM, tN]
//     and stores it via TSTORE_CUBE (CUBE cell -> ND conversion).
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_rm_mt(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {
    constexpr int kPeNum = 4;

    static_assert(in_c == 1 * 1 * in_c, "conv2d_1x1 expects kh=kw=1");

    constexpr int gM = in_h * in_w;
    constexpr int gN = out_c;
    constexpr int gK = in_c;

    static_assert(gM % tM == 0, "gM (in_h*in_w) must be divisible by tM");
    static_assert(gN % tN == 0, "gN (out_c) must be divisible by tN");
    static_assert(gK % tK == 0, "gK (in_c) must be divisible by tK");
    static_assert(tM <= 32, "CUBE_M16/M32 matmul supports tM <= 32");

    const uint32_t tid = get_thread_idx();

    // input/output are arrays of PE matrices. Select one complete matrix
    // before constructing the PE-local global iterators. weight keeps its
    // shared base.
    input_nchw_ptr += tid * gM * gK;
    output_ptr += tid * gM * gN;

    using gm_shapeInput  = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float, RowMajor<gM, gN>>;

    // PE-private CUBE lhs and accumulator tiles.
    using tileA = CubeTileA<dtype, tM, tK>;
    using tileC = CubeTileC<float, tM, tN>;

    // Private rhs operand: each PE loads its own CUBE_N8 copy of the block.
    using tileB = CubeTileN8<dtype, tK, tN>;

    using itA = global_iterator<gm_shapeInput, tileA>;
    using itB = global_iterator<gm_shapeWeight, tileB>;
    using itC = global_iterator<gm_shapeOutput, tileC>;

    itA gAIter(input_nchw_ptr);
    itB gBIter(weight_ptr);
    itC gCIter(output_ptr);

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
                tileB tB;

                auto gA = gAIter(i, 0);
                TLOAD_CUBE(tA, gA);
                auto gB = gBIter(0, j);
                TLOAD_CUBE(tB, gB);
                TMATMUL(tC, tA, tB);
            } else {
                {
                    tileA tA;
                    tileB tB;
                    auto gA = gAIter(i, 0);
                    auto gB = gBIter(0, j);
                    TLOAD_CUBE(tA, gA);
                    TLOAD_CUBE(tB, gB);
                    TMATMUL(tC, tA, tB);
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    tileA tA;
                    tileB tB;
                    auto gA = gAIter(i, k);
                    auto gB = gBIter(k, j);
                    TLOAD_CUBE(tA, gA);
                    TLOAD_CUBE(tB, gB);
                    TMATMUL_ACC(tC, tC, tA, tB);
                }
            }

            auto gC = gCIter(i, j);
            TSTORE_CUBE(gC, tC);
        }
    }
}
