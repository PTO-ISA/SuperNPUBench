#ifndef CONV2D_RM_TILEOP_KERNEL_HPP
#define CONV2D_RM_TILEOP_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <type_traits>

using namespace pto;

// CUBE cell layout contracts (TileOP-API f94bc12, PTO ISA 0.58.4):
//   - TMATMUL A/D must use CUBE_M16 or CUBE_M32, B must use CUBE_N8
//   - TLOAD_CUBE/TSTORE_CUBE perform the ND <-> CUBE cell layout conversion
//     in hardware (B.DATR ND2M16/ND2N8/M162ND), so GM data stays plain 2D.
template <typename E_, int R_, int C_, int VR_ = R_, int VC_ = C_>
using CubeTileA = std::conditional_t<
    (R_ <= 16), CubeTileM16<E_, R_, C_, VR_, VC_>,
    CubeTileM32<E_, R_, C_, VR_, VC_>>;

template <typename E_, int R_, int C_, int VR_ = R_, int VC_ = C_>
using CubeTileC = std::conditional_t<
    (R_ <= 16), CubeAccumulatorM16<E_, R_, C_, VR_, VC_>,
    CubeAccumulatorM32<E_, R_, C_, VR_, VC_>>;

namespace conv2d_rm_detail {

// Computes one output tile (i, j): full-K loop plus optional partial-K edge,
// using the supplied A/B/C tile types. AT_KE/BT_KE are the K-edge (partial
// column/row) tile types used for the trailing gK % tK strip.
template <typename AT, typename BT, typename AT_KE, typename BT_KE, typename CT,
          typename ItA, typename ItB, typename ItC, int Kb, bool HasKEdge>
inline void compute_tile(ItA &gAIter, ItB &gBIter, ItC &gCIter, int i, int j) {
    auto gC = gCIter(i, j);
    CT tACC;

    if constexpr (Kb > 0) {
        auto gA = gAIter(i, 0);
        auto gB = gBIter(0, j);
        AT tA;
        BT tB;
        TLOAD_CUBE(tA, gA);
        TLOAD_CUBE(tB, gB);
        TMATMUL(tACC, tA, tB);

        #pragma clang loop unroll(full)
        for (int k = 1; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, j);
            AT tA;
            BT tB;
            TLOAD_CUBE(tA, gA);
            TLOAD_CUBE(tB, gB);
            TMATMUL_ACC(tACC, tACC, tA, tB);
        }
    }

    if constexpr (HasKEdge) {
        auto gA = gAIter(i, Kb);
        auto gB = gBIter(Kb, j);
        AT_KE tA;
        BT_KE tB;
        TLOAD_CUBE(tA, gA);
        TLOAD_CUBE(tB, gB);
        if constexpr (Kb > 0) {
            TMATMUL_ACC(tACC, tACC, tA, tB);
        } else {
            TMATMUL(tACC, tA, tB);
        }
    }

    TSTORE_CUBE(gC, tACC);
}

}  // namespace conv2d_rm_detail

// conv2d_1x1 with RowMajor global tensors and CUBE cell local tiles.
//
// GM data contract (unchanged from the previous RowMajor scheme, no BFractal
// pre-conversion):
//   Input  : RowMajor<gM, gK>  (NHWC flattened)
//   Weight : RowMajor<gK, gN>  (B = W^T, standard RowMajor)
//   Output : RowMajor<gM, gN>  (NHWC flattened)
//
// TLOAD_CUBE converts GM ND data into CUBE_M16/N8 cell layout inside the tile
// register; TSTORE_CUBE converts CUBE_M16 accumulator back to ND on store.
//
// Boundary handling: gM/gN/gK need not be divisible by tM/tN/tK. Trailing
// partial tiles use ValidRow/ValidCol tile variants (upstream matmul pattern),
// so coverage is 100% for any shape.
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_rm_tileop(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {

    static_assert(in_c == 1 * 1 * in_c, "conv2d_1x1 expects kh=kw=1");

    constexpr int gM = in_h * in_w;
    constexpr int gN = out_c;
    constexpr int gK = in_c;

    static_assert(tM <= 32, "CUBE_M16/M32 matmul supports tM <= 32");

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr int rmd_M = gM % tM;
    constexpr int rmd_N = gN % tN;
    constexpr int rmd_K = gK % tK;

    using gm_shapeInput  = global_tensor<dtype,   RowMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype,   RowMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float,   RowMajor<gM, gN>>;

    using tileA_full   = CubeTileA<dtype, tM, tK>;
    using tileB_full   = CubeTileN8<dtype, tK, tN>;
    using tileC_full   = CubeTileC<float, tM, tN>;

    // K-edge (partial K) variants
    using tileA_kedge  = CubeTileA<dtype, tM, tK, tM, rmd_K>;
    using tileB_kedge  = CubeTileN8<dtype, tK, tN, rmd_K, tN>;
    using tileB_kcorner = CubeTileN8<dtype, tK, tN, rmd_K, rmd_N>;
    // N-edge (partial N) variants
    using tileB_nedge  = CubeTileN8<dtype, tK, tN, tK, rmd_N>;
    using tileC_nedge  = CubeTileC<float, tM, tN, tM, rmd_N>;
    // M-edge (partial M) variants
    using tileA_medge  = CubeTileA<dtype, tM, tK, rmd_M, tK>;
    using tileA_mcorner = CubeTileA<dtype, tM, tK, rmd_M, rmd_K>;
    using tileC_medge  = CubeTileC<float, tM, tN, rmd_M, tN>;
    // M-N corner
    using tileC_mncorner = CubeTileC<float, tM, tN, rmd_M, rmd_N>;

    using itA = global_iterator<gm_shapeInput, tileA_full>;
    using itB = global_iterator<gm_shapeWeight, tileB_full>;
    using itC = global_iterator<gm_shapeOutput, tileC_full>;

    itA gAIter(input_nchw_ptr);
    itB gBIter(weight_ptr);
    itC gCIter(output_ptr);

    // Interior + N edge
    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            conv2d_rm_detail::compute_tile<tileA_full, tileB_full,
                                           tileA_kedge, tileB_kedge,
                                           tileC_full, itA, itB, itC,
                                           Kb, rmd_K != 0>(gAIter, gBIter, gCIter, i, j);
        }
        if constexpr (rmd_N) {
            conv2d_rm_detail::compute_tile<tileA_full, tileB_nedge,
                                           tileA_kedge, tileB_kcorner,
                                           tileC_nedge, itA, itB, itC,
                                           Kb, rmd_K != 0>(gAIter, gBIter, gCIter, i, Nb);
        }
    }

    // M edge + M-N corner
    if constexpr (rmd_M) {
        for (int j = 0; j < Nb; ++j) {
            conv2d_rm_detail::compute_tile<tileA_medge, tileB_full,
                                           tileA_mcorner, tileB_kedge,
                                           tileC_medge, itA, itB, itC,
                                           Kb, rmd_K != 0>(gAIter, gBIter, gCIter, Mb, j);
        }
        if constexpr (rmd_N) {
            conv2d_rm_detail::compute_tile<tileA_medge, tileB_nedge,
                                           tileA_mcorner, tileB_kcorner,
                                           tileC_mncorner, itA, itB, itC,
                                           Kb, rmd_K != 0>(gAIter, gBIter, gCIter, Mb, Nb);
        }
    }
}

#endif
