#ifndef CONV2D_CM_TILEOP_KERNEL_HPP
#define CONV2D_CM_TILEOP_KERNEL_HPP

#include <common/pto_tileop.hpp>
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

// conv2d_1x1 with ColMajor global tensors and CUBE cell local tiles.
//
// GM data contract (natural NCHW layouts, no pre-conversion):
//   Input  : ColMajor<gM, gK>  = NCHW flattened  (input[ch*gM + pos])
//   Weight : ColMajor<gK, gN>  = W (OUT_C, IN_C) row-major flattened
//            (B(k,n) = W(n,k); offset k + n*gK == n*gK + k, identical memory)
//   Output : ColMajor<gM, gN>  = NCHW flattened  (output[n*gM + pos])
//
// NOTE: whether the gfrun CUBE TLOAD/TSTORE transport supports ColMajor GM
// blocks is under verification (the model computes GM addresses as
// base + row*stride + col*eleSize, i.e. RowMajor block semantics).
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_cm_tileop(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {

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

    using gm_shapeInput  = global_tensor<dtype,   ColMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype,   ColMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float,   ColMajor<gM, gN>>;

    using tileA_full   = CubeTileA<dtype, tM, tK>;
    using tileB_full   = CubeTileN8<dtype, tK, tN>;
    using tileC_full   = CubeTileC<float, tM, tN>;

    using tileA_kedge  = CubeTileA<dtype, tM, tK, tM, rmd_K>;
    using tileB_kedge  = CubeTileN8<dtype, tK, tN, rmd_K, tN>;
    using tileB_kcorner = CubeTileN8<dtype, tK, tN, rmd_K, rmd_N>;
    using tileB_nedge  = CubeTileN8<dtype, tK, tN, tK, rmd_N>;
    using tileC_nedge  = CubeTileC<float, tM, tN, tM, rmd_N>;
    using tileA_medge  = CubeTileA<dtype, tM, tK, rmd_M, tK>;
    using tileA_mcorner = CubeTileA<dtype, tM, tK, rmd_M, rmd_K>;
    using tileC_medge  = CubeTileC<float, tM, tN, rmd_M, tN>;
    using tileC_mncorner = CubeTileC<float, tM, tN, rmd_M, rmd_N>;

    using itA = global_iterator<gm_shapeInput, tileA_full>;
    using itB = global_iterator<gm_shapeWeight, tileB_full>;
    using itC = global_iterator<gm_shapeOutput, tileC_full>;

    itA gAIter(input_nchw_ptr);
    itB gBIter(weight_ptr);
    itC gCIter(output_ptr);

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            auto gC = gCIter(i, j);
            tileC_full tACC;

            auto gA = gAIter(i, 0);
            auto gB = gBIter(0, j);
            tileA_full tA;
            tileB_full tB;
            TLOAD_CUBE(tA, gA);
            TLOAD_CUBE(tB, gB);
            TMATMUL(tACC, tA, tB);

            #pragma clang loop unroll(full)
            for (int k = 1; k < Kb; ++k) {
                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                tileA_full tA;
                tileB_full tB;
                TLOAD_CUBE(tA, gA);
                TLOAD_CUBE(tB, gB);
                TMATMUL_ACC(tACC, tACC, tA, tB);
            }

            TSTORE_CUBE(gC, tACC);
        }
    }
}

#endif
