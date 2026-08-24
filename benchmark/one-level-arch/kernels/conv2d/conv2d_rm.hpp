#ifndef CONV2D_RM_TILEOP_KERNEL_HPP
#define CONV2D_RM_TILEOP_KERNEL_HPP

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using TileAcc = Tile<Location::Vec, E_, R_, C_, BLayout::RowMajor, VR_, VC_>;

template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void store_acc_tile_tileop(GmOut &Gout, TileAcc &tAcc){
    TSTORE(Gout, tAcc);
}

// conv2d_1x1 with RowMajor global tensors.
//
// All GM tensors use RowMajor to ensure TLOAD/TSTORE GetStride(3)=RowStride
// is the correct row stride (Cols * sizeof(DType)).
//
// Input  : RowMajor<gM, gK>  (NHWC flattened)
// Weight : RowMajor<gK, gN>  (B = W^T, pre-converted to BFractal ZN layout)
// Output : RowMajor<gM, gN>  (NHWC flattened)
//
// The weight data in src1.bin must be pre-converted: each tK x tN tile block
// of B is transposed, so that NORM TLOAD (no B.DATR) + ColMajor ZN physical
// layout produce the correct tile register contents via double-transpose.
//
// For FP16 (inner 16x16, single block): pre-conversion = transpose each 16x16 block.
// For FP32 (inner 8x16, two blocks): pre-conversion = BFractal ZN fractal permutation.
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_rm_tileop(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {

    static_assert(in_c == 1 * 1 * in_c, "conv2d_1x1 expects kh=kw=1");

    constexpr int gM = in_h * in_w;
    constexpr int gN = out_c;
    constexpr int gK = in_c;

    using gm_shapeInput  = global_tensor<dtype,   RowMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype,   RowMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float,   RowMajor<gM, gN>>;

    using tile_shapeA   = TileLeft<dtype, tM, tK>;
    using tile_shapeB   = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;

    using itA = global_iterator<gm_shapeInput, tile_shapeA>;
    using itB = global_iterator<gm_shapeWeight, tile_shapeB>;
    using itC = global_iterator<gm_shapeOutput, tile_shapeACC>;

    itA gAIter(input_nchw_ptr);
    itB gBIter(weight_ptr);
    itC gCIter(output_ptr);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            auto gC = gCIter(i, j);
            tile_shapeACC tACC;

            auto gA = gAIter(i, 0);
            auto gB = gBIter(0, j);
            tile_shapeA tA;
            tile_shapeB tB;
            TLOAD(tA, gA);
            TLOAD(tB, gB);
            TMATMUL(tACC, tA, tB);

            #pragma clang loop unroll(full)
            for (int k = 1; k < Kb; ++k) {
                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                tile_shapeA tA;
                tile_shapeB tB;
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                TMATMUL_ACC(tACC, tACC, tA, tB);
            }

            store_acc_tile_tileop(gC, tACC);
        }
    }
}

#endif
