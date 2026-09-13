#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// Four-PE cooperative conv2d 1x1 with Shared TLOAD (RowMajor global tensors).
//
// Mathematical semantics:
//   C = A * B  (conv2d 1x1 = matrix multiply)
//   A (input):  RowMajor<gM, gK>  (NHWC flattened, gM = in_h*in_w, gK = in_c)
//   B (weight): RowMajor<gK, gN>  (W^T, gK = in_c, gN = out_c)
//   C (output): RowMajor<gM, gN>  (NHWC flattened)
//
// All GM tensors use RowMajor to ensure TLOAD/TSTORE GetStride(3)=RowStride
// is the correct row stride (Cols * sizeof(DType)).  This matches the
// verified conv2d_rm kernel (see docs/conv2d_accuracy_verification.md §11).
//
// Difference from conv2d_rm:
//   - Both A and B are loaded into SharedTile via the GM->Shared TLOAD
//     (PTO v0.58 reissue), as in matmul_shared.hpp.
//   - A is a Shared Left tile, B is a Shared Right tile.
//   - TMATMUL is issued cooperatively by four PEs.  The shared A tile
//     covers the complete [tM, tK] block, while each PE keeps only its
//     contiguous [tM / 4, tN] row slice of C in a private Vec tile.
//   - Output is mapped via gIterC(i * kPeNum + tid, j) so each PE writes
//     its own row slice within the logical [tM, tN] output block.
template <typename dtype,
          const int in_c, const int in_h, const int in_w,
          const int out_c,
          const int tM, const int tN, const int tK>
void conv2d_1x1_rm_shared(float *output_ptr, dtype *input_nchw_ptr, dtype *weight_ptr) {
    constexpr int kPeNum = 4;

    static_assert(in_c == 1 * 1 * in_c, "conv2d_1x1 expects kh=kw=1");

    constexpr int gM = in_h * in_w;
    constexpr int gN = out_c;
    constexpr int gK = in_c;

    static_assert(gM % tM == 0, "gM (in_h*in_w) must be divisible by tM");
    static_assert(gN % tN == 0, "gN (out_c) must be divisible by tN");
    static_assert(gK % tK == 0, "gK (in_c) must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");

    const uint32_t tid = get_thread_idx();

    using gm_shapeInput  = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeWeight = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOutput = global_tensor<float, RowMajor<gM, gN>>;

    using tileALocal  = TileLeft<dtype, tM, tK>;
    using tileBLocal  = TileRight<dtype, tK, tN>;
    using tileAShared = SharedTile<tileALocal>;
    using tileBShared = SharedTile<tileBLocal>;
    using tileC =
        Tile<Location::Vec, float, tM / kPeNum, tN, BLayout::RowMajor>;

    using itA = global_iterator<gm_shapeInput, tileALocal>;
    using itB = global_iterator<gm_shapeWeight, tileBLocal>;
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
                auto gA = gAIter(i, 0);
                auto gB = gBIter(0, j);
                tileAShared tAShared;
                tileBShared tBShared;
                TLOAD<tileALocal, 1>(tAShared, gA);
                TLOAD<tileBLocal, 1>(tBShared, gB);
                TMATMUL(tC, tAShared, tBShared);
            } else {
                {
                    auto gA = gAIter(i, 0);
                    auto gB = gBIter(0, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileALocal, 1>(tAShared, gA);
                    TLOAD<tileBLocal, 1>(tBShared, gB);
                    TMATMUL(tC, tAShared, tBShared);
                }

                #pragma clang loop unroll(full)
                for (int k = 1; k < Kb; ++k) {
                    auto gA = gAIter(i, k);
                    auto gB = gBIter(k, j);
                    tileAShared tAShared;
                    tileBShared tBShared;
                    TLOAD<tileALocal, 1>(tAShared, gA);
                    TLOAD<tileBLocal, 1>(tBShared, gB);
                    TMATMUL_ACC(tC, tC, tAShared, tBShared);
                }
            }

            auto gC = gCIter(i * kPeNum + tid, j);
            TSTORE(gC, tC);
        }
    }
}
