#ifndef MATMUL_TEST_HPP
#define MATMUL_TEST_HPP

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename E_, int R_, int C_, int VR_ = R_, int VC_ = C_>
using TileAcc = Tile<Location::Vec, E_, R_, C_, BLayout::RowMajor, VR_, VC_>;

template <typename dtype, int tM, int tN, int tK>
__attribute__((noinline)) void matmul_test(__half *dst, dtype *src0, dtype *src1, int gM, int gN, int gK) {
    using gm_shapeA = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeC = global_tensor<__half, RowMajor<-1, -1>>;
    using tile_shapeA = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;
    using tile_shapeC = Tile<Location::Vec, __half, tM, tN, BLayout::RowMajor, -1, -1>;

    if (gM <= 0 || gN <= 0 || gK <= 0)
        return;

    for (int i = 0; i < gM; i += tM) {
        for (int j = 0; j < gN; j += tN) {
            int dyn_m = std::min(tM, gM - i);
            int dyn_n = std::min(tN, gN - j);

            tile_shapeACC tACC(dyn_m, dyn_n);
            for (int k = 0; k < gK; k += tK) {
                int dyn_k = gK - k > tK ? tK : gK - k;

                size_t offset_A = i * gK + k;
                size_t offset_B = k * gN + j;
                gm_shapeA gA(src0 + offset_A, gM, gK);
                gm_shapeB gB(src1 + offset_B, gK, gN);

                tile_shapeA tA(dyn_m, dyn_k);
                tile_shapeB tB(dyn_k, dyn_n);
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                if (k == 0) {
                    TMATMUL(tACC, tA, tB);
                } else {
                    TMATMUL_ACC(tACC, tACC, tA, tB);
                }
            }

            tile_shapeC tC(dyn_m, dyn_n);
            TCVT(tC, tACC);

            size_t offset_C = i * gN + j;
            gm_shapeC gC(dst + offset_C, gM, gN);
            TSTORE(gC, tC);
        }
    }
}

#endif