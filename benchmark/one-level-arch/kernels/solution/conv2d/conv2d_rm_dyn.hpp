#ifndef CONV2D_RM_DYN_TILEOP_KERNEL_HPP
#define CONV2D_RM_DYN_TILEOP_KERNEL_HPP

#include <common/pto_tileop.hpp>

using namespace pto;

template <typename E_, int R_, int C_, int VR_=R_, int VC_=C_>
using TileAcc = Tile<Location::Vec, E_, R_, C_, BLayout::RowMajor, VR_, VC_>;

template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void store_acc_tile_tileop_dyn(GmOut &Gout, TileAcc &tAcc){
    TSTORE(Gout, tAcc);
}

struct Conv2dParam {
    int in_c;
    int in_h;
    int in_w;
    int out_c;
    int tM;
    int tN;
    int tK;
};

template <typename dtype, const int tM, const int tN, const int tK>
void conv2d_1x1_rm_dyn_tileop(float *output_ptr, dtype *input_nchw_ptr,
                              dtype *weight_ptr, const Conv2dParam &param) {

    const int gM = param.in_h * param.in_w;
    const int gN = param.out_c;
    const int gK = param.in_c;

    using gm_shapeInput  = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeWeight = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_shapeOutput = global_tensor<float, RowMajor<-1, -1>>;

    using tile_shapeA   = TileLeft<dtype, tM, tK, -1, -1>;
    using tile_shapeB   = TileRight<dtype, tK, tN, -1, -1>;
    using tile_shapeACC = TileAcc<float, tM, tN, -1, -1>;

    const int Mb = (gM + tM - 1) / tM;
    const int Nb = (gN + tN - 1) / tN;
    const int Kb = (gK + tK - 1) / tK;

    const int rem_m = gM % tM;
    const int rem_n = gN % tN;
    const int rem_k = gK % tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            size_t offset_C = (size_t)i * gN * tM + (size_t)j * tN;
            gm_shapeOutput gC(output_ptr + offset_C, gM, gN);

            int dyn_m = (i + 1) * tM > gM ? rem_m : tM;
            int dyn_n = (j + 1) * tN > gN ? rem_n : tN;

            tile_shapeACC tACC(dyn_m, dyn_n);

            for (int k = 0; k < Kb; ++k) {
                size_t offset_A = (size_t)i * gK * tM + (size_t)k * tK;
                size_t offset_B = (size_t)k * gN * tK + (size_t)j * tN;

                gm_shapeInput gA(input_nchw_ptr + offset_A, gM, gK);
                gm_shapeWeight gB(weight_ptr + offset_B, gK, gN);

                int dyn_k = (k + 1) * tK > gK ? rem_k : tK;

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

            store_acc_tile_tileop_dyn(gC, tACC);
        }
    }
}

#endif
