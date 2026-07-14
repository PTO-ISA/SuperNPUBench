#ifndef FA_HIF4_HPP
#define FA_HIF4_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

template <typename SrcTile, typename CastTile, typename MaxTile, typename SumTile,
          typename ScaleTile, int scaleD>
void pto_flash_softmax_block(CastTile &src_exp, MaxTile &new_max, SumTile &new_sum,
                             ScaleTile &rescale, SrcTile &src, MaxTile &old_max,
                             SumTile &old_sum) {
    SrcTile scaled_src;
    TMULS(scaled_src, src, 1.0f / sqrt((float)scaleD));

    MaxTile local_max;
    TCOLMAX(local_max, scaled_src);
    TMAX(new_max, old_max, local_max);

    TSUB(rescale, old_max, new_max);
    TEXP(rescale, rescale);

    SumTile scaled_old_sum;
    TMUL(scaled_old_sum, old_sum, rescale);

    TCOLEXPANDSUB(scaled_src, scaled_src, new_max);
    TEXP(scaled_src, scaled_src);

    SumTile local_sum;
    TCOLSUM(local_sum, scaled_src);
    TADD(new_sum, scaled_old_sum, local_sum);

    TCVT(src_exp, scaled_src);
}

template <typename OutTile, typename OldTile, typename PvTile, typename ScaleTile>
void pto_online_update(OutTile &out, OldTile &old_out, PvTile &pv, ScaleTile &scale) {
    TCOLEXPANDMUL(out, old_out, scale);
    TADD(out, out, pv);
}

template <typename OutTile, typename SumTile>
void pto_normalize_by_sum(OutTile &out, SumTile &sum) {
    SumTile inv_sum;
    TRECIP(inv_sum, sum);
    TCOLEXPANDMUL(out, out, inv_sum);
}

template <typename QuantTile, typename SrcTile>
void pto_quantize_softmax_to_hif4(QuantTile &dst, SrcTile &src) {
    TQUANT(dst, src);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr,
                                    dtype* v_ptr, uint8_t* scale_q,
                                    uint8_t* scale_k, uint8_t* scale_v) {
    static_assert(qD == vD);

    using gmQ = global_tensor<dtype, RowMajor<Sq, qD / 2>>;
    using gmK = global_tensor<dtype, ColMajor<qD / 2, Skv>>;
    using gmV = global_tensor<dtype, RowMajor<Skv, vD / 2>>;
    using gmO = global_tensor<dtype, ColMajor<Sq, vD / 2>>;

    using gmQScale = global_tensor<uint8_t, RowMajor<Sq, qD / w_factor>>;
    using gmKScale = global_tensor<uint8_t, ColMajor<qD / w_factor, Skv>>;
    using gmVScale = global_tensor<uint8_t, RowMajor<Skv, vD / w_factor>>;

    using tileQ = TileLeft<dtype, kTm, (qD == 192 ? 256 : qD) / 2, kTm, qD / 2>;
    using tileK = TileRight<dtype, (qD == 192 ? 256 : qD) / 2, kTk, qD / 2, kTk>;
    using tileV = TileRight<dtype, kTk, vD>;

    using tileQScale = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor,
                            kTm, qD / w_factor, SLayout::NoneBox>;
    using tileKScale = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor,
                            qD / w_factor, kTk, SLayout::NoneBox>;
    using tileVScale = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor,
                            kTk, vD / w_factor, SLayout::NoneBox>;

    using tileWAcc = TileAcc<float, kTm, kTk>;
    using tileW = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileWCast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;
    using tilePHif4 = Tile<Location::Vec, __fp4_e1m2x2, 32, kTk / 2,
                           BLayout::ColMajor, kTm, kTk / 2, SLayout::ColMajor>;
    using tilePLeft = TileLeft<dtype, 32, kTk, kTm, kTk>;

    using tileOAcc = TileAcc<float, kTm, vD>;
    using tileO = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;
    using tileOCast = Tile<Location::Vec, dtype, 32, vD / 2,
                           BLayout::ColMajor, kTm, vD / 2, SLayout::ColMajor>;

    using tileMax = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileSum = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileScale = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileOCast>;
    using itQScale = global_iterator<gmQScale, tileQScale>;
    using itKScale = global_iterator<gmKScale, tileKScale>;
    using itVScale = global_iterator<gmVScale, tileVScale>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);
    itQScale gIterQScale(scale_q);
    itKScale gIterKScale(scale_k);
    itVScale gIterVScale(scale_v);

    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

#ifdef _2D_UNROLL_PTO
    static_assert(Qb % Xdim == 0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb % Ydim == 0, "Kb needs to be a multiple of Ydim");
#endif

    for (int i = 0; i < Qb; i += Xdim) {
        tileQ tQ[Xdim];
        tileQScale tQScale[Xdim];

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            auto gQ = gIterQ(i + x, 0);
            auto gQScale = gIterQScale(i + x, 0);
            TLOAD(tQ[x], gQ);
            TLOAD(tQScale[x], gQScale);
        }

        tileMax tMax[Xdim];
        tileSum tSum[Xdim];
        tileO tO[Xdim];
        tileO tPV[Xdim];
        tileScale tRescale[Xdim];

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            TEXPANDS(tMax[x], -1e30f);
            TEXPANDS(tSum[x], 0.0f);
            TEXPANDS(tO[x], 0.0f);
        }

        for (int j = 0; j < Kb; j += Ydim) {
            tileK tK[Ydim];
            tileKScale tKScale[Ydim];

            #pragma clang loop unroll(full)
            for (int y = 0; y < Ydim; ++y) {
                auto gK = gIterK(0, j + y);
                auto gKScale = gIterKScale(0, j + y);
                TLOAD(tK[y], gK);
                TLOAD(tKScale[y], gKScale);
            }

            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; ++x) {
                #pragma clang loop unroll(full)
                for (int y = 0; y < Ydim; ++y) {
                    tileWAcc tWAcc;
                    TMATMUL_MX(tWAcc, tQ[x], tQScale[x], tK[y], tKScale[y]);
                    ACCCVT(tW[x][y], tWAcc);
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];
            tileWCast tExpW[Xdim][Ydim];
            tilePHif4 tP[Xdim][Ydim];

            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; ++x) {
                tileMax tLocalMax[Ydim];

                #pragma clang loop unroll(full)
                for (int y = 0; y < Ydim; ++y) {
                    TCOLMAX(tLocalMax[y], tW[x][y]);
                }

#if Ydim == 1
                TMAX(tNewMax[x], tMax[x], tLocalMax[0]);
#elif Ydim == 2
                tileMax tMax01;
                TMAX(tMax01, tLocalMax[0], tLocalMax[1]);
                TMAX(tNewMax[x], tMax[x], tMax01);
#elif Ydim == 4
                tileMax tMax01;
                tileMax tMax23;
                tileMax tMax0123;
                TMAX(tMax01, tLocalMax[0], tLocalMax[1]);
                TMAX(tMax23, tLocalMax[2], tLocalMax[3]);
                TMAX(tMax0123, tMax01, tMax23);
                TMAX(tNewMax[x], tMax[x], tMax0123);
#else
                static_assert(Ydim == 1 || Ydim == 2 || Ydim == 4,
                              "PTO HIF4 FA currently supports Ydim 1/2/4");
#endif

                TSUB(tRescale[x], tMax[x], tNewMax[x]);
                TEXP(tRescale[x], tRescale[x]);

                tileSum tScaledOldSum;
                TMUL(tScaledOldSum, tSum[x], tRescale[x]);

                tileSum tLocalSum[Ydim];
                #pragma clang loop unroll(full)
                for (int y = 0; y < Ydim; ++y) {
                    TCOLEXPANDSUB(tW[x][y], tW[x][y], tNewMax[x]);
                    TEXP(tW[x][y], tW[x][y]);
                    TCOLSUM(tLocalSum[y], tW[x][y]);
                    TCVT(tExpW[x][y], tW[x][y]);
                    pto_quantize_softmax_to_hif4(tP[x][y], tExpW[x][y]);
                }

#if Ydim == 1
                TADD(tNewSum[x], tScaledOldSum, tLocalSum[0]);
#elif Ydim == 2
                tileSum tSum01;
                TADD(tSum01, tLocalSum[0], tLocalSum[1]);
                TADD(tNewSum[x], tScaledOldSum, tSum01);
#elif Ydim == 4
                tileSum tSum01;
                tileSum tSum23;
                tileSum tSum0123;
                TADD(tSum01, tLocalSum[0], tLocalSum[1]);
                TADD(tSum23, tLocalSum[2], tLocalSum[3]);
                TADD(tSum0123, tSum01, tSum23);
                TADD(tNewSum[x], tScaledOldSum, tSum0123);
#endif
            }

            tileV tV[Ydim];
            tileVScale tVScale[Ydim];
            #pragma clang loop unroll(full)
            for (int y = 0; y < Ydim; ++y) {
                auto gV = gIterV(j + y, 0);
                auto gVScale = gIterVScale(j + y, 0);
                TLOAD(tV[y], gV);
                TLOAD(tVScale[y], gVScale);
            }

            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; ++x) {
#if Ydim == 1
                tilePLeft tPLeft;
                TCVT(tPLeft, tP[x][0]);
                tileOAcc tPVAcc;
                TMATMUL_MX(tPVAcc, tPLeft, tQScale[x], tV[0], tVScale[0]);
                ACCCVT(tPV[x], tPVAcc);
#else
                tileO tPVSum;
                #pragma clang loop unroll(full)
                for (int y = 0; y < Ydim; ++y) {
                    tilePLeft tPLeft;
                    TCVT(tPLeft, tP[x][y]);
                    tileOAcc tPVAcc;
                    tileO tPVPart;
                    TMATMUL_MX(tPVAcc, tPLeft, tQScale[x], tV[y], tVScale[y]);
                    ACCCVT(tPVPart, tPVAcc);
                    if (y == 0) {
                        tPVSum = tPVPart;
                    } else {
                        TADD(tPVSum, tPVSum, tPVPart);
                    }
                }
                tPV[x] = tPVSum;
#endif

                pto_online_update(tO[x], tO[x], tPV[x], tRescale[x]);
            }

            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; ++x) {
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            tileOCast tOCast;
            pto_normalize_by_sum(tO[x], tSum[x]);
            TCVT(tOCast, tO[x]);
            auto gO = gIterO(i + x, 0);
            TSTORE(gO, tOCast);
        }
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4_nogather(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr,
                                             dtype* v_ptr, uint8_t* scale_q,
                                             uint8_t* scale_k, uint8_t* scale_v) {
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk, w_factor, casttype>(
        out_ptr, q_ptr, k_ptr, v_ptr, scale_q, scale_k, scale_v);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4_optsoftmax(dtype* out_ptr, dtype* q_ptr,
                                               dtype* k_ptr, dtype* v_ptr,
                                               uint8_t* scale_q, uint8_t* scale_k,
                                               uint8_t* scale_v) {
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk, w_factor, casttype>(
        out_ptr, q_ptr, k_ptr, v_ptr, scale_q, scale_k, scale_v);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_loadx2(dtype* out_ptr, dtype* q_ptr,
                                                      dtype* k_ptr, dtype* v_ptr,
                                                      uint8_t* scale_q, uint8_t* scale_k,
                                                      uint8_t* scale_v) {
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk, w_factor, casttype>(
        out_ptr, q_ptr, k_ptr, v_ptr, scale_q, scale_k, scale_v);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload(dtype* out_ptr, dtype* q_ptr,
                                                           dtype* k_ptr, dtype* v_ptr,
                                                           uint8_t* scale_q,
                                                           uint8_t* scale_k,
                                                           uint8_t* scale_v) {
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk, w_factor, casttype>(
        out_ptr, q_ptr, k_ptr, v_ptr, scale_q, scale_k, scale_v);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload2(dtype* out_ptr, dtype* q_ptr,
                                                            dtype* k_ptr, dtype* v_ptr,
                                                            uint8_t* scale_q,
                                                            uint8_t* scale_k,
                                                            uint8_t* scale_v) {
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk, w_factor, casttype>(
        out_ptr, q_ptr, k_ptr, v_ptr, scale_q, scale_k, scale_v);
}

#endif
