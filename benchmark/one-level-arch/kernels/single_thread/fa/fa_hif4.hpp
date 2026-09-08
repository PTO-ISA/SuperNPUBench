#ifndef FA_HIF4_HPP
#define FA_HIF4_HPP

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace pto;

template <typename T, int M, int K, int VM = M, int VK = K>
using FaCubeLeft = std::conditional_t<
    (M <= 16), CubeTileM16<T, M, K, VM, VK>,
    CubeTileM32<T, M, K, VM, VK>>;

template <typename T, int M, int N, int VM = M, int VN = N>
using FaCubeAcc = std::conditional_t<
    (M <= 16), CubeAccumulatorM16<T, M, N, VM, VN>,
    CubeAccumulatorM32<T, M, N, VM, VN>>;

// Convert two logical scalar columns into one packed-x2 CUBE element. This is
// the same active TileOP conversion used by the 4-PE FA kernel; the retired
// TQUANT<MXFP4> interface must not be used for HIF4 probability packing.
template <is_tile_data_v OutTile, is_tile_data_v InTile>
inline void fa_single_tcvt_packed_x2(OutTile &dst, InTile &src) {
    static_assert(OutTile::Rows == InTile::Rows,
                  "packed conversion must preserve rows");
    static_assert(InTile::Cols == OutTile::Cols * 2,
                  "packed-x2 destination must have half as many columns");
    const size_t valid_col = dst.GetValidCol();
    const size_t valid_row = dst.GetValidRow();
    asm volatile(
        "BSTART.TEPL 27, %c1\n"
        "B.DATR %c2, RNone\n"
        "B.IOT %3, mask=15, last, ->%0<%Z4>\n"
        "B.DIM %5, 0, ->lb0\n"
        "B.DIM %6, 0, ->lb1\n"
        "B.DIM zero, %c7, ->lb2\n"
        : "=Tr"(dst.data())
        : "i"(type_traits<typename InTile::DType>::TypeCode),
          "i"(type_traits<typename OutTile::DType>::TypeCode),
          "Tr"(src.data()),
          "i"(tile_type_traits<typename OutTile::TileDType>::TilesizeCode),
          "r"(valid_col), "r"(valid_row), "i"(OutTile::Cols));
}

// Single-PE HIF4 FlashAttention using the current persistent CUBE layouts.
// HIF4 is Matrix-MX-only: Q/K and V carry U32 scales (group size 64). The PV
// probability operand is converted from the FP32 softmax tile to packed HIF4
// and uses a unit U32 scale, avoiding the retired TQUANT interface.
template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_hif4_single_impl(vector_dtype *out_ptr,
                                      matrix_dtype *q_ptr,
                                      matrix_dtype *k_ptr,
                                      matrix_dtype *v_ptr,
                                      uint32_t *q_scale_ptr,
                                      uint32_t *k_scale_ptr,
                                      uint32_t *v_scale_ptr) {
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredSkv = Skv / PackedFactor;
    constexpr int kStoredTk = kTk / PackedFactor;

    static_assert(PackedFactor == 2,
                  "HIF4 single-thread kernel expects packed-x2 storage");
    static_assert(kTm == 16 || kTm == 32,
                  "local CUBE HIF4 FA supports Tm=16 or Tm=32");
    static_assert(Sq % kTm == 0 && Skv % kTk == 0,
                  "single-thread HIF4 FA currently requires full tiles");
    static_assert(qD % PackedFactor == 0 &&
                      Skv % PackedFactor == 0 &&
                      kTk % PackedFactor == 0,
                  "packed dimensions must be divisible by two");

    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    using tileQ = FaCubeLeft<matrix_dtype, kTm, qD>;
    using tileK = CubeTileN8<matrix_dtype, qD, kTk>;
    using tileW = FaCubeLeft<float, kTm, kTk>;
    using tileP = FaCubeLeft<matrix_dtype, kTm, kStoredTk>;
    using tileV = CubeTileN8<matrix_dtype, kStoredTk, vD>;
    using tileO = FaCubeAcc<float, kTm, vD>;
    using tileOCast = FaCubeAcc<vector_dtype, kTm, vD>;

    constexpr int kQKScaleCols = (qD + 63) / 64;
    constexpr int kPVScaleRows = (kStoredTk + 63) / 64;
    using tileQScale =
        Tile<Location::Vec, uint32_t, kTm, kQKScaleCols,
             BLayout::RowMajor>;
    using tileKScale =
        Tile<Location::Vec, uint32_t, kQKScaleCols, kTk,
             BLayout::RowMajor>;
    using tileVScale =
        Tile<Location::Vec, uint32_t, kPVScaleRows, vD,
             BLayout::RowMajor>;
    using tilePScale =
        Tile<Location::Vec, uint32_t, kTm, kPVScaleRows,
             BLayout::RowMajor>;

    // Match the vector-state layout used by the multi-thread FA path. The
    // physical M32 cell is retained while ValidRow selects this PE's rows.
    using tileState = VecTileM32<float, 32, 1, kTm, 1>;

    using itO = global_iterator<gmO, tileOCast>;

    itO gIterO(out_ptr);

    constexpr int Qb = Sq / kTm;
    constexpr int Kb = Skv / kTk;
    const float score_scale = 1.0f / sqrt((float)scaleD);

#pragma clang loop unroll(full)
    for (int i = 0; i < Qb; ++i) {
        tileQ tQ;
        using gmQBlock =
            global_tensor<matrix_dtype, RowMajor<kTm, kStoredQD>>;
        gmQBlock gQ(q_ptr + i * kTm * kStoredQD);
        TLOAD_CUBE(tQ, gQ);
        using gmQScaleBlock =
            global_tensor<uint32_t, RowMajor<kTm, kQKScaleCols>>;
        gmQScaleBlock gQS(q_scale_ptr + i * kTm * kQKScaleCols);
        tileQScale tQScale;
        TLOAD(tQScale, gQS);

        tileState tMax;
        tileState tSum;
        tileO tO;

#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            tileK tK;
            using gmKBlock = global_tensor<
                matrix_dtype,
                MatrixLayout<qD, kTk, 1, kStoredQD>>;
            gmKBlock gK(k_ptr + j * kTk * kStoredQD);
            TLOAD_CUBE(tK, gK);
            using gmKScaleBlock = global_tensor<
                uint32_t,
                MatrixLayout<kQKScaleCols, kTk, 1, kQKScaleCols>>;
            gmKScaleBlock gKS(k_scale_ptr + j * kTk * kQKScaleCols);
            tileKScale tKScale;
            TLOAD(tKScale, gKS);

            tileW tW;
            TMATMUL_MX(tW, tQ, tQScale, tK, tKScale,
                       fixp::keep_acc());
            TMULS(tW, tW, score_scale);

            tileState tLocalMax;
            tileState tNewMax;
            tileState tScale;
            TROWMAX(tLocalMax, tW);
            if (j == 0) {
                tNewMax = tLocalMax;
            } else {
                TMAX(tNewMax, tMax, tLocalMax);
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
                TROWEXPANDMUL(tO, tO, tScale);
            }

            TROWEXPANDEXPDIF(tW, tW, tNewMax);
            tileState tLocalSum;
            tileState tNewSum;
            TROWSUM(tLocalSum, tW);
            if (j == 0) {
                tNewSum = tLocalSum;
            } else {
                TFMA(tNewSum, tSum, tScale, tLocalSum);
            }

            tileP tP;
            fa_single_tcvt_packed_x2(tP, tW);
            tilePScale tPScale;
            // Unit HiF4 scale carrier (two BF16 1.0 values). The probability
            // conversion is a direct packed conversion, so no dynamic scale
            // tensor is produced by the retired quantization operation.
            TEXPANDS(tPScale, 0x3f803f80u);

            tileV tV;
            using gmVBlock =
                global_tensor<matrix_dtype, RowMajor<kStoredTk, vD>>;
            gmVBlock gV(v_ptr + j * kStoredTk * vD);
            TLOAD_CUBE(tV, gV);
            using gmVScaleBlock =
                global_tensor<uint32_t, RowMajor<kPVScaleRows, vD>>;
            gmVScaleBlock gVS(v_scale_ptr + (j * kTk / 64) * vD);
            tileVScale tVScale;
            TLOAD(tVScale, gVS);

            if (j == 0) {
                TMATMUL_MX(tO, tP, tPScale, tV, tVScale,
                           fixp::keep_acc());
            } else {
                TMATMUL_MX_ACC(tO, tO, tP, tPScale, tV, tVScale,
                               fixp::keep_acc());
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        TROWEXPANDDIV(tO, tO, tSum);
        tileOCast tOCast;
        TCVT(tOCast, tO);
        auto gO = gIterO(i, 0);
        TSTORE_CUBE(gO, tOCast);
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>
void flash_attention_2d_unroll_hif4(casttype *out_ptr, dtype *q_ptr,
                                    dtype *k_ptr, dtype *v_ptr,
                                    uint32_t *sq, uint32_t *sk, uint32_t *sv) {
    (void)w_factor;
    flash_attention_hif4_single_impl<dtype, casttype, 2, Sq, Skv, qD, vD,
                                     kTm, kTk>(out_ptr, q_ptr, k_ptr, v_ptr,
                                               sq, sk, sv);
}

#define FA_HIF4_FORWARD(Name)                                                 \
template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, \
          uint32_t w_factor = 64 / 4, typename casttype = __bf16>             \
void Name(casttype *out_ptr, dtype *q_ptr, dtype *k_ptr, dtype *v_ptr,        \
          uint32_t *sq, uint32_t *sk, uint32_t *sv) {                         \
    flash_attention_2d_unroll_hif4<dtype, Sq, Skv, qD, vD, kTm, kTk,         \
                                    w_factor, casttype>(                       \
        out_ptr, q_ptr, k_ptr, v_ptr, sq, sk, sv);                            \
}

FA_HIF4_FORWARD(flash_attention_2d_unroll_hif4_nogather)
FA_HIF4_FORWARD(flash_attention_2d_unroll_hif4_optsoftmax)
FA_HIF4_FORWARD(flash_attention_2d_unroll_hif4_optsoftmax_loadx2)
FA_HIF4_FORWARD(flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload)
FA_HIF4_FORWARD(flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload2)

#undef FA_HIF4_FORWARD

#endif
