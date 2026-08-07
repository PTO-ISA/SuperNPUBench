#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, cuBLAS scale (scaleAlg=1), FP8 output (E4M3 default, E5M2).
// cuBLAS consumes the bf16 VALUE view (abs -> TCOLMAX -> fp32 amax, guarded
// exponent extract). Two-pass structure keeps peak live tiles low.
template <int Axis, int Post, int BlockSize = 32, int TileN = 32, typename OutT = __fp8_e4m3,
          uint32_t MaxLowBoundBits = 0x2b8cbcccu>
void dynamic_mx_quant_nontail_cublas_fp8(__bf16 *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    // Axis must be whole blocks (a block is exactly BlockSize along the quant
    // axis); Post need NOT be a multiple of TileN: full column tiles + N_tail.
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");

    constexpr int numKb  = Axis / BlockSize;
    constexpr int numN   = Post / TileN;   // full column tiles
    constexpr int N_tail = Post % TileN;   // trailing partial column tile
    // AscendC scale even-pads the quant-axis block count (交织/interleaving):
    // scaleRows = ceil_even(numKb). The trailing padding block-row is left zero.
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   BlockSize, TileN, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor>;
    // Compact scale store (transposed): the cuBLAS core emits scale_byte/recip
    // boxed valid row=1 (one per-column scalar per block-row), so we narrow to
    // uint8 and store one byte per block — no intermediate TCOLMAX.
    using tile_sred   = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    // Per-column-scalar (valid row=1) reciprocal, reinterpreted + cast to fp32 and
    // fused into the data pass via TCOLEXPANDMUL (col broadcast mul).
    using tile_recip_bf1 = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x = global_tensor<__bf16,   RowMajor<Axis, Post>>;
    using gm_y = global_tensor<uint8_t,  RowMajor<Axis, Post>>;
    // AscendC scale layout: uint8 E8M0, one byte per block. Transposed (quant axis
    // is rows): compact [scaleRows, Post] with scaleRows = evenAlign(numKb); the
    // trailing padding block-row is left zero.
    using gm_s = global_tensor<uint8_t,  RowMajor<scaleRows, Post>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx = x_iter(kb, n);
            auto gy = y_iter(kb, n);
            // Compact scale: base pointer folds the block-row index (kb) since the
            // iterator's i-stride is the PHYSICAL tile height, not 1. Iterate the
            // Post columns via s_iter(0, n); each block-row writes TileN bytes.
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb * Post);
            auto gs = s_iter(0, n);

            tile_sred scale_byte;
            tile_sred recip;
            tile_x xq_s;
            TLOAD(xq_s, gx);
            compute_cublas_scale_not_tail<OutT, BlockSize, TileN, MaxLowBoundBits>(xq_s, scale_byte, recip);
            // scale_byte already boxed valid row=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            tile_recip_bf1 inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, BlockSize, TileN, 1, TileN>(recip, inv_bf16);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            tile_o oq;
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }

    // Tail column tile: N_tail (< TileN) leftover Post columns. Keep the PHYSICAL
    // tile shape at BlockSize x TileN (so logicalTileBytes stays >= 512B; a
    // TileN'=N_tail recursion would create sub-512B tiles and fail
    // IsValidActiveSize) but box every tile to ValidCol = N_tail so only the live
    // columns are touched. The column tile is addressed at index numN (iterator
    // j-stride uses PHYSICAL TileN).
    if constexpr (N_tail > 0) {
        using tile_x_r      = Tile<Location::Vec, __bf16,   BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_f_r      = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_o_r      = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_sred_r   = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_sstore_r = Tile<Location::Vec, uint8_t,  BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_recip_bf1_r = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_recip_f1_r  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;

        global_iterator<gm_x, tile_x_r> x_iter_r(x);
        global_iterator<gm_y, tile_o_r> y_iter_r(reinterpret_cast<uint8_t *>(y));

        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter_r(kb, numN);
            auto gy = y_iter_r(kb, numN);
            global_iterator<gm_s, tile_sstore_r> s_iter_r(scale + kb * Post);
            auto gs = s_iter_r(0, numN);

            tile_sred_r scale_byte;
            tile_sred_r recip;
            tile_x_r xq_s;
            TLOAD(xq_s, gx);
            compute_cublas_scale_not_tail<OutT, BlockSize, TileN, MaxLowBoundBits, N_tail>(xq_s, scale_byte, recip);
            tile_sstore_r scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8);

            tile_recip_bf1_r inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, BlockSize, TileN, 1, N_tail>(recip, inv_bf16);
            tile_recip_f1_r inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x_r xq;
            TLOAD(xq, gx);
            tile_f_r xf;
            TCVT(xf, xq);
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            tile_o_r oq;
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
