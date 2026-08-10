#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Tail-axis, cuBLAS scale (scaleAlg=1), FP8 output (E4M3 default, E5M2 valid).
// cuBLAS consumes the fp32 VALUE view (amax via TABS+TROWMAX, guarded exponent
// extract). Two-pass structure keeps peak live tiles low.
//
// BlockSize range: UNBOUNDED. Unlike the non-tail kernels, the tail contiguous
// axis is BlockSize (the quant axis, always a multiple of 32 -> naturally 32B
// aligned), and the free axis is M rows tiled by the tunable TileM. There is no
// alignment-vs-TileSize conflict on a single axis: the binding tile budget
// (TileM*BlockSize) is met by shrinking TileM, so ANY BlockSize is legal. The
// budget itself has two values from the fp32 reinterpret roundtrip (问题4):
// CURRENT (fp32 32b roundtrip) TileM*BlockSize <= 2048; FORMAL (post-bitcast,
// 16b) TileM*BlockSize <= 4096. Either way BlockSize is free — only TileM shrinks
// — so no static_assert on BlockSize is needed here.
template <int M, int K, int TileM = 8, int BlockSize = 32, typename OutT = __fp8_e4m3,
          uint32_t MaxLowBoundBits = 0x2b8cbcccu>
void dynamic_mx_quant_tail_cublas_fp8(__bf16 *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");

    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = K / BlockSize;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned (scaleColNum_ = CeilDiv(numKb,2)*2). The
    // trailing padding column is left zero. Mirrors dynamic_mx_quant_tail_axis_fp8.h:168.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    // Compact scale store: the cuBLAS core now emits scale_byte/recip already
    // boxed valid col=1 (one per-row scalar per block), so we narrow straight to
    // uint8 and store one byte per block — no intermediate TROWMAX.
    using tile_sred   = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    // Per-row-scalar (valid col=1) reciprocal, reinterpreted + cast to fp32 and
    // fused into the data pass via TROWEXPANDMUL (row broadcast mul).
    using tile_recip_bf1 = Tile<Location::Vec, __bf16, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_recip_f1  = Tile<Location::Vec, float,  TileM, BlockSize, BLayout::RowMajor, TileM, 1>;

    using gm_x = global_tensor<__bf16,   RowMajor<M, K>>;
    using gm_y = global_tensor<uint8_t,  RowMajor<M, K>>;
    using gm_s = global_tensor<uint8_t,  RowMajor<M, scaleCols>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));

    for (int m = 0; m < full_m; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter(m, kb);
            auto gy = y_iter(m, kb);
            // Compact scale: base pointer folds the block index (kb) since the
            // iterator's j-stride is the PHYSICAL tile width, not 1. Iterate rows
            // only via s_iter(m, 0); each row writes one byte at column kb.
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb);
            auto gs = s_iter(m, 0);

            // ComputeScale pass: reduce the amax in bf16, cast only the reduced
            // per-row scalar to fp32 (mirrors AscendC; no whole-tile CVT).
            tile_sred scale_byte;
            tile_sred recip;
            tile_x xq_s;
            TLOAD(xq_s, gx);
            compute_cublas_scale_tail<OutT, TileM, BlockSize, MaxLowBoundBits>(xq_s, scale_byte, recip);
            // scale_byte already boxed valid col=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            tile_recip_bf1 inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, TileM, BlockSize, TileM, 1>(recip, inv_bf16);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            // ComputeData pass: reload the bf16 value view now.
            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
            tile_o oq;
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }

    // Tail block: M_tail (< TileM) leftover rows. Keep the PHYSICAL tile shape at
    // TileM x BlockSize (so logicalTileBytes stays >= 512B; a TileM'=M_tail
    // recursion would create sub-512B tiles and fail IsValidActiveSize) but box
    // every tile to ValidRow = M_tail so only the live rows are touched. The row
    // block is addressed at index full_m (iterator i-stride uses PHYSICAL Rows).
    if constexpr (M_tail > 0) {
        using tile_x_r      = Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_f_r      = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_o_r      = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_scale_r  = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_sred_r   = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_sstore_r = Tile<Location::Vec, uint8_t,  TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_recip_bf1_r = Tile<Location::Vec, __bf16, TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_recip_f1_r  = Tile<Location::Vec, float,  TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;

        global_iterator<gm_x, tile_x_r> x_iter_r(x);
        global_iterator<gm_y, tile_o_r> y_iter_r(reinterpret_cast<uint8_t *>(y));

        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter_r(full_m, kb);
            auto gy = y_iter_r(full_m, kb);
            global_iterator<gm_s, tile_sstore_r> s_iter_r(scale + kb);
            auto gs = s_iter_r(full_m, 0);

            tile_sred_r scale_byte;
            tile_sred_r recip;
            tile_x_r xq_s;
            TLOAD(xq_s, gx);
            compute_cublas_scale_tail<OutT, TileM, BlockSize, MaxLowBoundBits, M_tail>(xq_s, scale_byte, recip);
            tile_sstore_r scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8);

            tile_recip_bf1_r inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, TileM, BlockSize, M_tail, 1>(recip, inv_bf16);
            tile_recip_f1_r inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x_r xq;
            TLOAD(xq, gx);
            tile_f_r xf;
            TCVT(xf, xq);
            TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
            tile_o_r oq;
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
