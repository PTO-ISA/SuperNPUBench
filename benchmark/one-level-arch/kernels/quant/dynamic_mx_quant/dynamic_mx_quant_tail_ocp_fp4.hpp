#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// One MX block of the tail-axis OCP-fp4 pipeline: boxed OCP scale (valid col=1),
// compact 1-byte-per-block scale store, then the reciprocal fused into the data
// via TROWEXPANDMUL, leaving the per-block result as a full [TileM,BlockSize]
// fp32 tile in `xf_out` (ready to TCONCAT with its pair partner). ValidM carries
// the live-row count so the M_tail path reuses this body with M_tail rows.
template <typename OutT, int M, int K, int TileM, int BlockSize, int scaleCols, int ValidM>
inline void ocp_fp4_block(
    __bf16 *x, uint8_t *scale, int m, int kb,
    pto::Tile<pto::Location::Vec, float, TileM, BlockSize, pto::BLayout::RowMajor, ValidM, BlockSize> &xf_out) {
    using namespace pto;

    using tile_x   = Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize>;
    using tile_xu  = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, BlockSize>;
    using tile_sred      = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    using tile_sstore    = Tile<Location::Vec, uint8_t,  TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;
    using tile_recip_f1  = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor, ValidM, 1>;

    using gm_x  = global_tensor<__bf16,   RowMajor<M, K>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<M, K>>;
    using gm_s  = global_tensor<uint8_t,  RowMajor<M, scaleCols>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);
    global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x));
    auto gx  = x_iter(m, kb);
    auto gxu = xu_iter(m, kb);
    // Compact scale: base pointer folds the block index (kb) since the iterator's
    // j-stride is the PHYSICAL tile width, not 1. Each row writes one byte at
    // column kb (see dynamic_mx_quant_tail_cublas_fp8).
    global_iterator<gm_s, tile_sstore> s_iter(scale + kb);
    auto gs = s_iter(m, 0);

    tile_sred scale_byte;
    tile_sred recip;
    tile_xu x_u16;
    TLOAD(x_u16, gxu);
    compute_ocp_scale_tail_boxed<OutT, TileM, BlockSize, ValidM>(x_u16, scale_byte, recip);
    tile_sstore scale_u8;
    TCVT(scale_u8, scale_byte);
    TSTORE(gs, scale_u8);

    tile_recip_bf1 inv_bf16;
    // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
    reinterpret_u16_to_bf16<2, TileM, BlockSize, ValidM, 1>(recip, inv_bf16);
    tile_recip_f1 inv_scale_f;
    TCVT(inv_scale_f, inv_bf16);

    tile_x xq;
    TLOAD(xq, gx);
    TCVT(xf_out, xq);
    TROWEXPANDMUL(xf_out, xf_out, inv_scale_f); // per-row scalar broadcast-mul
}

// One pair of adjacent MX blocks -> one legal fp4 output tile. Each block reduces
// independently (ocp_fp4_block), producing a [TileM,BlockSize] fp32 result. The
// two results are widened to a [TileM,2*BlockSize] fp32 tile via a scratch-HBM
// concat (NOT registers): holding both fp32 halves live at once raises register
// pressure until a small boxed tile spills, tripping the LinxV5 RegSize>=512
// assert. Round-tripping each half through HBM keeps live pressure at the
// single-block level (same as the fp8 kernels). One TCVT then packs the 64-value
// (2-block) fp32 tile to a BlockSize-byte fp4 tile (32B, alignment OK).
template <typename OutT, int M, int K, int TileM, int BlockSize, int scaleCols, int ValidM>
inline void ocp_fp4_pair(__bf16 *x, uint8_t *y, uint8_t *scale, int m, int p) {
    using namespace pto;
    static float cat_buf[TileM * 2 * BlockSize] __attribute__((aligned(4096)));
    using gm_cat = global_tensor<float, RowMajor<TileM, 2 * BlockSize>>;
    using tile_half = Tile<Location::Vec, float, TileM, BlockSize,     BLayout::RowMajor, ValidM, BlockSize>;
    using tile_cat  = Tile<Location::Vec, float, TileM, 2 * BlockSize, BLayout::RowMajor, ValidM, 2 * BlockSize>;
    using tile_o    = Tile<Location::Vec, OutT,  TileM, BlockSize,     BLayout::RowMajor, ValidM, BlockSize>;
    using gm_y = global_tensor<uint8_t, RowMajor<M, K / 2>>;

    global_iterator<gm_cat, tile_half> c_iter(cat_buf);
    for (int b = 0; b < 2; ++b) {
        tile_half xf;
        ocp_fp4_block<OutT, M, K, TileM, BlockSize, scaleCols, ValidM>(x, scale, m, 2 * p + b, xf);
        auto gc = c_iter(0, b); // column-half b of cat_buf
        TSTORE(gc, xf);
    }

    tile_cat xcat;
    global_iterator<gm_cat, tile_cat> cr_iter(cat_buf);
    auto gcr = cr_iter(0, 0);
    TLOAD(xcat, gcr);
    tile_o oq;
    TCVT(oq, xcat); // fp32 -> packed fp4_e2m1x2 (2 blocks -> BlockSize bytes)
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));
    auto gy = y_iter(m, p);
    TSTORE(gy, oq);
}

// Tail-axis, OCP scale (scaleAlg=0), FP4 output (E2M1 default, E1M2 valid).
// emax derived from OutT (fp4_e2m1 -> 0x0100).
//
// ALIGNMENT/PACKING (RECORD.md 问题2): a single MX block of fp4 output is
// BlockSize/2 packed bytes = 16B at BlockSize=32, which fails the 32B column
// alignment (pto_tile.hpp:649). The pack axis (halved cols) COINCIDES with the
// reduce axis, so we cannot simply widen the physical tile to 64 — that would
// merge two blocks into one TROWMAX. Instead we DECOUPLE reduce from pack:
// reduce each 32-col block independently (own scale/recip, TROWEXPANDMUL its own
// data to fp32), then TCONCAT the two [TileM,BlockSize] fp32 results into a
// [TileM,2*BlockSize] tile and emit ONE fp32->packed-fp4 TCVT of width
// BlockSize (= 2*BlockSize fp4 values = 32 packed bytes, alignment OK). The
// intermediate [TileM,BlockSize/2] single-block fp4 tile is never constructed.
// Hence K must be a multiple of 2*BlockSize (>= 2 MX blocks per output tile).
template <int M, int K, int TileM = 8, int BlockSize = 32, typename OutT = __fp4_e2m1x2>
void dynamic_mx_quant_tail_ocp_fp4(__bf16 *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");
    static_assert(BlockSize % 2 == 0, "fp4 packs 2/byte; BlockSize must be even");
    static_assert((K % (2 * BlockSize)) == 0,
                  "tail fp4 packs 2 MX blocks per output tile (a single block is "
                  "BlockSize/2 bytes < 32B and fails pto_tile.hpp:649); K must be "
                  "a multiple of 2*BlockSize");

    constexpr int full_m  = M / TileM;
    constexpr int M_tail  = M % TileM;
    constexpr int numKb   = K / BlockSize;
    constexpr int numPair = numKb / 2;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned. Mirrors dynamic_mx_quant_tail_axis_fp8.h:168.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    // Full-tile pass (ValidM == TileM).
    for (int m = 0; m < full_m; ++m) {
        for (int p = 0; p < numPair; ++p) {
            ocp_fp4_pair<OutT, M, K, TileM, BlockSize, scaleCols, TileM>(x, y_u8, scale, m, p);
        }
    }

    // Tail block: M_tail (< TileM) leftover rows. Keep the PHYSICAL tile shape at
    // TileM (>=512B / 32B-legal) but box every tile to ValidRow = M_tail so only
    // the live rows are touched (mirrors dynamic_mx_quant_tail_cublas_fp8).
    if constexpr (M_tail > 0) {
        for (int p = 0; p < numPair; ++p) {
            ocp_fp4_pair<OutT, M, K, TileM, BlockSize, scaleCols, M_tail>(x, y_u8, scale, full_m, p);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
