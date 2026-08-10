#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Tail-axis, OCP scale (scaleAlg=0), FP4 output (E2M1 default, E1M2 valid).
// emax derived from OutT (fp4_e2m1 -> 0x0100). Two-pass (full + M_tail) structure
// mirrors dynamic_mx_quant_tail_cublas_fp8: everything is inlined, the tail is an
// `if constexpr (M_tail>0)` block with M_tail baked into the tile aliases.
//
// SCHEME (padded-physical column-box). A single MX block of fp4 output is
// BlockSize/2 packed bytes (16B at BlockSize=32), which fails the 32B column
// alignment (pto_tile.hpp:408) whenever BlockSize/2 % 32 != 0. Instead of packing
// two blocks per output tile, we PAD the PHYSICAL tile width to
//   PW = next multiple of 64 >= BlockSize
// and COLUMN-BOX every op to the real BlockSize:
//   - value/scale tiles (uint16/bf16/fp32): physical PW, valid BlockSize. TROWMAX,
//     TCVT and TROWEXPANDMUL operate on ValidCol=BlockSize only (cpu_sim
//     TRowMax.hpp:13 loops j<ValidCol), so each block's reduction/scale stays
//     independent and correct; the PW padding is register-only.
//   - fp4 output tile: physical PW/2 bytes (32B-aligned because PW % 64 == 0),
//     valid BlockSize/2 bytes.
//   - boxed TLOAD/TSTORE transfer ValidCol columns, NOT the physical PW
//     (TLoadBackend blk_tload count = GetValidCol()), so the trailing block
//     reads/writes only its real BlockSize columns -> no HBM over-read past N.
//   - base-pointer fold places each block at its true column (value input
//     x + kb*BlockSize; fp4 output y + kb*(BlockSize/2)), since the physical
//     width PW != the per-block stride BlockSize, so we cannot let the iterator
//     step kb by the tile width.
// No scratch-HBM concat, no 2-block pairing. When BlockSize % 64 == 0, PW ==
// BlockSize and the col-box collapses to a full tile. Needs only N % BlockSize == 0.
//
// ODD numKb (scale even-pad): the AscendC scale layout is even-block-aligned
// (golden _pad_to_even pads with 2^-127, whose E8M0 byte is 0x00). We process
// only the numKb real blocks, so when numKb is odd we explicitly write one 0x00
// E8M0 byte to the padding scale column scale[numKb]. The data output is
// naturally N/2 real bytes/row (no padding data emitted).
template <int M, int N, int TileM = 8, int BlockSize = 32, typename OutT = __fp4_e2m1x2>
void dynamic_mx_quant_tail_ocp_fp4(__bf16 *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % BlockSize == 0, "N must be multiple of BlockSize");
    static_assert(BlockSize % 32 == 0,
                  "fp4 output block is BlockSize/2 packed bytes; BlockSize must be "
                  "a multiple of 32 so the padded physical fp4 width PW/2 is "
                  "32B-column-aligned (pto_tile.hpp:408)");

    using namespace pto;

    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = N / BlockSize;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned. Mirrors dynamic_mx_quant_tail_axis.h:217.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr bool oddTail  = (numKb % 2) != 0;   // padding scale col must be 0x00
    // Physical tile width padded to the next multiple of 64 so the packed fp4
    // output tile (physical PW/2 bytes) is 32B-column-aligned. BlockSize % 64 == 0
    // -> PW == BlockSize (col-box collapses to a full tile).
    constexpr int PW = ((BlockSize + 63) / 64) * 64;

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using gm_x  = global_tensor<__bf16,   RowMajor<M, N>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<M, N>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<M, N / 2>>;
    using gm_s  = global_tensor<uint8_t,  RowMajor<M, scaleCols>>;

    // Full-tile pass (ValidRow == TileM; boxed row collapses to NoneBox).
    {
        using tile_x         = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, TileM, BlockSize>;
        using tile_xu        = Tile<Location::Vec, uint16_t, TileM, PW,     BLayout::RowMajor, TileM, BlockSize>;
        using tile_sred      = Tile<Location::Vec, uint16_t, TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_sstore    = Tile<Location::Vec, uint8_t,  TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_recip_bf1 = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_recip_f1  = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, TileM, 1>;
        using tile_f         = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, TileM, BlockSize>;
        using tile_o         = Tile<Location::Vec, OutT,     TileM, PW / 2, BLayout::RowMajor, TileM, BlockSize / 2>;

        for (int m = 0; m < full_m; ++m) {
            for (int kb = 0; kb < numKb; ++kb) {
                // --- scale path: col-boxed load + reduce, compact scale store ---
                global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x) + kb * BlockSize);
                auto gxu = xu_iter(m, 0);
                tile_xu x_u16;
                TLOAD(x_u16, gxu);

                tile_sred scale_byte;
                tile_sred recip;
                compute_ocp_scale_tail_boxed_pw<OutT, TileM, PW, BlockSize>(x_u16, scale_byte, recip);
                tile_sstore scale_u8;
                TCVT(scale_u8, scale_byte);
                global_iterator<gm_s, tile_sstore> s_iter(scale + kb);
                auto gs = s_iter(m, 0);
                TSTORE(gs, scale_u8);

                tile_recip_bf1 inv_bf16;
                // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题4
                reinterpret_u16_to_bf16<2, TileM, PW, TileM, 1>(recip, inv_bf16);
                tile_recip_f1 inv_scale_f;
                TCVT(inv_scale_f, inv_bf16);

                // --- data path: col-boxed load, fp32 scale, narrowed fp4 store ---
                global_iterator<gm_x, tile_x> x_iter(x + kb * BlockSize);
                auto gx = x_iter(m, 0);
                tile_x xq;
                TLOAD(xq, gx);
                tile_f xf;
                TCVT(xf, xq);
                TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
                tile_o oq;
                TCVT(oq, xf);                        // fp32 (valid BlockSize) -> fp4 (valid BlockSize/2 bytes)
                global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * (BlockSize / 2));
                auto gy = y_iter(m, 0);
                TSTORE(gy, oq);                       // narrowed store at byte kb*(BlockSize/2)
            }
            // Even-pad the odd trailing scale column with a 0x00 E8M0 byte
            // (golden _pad_to_even uses 2^-127 == E8M0 0x00).
            if constexpr (oddTail) {
                tile_sred zpad16;
                TEXPANDS(zpad16, static_cast<uint16_t>(0));
                tile_sstore zpad;
                TCVT(zpad, zpad16);
                global_iterator<gm_s, tile_sstore> zs_iter(scale + numKb);
                auto gzs = zs_iter(m, 0);
                TSTORE(gzs, zpad);
            }
        }
    }

    // Tail rows: M_tail (< TileM), boxed to ValidRow = M_tail; row block full_m.
    if constexpr (M_tail > 0) {
        using tile_x         = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, M_tail, BlockSize>;
        using tile_xu        = Tile<Location::Vec, uint16_t, TileM, PW,     BLayout::RowMajor, M_tail, BlockSize>;
        using tile_sred      = Tile<Location::Vec, uint16_t, TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_sstore    = Tile<Location::Vec, uint8_t,  TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_recip_bf1 = Tile<Location::Vec, __bf16,   TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_recip_f1  = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, M_tail, 1>;
        using tile_f         = Tile<Location::Vec, float,    TileM, PW,     BLayout::RowMajor, M_tail, BlockSize>;
        using tile_o         = Tile<Location::Vec, OutT,     TileM, PW / 2, BLayout::RowMajor, M_tail, BlockSize / 2>;

        for (int kb = 0; kb < numKb; ++kb) {
            global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x) + kb * BlockSize);
            auto gxu = xu_iter(full_m, 0);
            tile_xu x_u16;
            TLOAD(x_u16, gxu);

            tile_sred scale_byte;
            tile_sred recip;
            compute_ocp_scale_tail_boxed_pw<OutT, TileM, PW, BlockSize, M_tail>(x_u16, scale_byte, recip);
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb);
            auto gs = s_iter(full_m, 0);
            TSTORE(gs, scale_u8);

            tile_recip_bf1 inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题4
            reinterpret_u16_to_bf16<2, TileM, PW, M_tail, 1>(recip, inv_bf16);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            global_iterator<gm_x, tile_x> x_iter(x + kb * BlockSize);
            auto gx = x_iter(full_m, 0);
            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TROWEXPANDMUL(xf, xf, inv_scale_f);
            tile_o oq;
            TCVT(oq, xf);
            global_iterator<gm_y, tile_o> y_iter(y_u8 + kb * (BlockSize / 2));
            auto gy = y_iter(full_m, 0);
            TSTORE(gy, oq);
        }
        if constexpr (oddTail) {
            tile_sred zpad16;
            TEXPANDS(zpad16, static_cast<uint16_t>(0));
            tile_sstore zpad;
            TCVT(zpad, zpad16);
            global_iterator<gm_s, tile_sstore> zs_iter(scale + numKb);
            auto gzs = zs_iter(full_m, 0);
            TSTORE(gzs, zpad);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
