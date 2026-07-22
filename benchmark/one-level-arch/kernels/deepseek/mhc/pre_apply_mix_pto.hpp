// =============================================================================
// pre_apply_mix_pto.hpp — apply pre-layer MHC mix fwd/bwd
// =============================================================================
//
// Source: TileKernels/tile_kernels/mhc/pre_apply_mix_kernel.py @ 36d9e45.
// Computes o[n,h] = sum_m mix[n,m] * x[n,m,h] and the matching source-level
// backward. All local tile fragments are >=128B.
// =============================================================================
#ifndef SUPERNPU_MHC_PRE_APPLY_MIX_PTO_HPP
#define SUPERNPU_MHC_PRE_APPLY_MIX_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int MhcMult, int Hidden, int TileH = 64>
void mhc_pre_apply_mix_fwd(__bf16 *x,
                           float *mix,
                           __bf16 *o) {
    static_assert(NumTokens > 0 && MhcMult > 0 && Hidden > 0, "dim must be positive");
    static_assert(MhcMult <= 8 && TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "valid tile shape required");
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_x = global_tensor<__bf16, RowMajor<NumTokens * MhcMult, Hidden>>;
    using gm_o = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using gm_m = global_tensor<float, RowMajor<NumTokens * MhcMult, 1>>;
    using tile_xbf = Tile<Location::Vec, __bf16, MhcMult, TileH, BLayout::RowMajor>;
    using tile_xf  = Tile<Location::Vec, float, MhcMult, TileH, BLayout::RowMajor>;
    using tile_o   = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_of  = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using tile_m   = Tile<Location::Vec, float, MhcMult, 8, BLayout::RowMajor, MhcMult, 1>;
    static_assert(tile_xbf::Rows * tile_xbf::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_xf::Rows * tile_xf::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_o::Rows * tile_o::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_m::Rows * tile_m::Cols * static_cast<int>(sizeof(float)) >= 128);

    using it_x = global_iterator<gm_x, tile_xbf>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_m = global_iterator<gm_m, tile_m>;
    it_x x_iter(x);
    it_o o_iter(o);
    it_m m_iter(mix);

    for (int n = 0; n < NumTokens; ++n) {
        auto gm = m_iter(n * MhcMult, 0);
        tile_m mix_col;
        TLOAD(mix_col, gm);
        for (int th = 0; th < kTiles; ++th) {
            auto gx = x_iter(n, th);
            auto go = o_iter(n, th);
            tile_xbf xb;
            tile_xf xf, mix_full, prod;
            tile_of sum;
            tile_o ob;
            TLOAD(xb, gx);
            TCVT(xf, xb);
            TROWEXPAND(mix_full, mix_col);
            TMUL(prod, xf, mix_full);
            TCOLSUM(sum, prod);
            TCVT(ob, sum);
            TSTORE(go, ob);
        }
    }
}

template <int NumTokens, int MhcMult, int Hidden, int TileH = 64>
void mhc_pre_apply_mix_bwd(__bf16 *o_grad,
                           __bf16 *x,
                           float *mix,
                           __bf16 *x_grad,
                           float *mix_grad) {
    static_assert(MhcMult <= 8 && TileH >= 64 && TileH % 8 == 0 && Hidden % TileH == 0, "valid tile shape required");
    constexpr int kTiles = Hidden / TileH;

    using namespace pto;
    using gm_x = global_tensor<__bf16, RowMajor<NumTokens * MhcMult, Hidden>>;
    using gm_o = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using gm_m = global_tensor<float, RowMajor<NumTokens * MhcMult, 1>>;
    using tile_xbf = Tile<Location::Vec, __bf16, MhcMult, TileH, BLayout::RowMajor>;
    using tile_xf  = Tile<Location::Vec, float, MhcMult, TileH, BLayout::RowMajor>;
    using tile_o   = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using tile_of  = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor>;
    using tile_m   = Tile<Location::Vec, float, MhcMult, 8, BLayout::RowMajor, MhcMult, 1>;
    static_assert(tile_xbf::Rows * tile_xbf::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_xf::Rows * tile_xf::Cols * static_cast<int>(sizeof(float)) >= 128);
    static_assert(tile_o::Rows * tile_o::Cols * static_cast<int>(sizeof(__bf16)) >= 128);
    static_assert(tile_m::Rows * tile_m::Cols * static_cast<int>(sizeof(float)) >= 128);

    using it_x = global_iterator<gm_x, tile_xbf>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_m = global_iterator<gm_m, tile_m>;
    it_x x_iter(x), xg_iter(x_grad);
    it_o og_iter(o_grad);
    it_m m_iter(mix), mg_iter(mix_grad);

    for (int n = 0; n < NumTokens; ++n) {
        auto gm = m_iter(n * MhcMult, 0);
        tile_m mix_col, mix_grad_acc;
        TLOAD(mix_col, gm);
        TEXPANDS(mix_grad_acc, 0.0f);
        for (int th = 0; th < kTiles; ++th) {
            auto gog = og_iter(n, th);
            auto gx = x_iter(n, th);
            auto gxg = xg_iter(n, th);
            tile_o ogb;
            tile_of ogf;
            tile_xbf xb, old_xgb, out_xgb;
            tile_xf xf, old_xgf, og_full, mix_full, xg_add, xg_out, prod;
            tile_m mg_part;
            TLOAD(ogb, gog);
            TLOAD(xb, gx);
            TLOAD(old_xgb, gxg);
            TCVT(ogf, ogb);
            TCVT(xf, xb);
            TCVT(old_xgf, old_xgb);
            TCOLEXPAND(og_full, ogf);
            TROWEXPAND(mix_full, mix_col);
            TMUL(xg_add, og_full, mix_full);
            TADD(xg_out, old_xgf, xg_add);
            TCVT(out_xgb, xg_out);
            TSTORE(gxg, out_xgb);
            TMUL(prod, og_full, xf);
            TROWSUM(mg_part, prod);
            TADD(mix_grad_acc, mix_grad_acc, mg_part);
        }
        auto gmg = mg_iter(n * MhcMult, 0);
        TSTORE(gmg, mix_grad_acc);
    }
}

} // namespace supernpu::tile_isa
#endif
