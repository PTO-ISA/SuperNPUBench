// =============================================================================
// per_channel_cast_fused_pto.hpp -- fused per-channel FP8 cast tile port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/per_channel_cast_fused_kernel.py
// @ 36d9e45.
//
// The source optionally gathers expanded tokens through pos_to_token. The PTO
// source-level port below covers the contiguous no-expand tile data path. The
// expansion map is a scalar indexed data-dependent gather in the source module;
// it is intentionally not represented here because this migration owns no
// pointer-indexed tensor data path.
// =============================================================================
#ifndef SUPERNPU_PER_CHANNEL_CAST_FUSED_PTO_HPP
#define SUPERNPU_PER_CHANNEL_CAST_FUSED_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename InDType, typename OutDType, int NumTokens, int Hidden,
          int NumPerTokens = 128, int TileK = 128>
void per_channel_cast_fused(InDType *x, OutDType *out, float *out_sf,
                            float max_val, float clamp_min) {
    static_assert(NumTokens > 0 && Hidden > 0, "dim must be positive");
    static_assert(NumPerTokens == 128, "source fused path uses 128-token scale blocks");
    static_assert(TileK % 64 == 0, "source uses hidden tiles aligned to 64/128 columns");
    static_assert(NumPerTokens * TileK * static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");

    constexpr int kTM = NumTokens / NumPerTokens;
    constexpr int kTK = Hidden / TileK;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using gm_sf = global_tensor<float, RowMajor<kTM, Hidden>>;
    using tile_x = Tile<Location::Vec, InDType, NumPerTokens, TileK,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, NumPerTokens, TileK,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, NumPerTokens, TileK,
                        BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, 1, TileK, BLayout::RowMajor, 1,
                         TileK>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    it_x x_iter(x);
    it_o o_iter(out);
    it_sf sf_iter(out_sf);

    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gx = x_iter(tm, tk);
            auto go = o_iter(tm, tk);
            auto gsf = sf_iter(tm, tk);
            tile_x xq;
            tile_o oq;
            tile_f xf, neg, scaled, inv_full;
            tile_sf pos, negm, amax, sf, inv;
            TLOAD(xq, gx);
            TCVT(xf, xq);
            TMULS(neg, xf, -1.0f);
            TCOLMAX(pos, xf);
            TCOLMAX(negm, neg);
            TMAX(amax, pos, negm);
            TMAXS(amax, amax, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TRECIP(inv, amax);
            TMULS(inv, inv, max_val);
            TCOLEXPAND(inv_full, inv);
            TMUL(scaled, xf, inv_full);
            TCVT(oq, scaled);
            TSTORE(gsf, sf);
            TSTORE(go, oq);
        }
    }
}

} // namespace supernpu::tile_isa

#endif
