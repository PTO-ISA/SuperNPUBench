// =============================================================================
// swiglu_forward_and_per_channel_cast_and_transpose_pto.hpp
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/
// swiglu_forward_and_per_channel_cast_and_transpose_kernel.py @ 36d9e45.
// =============================================================================
#ifndef SUPERNPU_SWIGLU_FORWARD_AND_PER_CHANNEL_CAST_AND_TRANSPOSE_PTO_HPP
#define SUPERNPU_SWIGLU_FORWARD_AND_PER_CHANNEL_CAST_AND_TRANSPOSE_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename OutDType, int NumTokens, int Hidden, int NumPerTokens = 128,
          int TileTokens = 128, int TileHidden = 64>
void swiglu_forward_and_per_channel_cast_and_transpose(
    __bf16 *x, OutDType *out, float *out_sf, float max_val, float clamp_min,
    float swiglu_clamp_value = 0.0f) {
    static_assert(NumTokens % 128 == 0, "source requires token count multiple of 128");
    static_assert(Hidden % 64 == 0, "logical hidden must be multiple of 64");
    static_assert(NumPerTokens == 32 || NumPerTokens == 128,
                  "source supports 32 or 128 tokens per scale");
    static_assert(TileTokens * TileHidden * static_cast<int>(sizeof(__bf16)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    (void)swiglu_clamp_value;

    constexpr int kTM = NumTokens / TileTokens;
    constexpr int kTK = Hidden / TileHidden;
    using namespace pto;
    using gm_x = global_tensor<__bf16, RowMajor<NumTokens, Hidden * 2>>;
    using gm_o = global_tensor<OutDType, RowMajor<Hidden, NumTokens>>;
    using gm_sf = global_tensor<float, RowMajor<NumTokens / NumPerTokens, Hidden>>;
    using tile_x = Tile<Location::Vec, __bf16, TileTokens, TileHidden,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, TileTokens, TileHidden,
                        BLayout::RowMajor>;
    using tile_t = Tile<Location::Vec, float, TileHidden, TileTokens,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, TileHidden, TileTokens,
                        BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, TileHidden, 8, BLayout::RowMajor,
                         TileHidden, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    it_x x_iter(x);
    it_o o_iter(out);
    it_sf sf_iter(out_sf);

    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gg = x_iter(tm, tk);
            auto gu = x_iter(tm, kTK + tk);
            auto go = o_iter(tk, tm);
            auto gsf = sf_iter(tm, tk);
            tile_x gq, uq;
            tile_f gf, uf, neg_g, expn, denom, sig, silu, sw, neg_s;
            tile_t sw_t, neg_t, scaled_t, inv_full;
            tile_o oq;
            tile_sf pos, negm, amax, sf, inv;
            TLOAD(gq, gg);
            TLOAD(uq, gu);
            TCVT(gf, gq);
            TCVT(uf, uq);
            TMULS(neg_g, gf, -1.0f);
            TEXP(expn, neg_g);
            TADDS(denom, expn, 1.0f);
            TRECIP(sig, denom);
            TMUL(silu, gf, sig);
            TMUL(sw, silu, uf);
            TMULS(neg_s, sw, -1.0f);
            TTRANS(sw_t, sw);
            TTRANS(neg_t, neg_s);
            TROWMAX(pos, sw_t);
            TROWMAX(negm, neg_t);
            TMAX(amax, pos, negm);
            TMAXS(amax, amax, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TRECIP(inv, amax);
            TMULS(inv, inv, max_val);
            TROWEXPAND(inv_full, inv);
            TMUL(scaled_t, sw_t, inv_full);
            TCVT(oq, scaled_t);
            TSTORE(gsf, sf);
            TSTORE(go, oq);
        }
    }
}

template <typename OutDType, int NumTokens, int Hidden, int NumPerTokens = 128,
          int TileTokens = 128, int TileHidden = 64>
void swiglu_forward_and_per_channel_cast_without_transpose(
    __bf16 *x, OutDType *out, float *out_sf, float max_val, float clamp_min,
    float swiglu_clamp_value = 0.0f) {
    static_assert(TileTokens * TileHidden * static_cast<int>(sizeof(__bf16)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    (void)swiglu_clamp_value;
    constexpr int kTM = NumTokens / TileTokens;
    constexpr int kTK = Hidden / TileHidden;
    using namespace pto;
    using gm_x = global_tensor<__bf16, RowMajor<NumTokens, Hidden * 2>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using gm_sf = global_tensor<float, RowMajor<NumTokens / NumPerTokens, Hidden>>;
    using tile_x = Tile<Location::Vec, __bf16, TileTokens, TileHidden,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, TileTokens, TileHidden,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, TileTokens, TileHidden,
                        BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, 1, TileHidden, BLayout::RowMajor,
                         1, TileHidden>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    it_x x_iter(x);
    it_o o_iter(out);
    it_sf sf_iter(out_sf);
    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gg = x_iter(tm, tk);
            auto gu = x_iter(tm, kTK + tk);
            auto go = o_iter(tm, tk);
            auto gsf = sf_iter(tm, tk);
            tile_x gq, uq;
            tile_f gf, uf, neg_g, expn, denom, sig, sw, silu, neg_s, scaled,
                inv_full;
            tile_o oq;
            tile_sf pos, negm, amax, sf, inv;
            TLOAD(gq, gg);
            TLOAD(uq, gu);
            TCVT(gf, gq);
            TCVT(uf, uq);
            TMULS(neg_g, gf, -1.0f);
            TEXP(expn, neg_g);
            TADDS(denom, expn, 1.0f);
            TRECIP(sig, denom);
            TMUL(silu, gf, sig);
            TMUL(sw, silu, uf);
            TMULS(neg_s, sw, -1.0f);
            TCOLMAX(pos, sw);
            TCOLMAX(negm, neg_s);
            TMAX(amax, pos, negm);
            TMAXS(amax, amax, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TRECIP(inv, amax);
            TMULS(inv, inv, max_val);
            TCOLEXPAND(inv_full, inv);
            TMUL(scaled, sw, inv_full);
            TCVT(oq, scaled);
            TSTORE(gsf, sf);
            TSTORE(go, oq);
        }
    }
}

} // namespace supernpu::tile_isa

#endif
