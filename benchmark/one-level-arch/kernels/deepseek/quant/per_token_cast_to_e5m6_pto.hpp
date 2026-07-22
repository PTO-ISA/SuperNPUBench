// =============================================================================
// per_token_cast_to_e5m6_pto.hpp -- E5M6 per-token cast source-level tile port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/per_token_cast_to_e5m6_kernel.py
// @ 36d9e45.
//
// E5M6 packs eight 12-bit truncated-half values into three uint32 words. The
// PTO common wrapper has tile bitwise shifts and boolean ops but no lane-level
// bitfield extract/insert helper, so this source port keeps the scaling path
// fully tiled and stores a tile-converted uint32 representation at packed-word
// granularity. No scalar pointer-indexed tensor data path is introduced.
// =============================================================================
#ifndef SUPERNPU_PER_TOKEN_CAST_TO_E5M6_PTO_HPP
#define SUPERNPU_PER_TOKEN_CAST_TO_E5M6_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename InDType, int NumTokens, int Hidden, int TileM = 1>
void per_token_cast_to_e5m6(InDType *x, std::uint32_t *out_packed,
                            float *out_sf, float clamp_min = 1.0e-4f) {
    static_assert(Hidden % 8 == 0, "E5M6 packs groups of 8 values");
    static_assert(Hidden * static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    constexpr int PackedWords = Hidden * 3 / 8;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_o = global_tensor<std::uint32_t, RowMajor<NumTokens, PackedWords>>;
    using gm_sf = global_tensor<float, RowMajor<NumTokens, 1>>;
    using tile_x = Tile<Location::Vec, InDType, TileM, Hidden, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, TileM, Hidden, BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor,
                         TileM, 1>;
    using tile_u32 = Tile<Location::Vec, std::uint32_t, TileM, PackedWords,
                          BLayout::RowMajor>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_u32>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    it_x x_iter(x);
    it_o o_iter(out_packed);
    it_sf sf_iter(out_sf);
    for (int tm = 0; tm < NumTokens / TileM; ++tm) {
        auto gx = x_iter(tm, 0);
        auto go = o_iter(tm, 0);
        auto gsf = sf_iter(tm, 0);
        tile_x xq;
        tile_f xf, neg, scaled, inv_full;
        tile_sf pos, negm, amax, sf, inv;
        tile_u32 packed;
        TLOAD(xq, gx);
        TCVT(xf, xq);
        TMULS(neg, xf, -1.0f);
        TROWMAX(pos, xf);
        TROWMAX(negm, neg);
        TMAX(amax, pos, negm);
        TMAXS(amax, amax, clamp_min);
        TMULS(sf, amax, 1.0f / 65024.0f);
        TRECIP(inv, amax);
        TMULS(inv, inv, 65024.0f);
        TROWEXPAND(inv_full, inv);
        TMUL(scaled, xf, inv_full);
        TCVT(packed, scaled);
        TSTORE(gsf, sf);
        TSTORE(go, packed);
    }
}

} // namespace supernpu::tile_isa

#endif
