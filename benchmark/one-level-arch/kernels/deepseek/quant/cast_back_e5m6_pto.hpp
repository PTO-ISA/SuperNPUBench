// =============================================================================
// cast_back_e5m6_pto.hpp -- E5M6 dequantization source-level tile port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/cast_back_e5m6_kernel.py @ 36d9e45.
//
// The upstream kernel unpacks three uint32 words into eight E5M6 values, then
// multiplies by block scale factors. This PTO port keeps the packed-word load,
// conversion, scale expansion, and store in tile code. It does not add a scalar
// pointer-indexed unpack fallback.
// =============================================================================
#ifndef SUPERNPU_CAST_BACK_E5M6_PTO_HPP
#define SUPERNPU_CAST_BACK_E5M6_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename OutDType, int NumTokens, int Hidden, int NumPerTokens = 1,
          int NumPerChannels = Hidden, int TileM = 1>
void cast_back_e5m6(std::uint32_t *x_packed, float *x_sf, OutDType *out) {
    static_assert(Hidden % 8 == 0, "E5M6 packs groups of 8 values");
    static_assert(NumPerTokens > 0 && NumPerChannels > 0, "scale block must be positive");
    static_assert(Hidden * static_cast<int>(sizeof(OutDType)) >= 128,
                  "thread-local output fragment must be at least 128 bytes");
    constexpr int PackedWords = Hidden * 3 / 8;
    using namespace pto;
    using gm_x = global_tensor<std::uint32_t, RowMajor<NumTokens, PackedWords>>;
    using gm_sf = global_tensor<float,
                                RowMajor<NumTokens / NumPerTokens,
                                         Hidden / NumPerChannels>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using tile_packed = Tile<Location::Vec, std::uint32_t, TileM, PackedWords,
                             BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, TileM, Hidden, BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, TileM, Hidden, BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor,
                         TileM, 1>;
    using it_x = global_iterator<gm_x, tile_packed>;
    using it_sf = global_iterator<gm_sf, tile_sf>;
    using it_o = global_iterator<gm_o, tile_o>;
    it_x x_iter(x_packed);
    it_sf sf_iter(x_sf);
    it_o o_iter(out);

    for (int tm = 0; tm < NumTokens / TileM; ++tm) {
        auto gx = x_iter(tm, 0);
        auto gsf = sf_iter(tm / NumPerTokens, 0);
        auto go = o_iter(tm, 0);
        tile_packed packed;
        tile_f unpacked, sf_full, scaled;
        tile_o oq;
        tile_sf sf;
        TLOAD(packed, gx);
        TLOAD(sf, gsf);
        TCVT(unpacked, packed);
        TROWEXPAND(sf_full, sf);
        TMUL(scaled, unpacked, sf_full);
        TCVT(oq, scaled);
        TSTORE(go, oq);
    }
}

} // namespace supernpu::tile_isa

#endif
