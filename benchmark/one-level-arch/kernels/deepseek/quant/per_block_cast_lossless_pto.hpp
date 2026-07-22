// =============================================================================
// per_block_cast_lossless_pto.hpp -- FP4->FP8 lossless recast tile port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/per_block_cast_lossless_kernel.py
// @ 36d9e45.
// =============================================================================
#ifndef SUPERNPU_PER_BLOCK_CAST_LOSSLESS_PTO_HPP
#define SUPERNPU_PER_BLOCK_CAST_LOSSLESS_PTO_HPP

#include "per_block_cast_pto.hpp"
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename InDType, typename OutDType, int NumTokens, int Hidden,
          int InTokens, int InChannels, int OutTokens, int OutChannels>
void per_block_cast_lossless(InDType *x, float *x_sf, OutDType *out,
                             float *out_sf) {
    static_assert(OutTokens % InTokens == 0 && OutChannels % InChannels == 0,
                  "output scale block must be a multiple of input scale block");
    static_assert(OutTokens * OutChannels * static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    constexpr int kTM = NumTokens / OutTokens;
    constexpr int kTK = Hidden / OutChannels;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_xsf = global_tensor<float, RowMajor<NumTokens / InTokens,
                                                Hidden / InChannels>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using gm_osf = global_tensor<float, RowMajor<NumTokens / OutTokens,
                                                Hidden / OutChannels>>;
    using tile_x = Tile<Location::Vec, InDType, OutTokens, OutChannels,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, OutTokens, OutChannels,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, OutTokens, OutChannels,
                        BLayout::RowMajor>;
    using tile_sf = Tile<Location::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_xsf = global_iterator<gm_xsf, tile_sf>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_osf = global_iterator<gm_osf, tile_sf>;
    it_x x_iter(x);
    it_xsf xsf_iter(x_sf);
    it_o o_iter(out);
    it_osf osf_iter(out_sf);

    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gx = x_iter(tm, tk);
            auto go = o_iter(tm, tk);
            auto gxs = xsf_iter(tm * (OutTokens / InTokens),
                                tk * (OutChannels / InChannels));
            auto gos = osf_iter(tm, tk);
            tile_x xq;
            tile_o oq;
            tile_f xf, sf_full, osf_full, osf_inv_full, adjusted;
            tile_sf in_sf, out_scale;
            TLOAD(xq, gx);
            TLOAD(in_sf, gxs);
            TCVT(xf, xq);
            TMULS(out_scale, in_sf, 1.0f / 64.0f);
            per_block_expand_scalar(sf_full, in_sf);
            per_block_expand_scalar(osf_full, out_scale);
            TRECIP(osf_full, osf_full);
            TMUL(adjusted, xf, sf_full);
            TMUL(adjusted, adjusted, osf_full);
            TCVT(oq, adjusted);
            TSTORE(gos, out_scale);
            TSTORE(go, oq);
        }
    }
}

} // namespace supernpu::tile_isa

#endif
