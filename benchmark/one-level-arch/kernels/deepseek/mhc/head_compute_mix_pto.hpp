// =============================================================================
// head_compute_mix_pto.hpp — MHC head mix activation fwd/bwd
// =============================================================================
//
// Source: TileKernels/tile_kernels/mhc/head_compute_mix_kernel.py @ 36d9e45.
// Implements sigmoid(input * scale + base) (+ eps in fwd) with official PTO
// tile ops only. All local tile fragments are >=128B.
// =============================================================================
#ifndef SUPERNPU_MHC_HEAD_COMPUTE_MIX_PTO_HPP
#define SUPERNPU_MHC_HEAD_COMPUTE_MIX_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {
namespace mhc_head_compute_mix_detail {

template <typename TileT>
inline void sigmoid(TileT &out, const TileT &x) {
    using namespace pto;
    TileT neg, ex, denom;
    TMULS(neg, x, -1.0f);
    TEXP(ex, neg);
    TADDS(denom, ex, 1.0f);
    TRECIP(out, denom);
}

} // namespace mhc_head_compute_mix_detail

template <int NumTokens, int MhcMult, int TileN = 16>
void mhc_head_compute_mix_fwd(float *input_mix,
                              const float *mhc_scale,
                              const float *mhc_base,
                              float *output_mix,
                              float mhc_pre_eps = 0.0f) {
    static_assert(NumTokens > 0 && MhcMult > 0, "dim must be positive");
    static_assert(MhcMult <= 8, "source-level port uses one physical 8-wide mix tile");
    static_assert(TileN * 8 * static_cast<int>(sizeof(float)) >= 128, "mix tile must be >=128B");
    constexpr int kTiles = NumTokens / TileN;

    using namespace pto;
    using gm_mix = global_tensor<float, RowMajor<NumTokens, MhcMult>>;
    using tile_m = Tile<Location::Vec, float, TileN, 8, BLayout::RowMajor, TileN, MhcMult>;
    using tile_c = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, MhcMult>;
    using tile_s = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    using it_mix = global_iterator<gm_mix, tile_m>;
    it_mix in_iter(input_mix), out_iter(output_mix);

    for (int tn = 0; tn < kTiles; ++tn) {
        auto gi = in_iter(tn, 0);
        auto go = out_iter(tn, 0);
        tile_s scale_s;
        tile_c base_c;
        tile_m x, scale_full, base_full, scaled, affine, sig, out;
        TLOAD(scale_s, mhc_scale);
        TLOAD(base_c, mhc_base);
        TLOAD(x, gi);
        TCOLEXPAND(scale_full, scale_s);
        TROWEXPAND(scale_full, scale_full);
        TCOLEXPAND(base_full, base_c);
        TMUL(scaled, x, scale_full);
        TADD(affine, scaled, base_full);
        mhc_head_compute_mix_detail::sigmoid(sig, affine);
        TADDS(out, sig, mhc_pre_eps);
        TSTORE(go, out);
    }
}

template <int NumTokens, int MhcMult, int TileN = 16>
void mhc_head_compute_mix_bwd(float *output_mix_grad,
                              float *input_mix,
                              const float *mhc_scale,
                              const float *mhc_base,
                              float *input_mix_grad,
                              float *mhc_scale_grad_partial,
                              float *mhc_base_grad_partial) {
    static_assert(NumTokens > 0 && MhcMult > 0 && MhcMult <= 8, "valid dim required");
    static_assert(TileN * 8 * static_cast<int>(sizeof(float)) >= 128, "mix tile must be >=128B");
    constexpr int kTiles = NumTokens / TileN;

    using namespace pto;
    using gm_mix = global_tensor<float, RowMajor<NumTokens, MhcMult>>;
    using gm_s   = global_tensor<float, RowMajor<1, MhcMult>>;
    using tile_m = Tile<Location::Vec, float, TileN, 8, BLayout::RowMajor, TileN, MhcMult>;
    using tile_c = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, MhcMult>;
    using tile_s = Tile<Location::Vec, float, 4, 8, BLayout::RowMajor, 1, 1>;
    using it_mix = global_iterator<gm_mix, tile_m>;
    using it_s   = global_iterator<gm_s, tile_c>;
    it_mix og_iter(output_mix_grad), in_iter(input_mix), ig_iter(input_mix_grad);
    it_s bgrad_iter(mhc_base_grad_partial);

    tile_s scale_grad;
    tile_c base_grad;
    TEXPANDS(scale_grad, 0.0f);
    TEXPANDS(base_grad, 0.0f);
    for (int tn = 0; tn < kTiles; ++tn) {
        auto gog = og_iter(tn, 0);
        auto gin = in_iter(tn, 0);
        auto gig = ig_iter(tn, 0);
        tile_s scale_s;
        tile_c base_c;
        tile_m og, x, scale_full, base_full, scaled, affine, sig, one_minus, deriv, in_grad, scale_part;
        tile_c base_part;
        tile_s scale_part_sum;
        TLOAD(scale_s, mhc_scale);
        TLOAD(base_c, mhc_base);
        TLOAD(og, gog);
        TLOAD(x, gin);
        TCOLEXPAND(scale_full, scale_s);
        TROWEXPAND(scale_full, scale_full);
        TCOLEXPAND(base_full, base_c);
        TMUL(scaled, x, scale_full);
        TADD(affine, scaled, base_full);
        mhc_head_compute_mix_detail::sigmoid(sig, affine);
        TSUBS(one_minus, sig, 1.0f);
        TMULS(one_minus, one_minus, -1.0f);
        TMUL(deriv, sig, one_minus);
        TMUL(deriv, deriv, og);
        TMUL(in_grad, deriv, scale_full);
        TSTORE(gig, in_grad);
        TMUL(scale_part, deriv, x);
        TROWSUM(scale_part_sum, scale_part);
        TADD(scale_grad, scale_grad, scale_part_sum);
        TCOLSUM(base_part, deriv);
        TADD(base_grad, base_grad, base_part);
    }

    using it_scale = global_iterator<global_tensor<float, RowMajor<1, 1>>, tile_s>;
    auto gs = it_scale(mhc_scale_grad_partial)(0, 0);
    auto gb = bgrad_iter(0, 0);
    TSTORE(gs, scale_grad);
    TSTORE(gb, base_grad);
}

} // namespace supernpu::tile_isa
#endif
