// =============================================================================
// per_block_cast_pto.hpp -- block FP8/FP4 quantization, source-level PTO tile port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/per_block_cast_kernel.py @ 36d9e45.
//
// Computes one scale per (NumPerTokens x NumPerChannels) block:
//   amax = max(abs(x block))
//   sf = amax / max_val
//   out = x * (max_val / amax)
//
// The source module also exposes sf-only and cast-with-precomputed-sf modes.
// They are represented as separate tile entry points below. Scalar C++ controls
// tile loops only; tensor data moves through global_iterator + tile intrinsics.
// =============================================================================
#ifndef SUPERNPU_PER_BLOCK_CAST_PTO_HPP
#define SUPERNPU_PER_BLOCK_CAST_PTO_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename TileF, typename TileScalar>
inline void per_block_absmax_tile(TileScalar &amax, TileF &xf,
                                  float clamp_min) {
    using namespace pto;
    TileF neg;
    TileScalar row_pos, row_neg, row_abs_pos, row_abs_neg;
    TMULS(neg, xf, -1.0f);
    TROWMAX(row_pos, xf);
    TROWMAX(row_neg, neg);
    TCOLMAX(row_abs_pos, row_pos);
    TCOLMAX(row_abs_neg, row_neg);
    TMAX(amax, row_abs_pos, row_abs_neg);
    TMAXS(amax, amax, clamp_min);
}

template <typename TileF, typename TileScalar>
inline void per_block_expand_scalar(TileF &expanded, TileScalar &scalar) {
    using namespace pto;
    using TileCol = pto::Tile<pto::Location::Vec, float, TileF::Rows, 8,
                              pto::BLayout::RowMajor, TileF::Rows, 1>;
    TileCol col;
    TCOLEXPAND(col, scalar);
    TROWEXPAND(expanded, col);
}

template <typename InDType, typename OutDType, int NumTokens, int Hidden,
          int NumPerTokens, int NumPerChannels>
void per_block_cast(InDType *x, OutDType *out, float *out_sf,
                    float max_val, float clamp_min) {
    static_assert(NumTokens > 0 && Hidden > 0, "dim must be positive");
    static_assert(NumPerTokens == 32 || NumPerTokens == 128,
                  "source supports 32 or 128 tokens per scale block");
    static_assert(NumPerChannels == 32 || NumPerChannels == 128,
                  "source supports 32 or 128 channels per scale block");
    static_assert(NumPerTokens * NumPerChannels *
                      static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    static_assert(NumPerTokens * NumPerChannels *
                      static_cast<int>(sizeof(float)) >= 128,
                  "thread-local fp32 fragment must be at least 128 bytes");

    constexpr int kTM = NumTokens / NumPerTokens;
    constexpr int kTK = Hidden / NumPerChannels;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using gm_sf = global_tensor<float, RowMajor<kTM, kTK>>;
    using tile_x = Tile<Location::Vec, InDType, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_scalar = Tile<Location::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_sf = global_iterator<gm_sf, tile_scalar>;
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
            tile_f xf, sf_full, inv_full, scaled;
            tile_scalar amax, sf, inv;
            TLOAD(xq, gx);
            TCVT(xf, xq);
            per_block_absmax_tile(amax, xf, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TRECIP(inv, amax);
            TMULS(inv, inv, max_val);
            per_block_expand_scalar(sf_full, sf);
            per_block_expand_scalar(inv_full, inv);
            TMUL(scaled, xf, inv_full);
            TCVT(oq, scaled);
            TSTORE(gsf, sf);
            TSTORE(go, oq);
        }
    }
}

template <typename InDType, int NumTokens, int Hidden, int NumPerTokens,
          int NumPerChannels>
void per_block_cast_sf_only(InDType *x, float *out_sf, float max_val,
                            float clamp_min) {
    static_assert(NumPerTokens * NumPerChannels *
                      static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    constexpr int kTM = NumTokens / NumPerTokens;
    constexpr int kTK = Hidden / NumPerChannels;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_sf = global_tensor<float, RowMajor<kTM, kTK>>;
    using tile_x = Tile<Location::Vec, InDType, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_scalar = Tile<Location::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_sf = global_iterator<gm_sf, tile_scalar>;
    it_x x_iter(x);
    it_sf sf_iter(out_sf);
    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gx = x_iter(tm, tk);
            auto gsf = sf_iter(tm, tk);
            tile_x xq;
            tile_f xf;
            tile_scalar amax, sf;
            TLOAD(xq, gx);
            TCVT(xf, xq);
            per_block_absmax_tile(amax, xf, clamp_min);
            TMULS(sf, amax, 1.0f / max_val);
            TSTORE(gsf, sf);
        }
    }
}

template <typename InDType, typename OutDType, int NumTokens, int Hidden,
          int NumPerTokens, int NumPerChannels>
void per_block_cast_with_precomputed_sf(InDType *x, OutDType *out,
                                        float *sf_inv) {
    static_assert(NumPerTokens * NumPerChannels *
                      static_cast<int>(sizeof(InDType)) >= 128,
                  "thread-local input fragment must be at least 128 bytes");
    constexpr int kTM = NumTokens / NumPerTokens;
    constexpr int kTK = Hidden / NumPerChannels;
    using namespace pto;
    using gm_x = global_tensor<InDType, RowMajor<NumTokens, Hidden>>;
    using gm_o = global_tensor<OutDType, RowMajor<NumTokens, Hidden>>;
    using gm_sf = global_tensor<float, RowMajor<kTM, kTK>>;
    using tile_x = Tile<Location::Vec, InDType, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, OutDType, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, NumPerTokens, NumPerChannels,
                        BLayout::RowMajor>;
    using tile_scalar = Tile<Location::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    using it_x = global_iterator<gm_x, tile_x>;
    using it_o = global_iterator<gm_o, tile_o>;
    using it_sf = global_iterator<gm_sf, tile_scalar>;
    it_x x_iter(x);
    it_o o_iter(out);
    it_sf sf_iter(sf_inv);
    for (int tm = 0; tm < kTM; ++tm) {
        for (int tk = 0; tk < kTK; ++tk) {
            auto gx = x_iter(tm, tk);
            auto go = o_iter(tm, tk);
            auto gsf = sf_iter(tm, tk);
            tile_x xq;
            tile_o oq;
            tile_f xf, inv_full, scaled;
            tile_scalar inv;
            TLOAD(xq, gx);
            TLOAD(inv, gsf);
            TCVT(xf, xq);
            per_block_expand_scalar(inv_full, inv);
            TMUL(scaled, xf, inv_full);
            TCVT(oq, scaled);
            TSTORE(go, oq);
        }
    }
}

} // namespace supernpu::tile_isa

#endif
