// =============================================================================
// rms_norm.hpp — RMSNorm (one-level PTO)
// =============================================================================
//
// Shape dims: A (outer / row), R (reduce / col).
//
//   out[a] = x[a] * rsqrt(mean(x[a]^2) + eps)
//
// Entry:
//   rms_norm<dtype>(x, tiling, out, eps);
//   tiling[4] = {g_a, g_r, tile_a, tile_r}  (int64_t)
//   tile_r <= 0 means use g_r (full-row tile).
//
// Pipeline (fp16 in/out, fp32 compute):
//   TLOAD → TCVT → TMUL(x,x) → TROWSUM → TMULS(1/g_r) → TADDS(eps)
//   → Newton rsqrt → TROWEXPANDMUL → TCVT → TSTORE
//
// Dynamic ValidRow/ValidCol: Tile Valid = -1, ctor passes runtime values.
// Full A tiles in the main loop; trailing rows handled separately.
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_PTO_HPP
#define SUPERNPU_RMS_NORM_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail {

template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    TRSQRT(out, a);
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_v>
inline void rms_norm_tile(dtype *x, dtype *out, int64_t gA, int64_t gR,
                          int64_t a_off, int64_t active_a, int64_t active_r,
                          float inv_r, float eps) {
    const int64_t offset = a_off * gR;
    gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
    gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));

    tile_h src_h(static_cast<size_t>(active_a),
                 static_cast<size_t>(active_r));
    tile_h dst_h(static_cast<size_t>(active_a),
                 static_cast<size_t>(active_r));
    tile_f src(static_cast<size_t>(active_a),
               static_cast<size_t>(active_r));
    tile_f squared(static_cast<size_t>(active_a),
                   static_cast<size_t>(active_r));
    tile_f dst(static_cast<size_t>(active_a),
               static_cast<size_t>(active_r));
    tile_v sqrsum(static_cast<size_t>(active_a));
    tile_v mean(static_cast<size_t>(active_a));
    tile_v denom(static_cast<size_t>(active_a));
    tile_v rms(static_cast<size_t>(active_a));

    TLOAD(src_h, gi);
    TCVT(src, src_h);
    TMUL(squared, src, src);
    TROWSUM(sqrsum, squared);
    TMULS(mean, sqrsum, inv_r);
    TADDS(denom, mean, eps);
    rsqrt_newton(rms, denom);
    TROWEXPANDMUL(dst, src, rms);
    TCVT(dst_h, dst);
    TSTORE(go, dst_h);
}

} // namespace rms_detail

// tiling: [g_a, g_r, tile_a, tile_r]
template <typename dtype>
void rms_norm(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    // Physical capacity (Rows×Cols); Valid comes from tiling (tile_a,tile_r).
    // Size must cover ValidRow×ValidCol; SoftCore should not require Rows≥ValidRow.
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t gA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_a = tiling[2] > 0 ? tiling[2] : tA;
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : gR;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    // Row expansion requires the broadcast source to have physical Cols=1.
    // Keep a 512-byte carrier by placing the padding in Rows instead.
    using tile_v =
        Tile<Location::Vec, float, 128, 1, BLayout::RowMajor, -1, 1>;

    const float inv_r = 1.0f / static_cast<float>(gR);

    // Full A tiles; peel the last iteration for the trailing block.
    int64_t ia = 0;
    for (; ia + tile_a < gA; ia += tile_a) {
        rms_detail::rms_norm_tile<dtype, gm_t, tile_h, tile_f, tile_v>(
            x, out, gA, gR, ia, tile_a, tile_r, inv_r, eps);
    }
    // Tail (or sole) block: ValidRow = remaining rows along A.
    rms_detail::rms_norm_tile<dtype, gm_t, tile_h, tile_f, tile_v>(
        x, out, gA, gR, ia, gA - ia, tile_r, inv_r, eps);
}

// Compile-time shape / tiling (gelu/gather style). tR must cover the full row.
template <typename dtype, int gA, int gR, int tA, int tR>
void rms_norm(dtype *x, dtype *out, float eps = 1e-6f) {
    static_assert(gA > 0 && gR > 0 && tA > 0 && tR > 0);
    static_assert(tR == gR, "static rms_norm is a single R-tile; use rms_norm_binary for R-split");
    constexpr int Mb = gA / tA;
    constexpr int rmd_A = gA % tA;
    constexpr float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<gA, gR>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor>;
    static_assert(tA <= 128, "RMSNorm row-state carrier supports tA <= 128");
    using tile_v =
        Tile<Location::Vec, float, 128, 1, BLayout::RowMajor, tA, 1>;
    using it_t = global_iterator<gm_t, tile_h>;

    it_t gI(x);
    it_t gO(out);

    for (int ia = 0; ia < Mb; ++ia) {
        tile_h src_h, dst_h;
        tile_f src, squared, dst;
        tile_v sqrsum, mean, denom, rms;
        auto gi = gI(ia, 0);
        auto go = gO(ia, 0);
        TLOAD(src_h, gi);
        TCVT(src, src_h);
        TMUL(squared, src, src);
        TROWSUM(sqrsum, squared);
        TMULS(mean, sqrsum, inv_r);
        TADDS(denom, mean, eps);
        rms_detail::rsqrt_newton(rms, denom);
        TROWEXPANDMUL(dst, src, rms);
        TCVT(dst_h, dst);
        TSTORE(go, dst_h);
    }
    if constexpr (rmd_A) {
        using tile_h_r = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, rmd_A, tR>;
        using tile_f_r = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, rmd_A, tR>;
        using tile_v_r =
            Tile<Location::Vec, float, 128, 1, BLayout::RowMajor, rmd_A, 1>;
        tile_h_r src_h, dst_h;
        tile_f_r src, squared, dst;
        tile_v_r sqrsum, mean, denom, rms;
        auto gi = gI(Mb, 0);
        auto go = gO(Mb, 0);
        TLOAD(src_h, gi);
        TCVT(src, src_h);
        TMUL(squared, src, src);
        TROWSUM(sqrsum, squared);
        TMULS(mean, sqrsum, inv_r);
        TADDS(denom, mean, eps);
        rms_detail::rsqrt_newton(rms, denom);
        TROWEXPANDMUL(dst, src, rms);
        TCVT(dst_h, dst);
        TSTORE(go, dst_h);
    }
}

#endif // SUPERNPU_RMS_NORM_PTO_HPP
