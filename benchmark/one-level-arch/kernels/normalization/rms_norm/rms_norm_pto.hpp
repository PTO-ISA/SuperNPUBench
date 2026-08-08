// =============================================================================
// rms_norm_pto.hpp — RMSNorm (one-level PTO)
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
    // Dynamic Valid: TEPL reads GetValid* from src0. Temps must carry ValidRow.
    const size_t vr = static_cast<size_t>(a.GetValidRow());
    TileVec x(vr), t1(vr), t2(vr);
    TRECIP(x, a);
    for (int64_t i = 0; i < 4; ++i) {
        TMUL(t1, x, x);
        TMUL(t2, t1, a);
        TMULS(t2, t2, -0.5f);
        TADDS(t2, t2, 1.5f);
        TMUL(x, x, t2);
    }
    TMULS(out, x, 1.0f);
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
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t gA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_a = tiling[2] > 0 ? tiling[2] : tA;
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : gR;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    // ValidCol=1; Cols=128 → 512B (TileOP IsValidActiveSize / TSize=1..7).
    using tile_v = Tile<Location::Vec, float, tA, 128, BLayout::RowMajor, -1, 1>;

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

#endif // SUPERNPU_RMS_NORM_PTO_HPP
