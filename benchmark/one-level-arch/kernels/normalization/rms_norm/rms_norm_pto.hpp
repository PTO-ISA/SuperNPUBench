// =============================================================================
// rms_norm_pto.hpp — RMSNorm (one-level PTO)
// =============================================================================
//
// Shape dims: A (outer / row), R (reduce / col).
//
//   out[a] = x[a] * rsqrt(mean(x[a]^2) + eps)
//
// Entry:
//   rms_norm<dtype, peNum>(x, tiling, out, eps);
//   peNum defaults to 1; PE partitioning stays inside the kernel.
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
    auto body = [&](auto &x, auto &t1, auto &t2) {
        TRECIP(x, a);
        for (int64_t i = 0; i < 4; ++i) {
            TMUL(t1, x, x);
            TMUL(t2, t1, a);
            TMULS(t2, t2, -0.5f);
            TADDS(t2, t2, 1.5f);
            TMUL(x, x, t2);
        }
        TMULS(out, x, 1.0f);
    };
    if constexpr (TileVec::ValidRow > 0) {
        TileVec x, t1, t2;
        body(x, t1, t2);
    } else {
        const size_t vr = static_cast<size_t>(a.GetValidRow());
        TileVec x(vr), t1(vr), t2(vr);
        body(x, t1, t2);
    }
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
template <typename dtype, int peNum>
void rms_norm(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");

    // Physical capacity (Rows×Cols); Valid comes from tiling (tile_a,tile_r).
    // Size must cover ValidRow×ValidCol; SoftCore should not require Rows≥ValidRow.
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 8192;

    const int64_t globalA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t peA = globalA / peNum;
    const int64_t tile_a = tiling[2] > 0 ? tiling[2] : tA;
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : gR;
    const uint32_t tid = get_thread_idx();
    const int64_t pe_offset = static_cast<int64_t>(tid) * peA * gR;

    // The runtime tiling is host data, so divisibility cannot be a static
    // assertion. Invalid configurations do no work rather than overlap rows.
    if (globalA <= 0 || globalA % peNum != 0 || peA < tile_a) {
        return;
    }
    x += pe_offset;
    out += pe_offset;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    // Row-reduction output and row-broadcast input use physical Columns=1.
    using tile_v = Tile<Location::Vec, float, tA, 1, BLayout::RowMajor, -1, 1>;

    const float inv_r = 1.0f / static_cast<float>(gR);

    // Full A tiles; peel the last iteration for the trailing block.
    int64_t ia = 0;
    for (; ia + tile_a < peA; ia += tile_a) {
        rms_detail::rms_norm_tile<dtype, gm_t, tile_h, tile_f, tile_v>(
            x, out, peA, gR, ia, tile_a, tile_r, inv_r, eps);
    }
    // Tail (or sole) block: ValidRow = remaining rows along A.
    rms_detail::rms_norm_tile<dtype, gm_t, tile_h, tile_f, tile_v>(
        x, out, peA, gR, ia, peA - ia, tile_r, inv_r, eps);
}

#endif // SUPERNPU_RMS_NORM_PTO_HPP
