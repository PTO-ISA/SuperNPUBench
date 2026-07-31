// =============================================================================
// rms_norm_pto.hpp — RMSNorm (one-level PTO), single N-tile
// =============================================================================
//
//   out[row] = x[row] * rsqrt(mean(x[row]^2) + eps)
//
// Entry:
//   rms_norm<dtype>(x, tiling, out, eps);
//   tiling[4] = {g_m, g_n, tile_m, tile_n}  (tile_* <=0 → capacity)
//
// Requires g_n <= tile_n. Dynamic Valid for M/N tails:
//   using tile_h = Tile<..., tM, tN, ..., -1, -1>;
//   tile_h src_h(active_row, active_col);
//   TCOPYIN(src_h, gi);
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_PTO_HPP
#define SUPERNPU_RMS_NORM_PTO_HPP

#include "normalization/rms_norm_dyn_ops.hpp"

#include <cstdint>

// tiling: [g_m, g_n, tile_m, tile_n]
template <typename dtype>
void rms_norm(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    constexpr int64_t tM = 1;
    constexpr int64_t tN = 1024;

    const int64_t gM = tiling[0];
    const int64_t gN = tiling[1];
    const int64_t tile_m = tiling[2] > 0 ? tiling[2] : tM;
    const int64_t tile_n = tiling[3] > 0 ? tiling[3] : tN;
    const float inv_n = 1.0f / static_cast<float>(gN);

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, tM, 8, BLayout::RowMajor, -1, 1>;

    for (int64_t i = 0; i < gM; i += tile_m) {
        const int64_t active_row = rms_dyn::min64(tile_m, gM - i);
        const int64_t active_col = gN;

        const size_t offset =
            static_cast<size_t>(i) * static_cast<size_t>(gN);
        gm_t gi(x + offset, static_cast<int>(gM), static_cast<int>(gN));
        gm_t go(out + offset, static_cast<int>(gM), static_cast<int>(gN));

        tile_h src_h(static_cast<size_t>(active_row),
                     static_cast<size_t>(active_col));
        tile_h dst_h(static_cast<size_t>(active_row),
                     static_cast<size_t>(active_col));
        tile_f src(static_cast<size_t>(active_row),
                   static_cast<size_t>(active_col));
        tile_f squared(static_cast<size_t>(active_row),
                       static_cast<size_t>(active_col));
        tile_f dst(static_cast<size_t>(active_row),
                   static_cast<size_t>(active_col));
        tile_v sqrsum(static_cast<size_t>(active_row));
        tile_v mean(static_cast<size_t>(active_row));
        tile_v denom(static_cast<size_t>(active_row));
        tile_v rms(static_cast<size_t>(active_row));

        TCOPYIN(src_h, gi);
        rms_dyn::tcvt_rm(src, src_h);
        rms_dyn::tmul(squared, src, src);
        rms_dyn::trowsum(sqrsum, squared);
        rms_dyn::tmuls(mean, sqrsum, inv_n);
        rms_dyn::tadds(denom, mean, eps);
        rms_dyn::rsqrt_newton(rms, denom);
        rms_dyn::trowexpandmul(dst, src, rms);
        rms_dyn::tcvt_rm(dst_h, dst);
        TCOPYOUT(go, dst_h);
    }
}

#endif // SUPERNPU_RMS_NORM_PTO_HPP
