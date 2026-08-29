#pragma once

#include "single_thread/normalization/rms_norm/rms_norm.hpp"

// Compile-time four-PE RMSNorm. The A dimension is divided equally across
// PEs; each PE processes one contiguous row range and writes its own output.
template <typename dtype, int peA, int gA, int gR, int tA, int tR>
void rms_norm(dtype *x, dtype *out, float eps = 1e-6f) {
    static_assert(gA > 0 && gR > 0 && tA > 0 && tR > 0);
    static_assert(peA > 0 && gA % peA == 0, "gA must be divisible by peA");
    static_assert(peA >= tA, "peA must cover one tile_a");
    static_assert(tA <= 128, "RMSNorm row-state carrier supports tA <= 128");
    static_assert(tR == gR, "static rms_norm is a single R-tile; use rms_norm_binary for R-split");
    constexpr int Mb = peA / tA;
    constexpr int rmd_A = peA % tA;
    constexpr float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<peA, gR>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor>;
    using tile_v =
        Tile<Location::Vec, float, 128, 1, BLayout::RowMajor, tA, 1>;
    using it_t = global_iterator<gm_t, tile_h>;

    const uint32_t tid = get_thread_idx();
    const uint32_t gm_offset = tid * static_cast<uint32_t>(peA * gR);

    it_t gI(x + gm_offset);
    it_t gO(out + gm_offset);

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
        using tile_h_r = Tile<Location::Vec, dtype, tA, tR,
                            BLayout::RowMajor, rmd_A, tR>;
        using tile_f_r = Tile<Location::Vec, float, tA, tR,
                            BLayout::RowMajor, rmd_A, tR>;
        using tile_v_r = Tile<Location::Vec, float, 128, 1,
                            BLayout::RowMajor, rmd_A, 1>;
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
