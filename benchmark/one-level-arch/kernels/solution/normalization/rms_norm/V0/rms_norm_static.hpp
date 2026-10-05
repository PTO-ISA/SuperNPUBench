// rms_norm_static: [128,8192].
// Fixed-shape 4PE implementation with compile-time Tile valid dimensions.
// Kernel entry points do not accept runtime tiling. Dynamic counterpart is unchanged.
#ifndef SUPERNPU_RMS_NORM_PTO_STATIC_HPP
#define SUPERNPU_RMS_NORM_PTO_STATIC_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail_static {

constexpr float kEpsilon = 1e-6f;

template <typename TileVec>
__attribute__((always_inline)) inline void rsqrt_regbase(TileVec &out, TileVec &a) {
    auto body = [&](auto &recip, auto &y, auto &tmp) __attribute__((always_inline)) {
        // Match the regbase formula: y=sqrt(1/a), one Newton step, then a
        // compensated residual correction using the original reciprocal.
        TRECIP(recip, a);
        TSQRT(y, recip);

        TMULS(tmp, a, -0.5f);
        TMUL(tmp, tmp, y);
        TMUL(tmp, tmp, y);
        TADDS(tmp, tmp, 1.5f);
        TMUL(y, y, tmp);

        // residual = (1 - a*recip) + a*(recip - y*y)
        TMULS(tmp, a, -1.0f);
        TMUL(tmp, tmp, recip);
        TADDS(out, tmp, 1.0f);
        TMULS(tmp, y, -1.0f);
        TMUL(tmp, tmp, y);
        TADD(tmp, recip, tmp);
        TMUL(tmp, a, tmp);
        TADD(out, out, tmp);
        TMUL(out, out, y);
        TMULS(out, out, 0.5f);
        TADD(out, y, out);
    };
    if constexpr (TileVec::ValidRow > 0) {
        TileVec recip, y, tmp;
        body(recip, y, tmp);
    } else {
        const size_t vr = static_cast<size_t>(a.GetValidRow());
        TileVec recip, y, tmp;
        body(recip, y, tmp);
    }
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_v>
inline void rms_norm_tile_static(dtype *x, const dtype *gamma, dtype *out,
                          int64_t gA, int64_t gR,
                          int64_t a_off, int64_t active_a, int64_t active_r,
                          float inv_r) {
    // Reduce the entire row using <=2 KiB FP32 strips, then normalize.
    const int64_t offset = a_off * gR;
    tile_v sum, partial, mean, denom, rms;
    TEXPANDS(sum, 0.0f);
    for (int64_t col = 0; col < gR; col += active_r) {
        const size_t width = gR-col < active_r ? gR-col : active_r;
        gm_t gi(x+offset+col, 1, static_cast<int>(gR));
        tile_h h;
        tile_f src, squared;
        TLOAD(h, gi); TCVT(src, h);
        TMUL(squared, src, src);
        using reduce_row = Tile<Location::Vec, float, 1, tile_f::Cols,
                                BLayout::CubeM32, 1, 1>;
        reduce_row row_sum;
        TROWSUM(row_sum, squared);
        TCOLSUM(partial, row_sum);
        TADD(sum, sum, partial);
    }
    TMULS(mean, sum, inv_r);
    TADDS(denom, mean, rms_detail_static::kEpsilon);
    rsqrt_regbase(rms, denom);
    for (int64_t col = 0; col < gR; col += active_r) {
        const size_t width = gR-col < active_r ? gR-col : active_r;
        gm_t gi(x+offset+col, 1, static_cast<int>(gR));
        gm_t gg(const_cast<dtype *>(gamma)+col, 1, static_cast<int>(gR));
        gm_t go(out+offset+col, 1, static_cast<int>(gR));
        tile_h h, gamma_h;
        tile_f src, normalized, gamma_f, dst;
        TLOAD(h, gi); TCVT(src, h);
        TLOAD(gamma_h, gg); TCVT(gamma_f, gamma_h);
        TROWEXPANDMUL(normalized, src, rms);
        TMUL(dst, normalized, gamma_f);
        TCVT(h, dst); TSTORE(go, h);
    }
}

} // namespace rms_detail_static

// Fixed shape [128,8192].
template <typename dtype, int peNum>
void rms_norm_static(dtype *x, const dtype *gamma, dtype *out) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");

    // Physical and valid Tile shapes are compile-time constants.
    // Size must cover ValidRow×ValidCol; SoftCore should not require Rows≥ValidRow.
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 512;

    constexpr int64_t globalA = 128;
    constexpr int64_t gR = 8192;
    constexpr int64_t tile_a = 1;
    constexpr int64_t tile_r = 512;
    const uint32_t tid = get_thread_idx();

    if (globalA <= 0 || gR <= 0 || tile_a != 1 || tile_r <= 0 ||
        tile_r > tR || tid >= static_cast<uint32_t>(peNum)) {
        return;
    }

    // Ceil partition: the first PEs take rows_per_pe rows and the last
    // active PE owns the remainder. For M=333 and 4PE: 84, 84, 84, 81.
    const int64_t rows_per_pe = (globalA + peNum - 1) / peNum;
    const int64_t pe_start = static_cast<int64_t>(tid) * rows_per_pe;
    if (pe_start >= globalA) {
        return;
    }
    const int64_t remaining = globalA - pe_start;
    const int64_t peA =
        remaining < rows_per_pe ? remaining : rows_per_pe;
    if (peA < tile_a) {
        return;
    }
    const int64_t pe_offset = pe_start * gR;
    x += pe_offset;
    out += pe_offset;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::CubeM32, 1, 512>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::CubeM32, 1, 512>;
    // TCOLSUM materializes the scalar, as in R-tree/R-simt.
    using tile_v = Tile<Location::Vec, float, tA, tR, BLayout::CubeM32, 1, 1>;

    const float inv_r = 1.0f / static_cast<float>(gR);

    // Full A tiles; peel the last iteration for the trailing block.
    int64_t ia = 0;
    for (; ia + tile_a < peA; ia += tile_a) {
        rms_detail_static::rms_norm_tile_static<dtype, gm_t, tile_h, tile_f, tile_v>(
            x, gamma, out, peA, gR, ia, tile_a, tile_r, inv_r);
    }
    // Tail (or sole) block: ValidRow = remaining rows along A.
    rms_detail_static::rms_norm_tile_static<dtype, gm_t, tile_h, tile_f, tile_v>(
        x, gamma, out, peA, gR, ia, peA - ia, tile_r, inv_r);
}

#endif // SUPERNPU_RMS_NORM_PTO_STATIC_HPP
