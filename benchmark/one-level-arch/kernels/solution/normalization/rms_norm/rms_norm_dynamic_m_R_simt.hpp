// rms_norm_dynamic_m_R_simt: default test shape [128,8192].
// Fixed-shape 4PE implementation with R=[16,32,16], Tile=[32,16].
#ifndef SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_SIMT_HPP
#define SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_SIMT_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail_simt_dynamic_m_R_simt {

constexpr float kEpsilon = 1e-6f;

template <typename TileVec>
__attribute__((always_inline)) inline void rsqrt_regbase(TileVec &out,
                                                         TileVec &a) {
    auto body = [&](auto &recip, auto &y, auto &tmp)
                    __attribute__((always_inline)) {
        TRECIP(recip, a);
        TSQRT(y, recip);

        TMULS(tmp, a, -0.5f);
        TMUL(tmp, tmp, y);
        TMUL(tmp, tmp, y);
        TADDS(tmp, tmp, 1.5f);
        TMUL(y, y, tmp);

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
    TileVec recip, y, tmp;
    body(recip, y, tmp);
}

template <typename gm_t, typename tile_h, typename tile_f, typename tile_m_v>
inline void reduce_sequential(gm_t &base, int64_t pair_count,
                              int64_t tile_elems, int64_t tile_m,
                              int64_t tile_r, tile_m_v &sum_rows) {
    tile_f accum(tile_m, tile_r);
    TEXPANDS(accum, 0.0f);
    const int64_t block_count = pair_count * 2;
    for (int64_t block = 0; block < block_count; ++block) {
        gm_t g(base.data() + block * tile_elems, static_cast<int>(tile_m),
               static_cast<int>(tile_r));
        tile_h h(tile_m, tile_r);
        tile_f x(tile_m, tile_r), squared(tile_m, tile_r);
        TLOAD(h, g);
        TCVT(x, h);
        TMUL(squared, x, x);
        TADD(accum, accum, squared);
    }
    TROWSUM(sum_rows, accum);
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_m_v, typename tile_m_matrix, typename tile_v,
          typename tile_s>
inline void rms_norm_tile(dtype *x, const dtype *gamma, dtype *out,
                                 int64_t gR, int64_t pair_count, int64_t a_off,
                                 int64_t tile_r, float inv_r) {
    const int64_t offset = a_off * gR;
    // tile_r is the number of elements in one linear R block. Map that block
    // onto the physical [32,16] Tile without sharing the outer-A tile size.
    const int64_t curtile_factal_r = (tile_r + 31) / 32;
    const int64_t curtile_factal_a =
        (tile_r + curtile_factal_r - 1) / curtile_factal_r;
    gm_t input_row(x + offset, 1, static_cast<int>(gR));

    // Sequentially reduce every R block into one row-sum vector.
    // ValidCol is static: the one-argument constructor sets ValidRow.
    tile_m_v sum_rows(curtile_factal_a);
    reduce_sequential<gm_t, tile_h, tile_f, tile_m_v>(
        input_row, pair_count, tile_r, curtile_factal_a,
        curtile_factal_r, sum_rows);
    tile_s tile_sum, mean, denom, rms;
    TCOLSUM(tile_sum, sum_rows);
    TMULS(mean, tile_sum, inv_r);
    TADDS(denom, mean, kEpsilon);
    rsqrt_regbase(rms, denom);

    tile_v rms_rows(curtile_factal_a);
    TCOLEXPAND(rms_rows, rms);
    for (int64_t r = 0; r < gR; r += tile_r) {
        gm_t gi(x + offset + r, static_cast<int>(curtile_factal_a),
                static_cast<int>(curtile_factal_r));
        gm_t gg(const_cast<dtype *>(gamma) + r,
                static_cast<int>(curtile_factal_a), static_cast<int>(curtile_factal_r));
        gm_t go(out + offset + r,
            static_cast<int>(curtile_factal_a), static_cast<int>(curtile_factal_r));
        tile_h h(curtile_factal_a, curtile_factal_r), gh(curtile_factal_a, curtile_factal_r);
        tile_f src(curtile_factal_a, curtile_factal_r), gf(curtile_factal_a, curtile_factal_r), normalized(curtile_factal_a, curtile_factal_r), dst(curtile_factal_a, curtile_factal_r) ;
        TLOAD(h, gi);
        TCVT(src, h);
        TLOAD(gh, gg);
        TCVT(gf, gh);
        TROWEXPANDMUL(normalized, src, rms_rows);
        TMUL(dst, normalized, gf);
        TCVT(h, dst);
        TSTORE(go, h);
    }
}

} // namespace rms_detail_simt_dynamic_m_R_simt

template <typename dtype, int peNum, typename TilingData>
void rms_norm_dynamic_m_R_simt(dtype *x, const dtype *gamma, const TilingData *tiling, dtype *out) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");

    const int64_t globalA = tiling->g_a;
    const int64_t gR = tiling->g_r;
    const int64_t powR = tiling->powR;
    const int64_t tile_r = tiling->tile_r > 0 ? tiling->tile_r : (gR + 511) / 512;
    const int64_t pair_count = tile_r > 0 ? powR / tile_r : 0;

    constexpr int64_t kMaxReduceR = 8192;
    const uint32_t tid = get_thread_idx();
    if (globalA <= 0 || gR <= 0 || powR <= 0 || powR > gR ||
        gR > kMaxReduceR || tile_r <= 0 || tile_r > 512 ||
        pair_count <= 0 || gR % (tile_r * 2) != 0 ||
        pair_count * tile_r * 2 != gR ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }
    const int64_t rows_per_pe = (globalA + peNum - 1) / peNum;
    const int64_t pe_start = static_cast<int64_t>(tid) * rows_per_pe;
    if (pe_start >= globalA) {
        return;
    }
    const int64_t remaining = globalA - pe_start;
    const int64_t peA =
        remaining < rows_per_pe ? remaining : rows_per_pe;
    x += pe_start * gR;
    out += pe_start * gR;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, 32, 16,
                        BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, 32, 16,
                        BLayout::RowMajor, -1, -1>;
    using tile_m_v = Tile<Location::Vec, float, 32, 1,
                          BLayout::RowMajor, -1, 1>;
    using tile_m_matrix = Tile<Location::Vec, float, 32, 8,
                               BLayout::RowMajor, -1, 8>;
    using tile_v = Tile<Location::Vec, float, 32, 1,
                        BLayout::RowMajor, -1, 1>;
    using tile_s = Tile<Location::Vec, float, 1, 1,
                        BLayout::RowMajor, 1, 1>;

    const float inv_r = 1.0f / static_cast<float>(gR);
    for (int64_t ia = 0; ia < peA; ++ia) {
        rms_detail_simt_dynamic_m_R_simt::rms_norm_tile<
            dtype, gm_t, tile_h, tile_f, tile_m_v, tile_m_matrix, tile_v,
            tile_s>(
            x, gamma, out, gR, pair_count, ia, tile_r, inv_r);
    }
}

#endif // SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_SIMT_HPP
