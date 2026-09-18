// rms_norm_dynamic_m_R_tree: [512,8192].
// Fixed-shape 4PE implementation with R=[16,32,16], Tile=[32,16].
#ifndef SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_TREE_HPP
#define SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_TREE_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail_simt_dynamic_m_R_tree {

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

template <typename gm_t, typename tile_h, typename tile_f, typename parent_t>
inline void reduce_pairs(gm_t &base, int64_t gR, int64_t pair_count, int64_t tile_elems, int64_t tile_m, int64_t tile_r, parent_t &partial_matrix) {
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        gm_t g0(base.data() + pair * tile_elems, static_cast<int>(tile_m), static_cast<int>(tile_r));
        gm_t g1(base.data() + (pair + pair_count) * tile_elems, static_cast<int>(tile_m), static_cast<int>(tile_r));
        tile_h h0(tile_m, tile_r), h1(tile_m, tile_r);
        tile_f x0(tile_m, tile_r), x1(tile_m, tile_r), sq0(tile_m, tile_r), sq1(tile_m, tile_r), sq_pair(tile_m, tile_r);
        TLOAD(h0, g0); TCVT(x0, h0); TMUL(sq0, x0, x0);
        TLOAD(h1, g1); TCVT(x1, h1); TMUL(sq1, x1, x1);
        TADD(sq_pair, sq0, sq1);
        if (pair == 0) { auto rows = pto::range::assemble<1>(partial_matrix, pair); TROWSUM(rows, sq_pair); }
        else if (pair + 1 == pair_count) { auto rows = pto::range::assemble_last<1>(partial_matrix, pair); TROWSUM_ASS(rows, sq_pair); }
        else { auto rows = pto::range::assemble_middle<1>(partial_matrix, pair); TROWSUM_ASS(rows, sq_pair); }
    }
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_m_v, typename tile_m_matrix, typename tile_v,
          typename tile_s>
inline void rms_norm_tile(dtype *x, const dtype *gamma, dtype *out,
                                 int64_t gR, int64_t pair_count, int64_t a_off,
                                 int64_t tile_m, int64_t tile_r,
                                 float inv_r) {
    const int64_t offset = a_off * gR;
    const int64_t tile_elems = tile_m * tile_r;
    gm_t input_row(x + offset, 1, static_cast<int>(gR));

    // First pair outer-R blocks (0,8), (1,9), ... (7,15). Assemble the
    // eight reduced [32,1] results by columns into one [32,8] Tile.
    tile_m_matrix partial_matrix(tile_m, pair_count);
    reduce_pairs<gm_t, tile_h, tile_f, tile_m_matrix>(
        input_row, gR, pair_count, tile_elems, tile_m, tile_r, partial_matrix);

    tile_m_v sum_rows(tile_m, 1);
    TROWSUM(sum_rows, partial_matrix);
    tile_s tile_sum, mean, denom, rms;
    TCOLSUM(tile_sum, sum_rows);
    TMULS(mean, tile_sum, inv_r);
    TADDS(denom, mean, kEpsilon);
    rsqrt_regbase(rms, denom);

    tile_v rms_rows(tile_m, 1);
    TCOLEXPAND(rms_rows, rms);
    for (int64_t r = 0; r < gR; r += tile_elems) {
        gm_t gi(x + offset + r, static_cast<int>(tile_m),
                static_cast<int>(tile_r));
        gm_t gg(const_cast<dtype *>(gamma) + r,
                static_cast<int>(tile_m), static_cast<int>(tile_r));
        gm_t go(out + offset + r, static_cast<int>(tile_m),
                static_cast<int>(tile_r));
        tile_h h(tile_m, tile_r), gh(tile_m, tile_r);
        tile_f src(tile_m, tile_r), gf(tile_m, tile_r), normalized(tile_m, tile_r), dst(tile_m, tile_r) ;
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

} // namespace rms_detail_simt_dynamic_m_R_tree

template <typename dtype, int peNum, typename TilingData>
void rms_norm_dynamic_m_R_tree(dtype *x, const dtype *gamma, const TilingData *tiling, dtype *out) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");

    const int64_t globalA = tiling->g_a;
    const int64_t gR = tiling->g_r;
    const int64_t powR = tiling->powR;
    constexpr int64_t tile_m = 32;
    const int64_t tile_r = tiling->tile_r > 0 ? tiling->tile_r : (gR + 511) / 512;
    const int64_t tile_elems = tile_m * tile_r;
    const int64_t pair_count = powR / 512;

    constexpr int64_t kMaxReduceR = 8192;
    const uint32_t tid = get_thread_idx();
    if (globalA <= 0 || gR <= 0 || powR <= 0 || powR > gR || gR > kMaxReduceR || tile_r <= 0 || tile_r > 16 || gR % (tile_elems * 2) != 0 || tid >= static_cast<uint32_t>(peNum)) {
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
    for (int64_t ia = 0; ia < peA; ia += tile_m) {
        int64_t cur_tile_m = tile_m;
        if (ia + cur_tile_m > peA) {
            cur_tile_m = peA - ia;
        }
        rms_detail_simt_dynamic_m_R_tree::rms_norm_tile<
            dtype, gm_t, tile_h, tile_f, tile_m_v, tile_m_matrix, tile_v,
            tile_s>(
            x, gamma, out, gR, pair_count, ia, cur_tile_m, tile_r, inv_r);
    }
}

#endif // SUPERNPU_RMS_NORM_SIMT_DYNAMIC_M_R_TREE_HPP
