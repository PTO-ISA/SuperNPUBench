// rms_norm_dynamic_tree_32k: default test shapes [128,8192] and [32,16384].
// Dynamic 4PE implementation with FP32 Tile=[32,256] (32 KiB).
// Full 8192-element R blocks; up to 16 blocks. Original 2 KiB kernel is unchanged.
// R == 8192: two 4096-element halves in 16 KiB tiles, squares added, then one reduction.
// R == 16384: two 8192-element blocks in one 32 KiB tile, then the same reduction.
// R >  16384: paired blocks, partial row sums spilled to the workspace.
#ifndef SUPERNPU_RMS_NORM_SIMT_DYNAMIC_TREE_32K_HPP
#define SUPERNPU_RMS_NORM_SIMT_DYNAMIC_TREE_32K_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail_simt_dynamic_tree_32k {

constexpr float kEpsilon = 1e-6f;
constexpr int64_t kTileM = 32;
constexpr int64_t kMaxPairCount = 8;

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

template <typename gm_t, typename gm_f_col, typename tile_h, typename tile_f,
          typename tile_m_v>
inline void reduce_pairs(gm_t &base, int64_t gR, int64_t pair_count,
                         int64_t tile_elems, int64_t tile_m,
                         int64_t tile_r, float *partial_workspace) {
    // Two in-range pairs per iteration. Same stage on both pairs runs back
    // to back: square the low block, square the high block, then reduce and
    // store. pair_count can be odd, so the last pair is reduced alone.
    for (int64_t pair = 0; pair + 1 < pair_count; pair += 2) {
        const int64_t p0 = pair;
        const int64_t p1 = pair + 1;
        gm_t g0_0(base.data() + p0 * tile_elems, static_cast<int>(tile_m),
                  static_cast<int>(tile_r));
        gm_t g1_0(base.data() + (p0 + pair_count) * tile_elems,
                  static_cast<int>(tile_m), static_cast<int>(tile_r));
        gm_t g0_1(base.data() + p1 * tile_elems, static_cast<int>(tile_m),
                  static_cast<int>(tile_r));
        gm_t g1_1(base.data() + (p1 + pair_count) * tile_elems,
                  static_cast<int>(tile_m), static_cast<int>(tile_r));
        tile_h h0_0(tile_m, tile_r), h1_0(tile_m, tile_r);
        tile_h h0_1(tile_m, tile_r), h1_1(tile_m, tile_r);
        tile_f x0_0(tile_m, tile_r), x1_0(tile_m, tile_r), sq0_0(tile_m, tile_r),
            sq1_0(tile_m, tile_r), sqp_0(tile_m, tile_r);
        tile_f x0_1(tile_m, tile_r), x1_1(tile_m, tile_r), sq0_1(tile_m, tile_r),
            sq1_1(tile_m, tile_r), sqp_1(tile_m, tile_r);

        TLOAD(h0_0, g0_0);
        TCVT(x0_0, h0_0);
        TMUL(sq0_0, x0_0, x0_0);
        TLOAD(h0_1, g0_1);
        TCVT(x0_1, h0_1);
        TMUL(sq0_1, x0_1, x0_1);

        if ((p0 + pair_count) * tile_elems < gR) {
            TLOAD(h1_0, g1_0);
            TCVT(x1_0, h1_0);
            TMUL(sq1_0, x1_0, x1_0);
        } else {
            TEXPANDS(sq1_0, 0.0f);
        }
        if ((p1 + pair_count) * tile_elems < gR) {
            TLOAD(h1_1, g1_1);
            TCVT(x1_1, h1_1);
            TMUL(sq1_1, x1_1, x1_1);
        } else {
            TEXPANDS(sq1_1, 0.0f);
        }

        // ValidCol is static: (value, row) would select the fill constructor.
        // Each TSTORE writes one column. The source valid shape [tile_m, 1]
        // limits each row to one element.
        TADD(sqp_0, sq0_0, sq1_0);
        tile_m_v rows_0(tile_m);
        TROWSUM(rows_0, sqp_0);
        gm_f_col partial_out_0(partial_workspace + p0);
        TSTORE(partial_out_0, rows_0);

        TADD(sqp_1, sq0_1, sq1_1);
        tile_m_v rows_1(tile_m);
        TROWSUM(rows_1, sqp_1);
        gm_f_col partial_out_1(partial_workspace + p1);
        TSTORE(partial_out_1, rows_1);
    }
    if ((pair_count & 1) != 0) {
        const int64_t pair = pair_count - 1;
        gm_t g0(base.data() + pair * tile_elems, static_cast<int>(tile_m),
                static_cast<int>(tile_r));
        gm_t g1(base.data() + (pair + pair_count) * tile_elems,
                static_cast<int>(tile_m), static_cast<int>(tile_r));
        tile_h h0(tile_m, tile_r), h1(tile_m, tile_r);
        tile_f x0(tile_m, tile_r), x1(tile_m, tile_r), sq0(tile_m, tile_r),
            sq1(tile_m, tile_r), sqp(tile_m, tile_r);
        TLOAD(h0, g0);
        TCVT(x0, h0);
        TMUL(sq0, x0, x0);
        if ((pair + pair_count) * tile_elems < gR) {
            TLOAD(h1, g1);
            TCVT(x1, h1);
            TMUL(sq1, x1, x1);
        } else {
            TEXPANDS(sq1, 0.0f);
        }
        TADD(sqp, sq0, sq1);
        tile_m_v rows(tile_m);
        TROWSUM(rows, sqp);
        gm_f_col partial_out(partial_workspace + pair);
        TSTORE(partial_out, rows);
    }
}

template <typename dtype, typename gm_t, typename gm_f_col,
          typename gm_f_matrix, typename tile_h, typename tile_f,
          typename tile_m_v, typename tile_m_matrix, typename tile_v,
          typename tile_s>
inline void rms_norm_tile(dtype *x, const dtype *gamma, dtype *out,
                          float *workspace, int64_t gR, int64_t pair_count,
                          int64_t a_off, int64_t tile_r, float inv_r) {
    const int64_t offset = a_off * gR;
    // tile_r is the number of elements in one linear R block. Map that block
    // onto the physical [32,256] Tile independently from the outer A axis.
    const int64_t curtile_factal_r = (tile_r + 31) / 32;
    const int64_t curtile_factal_a =
        (tile_r + curtile_factal_r - 1) / curtile_factal_r;
    gm_t input_row(x + offset, 1, static_cast<int>(gR));

    // Pair the first and second halves of the full R blocks. Spill up to eight
    // reduced [32,1] results to GM, then load them as one [32,8] Tile for the
    // second row reduction.
    // Each PE processes rows sequentially; reuse its private scratch matrix.
    float *partial_workspace = workspace;
    // Define unused pair slots as zero before reusing this PE-private matrix.
    // The second reduction consumes its full physical [32,8] shape.
    gm_f_matrix partial_gm(partial_workspace);
    tile_m_matrix zeros(curtile_factal_a, kMaxPairCount);
    TEXPANDS(zeros, 0.0f);
    TSTORE(partial_gm, zeros);
    reduce_pairs<gm_t, gm_f_col, tile_h, tile_f, tile_m_v>(
        input_row, gR, pair_count, tile_r, curtile_factal_a,
        curtile_factal_r, partial_workspace);

    tile_m_matrix partial_matrix(curtile_factal_a, kMaxPairCount);
    TLOAD(partial_matrix, partial_gm);

    // M32 row reductions retain the source physical columns.
    using tile_sum_rows = Tile<Location::Vec, float, 32, 8,
                               BLayout::CubeM32, -1, 1>;
    tile_sum_rows sum_rows(curtile_factal_a);
    TROWSUM(sum_rows, partial_matrix);
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
                static_cast<int>(curtile_factal_a),
                static_cast<int>(curtile_factal_r));
        gm_t go(out + offset + r, static_cast<int>(curtile_factal_a),
                static_cast<int>(curtile_factal_r));
        tile_h h(curtile_factal_a, curtile_factal_r),
            gh(curtile_factal_a, curtile_factal_r);
        tile_f src(curtile_factal_a, curtile_factal_r),
            gf(curtile_factal_a, curtile_factal_r),
            normalized(curtile_factal_a, curtile_factal_r),
            dst(curtile_factal_a, curtile_factal_r);
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

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_m_v, typename tile_s>
inline void rms_norm_tile_16k(dtype *x, const dtype *gamma, dtype *out,
                              int64_t gR, int64_t peA, float inv_r) {
    // [32,128] FP32 is 16 KiB and holds 4096 elements. R == 8192 is two
    // halves; their squares are added before one reduction. Gamma is the
    // same on every row, so both halves are loaded once.
    constexpr int fa = 32;
    constexpr int fr = 128;
    constexpr int64_t kSpan = kTileM * fr;
    gm_t gg0(const_cast<dtype *>(gamma), fa, fr);
    gm_t gg1(const_cast<dtype *>(gamma) + kSpan, fa, fr);
    tile_f gf0(fa, fr), gf1(fa, fr);
    {
        tile_h gh0(fa, fr), gh1(fa, fr);
        TLOAD(gh0, gg0);
        TCVT(gf0, gh0);
        TLOAD(gh1, gg1);
        TCVT(gf1, gh1);
    }
    for (int64_t ia = 0; ia < peA; ++ia) {
        const int64_t offset = ia * gR;
        gm_t gi0(x + offset, fa, fr);
        gm_t gi1(x + offset + kSpan, fa, fr);
        gm_t go0(out + offset, fa, fr);
        gm_t go1(out + offset + kSpan, fa, fr);
        tile_h h0(fa, fr), h1(fa, fr);
        tile_f x0(fa, fr), x1(fa, fr), sq0(fa, fr), sq1(fa, fr);
        TLOAD(h0, gi0);
        TCVT(x0, h0);
        TMUL(sq0, x0, x0);
        TLOAD(h1, gi1);
        TCVT(x1, h1);
        TMUL(sq1, x1, x1);
        TADD(sq0, sq0, sq1);

        tile_m_v row_sums(fa);
        TROWSUM(row_sums, sq0);
        // TCOLSUM still needs the wide carrier. The scalar lives in its first
        // 128 B cell; the prefix view exposes that cell as [32,1].
        using tile_sum_wide =
            Tile<tile_s::Loc, typename tile_s::DType, tile_s::Rows,
                 tile_m_v::Cols, tile_s::BFractal, 1, 1>;
        tile_sum_wide sum_wide;
        TCOLSUM(sum_wide, row_sums);
        auto tile_sum = TREDUCEPREFIXVIEW<tile_s>(sum_wide);
        tile_s mean, denom, rms;
        TMULS(mean, tile_sum, inv_r);
        TADDS(denom, mean, kEpsilon);
        rsqrt_regbase(rms, denom);

        tile_m_v rms_rows(fa);
        TCOLEXPAND(rms_rows, rms);
        tile_h h_out0(fa, fr), h_out1(fa, fr);
        tile_f dst(fa, fr);
        TROWEXPANDMUL(dst, x0, rms_rows);
        TMUL(dst, dst, gf0);
        TCVT(h_out0, dst);
        TSTORE(go0, h_out0);
        TROWEXPANDMUL(dst, x1, rms_rows);
        TMUL(dst, dst, gf1);
        TCVT(h_out1, dst);
        TSTORE(go1, h_out1);
    }
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_m_v, typename tile_s>
inline void rms_norm_tile_single_block(dtype *x, const dtype *gamma,
                                       dtype *out, int64_t gR, int64_t peA,
                                       float inv_r) {
    // One physical [32,256] tile holds 8192 elements. A 16384-row is two
    // blocks; their squares are added before this same reduction.
    constexpr int64_t kSpan = kTileM * 256;
    const int64_t span = gR > kSpan ? kSpan : gR;
    const int64_t curtile_factal_r = (span + 31) / 32;
    const int64_t curtile_factal_a =
        (span + curtile_factal_r - 1) / curtile_factal_r;
    if (gR > kSpan) {
        const int fa = static_cast<int>(curtile_factal_a);
        const int fr = static_cast<int>(curtile_factal_r);
        for (int64_t ia = 0; ia < peA; ++ia) {
            const int64_t offset = ia * gR;
            gm_t gi0(x + offset, fa, fr);
            gm_t gi1(x + offset + span, fa, fr);
            gm_t gg0(const_cast<dtype *>(gamma), fa, fr);
            gm_t gg1(const_cast<dtype *>(gamma) + span, fa, fr);
            gm_t go0(out + offset, fa, fr);
            gm_t go1(out + offset + span, fa, fr);
            tile_h h0(curtile_factal_a, curtile_factal_r),
                h1(curtile_factal_a, curtile_factal_r);
            tile_f x0(curtile_factal_a, curtile_factal_r),
                x1(curtile_factal_a, curtile_factal_r),
                sq0(curtile_factal_a, curtile_factal_r),
                sq1(curtile_factal_a, curtile_factal_r);
            TLOAD(h0, gi0);
            TCVT(x0, h0);
            TMUL(sq0, x0, x0);
            TLOAD(h1, gi1);
            TCVT(x1, h1);
            TMUL(sq1, x1, x1);
            TADD(sq0, sq0, sq1);

            tile_m_v row_sums(curtile_factal_a);
            TROWSUM(row_sums, sq0);
            // TCOLSUM still needs the wide carrier. The scalar lives in its
            // first 128 B cell; the prefix view exposes that cell as [32,1].
            using tile_sum_wide =
                Tile<tile_s::Loc, typename tile_s::DType, tile_s::Rows,
                     tile_m_v::Cols, tile_s::BFractal, 1, 1>;
            tile_sum_wide sum_wide;
            TCOLSUM(sum_wide, row_sums);
            auto tile_sum = TREDUCEPREFIXVIEW<tile_s>(sum_wide);
            tile_s mean, denom, rms;
            TMULS(mean, tile_sum, inv_r);
            TADDS(denom, mean, kEpsilon);
            rsqrt_regbase(rms, denom);

            tile_m_v rms_rows(curtile_factal_a);
            TCOLEXPAND(rms_rows, rms);
            tile_h gh(curtile_factal_a, curtile_factal_r),
                h_out0(curtile_factal_a, curtile_factal_r),
                h_out1(curtile_factal_a, curtile_factal_r);
            tile_f gf(curtile_factal_a, curtile_factal_r),
                dst(curtile_factal_a, curtile_factal_r);
            TLOAD(gh, gg0);
            TCVT(gf, gh);
            TROWEXPANDMUL(dst, x0, rms_rows);
            TMUL(dst, dst, gf);
            TCVT(h_out0, dst);
            TSTORE(go0, h_out0);
            TLOAD(gh, gg1);
            TCVT(gf, gh);
            TROWEXPANDMUL(dst, x1, rms_rows);
            TMUL(dst, dst, gf);
            TCVT(h_out1, dst);
            TSTORE(go1, h_out1);
        }
        return;
    }
    // Gamma is identical for every row. Load it once here so the FP32 tile
    // and the per-row TMUL destination share this function's M carrier.
    gm_t gg(const_cast<dtype *>(gamma), static_cast<int>(curtile_factal_a),
            static_cast<int>(curtile_factal_r));
    tile_f gf(curtile_factal_a, curtile_factal_r);
    {
        tile_h gh(curtile_factal_a, curtile_factal_r);
        TLOAD(gh, gg);
        TCVT(gf, gh);
    }
    for (int64_t ia = 0; ia < peA; ++ia) {
        const int64_t offset = ia * gR;
        gm_t gi(x + offset, static_cast<int>(curtile_factal_a),
                static_cast<int>(curtile_factal_r));
        gm_t go(out + offset, static_cast<int>(curtile_factal_a),
                static_cast<int>(curtile_factal_r));
        // h_in and h_out are distinct Tiles: the next row's TLOAD must not WAW
        // against this row's in-flight TSTORE.
        tile_h h_in(curtile_factal_a, curtile_factal_r),
            h_out(curtile_factal_a, curtile_factal_r);
        tile_f src(curtile_factal_a, curtile_factal_r),
            sq(curtile_factal_a, curtile_factal_r),
            dst(curtile_factal_a, curtile_factal_r);
        TLOAD(h_in, gi);
        TCVT(src, h_in);
        TMUL(sq, src, src);

        // TCOLSUM still needs the wide carrier. The scalar lives in its first
        // 128 B cell; the prefix view exposes that cell as [32,1].
        tile_m_v row_sums(curtile_factal_a);
        TROWSUM(row_sums, sq);
        using tile_sum_wide =
            Tile<tile_s::Loc, typename tile_s::DType, tile_s::Rows,
                 tile_m_v::Cols, tile_s::BFractal, 1, 1>;
        tile_sum_wide sum_wide;
        TCOLSUM(sum_wide, row_sums);
        auto tile_sum = TREDUCEPREFIXVIEW<tile_s>(sum_wide);
        tile_s mean, denom, rms;
        TMULS(mean, tile_sum, inv_r);
        TADDS(denom, mean, kEpsilon);
        rsqrt_regbase(rms, denom);

        tile_m_v rms_rows(curtile_factal_a);
        TCOLEXPAND(rms_rows, rms);
        TROWEXPANDMUL(dst, src, rms_rows);
        TMUL(dst, dst, gf);
        TCVT(h_out, dst);
        TSTORE(go, h_out);
    }
}

} // namespace rms_detail_simt_dynamic_tree_32k

template <typename dtype, int peNum, typename TilingData>
void rms_norm_dynamic_tree_32k(dtype *x, const dtype *gamma,
                               const TilingData *tiling, dtype *out,
                               float *workspace) {
    const int64_t globalA = tiling->g_a;
    const int64_t gR = tiling->g_r;
    const int64_t tile_r = tiling->tile_r;
    // The PE split is computed on the host and carried by the tiling info;
    // the kernel only applies it.
    const int64_t rows_per_pe = tiling->rows_per_pe;
    const int64_t workspace_rows_per_pe = tiling->workspace_rows_per_pe;
    const int64_t block_count = gR / tile_r;
    const int64_t pair_count = (block_count + 1) / 2;

    const uint32_t tid = get_thread_idx();
    const int64_t pe_start = static_cast<int64_t>(tid) * rows_per_pe;
    if (pe_start >= globalA) {
        return;
    }
    const int64_t remaining = globalA - pe_start;
    const int64_t peA =
        remaining < rows_per_pe ? remaining : rows_per_pe;
    x += pe_start * gR;
    out += pe_start * gR;
    const float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, 32, 256,
                        BLayout::CubeM32, -1, -1>;
    using tile_f = Tile<Location::Vec, float, 32, 256,
                        BLayout::CubeM32, -1, -1>;
    using tile_m_v = Tile<Location::Vec, float, 32, 256,
                          BLayout::CubeM32, -1, 1>;

    if (gR <= tile_r) {
        using tile_h16 = Tile<Location::Vec, dtype, 32, 128,
                              BLayout::CubeM32, -1, -1>;
        using tile_f16 = Tile<Location::Vec, float, 32, 128,
                              BLayout::CubeM32, -1, -1>;
        using tile_m16 = Tile<Location::Vec, float, 32, 128,
                              BLayout::CubeM32, -1, 1>;
        using tile_s16 = Tile<Location::Vec, float, 32, 1,
                              BLayout::CubeM32, 1, 1>;
        rms_detail_simt_dynamic_tree_32k::rms_norm_tile_16k<
            dtype, gm_t, tile_h16, tile_f16, tile_m16, tile_s16>(
            x, gamma, out, gR, peA, inv_r);
        return;
    }
    if (gR <= tile_r * 2) {
        using tile_s32 = Tile<Location::Vec, float, 32, 1,
                              BLayout::CubeM32, 1, 1>;
        rms_detail_simt_dynamic_tree_32k::rms_norm_tile_single_block<
            dtype, gm_t, tile_h, tile_f, tile_m_v, tile_s32>(
            x, gamma, out, gR, peA, inv_r);
        return;
    }

    workspace += static_cast<int64_t>(tid) * workspace_rows_per_pe *
                 rms_detail_simt_dynamic_tree_32k::kMaxPairCount;

    using gm_f_matrix = global_tensor<
        float, RowMajor<rms_detail_simt_dynamic_tree_32k::kTileM,
                        rms_detail_simt_dynamic_tree_32k::kMaxPairCount>>;
    using gm_f_col = gm_f_matrix;
    using tile_m_matrix = Tile<Location::Vec, float, 32, 8,
                               BLayout::CubeM32, -1, -1>;
    using tile_v = Tile<Location::Vec, float, 32, 8,
                        BLayout::CubeM32, -1, 1>;
    using tile_s = Tile<Location::Vec, float, 1, 8,
                        BLayout::CubeM32, 1, 1>;

    for (int64_t ia = 0; ia < peA; ++ia) {
        rms_detail_simt_dynamic_tree_32k::rms_norm_tile<
            dtype, gm_t, gm_f_col, gm_f_matrix, tile_h, tile_f, tile_m_v,
            tile_m_matrix, tile_v, tile_s>(
            x, gamma, out, workspace, gR, pair_count, ia, tile_r, inv_r);
    }
}

#endif // SUPERNPU_RMS_NORM_SIMT_DYNAMIC_TREE_32K_HPP
