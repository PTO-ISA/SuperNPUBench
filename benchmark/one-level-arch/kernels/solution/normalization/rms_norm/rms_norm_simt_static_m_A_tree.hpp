// rms_norm_simt_static_m_A_tree: [512,8192].
// Fixed-shape 4PE implementation with a [32,16] FP32 compute Tile.
// Kernel entry points do not accept runtime tiling. Dynamic counterpart is unchanged.
#ifndef SUPERNPU_RMS_NORM_SIMT_STATIC_M_A_TREE_HPP
#define SUPERNPU_RMS_NORM_SIMT_STATIC_M_A_TREE_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_detail_simt_static_m_A_tree {

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

template <typename gm_t, typename tile_h, typename tile_f,
          typename tile_reduce_matrix>
__attribute__((always_inline)) inline void square_pair_reduce(
    gm_t &base, int64_t outer_index, int64_t pair_index,
    int64_t active_a, int64_t gR, int64_t tile_r,
    tile_reduce_matrix &level1) {
    // One outer group contains 16 adjacent pairs. Each pair covers two
    // [32,16] R slices, which are squared and added before the first row sum.
    const int64_t pair_base = (outer_index * 16 + pair_index) * 2;
    const int64_t r0 = pair_base * tile_r;
    const int64_t r1 = (pair_base + 1) * tile_r;
    gm_t g0(base.data() + r0, static_cast<int>(active_a),
            static_cast<int>(gR));
    gm_t g1(base.data() + r1, static_cast<int>(active_a),
            static_cast<int>(gR));
    tile_h h0, h1;
    tile_f x0, x1, sq0, sq1, sq_pair;
    TLOAD(h0, g0);
    TCVT(x0, h0);
    TMUL(sq0, x0, x0);
    TLOAD(h1, g1);
    TCVT(x1, h1);
    TMUL(sq1, x1, x1);
    TADD(sq_pair, sq0, sq1);
    if (pair_index == 0) {
        auto rows = pto::range::assemble<1>(level1, pair_index);
        TROWSUM(rows, sq_pair);
    } else if (pair_index == 15) {
        auto rows = pto::range::assemble_last<1>(level1, pair_index);
        TROWSUM_ASS(rows, sq_pair);
    } else {
        auto rows = pto::range::assemble_middle<1>(level1, pair_index);
        TROWSUM_ASS(rows, sq_pair);
    }
}

template <typename gm_t, typename tile_h, typename tile_f,
          typename tile_reduce_matrix>
__attribute__((always_inline)) inline void reduce_outer_group(
    gm_t &base, int64_t outer_index, int64_t active_a, int64_t gR,
    int64_t tile_r, tile_reduce_matrix &level2) {
    tile_reduce_matrix level1;
    for (int64_t pair = 0; pair < 16; ++pair) {
        square_pair_reduce<gm_t, tile_h, tile_f, tile_reduce_matrix>(
            base, outer_index, pair, active_a, gR, tile_r, level1);
    }

    // The second row sum reduces the 16 first-level results for each M row.
    // Assemble the 16 outer-group results for the final row sum.
    if (outer_index == 0) {
        auto rows = pto::range::assemble<1>(level2, outer_index);
        TROWSUM(rows, level1);
    } else if (outer_index == 15) {
        auto rows = pto::range::assemble_last<1>(level2, outer_index);
        TROWSUM_ASS(rows, level1);
    } else {
        auto rows = pto::range::assemble_middle<1>(level2, outer_index);
        TROWSUM_ASS(rows, level1);
    }
}

template <typename dtype, typename gm_t, typename tile_h, typename tile_f,
          typename tile_v, typename tile_reduce_matrix, typename gamma_h,
          typename gamma_f>
inline void rms_norm_tile_static(dtype *x, const dtype *gamma, dtype *out,
                                 int64_t gR, int64_t a_off,
                                 int64_t active_a, int64_t active_r,
                                 float inv_r) {
    // [32,16] holds 32 A rows and a 16-element R slice.
    const int64_t offset = a_off * gR;
    gm_t input_block(x + offset, static_cast<int>(active_a),
                     static_cast<int>(gR));
    tile_reduce_matrix level2;
    for (int64_t outer = 0; outer < 16; ++outer) {
        reduce_outer_group<gm_t, tile_h, tile_f, tile_reduce_matrix>(
            input_block, outer, active_a, gR, active_r, level2);
    }

    // Third and final row sum. M is the A axis, so all 32 rows remain
    // independent and there is deliberately no column reduction.
    tile_v sum, mean, denom, rms;
    TROWSUM(sum, level2);
    TMULS(mean, sum, inv_r);
    TADDS(denom, mean, rms_detail_simt_static_m_A_tree::kEpsilon);
    rsqrt_regbase(rms, denom);
    for (int64_t r = 0; r < gR; r += active_r) {
        gm_t gi(x + offset + r, static_cast<int>(active_a),
                static_cast<int>(gR));
        gm_t gg(const_cast<dtype *>(gamma) + r, 1,
                static_cast<int>(gR));
        gm_t go(out + offset + r, static_cast<int>(active_a),
                static_cast<int>(gR));
        tile_h h;
        gamma_h gh;
        tile_f src, normalized, dst;
        gamma_f gf;
        TLOAD(h, gi); TCVT(src, h);
        TLOAD(gh, gg); TCVT(gf, gh);
        TROWEXPANDMUL(normalized, src, rms);
        TCOLEXPANDMUL(dst, normalized, gf);
        TCVT(h, dst); TSTORE(go, h);
    }
}

} // namespace rms_detail_simt_static_m_A_tree

// Fixed shape [512,8192].
template <typename dtype, int peNum>
void rms_norm_simt_static_m_A_tree(dtype *x, const dtype *gamma, dtype *out) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");

    // The FP32 compute/reduction-source Tile is [32,16], i.e. 2 KiB.
    constexpr int64_t tRows = 32;
    constexpr int64_t tCols = 16;

    constexpr int64_t globalA = 512;
    constexpr int64_t gR = 8192;
    constexpr int64_t tile_a = 32;
    constexpr int64_t tile_r = 16;
    const uint32_t tid = get_thread_idx();

    if (globalA <= 0 || gR <= 0 ||
        tid >= static_cast<uint32_t>(peNum)) {
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
    if (peA < tile_a || peA % tile_a != 0) {
        return;
    }
    const int64_t pe_offset = pe_start * gR;
    x += pe_offset;
    out += pe_offset;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tRows, tCols,
                        BLayout::RowMajor, 32, 16>;
    using tile_f = Tile<Location::Vec, float, tRows, tCols,
                        BLayout::RowMajor, 32, 16>;
    using tile_v = Tile<Location::Vec, float, tRows, 1,
                        BLayout::RowMajor, 32, 1>;
    using tile_reduce_matrix = Tile<Location::Vec, float, tRows, 16,
                                    BLayout::RowMajor, 32, 16>;
    using gamma_h = Tile<Location::Vec, dtype, 1, 32,
                         BLayout::RowMajor, 1, 16>;
    using gamma_f = Tile<Location::Vec, float, 1, tCols,
                         BLayout::RowMajor, 1, 16>;

    const float inv_r = 1.0f / static_cast<float>(gR);

    for (int64_t ia = 0; ia < peA; ia += tile_a) {
        rms_detail_simt_static_m_A_tree::rms_norm_tile_static<
            dtype, gm_t, tile_h, tile_f, tile_v, tile_reduce_matrix, gamma_h,
            gamma_f>(
            x, gamma, out, gR, ia, tile_a, tile_r, inv_r);
    }
}

#endif // SUPERNPU_RMS_NORM_SIMT_STATIC_M_A_TREE_HPP
