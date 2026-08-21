// =============================================================================
// rms_norm_binary_pto.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// tiling[5] = {g_a, g_r, tile_a, tile_r, pow_r}
//
// 每块 RowSum 后立刻 UpdateCache。cache 在 Local tile 数组里（不再走 GM）：
//   TROWSUM(cur, x^2)
//   for (j = 0; j < cid; ++j) TADD(cur, cur, updateTile[j]);
//   TMULS(updateTile[cid], cur, 1);
//   cid = GetCacheId(idx) = ctz(idx+1)
//   TMULS(sum, cur, 1)  — 最后一次 merge 后 cur 就是全量和（r 为 2^k）
//   不要 TMULS(sum, upd[GetCacheId(r-1)], 1)：dyn 运行时 rid 会编成错误的
//   ctz/SWAR，fold 到 upd[0]（pair2），rsqrt Newton 初值 1/a 越号成约 −2×。
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_BINARY_PTO_HPP
#define SUPERNPU_RMS_NORM_BINARY_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_bin {

constexpr int kWsCols = 32;
constexpr int kMaxLevels = 8;

inline int64_t GetCacheId(int64_t idx) {
    return static_cast<int64_t>(
        __builtin_ctzll(static_cast<unsigned long long>(idx + 1)));
}

template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    TileVec x, t1, t2;
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

// Local tile list。同类型拷贝用 TMULS(dst, src, 1)（device 无 TCOPY；TMOV 运行时下标会断言）。
template <typename TileVec>
inline void update_cache(TileVec (&upd)[kMaxLevels], TileVec &cur, int64_t &r) {
    const int cid = static_cast<int>(GetCacheId(r));
    for (int j = 0; j < cid; ++j) {
        TADD(cur, cur, upd[j]);
    }
    TMULS(upd[cid], cur, 1.0f);
    ++r;
}

// 用最后一次 update 留下的 cur，不要运行时 upd[rid]。
// dyn 上 GetCacheId(r-1) 会编成 hl.bfi SWAR，rid 变成 0。
template <typename TileVec>
inline void fold_cache(TileVec &sum, TileVec &cur) {
    TMULS(sum, cur, 1.0f);
}

} // namespace rms_bin

template <typename dtype>
void rms_norm_binary(dtype *x, const int64_t *tiling, dtype *out,
                     float eps = 1e-6f) {
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t gA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : tR;
    const int64_t powR = tiling[4];

    const int64_t remR = gR - powR;
    const int64_t headR = powR - remR;
    const int64_t n_rem_full = remR / tile_r;
    const int64_t rem_tail = remR - n_rem_full * tile_r;
    const int64_t n_head_full = headR / tile_r;
    const int64_t head_tail = headR - n_head_full * tile_r;
    const int64_t n_full = gR / tile_r;
    const int64_t tail_r = gR - n_full * tile_r;
    const float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, tA, rms_bin::kWsCols,
                        BLayout::RowMajor, 1, 1>;
    tile_v updateTile[rms_bin::kMaxLevels];

    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;
        const size_t full_r = static_cast<size_t>(tile_r);

        tile_v cur, sum, mean, denom, rms;

        int64_t r = 0;

        for (int64_t tr = 0; tr < n_rem_full; ++tr) {
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi0(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gi1(x + offset + powR, static_cast<int>(gA),
                     static_cast<int>(gR));
            tile_h src0_h(active_a, full_r);
            tile_h src1_h(active_a, full_r);
            tile_f src0(active_a, full_r);
            tile_f src1(active_a, full_r);
            tile_f sq0(active_a, full_r);
            tile_f sq1(active_a, full_r);

            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            TADD(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            rms_bin::update_cache(updateTile, cur, r);
        }

        if (rem_tail > 0) {
            const int64_t offset = ia * gR + n_rem_full * tile_r;
            const size_t ar = static_cast<size_t>(rem_tail);
            gm_t gi0(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gi1(x + offset + powR, static_cast<int>(gA),
                     static_cast<int>(gR));
            tile_h src0_h(active_a, ar);
            tile_h src1_h(active_a, ar);
            tile_f src0(active_a, ar);
            tile_f src1(active_a, ar);
            tile_f sq0(active_a, ar);
            tile_f sq1(active_a, ar);

            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            TADD(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            rms_bin::update_cache(updateTile, cur, r);
        }

        for (int64_t tr = 0; tr < n_head_full; ++tr) {
            const int64_t offset = ia * gR + remR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, full_r);
            tile_f src(active_a, full_r);
            tile_f sq(active_a, full_r);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TMUL(sq, src, src);
            TROWSUM(cur, sq);
            rms_bin::update_cache(updateTile, cur, r);
        }
        if (head_tail > 0) {
            const int64_t offset = ia * gR + remR + n_head_full * tile_r;
            const size_t ar = static_cast<size_t>(head_tail);
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, ar);
            tile_f src(active_a, ar);
            tile_f sq(active_a, ar);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TMUL(sq, src, src);
            TROWSUM(cur, sq);
            rms_bin::update_cache(updateTile, cur, r);
        }

        rms_bin::fold_cache(sum, cur);

        TMULS(mean, sum, inv_r);
        TADDS(denom, mean, eps);
        rms_bin::rsqrt_newton(rms, denom);

        for (int64_t tr = 0; tr < n_full; ++tr) {
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, full_r);
            tile_h dst_h(active_a, full_r);
            tile_f src(active_a, full_r);
            tile_f dst(active_a, full_r);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
        if (tail_r > 0) {
            const int64_t offset = ia * gR + n_full * tile_r;
            const size_t ar = static_cast<size_t>(tail_r);
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, ar);
            tile_h dst_h(active_a, ar);
            tile_f src(active_a, ar);
            tile_f dst(active_a, ar);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
    }
}

// Compile-time shape / tiling. Same algorithm as the dynamic entry.
template <typename dtype, int gA, int gR, int tA, int tR, int powR>
void rms_norm_binary(dtype *x, dtype *out, float eps = 1e-6f) {
    static_assert(gA > 0 && gR > 0 && tA == 1 && tR > 0 && powR > 0);
    static_assert(powR < gR && gR <= 2 * powR);
    constexpr int remR = gR - powR;
    constexpr int headR = powR - remR;
    constexpr int n_rem_full = remR / tR;
    constexpr int rem_tail = remR % tR;
    constexpr int n_head_full = headR / tR;
    constexpr int head_tail = headR % tR;
    constexpr int n_full = gR / tR;
    constexpr int tail_r = gR % tR;
    constexpr float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<gA, gR>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor>;
    using tile_v = Tile<Location::Vec, float, tA, rms_bin::kWsCols,
                        BLayout::RowMajor, 1, 1>;
    tile_v updateTile[rms_bin::kMaxLevels];

    for (int64_t ia = 0; ia < gA; ++ia) {
        tile_v cur, sum, mean, denom, rms;

        int64_t r = 0;

        for (int tr = 0; tr < n_rem_full; ++tr) {
            const int64_t offset = ia * gR + tr * tR;
            gm_t gi0(x + offset);
            gm_t gi1(x + offset + powR);
            tile_h src0_h, src1_h;
            tile_f src0, src1, sq0, sq1;
            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            TADD(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            rms_bin::update_cache(updateTile, cur, r);
        }
        if constexpr (rem_tail) {
            using tile_h_r =
                Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, tA, rem_tail>;
            using tile_f_r =
                Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, tA, rem_tail>;
            const int64_t offset = ia * gR + n_rem_full * tR;
            gm_t gi0(x + offset);
            gm_t gi1(x + offset + powR);
            tile_h_r src0_h, src1_h;
            tile_f_r src0, src1, sq0, sq1;
            TLOAD(src0_h, gi0);
            TLOAD(src1_h, gi1);
            TCVT(src0, src0_h);
            TCVT(src1, src1_h);
            TMUL(sq0, src0, src0);
            TMUL(sq1, src1, src1);
            TADD(sq0, sq0, sq1);
            TROWSUM(cur, sq0);
            rms_bin::update_cache(updateTile, cur, r);
        }
        for (int tr = 0; tr < n_head_full; ++tr) {
            const int64_t offset = ia * gR + remR + tr * tR;
            gm_t gi(x + offset);
            tile_h src_h;
            tile_f src, sq;
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TMUL(sq, src, src);
            TROWSUM(cur, sq);
            rms_bin::update_cache(updateTile, cur, r);
        }
        if constexpr (head_tail) {
            using tile_h_r =
                Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, tA, head_tail>;
            using tile_f_r =
                Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, tA, head_tail>;
            const int64_t offset = ia * gR + remR + n_head_full * tR;
            gm_t gi(x + offset);
            tile_h_r src_h;
            tile_f_r src, sq;
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TMUL(sq, src, src);
            TROWSUM(cur, sq);
            rms_bin::update_cache(updateTile, cur, r);
        }

        rms_bin::fold_cache(sum, cur);

        TMULS(mean, sum, inv_r);
        TADDS(denom, mean, eps);
        rms_bin::rsqrt_newton(rms, denom);

        for (int tr = 0; tr < n_full; ++tr) {
            const int64_t offset = ia * gR + tr * tR;
            gm_t gi(x + offset);
            gm_t go(out + offset);
            tile_h src_h, dst_h;
            tile_f src, dst;
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
        if constexpr (tail_r) {
            using tile_h_r =
                Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, tA, tail_r>;
            using tile_f_r =
                Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, tA, tail_r>;
            const int64_t offset = ia * gR + n_full * tR;
            gm_t gi(x + offset);
            gm_t go(out + offset);
            tile_h_r src_h, dst_h;
            tile_f_r src, dst;
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TROWEXPANDMUL(dst, src, rms);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
    }
}

#endif // SUPERNPU_RMS_NORM_BINARY_PTO_HPP
