// =============================================================================
// rms_norm_binary_pto.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// tiling[5] = {g_a, g_r, tile_a, tile_r, pow_r}
//
// 每块 RowSum 后立刻 UpdateCache（workspace = cacheBuffer），对齐 AscendC：
//   DataCopy(aReg, src);
//   for (j = 0; j < cid; ++j) {
//       DataCopy(bReg, cache + j * stride);
//       Add(aReg, aReg, bReg);
//   }
//   DataCopy(cache + cid * stride, aReg);
//   cid = GetCacheId(idx) = ctz(idx+1)
//   sum = cache[GetCacheId(r-1)]   （r 为 2^k）
//
// workspace: [0, kMaxLevels) cache 档
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_BINARY_PTO_HPP
#define SUPERNPU_RMS_NORM_BINARY_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_bin {

// Row-reduction results have physical Columns=1. Workspace cache entries
// must preserve that layout so TLOAD and TADD match the TROWSUM output.
constexpr int kWsCols = 1;
constexpr int kMaxLevels = 6;


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

} // namespace rms_bin

template <typename dtype, int peNum>
void rms_norm_binary(dtype *x, const int64_t *tiling, dtype *out,
                     float *workspace, float eps = 1e-6f) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 1024;

    const int64_t globalA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t gA = globalA / peNum;
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
    const uint32_t tid = get_thread_idx();

    if (globalA <= 0 || globalA % peNum != 0 || gA < tA) {
        return;
    }

    const int64_t pe_offset = static_cast<int64_t>(tid) * gA * gR;
    x += pe_offset;
    out += pe_offset;
    workspace += static_cast<int64_t>(tid) * gA * rms_bin::kWsCols;

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_v = Tile<Location::Vec, float, tA, rms_bin::kWsCols,
                        BLayout::RowMajor, 1, 1>;

    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;
        const size_t full_r = static_cast<size_t>(tile_r);

        tile_v cur, buf, sum, mean, denom, rms, zero;
        TEXPANDS(zero, 0.0f);

        float *cache = workspace + ia * rms_bin::kWsCols;
        const int64_t stride = globalA * rms_bin::kWsCols;

        for (int64_t lv = 0; lv < rms_bin::kMaxLevels; ++lv) {
            gm_f go(cache + lv * stride, 1, rms_bin::kWsCols);
            TSTORE(go, zero);
        }

        int64_t r = 0;

        // UpdateCache（AscendC 同构）
#define RMS_BIN_UPDATE_CACHE()                                                  \
    do {                                                                       \
        const uint16_t cid =                                                   \
            static_cast<uint16_t>(rms_bin::GetCacheId(r));                     \
        for (uint16_t j = 0; j < cid; ++j) {                                   \
            gm_f gj(cache + static_cast<int64_t>(j) * stride, 1,               \
                    rms_bin::kWsCols);                                         \
            TLOAD(buf, gj);                                                    \
            TADD(cur, cur, buf);                                               \
        }                                                                      \
        gm_f gc(cache + static_cast<int64_t>(cid) * stride, 1,                 \
                rms_bin::kWsCols);                                             \
        TSTORE(gc, cur);                                                       \
        ++r;                                                                   \
    } while (0)

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
            RMS_BIN_UPDATE_CACHE();
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
            RMS_BIN_UPDATE_CACHE();
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
            RMS_BIN_UPDATE_CACHE();
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
            RMS_BIN_UPDATE_CACHE();
        }
#undef RMS_BIN_UPDATE_CACHE

        {
            const int64_t rid = r > 0 ? rms_bin::GetCacheId(r - 1) : 0;
            gm_f gr(cache + rid * stride, 1, rms_bin::kWsCols);
            TLOAD(sum, gr);
        }

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

#endif // SUPERNPU_RMS_NORM_BINARY_PTO_HPP
