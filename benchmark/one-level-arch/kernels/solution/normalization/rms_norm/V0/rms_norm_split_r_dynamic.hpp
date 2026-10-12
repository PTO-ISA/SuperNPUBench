// =============================================================================
// rms_norm_split_r_dynamic.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// tiling = {g_a, g_r, tile_a, tile_r, pow_r, n_padded}
//
// 每块 RowSum 后立刻 UpdateCache（workspace = cacheBuffer），对齐 AscendC：
//   DataCopy(aReg, src);
//   for (j = 0; j < cid; ++j) {
//       DataCopy(bReg, cache + j * stride);
//       Add(aReg, aReg, bReg);
//   }
//   DataCopy(cache + cid * stride, aReg);
//   cid = GetCacheId(idx) = ctz(idx+1)
//   sum = cache[ctz(n_padded)]   （n_padded 为 2^k，含补零块）
//
// 二分累加方案要求 partial 总数必须是 2 的幂（否则 cache[ctz(n)] 只包含
// 末尾部分和）。host 在 tiling 侧算好 n_padded（实际 partial 数向上补零
// 到下一个 2 的幂）传入 kernel；kernel 只在块循环末尾补发 n_padded -
// n_actual 个零 partial。补零块对总和贡献为 0，不改变结果，但保证最终
// 读取的 cache 档包含全和。kernel 校验 n_padded 是 2 的幂且不超过
// 2^(kMaxLevels-1)（cid 最大 kMaxLevels-1，不越界）。
//
// workspace: [0, kMaxLevels) cache 档
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_SPLIT_R_PTO_HPP
#define SUPERNPU_RMS_NORM_SPLIT_R_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_split_r {

struct RmsNormSplitRTilingData {
    int64_t g_a;
    int64_t g_r;
    int64_t tile_a;
    int64_t tile_r;
    int64_t pow_r;
    int64_t n_padded;
};

constexpr float kEpsilon = 1e-6f;

// Only one FP32 value is stored per GM cache entry. The local reduction
// carrier may be wider; padding is not part of the workspace layout.
constexpr int kWsCols = 1;
constexpr int kMaxLevels = 6;


inline int64_t GetCacheId(int64_t idx) {
    return static_cast<int64_t>(
        __builtin_ctzll(static_cast<unsigned long long>(idx + 1)));
}

template <typename TileVec>
__attribute__((always_inline)) inline void rsqrt_regbase(TileVec &out, TileVec &a) {
    TileVec recip, y, tmp;
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
}

} // namespace rms_split_r

template <typename dtype, int peNum>
void rms_norm_split_r(dtype *x, const dtype *gamma,
                      const rms_split_r::RmsNormSplitRTilingData *tiling,
                      dtype *out, float *workspace) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 512;

    const int64_t globalA = tiling->g_a;
    const int64_t gR = tiling->g_r;
    const int64_t tile_r = tiling->tile_r > 0 ? tiling->tile_r : tR;
    const int64_t powR = tiling->pow_r;
    const int64_t nPadded = tiling->n_padded;
    const uint32_t tid = get_thread_idx();

    if (globalA <= 0 || gR <= 1 || tile_r <= 0 || tile_r > tR ||
        powR <= 0 || powR >= gR || gR > 2 * powR ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }

    // Ceil partition: for M=333 and 4PE, rows are split 84, 84, 84, 81.
    const int64_t rows_per_pe = (globalA + peNum - 1) / peNum;
    const int64_t pe_start = static_cast<int64_t>(tid) * rows_per_pe;
    if (pe_start >= globalA) {
        return;
    }
    const int64_t remaining = globalA - pe_start;
    const int64_t gA =
        remaining < rows_per_pe ? remaining : rows_per_pe;
    if (gA < tA) {
        return;
    }

    const int64_t pe_offset = pe_start * gR;
    x += pe_offset;
    out += pe_offset;
    // Workspace is level-major: [level][global row].
    workspace += pe_start * rms_split_r::kWsCols;

    const int64_t remR = gR - powR;
    const int64_t headR = powR - remR;
    const int64_t n_rem_full = remR / tile_r;
    const int64_t rem_tail = remR - n_rem_full * tile_r;
    const int64_t n_head_full = headR / tile_r;
    const int64_t head_tail = headR - n_head_full * tile_r;
    const int64_t n_full = gR / tile_r;
    const int64_t tail_r = gR - n_full * tile_r;
    const float inv_r = 1.0f / static_cast<float>(gR);

    // 实际 partial 数（含 tail 块）
    const int64_t n_actual = n_rem_full + (rem_tail > 0 ? 1 : 0) +
                             n_head_full + (head_tail > 0 ? 1 : 0);
    // n_padded 由 host 在 tiling 侧算好传入；kernel 只做契约校验：
    // 必须是 2 的幂、覆盖全部实际块、不超过 cache 档容量（fail-closed）。
    if (nPadded < n_actual || nPadded <= 0 ||
        (nPadded & (nPadded - 1)) != 0 ||
        nPadded > (int64_t(1) << (rms_split_r::kMaxLevels - 1))) {
        return;
    }

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::CubeM32, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::CubeM32, -1, -1>;
    using reduce_row = Tile<Location::Vec, float, tA, tR,
                            BLayout::CubeM32, 1, 1>;
    using tile_v = Tile<Location::Vec, float, tA, tR,
                        BLayout::CubeM32, 1, 1>;

    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;
        const size_t full_r = static_cast<size_t>(tile_r);

        tile_v cur, buf, sum, mean, denom, rms, zero;
        TEXPANDS(zero, 0.0f);

        float *cache = workspace + ia * rms_split_r::kWsCols;
        const int64_t stride = globalA * rms_split_r::kWsCols;

        for (int64_t lv = 0; lv < rms_split_r::kMaxLevels; ++lv) {
            gm_f go(cache + lv * stride, 1, rms_split_r::kWsCols);
            TSTORE(go, zero);
        }

        int64_t r = 0;

        // UpdateCache（AscendC 同构）
#define RMS_BIN_UPDATE_CACHE()                                                  \
    do {                                                                       \
        const uint16_t cid =                                                   \
            static_cast<uint16_t>(rms_split_r::GetCacheId(r));                     \
        for (uint16_t j = 0; j < cid; ++j) {                                   \
            gm_f gj(cache + static_cast<int64_t>(j) * stride, 1,               \
                    rms_split_r::kWsCols);                                         \
            TLOAD(buf, gj);                                                    \
            TADD(cur, cur, buf);                                               \
        }                                                                      \
        gm_f gc(cache + static_cast<int64_t>(cid) * stride, 1,                 \
                rms_split_r::kWsCols);                                             \
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
            reduce_row row_sum;
            TROWSUM(row_sum, sq0);
            TCOLSUM(cur, row_sum);
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
            reduce_row row_sum;
            TROWSUM(row_sum, sq0);
            TCOLSUM(cur, row_sum);
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
            reduce_row row_sum;
            TROWSUM(row_sum, sq);
            TCOLSUM(cur, row_sum);
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
            reduce_row row_sum;
            TROWSUM(row_sum, sq);
            TCOLSUM(cur, row_sum);
            RMS_BIN_UPDATE_CACHE();
        }

        // 补零块：partial 数补到 nPadded（2 的幂，host 传入）。零块对总和
        // 贡献为 0，但保证二分累加的最终 cache[ctz(nPadded)] 档包含全和。
        for (int64_t p = n_actual; p < nPadded; ++p) {
            TEXPANDS(cur, 0.0f);
            RMS_BIN_UPDATE_CACHE();
        }
#undef RMS_BIN_UPDATE_CACHE

        {
            const int64_t rid = r > 0 ? rms_split_r::GetCacheId(r - 1) : 0;
            gm_f gr(cache + rid * stride, 1, rms_split_r::kWsCols);
            TLOAD(sum, gr);
        }

        TMULS(mean, sum, inv_r);
        TADDS(denom, mean, rms_split_r::kEpsilon);
        rms_split_r::rsqrt_regbase(rms, denom);

        for (int64_t tr = 0; tr < n_full; ++tr) {
            const int64_t offset = ia * gR + tr * tile_r;
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gg(const_cast<dtype *>(gamma) + tr * tile_r, 1,
                    static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, full_r);
            tile_h gamma_h(active_a, full_r), dst_h(active_a, full_r);
            tile_f src(active_a, full_r);
            tile_f normalized(active_a, full_r), gamma_f(active_a, full_r);
            tile_f dst(active_a, full_r);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TLOAD(gamma_h, gg);
            TCVT(gamma_f, gamma_h);
            TROWEXPANDMUL(normalized, src, rms);
            TMUL(dst, normalized, gamma_f);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
        if (tail_r > 0) {
            const int64_t offset = ia * gR + n_full * tile_r;
            const size_t ar = static_cast<size_t>(tail_r);
            gm_t gi(x + offset, static_cast<int>(gA), static_cast<int>(gR));
            gm_t gg(const_cast<dtype *>(gamma) + n_full * tile_r, 1,
                    static_cast<int>(gR));
            gm_t go(out + offset, static_cast<int>(gA), static_cast<int>(gR));
            tile_h src_h(active_a, ar);
            tile_h gamma_h(active_a, ar), dst_h(active_a, ar);
            tile_f src(active_a, ar);
            tile_f normalized(active_a, ar), gamma_f(active_a, ar);
            tile_f dst(active_a, ar);
            TLOAD(src_h, gi);
            TCVT(src, src_h);
            TLOAD(gamma_h, gg);
            TCVT(gamma_f, gamma_h);
            TROWEXPANDMUL(normalized, src, rms);
            TMUL(dst, normalized, gamma_f);
            TCVT(dst_h, dst);
            TSTORE(go, dst_h);
        }
    }
}

#endif // SUPERNPU_RMS_NORM_SPLIT_R_PTO_HPP
