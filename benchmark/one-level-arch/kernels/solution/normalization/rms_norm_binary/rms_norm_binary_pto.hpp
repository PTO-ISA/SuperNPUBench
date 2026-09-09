// =============================================================================
// rms_norm_binary_pto.hpp — RMSNorm for g_r > tile_r (R-split)
// =============================================================================
//
// tiling[5] = {g_a, g_r, tile_a, tile_r, pow_r}
//
// RowSum partials are carry-merged in a tile parent split into subviews.
// TPARTVIEW reads cache slots; TileArray + TASSEMBLY updates the parent.
// A scalar is replicated inside each slot so region operations use legal
// 128-byte-granular ranges. No GM workspace is used.
// =============================================================================
#ifndef SUPERNPU_RMS_NORM_BINARY_PTO_HPP
#define SUPERNPU_RMS_NORM_BINARY_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <utility>

namespace rms_bin {

// Row-reduction results keep physical Columns=1 as required by TROWSUM.
constexpr int kReductionCols = 1;
// Eight 1 KiB fragments form one 8 KiB RowMajor parent tile.
// Each cache scalar is replicated across its 32x8 FP32 fragment.
constexpr int kCacheSlots = 8;
constexpr int kCacheFragmentCols = 8;


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
void rms_norm_binary(dtype *x, const int64_t *tiling, dtype *out, float eps = 1e-6f) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    constexpr int64_t tA = 1;
    constexpr int64_t tR = 8192;

    const int64_t globalA = tiling[0];
    const int64_t gR = tiling[1];
    const int64_t tile_r = tiling[3] > 0 ? tiling[3] : tR;
    const int64_t powR = tiling[4];
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

    const int64_t remR = gR - powR;
    const int64_t headR = powR - remR;
    const int64_t n_rem_full = remR / tile_r;
    const int64_t rem_tail = remR - n_rem_full * tile_r;
    const int64_t n_head_full = headR / tile_r;
    const int64_t head_tail = headR - n_head_full * tile_r;
    const int64_t reduction_blocks =
        n_rem_full + (rem_tail > 0 ? 1 : 0) + n_head_full +
        (head_tail > 0 ? 1 : 0);
    if (reduction_blocks > (int64_t{1} << rms_bin::kCacheSlots) - 1) {
        return;
    }
    const int64_t n_full = gR / tile_r;
    const int64_t tail_r = gR - n_full * tile_r;
    const float inv_r = 1.0f / static_cast<float>(gR);

    using gm_t = global_tensor<dtype, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_f = Tile<Location::Vec, float, tA, tR, BLayout::RowMajor, -1, -1>;
    using tile_v = VecTileM32<float, 32, rms_bin::kReductionCols, 1,
                              rms_bin::kReductionCols>;
    using cache_fragment =
        Tile<Location::Vec, float, 32, rms_bin::kCacheFragmentCols,
             BLayout::RowMajor>;
    using cache_parent =
        Tile<Location::Vec, float, 32,
             rms_bin::kCacheSlots * rms_bin::kCacheFragmentCols,
             BLayout::RowMajor>;
    using cache_vec =
        Tile<Location::Vec, float, 32, rms_bin::kCacheFragmentCols,
             BLayout::RowMajor>;
    using cache_column = VecTileM32<float, 32, 1>;
    for (int64_t ia = 0; ia < gA; ++ia) {
        constexpr size_t active_a = 1;
        const size_t full_r = static_cast<size_t>(tile_r);

        tile_v cur, buf, sum, mean, denom, rms;
        cache_vec zero_cache;
        TEXPANDS(zero_cache, 0.0f);
        TileArray<cache_fragment, 1, rms_bin::kCacheSlots> initial_cache;
        TMULS(initial_cache[0][0], zero_cache, 1.0f);
        TMULS(initial_cache[0][1], zero_cache, 1.0f);
        TMULS(initial_cache[0][2], zero_cache, 1.0f);
        TMULS(initial_cache[0][3], zero_cache, 1.0f);
        TMULS(initial_cache[0][4], zero_cache, 1.0f);
        TMULS(initial_cache[0][5], zero_cache, 1.0f);
        TMULS(initial_cache[0][6], zero_cache, 1.0f);
        TMULS(initial_cache[0][7], zero_cache, 1.0f);
        cache_parent cache_tile =
            TASSEMBLY<cache_parent>(std::move(initial_cache));

        int64_t r = 0;

        // Carry-merge the reduction into subview-backed tile cache slots.
#define RMS_BIN_UPDATE_CACHE()                                                  \
    do {                                                                       \
        const uint16_t cid =                                                   \
            static_cast<uint16_t>(rms_bin::GetCacheId(r));                     \
        auto cache_views =                                                     \
            TPARTVIEW<cache_fragment, 1, rms_bin::kCacheSlots>(cache_tile);     \
        auto merge_slot = [&]<uint16_t Slot>() {                              \
            if (Slot < cid) {                                                 \
                auto cached = cache_views[0][Slot];                           \
                cache_vec copied;                                             \
                TMULS(copied, cached, 1.0f);                                  \
                cache_column cached_rows;                                     \
                TROWMAX(cached_rows, copied);                                 \
                TCOLMAX(buf, cached_rows);                                    \
                TADD(cur, cur, buf);                                          \
            }                                                                \
        };                                                                    \
        merge_slot.template operator()<0>();                                  \
        merge_slot.template operator()<1>();                                  \
        merge_slot.template operator()<2>();                                  \
        merge_slot.template operator()<3>();                                  \
        merge_slot.template operator()<4>();                                  \
        merge_slot.template operator()<5>();                                  \
        merge_slot.template operator()<6>();                                  \
        merge_slot.template operator()<7>();                                  \
        cache_column repeated_cur;                                            \
        TCOLEXPAND(repeated_cur, cur);                                        \
        cache_vec expanded_cur;                                                \
        TROWEXPAND(expanded_cur, repeated_cur);                                \
        TileArray<cache_fragment, 1, rms_bin::kCacheSlots> next_cache;          \
        auto write_slot = [&]<uint16_t Slot>() {                               \
            if (Slot == cid) {                                                 \
                TMULS(next_cache[0][Slot], expanded_cur, 1.0f);                 \
            } else {                                                           \
                auto cached = cache_views[0][Slot];                            \
                cache_vec copied;                                              \
                TMULS(copied, cached, 1.0f);                                   \
                TMULS(next_cache[0][Slot], copied, 1.0f);                       \
            }                                                                  \
        };                                                                     \
        write_slot.template operator()<0>();                                   \
        write_slot.template operator()<1>();                                   \
        write_slot.template operator()<2>();                                   \
        write_slot.template operator()<3>();                                   \
        write_slot.template operator()<4>();                                   \
        write_slot.template operator()<5>();                                   \
        write_slot.template operator()<6>();                                   \
        write_slot.template operator()<7>();                                   \
        cache_tile = TASSEMBLY<cache_parent>(std::move(next_cache));            \
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
            auto cache_views =
                TPARTVIEW<cache_fragment, 1, rms_bin::kCacheSlots>(cache_tile);
            auto cached = cache_views[0][rid];
            cache_vec copied;
            TMULS(copied, cached, 1.0f);
            cache_column cached_rows;
            TROWMAX(cached_rows, copied);
            TCOLMAX(sum, cached_rows);
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
