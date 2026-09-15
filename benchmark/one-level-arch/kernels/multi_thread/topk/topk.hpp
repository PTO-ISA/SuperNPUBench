#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// Top-K radix select: FP16 coarse pass + four FP32 byte passes, per
// incoming/topk_pseudocode/topk_scatter_atomic_add_tileop.md.
//
// Semantics: for each batch row bx, select the kTopK largest values of
// input[bx, start:end) and write their in-row element indices to
// output[bx, 0:kTopK]. The output is an unordered set (source note 11).
//
// SPMD: 4 PEs, PE tid handles rows bx = tid, tid+4, ...  All per-row state
// (histogram, ping-pong candidates, chunk scratch) is private to one PE, so
// the listing's cross-thread barriers collapse to program order.
//
// Deliberate deviations from the listing, forced by what gfrun actually
// models (SuperScalarModel TMAEngine / TileOpManager):
//  - The GM atomic family (MGATHER_ADD/MSCATTER_ADD, TLSU function 12/21) is
//    not implemented in gfrun (functions 4..8 only). Histogram accumulation,
//    suffix cumsum, threshold search and slot allocation therefore run on the
//    PE's scalar core over the PE-private GM histogram. Each 32-lane chunk's
//    bins are produced by tile ops, TSTOREd to scratch, then accumulated
//    scalar; program order replaces the atomic RMW (duplicates within a
//    chunk are serialized by the scalar loop, same result as the atomics).
//  - Tiles are RowMajor <1,32>, not CUBE_M32: the indexed TLSU wrappers
//    reject CUBE layouts and TCMPS/TSELS do not emit CUBE layout selectors.
//  - gfrun's MGATHER/MSCATTER_MASK consume the index tile as a *byte
//    displacement* (TMAEngine.cpp), so offset tiles carry index*4.
//  - The FP16 payload is reinterpreted via a GM round trip
//    (TSTORE fp16 -> TLOAD u16) instead of reinterpret_tile: the model tags
//    each tile register with the dtype of its last writer and validates it on
//    TSTORE (AccumulateBlockInfo.cpp ValidateLocalTlsu); the round trip keeps
//    every block's declared dtype equal to the tag.
//  - No TCMPS/TSEL predicates: the sign masks come from bit arithmetic
//    (key = bits ^ (sign ? ~0 : signbit)) and the scatter mask is a u8 tile
//    materialized through GM scratch.

namespace topk_radix {

using namespace pto;

constexpr int kBatch = 4;
constexpr int kCols = 8192;
constexpr int kTopK = 512;
constexpr int kCandCap = 4096;  // candidate workspace bound (source note 12)
constexpr int kLane = 32;

using F32Tile = Tile<Location::Vec, float, 1, kLane, BLayout::RowMajor>;
// Physical width padded to 128 B so sub-32-bit tiles keep one full cell;
// the logical valid region stays <1,32>.
using F16Tile = Tile<Location::Vec, __half, 1, 2 * kLane, BLayout::RowMajor, 1, kLane>;
using U16Tile = Tile<Location::Vec, uint16_t, 1, 2 * kLane, BLayout::RowMajor, 1, kLane>;
using U32Tile = Tile<Location::Vec, uint32_t, 1, kLane, BLayout::RowMajor>;
using I32Tile = Tile<Location::Vec, int32_t, 1, kLane, BLayout::RowMajor>;
using U8Tile = Tile<Location::Vec, uint8_t, 1, 4 * kLane, BLayout::RowMajor, 1, kLane>;

struct Scratch {
    int32_t hist[257];      // hist[0:256] bins, hist[256] sentinel
    int32_t num[2];         // ping-pong candidate counts
    int32_t error;          // 0 ok, 1 candidate overflow, 2 empty range
    uint16_t bin16[kLane];
    uint32_t word[kLane];
    uint32_t off[kLane];    // byte displacements for MSCATTER_MASK/MGATHER
    int32_t idx[kLane];
    uint8_t msk[kLane];
    int32_t cand[2][kCandCap];
};

inline int32_t clamp_lo(int32_t v, int32_t lo) { return v < lo ? lo : v; }
inline int32_t clamp_hi(int32_t v, int32_t hi) { return v > hi ? hi : v; }
inline int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

// hist[bin] = sum_{j=bin..255} original_hist[j]; hist[256] stays 0.
inline void suffix_cumsum(int32_t *hist) {
    int32_t s = 0;
    for (int b = 255; b >= 0; --b) {
        s += hist[b];
        hist[b] = s;
    }
}

// First bin with hist[bin] >= rem && hist[bin+1] < rem; default 0 (source 3.4).
inline int find_threshold(const int32_t *hist, int32_t rem) {
    for (int b = 0; b < 256; ++b) {
        if (hist[b] >= rem && hist[b + 1] < rem) return b;
    }
    return 0;
}

// Write the masked lanes of sc.idx to out[sc.off/4] via MSCATTER_MASK.
// sc.off holds byte displacements because gfrun consumes them unscaled.
inline void scatter_masked(int32_t *out, Scratch &sc) {
    bool any = false;
    for (int l = 0; l < kLane; ++l) any |= sc.msk[l] != 0;
    if (!any) return;
    global_tensor<uint32_t, RowMajor<1, kLane>> goff(sc.off);
    global_tensor<int32_t, RowMajor<1, kLane>> gidx(sc.idx);
    global_tensor<uint8_t, RowMajor<1, kLane>> gmsk(sc.msk);
    U32Tile off;
    I32Tile idx;
    U8Tile msk;
    TLOAD(off, goff);
    TLOAD(idx, gidx);
    TLOAD(msk, gmsk);
    global_tensor<int32_t, RowMajor<1, kTopK>> gout(out);
    MSCATTER_MASK(gout, idx, off, msk);
}

// Stage 1: bin = high 8 bits of the FP16 sortable key of each input element.
// key16 = bits16 ^ (sign ? 0xFFFF : 0x8000), bin = key16 >> 8.
// collect=false: histogram. collect=true: bin>thr -> out slot hist[bin+1]++;
// bin==thr -> append to cand[0].
inline void stage1_scan(const float *row, int32_t base0, int32_t n, bool collect,
                        int32_t thr, int32_t *out, Scratch &sc) {
    for (int32_t ch = 0; ch * kLane < n; ++ch) {
        const int32_t base = base0 + ch * kLane;
        const int32_t vc = min_i32(kLane, n - ch * kLane);

        global_tensor<float, RowMajor<1, kCols>> gsrc(row + base);
        F32Tile f32;
        TLOAD(f32, gsrc);
        F16Tile f16;
        TCVT(f16, f32);  // RNE
        global_tensor<__half, RowMajor<1, kLane>> gh(reinterpret_cast<__half *>(sc.bin16));
        TSTORE(gh, f16);  // dtype tag stays FP16: legal store
        global_tensor<uint16_t, RowMajor<1, kLane>> gu(sc.bin16);
        U16Tile bits, sign, scaled, mask, key, bin;
        TLOAD(bits, gu);  // same payload, u16 tag from here on
        TSHRS(sign, bits, static_cast<uint16_t>(15));
        TMULS(scaled, sign, static_cast<uint16_t>(0x7FFF));
        TADDS(mask, scaled, static_cast<uint16_t>(0x8000));
        TXOR(key, bits, mask);
        TSHRS(bin, key, static_cast<uint16_t>(8));
        TSTORE(gu, bin);

        bool any = false;
        for (int32_t lane = 0; lane < kLane; ++lane) {
            sc.msk[lane] = 0;
            if (lane >= vc) continue;
            const int32_t b = sc.bin16[lane];
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            const int32_t in_idx = base + lane;
            if (b > thr) {
                sc.off[lane] = static_cast<uint32_t>(sc.hist[b + 1]++) * 4u;
                sc.idx[lane] = in_idx;
                sc.msk[lane] = 1;
                any = true;
            } else if (b == thr) {
                const int32_t p = sc.num[0]++;
                if (p < kCandCap) {
                    sc.cand[0][p] = in_idx;
                } else {
                    sc.error = 1;
                }
            }
        }
        if (collect && any) scatter_masked(out, sc);
    }
}

// Stage 2 round: byte (24-8*round) of the FP32 sortable key of each candidate.
// key32 = bits ^ (sign ? 0xFFFFFFFF : 0x80000000); sign = bits >> 31.
// collect=false: histogram. collect=true: bin>thr -> out slot prefix+hist[bin+1]++;
// bin==thr -> next-round candidate (rounds 0..2) or, on round 3, direct output
// while pos < kTopK.
inline void round_scan(const float *row, int round, bool collect, int32_t thr,
                       int32_t prefix, int32_t *out, Scratch &sc) {
    const int r = round & 1;
    const int nr = r ^ 1;
    const uint32_t shift = static_cast<uint32_t>(24 - 8 * round);
    const int32_t count = sc.num[r];
    for (int32_t t = 0; t * kLane < count; ++t) {
        const int32_t vc = min_i32(kLane, count - t * kLane);

        global_tensor<uint32_t, RowMajor<1, kCandCap>> gcand(
            reinterpret_cast<const uint32_t *>(&sc.cand[r][t * kLane]));
        U32Tile cand, off, bits, sign, scaled, mask, key, shf, byte;
        TLOAD(cand, gcand);
        TMULS(off, cand, 4u);  // gfrun MGATHER: index tile is a byte displacement
        global_tensor<uint32_t, RowMajor<1, kCols>> gin(
            reinterpret_cast<const uint32_t *>(row));
        MGATHER(bits, gin, off);
        TSHRS(sign, bits, 31u);
        TMULS(scaled, sign, 0x7FFFFFFFu);
        TADDS(mask, scaled, 0x80000000u);
        TXOR(key, bits, mask);
        TSHRS(shf, key, shift);
        TANDS(byte, shf, 0xFFu);
        global_tensor<uint32_t, RowMajor<1, kLane>> gw(sc.word);
        TSTORE(gw, byte);

        bool any = false;
        for (int32_t lane = 0; lane < kLane; ++lane) {
            sc.msk[lane] = 0;
            if (lane >= vc) continue;
            const int32_t b = static_cast<int32_t>(sc.word[lane]);
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            const int32_t in_idx = sc.cand[r][t * kLane + lane];
            if (b > thr) {
                const int32_t pos = prefix + sc.hist[b + 1]++;
                sc.off[lane] = static_cast<uint32_t>(pos) * 4u;
                sc.idx[lane] = in_idx;
                sc.msk[lane] = 1;
                any = true;
            } else if (b == thr) {
                if (round < 3) {
                    const int32_t p = sc.num[nr]++;
                    if (p < kCandCap) {
                        sc.cand[nr][p] = in_idx;
                    } else {
                        sc.error = 1;
                    }
                } else {
                    const int32_t pos = prefix + sc.hist[b + 1]++;
                    if (pos < kTopK) {
                        sc.off[lane] = static_cast<uint32_t>(pos) * 4u;
                        sc.idx[lane] = in_idx;
                        sc.msk[lane] = 1;
                        any = true;
                    }
                }
            }
        }
        if (collect && any) scatter_masked(out, sc);
    }
}

inline void run(int32_t *output, int32_t *errors, const float *input,
                const int32_t *starts, const int32_t *ends, Scratch *scratches) {
    const uint32_t tid = get_thread_idx();
    if (tid >= 4) return;

    for (int bx = static_cast<int>(tid); bx < kBatch; bx += 4) {
        Scratch &sc = scratches[bx];
        const float *row = input + bx * kCols;
        int32_t *out = output + bx * kTopK;

        for (int i = 0; i < 257; ++i) sc.hist[i] = 0;
        sc.num[0] = 0;
        sc.num[1] = 0;
        sc.error = 0;
        for (int i = 0; i < kCandCap; ++i) {
            sc.cand[0][i] = 0;
            sc.cand[1][i] = 0;
        }

        const int32_t x2 = clamp_lo(starts[bx], 0);
        const int32_t x3 = clamp_hi(ends[bx], kCols);
        const int32_t n = clamp_lo(x3 - x2, 0);
        if (n < kTopK) {  // degenerate range; test inputs never take this path
            sc.error = 2;
            errors[bx] = sc.error;
            continue;
        }

        int32_t rem = kTopK;
        stage1_scan(row, x2, n, false, 0, out, sc);
        suffix_cumsum(sc.hist);
        int32_t thr = find_threshold(sc.hist, rem);
        rem -= sc.hist[thr + 1];
        stage1_scan(row, x2, n, true, thr, out, sc);

        for (int round = 0; round < 4 && rem > 0; ++round) {
            const int nr = (round & 1) ^ 1;
            const int32_t prefix = kTopK - rem;
            for (int i = 0; i < 257; ++i) sc.hist[i] = 0;
            sc.num[nr] = 0;
            round_scan(row, round, false, 0, 0, out, sc);
            suffix_cumsum(sc.hist);
            thr = find_threshold(sc.hist, rem);
            rem -= sc.hist[thr + 1];
            round_scan(row, round, true, thr, prefix, out, sc);
        }
        errors[bx] = sc.error;
    }
}

}  // namespace topk_radix
