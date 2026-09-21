#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp"

// Top-K radix select: FP16 coarse pass + four FP32 byte passes, per
// topk_scatter_atomic_add_tileop.md (same directory). The 256-bin suffix
// cumsum is the standalone histogram_cumsum_m32 operator (same kernels dir).
//
// Semantics: for each batch row bx, select the kTopK largest values of
// input[bx, start:end) and write their in-row element indices to
// output[bx, 0:kTopK]. The output is an unordered set (source note 11).
//
// SPMD: 4 PEs, PE tid handles rows bx = tid, tid+4, ...  All per-row state
// (histogram, ping-pong candidates, chunk scratch) is private to one PE, so
// the listing's cross-thread barriers collapse to program order.
//
// The FP16 coarse bin/key computation stays on CUBE_M32 vector tiles
// (TLOAD f32 -> TCVT f16 -> sortable key -> high-byte bin). The current
// gfrun + TileOP contracts force the rest of the pipeline scalar over the
// PE-private GM scratch:
//  - TCVT: the TileOP wrapper emits the PTO #291 form
//    `B.DATR CUBE_M32, <dtype>, Null, RNONE`, but gfrun's
//    ValidateOperandContract still requires the B.DATR Layout field to be NORM
//    for a CUBE_M16/M32 TCVT; tcvt_cube hand-writes the NORM spelling.
//  - Threshold search: a CUBE_M32 TCOLSUM is capped at 32 valid rows and now
//    requires both operands to carry CUBE descriptors, which the un-laid-out
//    TCMPS/TSELS results do not; find_threshold is scalar over the suffix sum.
//  - Suffix cumsum: the standalone histogram_cumsum_m32 operator (included
//    above) publishes the suffix sum (one grouped ND2M32 TLOAD + pair-tree
//    within-group suffix over B.SUBVIEW + a 32-lane TSHUF across-group scan).
//  - Indexed TLSU: gfrun's MGATHER ignores the IndexTile (every lane reads
//    offset 0), and the #291 forward contract rejects a CUBE-selecting TMUL
//    whose TCMPS/TSELS sources are RowMajor. The candidate append, the FP32
//    byte rounds and the output writes are therefore scalar, and each
//    candidate's sortable key is captured at stage 1 (candkey) instead of
//    being re-gathered from the source row.
//  - The FP16 payload is reinterpreted via a GM round trip
//    (TSTORE fp16 -> TLOAD u16) instead of reinterpret_tile: the model tags
//    each tile register with the dtype of its last writer and validates it
//    on TSTORE (AccumulateBlockInfo.cpp ValidateLocalTlsu); the round trip
//    keeps every block's declared dtype equal to the tag.

namespace topk_radix {

using namespace pto;
using histogram_cumsum_m32::GroupTileS32;
using histogram_cumsum_m32::group_texpands;
using histogram_cumsum_m32::group_tstore;

constexpr int kBatch = 4;
constexpr int kCols = 8192;
constexpr int kTopK = 512;
constexpr int kCandCap = 4096;  // candidate workspace bound (source note 12)
constexpr int kLane = 32;

// All vector tiles are CUBE_M32 single-cell column vectors (ValidCol=1,
// ValidRow=32). Sub-32-bit dtypes pad the physical Col to keep 128 B.
using F32Tile = VecTileM32<float, kLane, 1>;
using F16Tile = VecTileM32<__half, kLane, 2, kLane, 1>;
using U16Tile = VecTileM32<uint16_t, kLane, 2, kLane, 1>;
using U32Tile = VecTileM32<uint32_t, kLane, 1>;
using I32Tile = VecTileM32<int32_t, kLane, 1>;
using U8Tile = VecTileM32<uint8_t, kLane, 4, kLane, 1>;

struct Scratch {
    int32_t hist[384];      // [0,256) bins; [256,384) zero pad the grouped
                            // cumsum TLOAD reads past the sentinel, never written
    int32_t num[2];         // ping-pong candidate counts
    int32_t error;          // 0 ok, 1 candidate overflow, 2 empty range
    uint32_t scan[2 * kLane];  // float-bits staging for the stage-1 candidate
                               // key capture
    uint16_t bin16[kLane];
    int32_t cand[2][kCandCap];      // candidate element indices
    uint32_t candkey[2][kCandCap];  // candidate FP32 sortable keys, parallel to
                                    // cand; captured scalar at stage 1 so the
                                    // byte rounds need no MGATHER
};

inline int32_t clamp_lo(int32_t v, int32_t lo) { return v < lo ? lo : v; }
inline int32_t clamp_hi(int32_t v, int32_t hi) { return v > hi ? hi : v; }
inline int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

// CUBE_M32 <32,1> TCVT. The installed TileOP wrapper emits the PTO #291 form
//   B.DATR CUBE_M32, <dtype>, Null, RNONE
// but this gfrun build's ValidateOperandContract (isa/Block.cpp) still demands
// the B.DATR Layout field be NORM for a CUBE_M16/M32 TCVT and carries the
// CUBE-ness only in the operand descriptors. Hand-write the NORM spelling
// until the model adopts the #291 layout encoding.
template <typename DstTile, typename SrcTile>
inline __attribute__((always_inline)) void tcvt_cube(DstTile &dst, SrcTile &src) {
    static_assert(SrcTile::ValidCol > 0 && SrcTile::ValidRow > 0,
                  "tcvt_cube needs a fully static source shape");
    asm volatile(
        "BSTART.TEPL 27, %D[SrcType]\n"
        "B.DATR NORM, %D[DstType], Null, RNONE\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.IOT %[src], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        : [dst] "=Tr"(dst.data())
        : [src] "Tr"(src.data()),
          [SrcType] "i"(type_traits<typename SrcTile::DType>::TypeCode),
          [DstType] "i"(type_traits<typename DstTile::DType>::TypeCode),
          [DstSize] "i"(tile_type_traits<typename DstTile::TileDType>::TilesizeCode),
          [ValidCol] "i"(SrcTile::ValidCol),
          [ValidRow] "i"(SrcTile::ValidRow)
        : "memory");
}

// Zero a GM word range with tile ops instead of scalar stores: one broadcast
// zero tile, then grouped 1KB TSTOREs plus a 32-word cell TSTORE tail.
// `words` must be a multiple of kLane.
inline void tile_zero_i32(int32_t *base, int32_t words) {
    GroupTileS32 z;
    group_texpands(z, 0);
    int32_t off = 0;
    for (; off + 8 * kLane <= words; off += 8 * kLane) {
        group_tstore(base + off, z);
    }
    if (off < words) {
        I32Tile c;
        TEXPANDS(c, 0);
        global_tensor<int32_t, RowMajor<kLane, 1>> g(base + off);
        TSTORE(g, c);
    }
}

// Threshold search (source 3.4). The source's tile-op form is a TCMPS
// predicate + TSELS 0/1 + TCOLSUM count, but the current gfrun updated the
// PTO #311 CUBE reduction contract so that a CUBE_M32 TCOLSUM needs BOTH its
// source and destination to be CUBE descriptors. The predicate/select chain
// emits no B.DATR layout, so its result is a RowMajor descriptor and the
// CUBE destination is rejected (Block.cpp ValidateOperandContract). The
// histogram is already accumulated scalar over the PE-private GM array, so
// count the crossing scalar too and keep the computed suffix-cumsum. hist is
// non-increasing after suffix_cumsum with hist[0] = n >= rem, so
//   thr = #{bin : hist[bin] >= rem} - 1
// is exactly the first bin with hist[bin] >= rem && hist[bin+1] < rem (the
// source's no-crossing default of 0 is unreachable: bin 0 always counts).
inline int find_threshold(Scratch &sc, int32_t rem) {
    int32_t bin = 0;
    while (bin < 256 && sc.hist[bin] >= rem) ++bin;
    return bin - 1;
}

// Candidate append and output write. The source design tiles both
// (MGATHER for the refinement bytes, TCMPS/TSELS/TMUL + prefix scan +
// MSCATTER_MASK for the compacted appends and slot writes), but the current
// gfrun blocks both tile routes:
//   - MGATHER's IndexTile is ignored (every lane reads offset 0), and
//   - the PTO #291 forward layout contract rejects a CUBE-selecting TEPL
//     elementwise TMUL whose TCMPS/TSELS sources are RowMajor.
// The histogram and the suffix sum are already scalar over the PE-private GM
// scratch, so the candidate append, the per-round byte extraction and the
// output writes are scalar too; only the FP16 bin/key computation stays on
// tiles.

// Stage 1: bin = high 8 bits of the FP16 sortable key of each input element.
// key16 = bits16 ^ (sign ? 0xFFFF : 0x8000), bin = key16 >> 8.
// collect=false: histogram. collect=true: bin>thr -> out slot hist[bin+1]++
// (scalar, needs the atomic RMW); bin==thr -> append to cand[0] (scalar).
inline void stage1_scan(const float *row, int32_t base0, int32_t n, bool collect,
                        int32_t thr, int32_t *out, Scratch &sc) {
    for (int32_t ch = 0; ch * kLane < n; ++ch) {
        const int32_t base = base0 + ch * kLane;
        const int32_t vc = min_i32(kLane, n - ch * kLane);

        global_tensor<float, RowMajor<kLane, 1>> gsrc(row + base);
        F32Tile f32;
        TLOAD(f32, gsrc);
        // Stage the raw FP32 payload so an eq-bin candidate's sortable key can
        // be captured scalar, in parallel with its index. The rounds then read
        // candkey instead of MGATHER-indexing the source row.
        global_tensor<float, RowMajor<kLane, 1>> gfb(
            reinterpret_cast<float *>(sc.scan));
        TSTORE(gfb, f32);
        const uint32_t *keybits = reinterpret_cast<const uint32_t *>(sc.scan);
        F16Tile f16;
        tcvt_cube(f16, f32);  // RNE
        global_tensor<__half, RowMajor<kLane, 1>> gh(reinterpret_cast<__half *>(sc.bin16));
        TSTORE(gh, f16);  // dtype tag stays FP16: legal store
        global_tensor<uint16_t, RowMajor<kLane, 1>> gu(sc.bin16);
        U16Tile bits, sign, scaled, mask, key, bin;
        TLOAD(bits, gu);  // same payload, u16 tag from here on
        TSHRS(sign, bits, static_cast<uint16_t>(15));
        TMULS(scaled, sign, static_cast<uint16_t>(0x7FFF));
        TADDS(mask, scaled, static_cast<uint16_t>(0x8000));
        TXOR(key, bits, mask);
        TSHRS(bin, key, static_cast<uint16_t>(8));
        TSTORE(gu, bin);

        for (int32_t lane = 0; lane < kLane; ++lane) {
            if (lane >= vc) continue;
            const int32_t b = sc.bin16[lane];
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            if (b == thr) {
                if (sc.num[0] < kCandCap) {
                    const uint32_t raw = keybits[lane];
                    sc.cand[0][sc.num[0]] = base + lane;
                    sc.candkey[0][sc.num[0]] =
                        raw & 0x80000000u ? ~raw : (raw ^ 0x80000000u);
                    ++sc.num[0];
                } else {
                    sc.error = 1;
                }
                continue;
            }
            if (b > thr) {
                out[sc.hist[b + 1]++] = base + lane;
            }
        }
    }
}

// Stage 2 round: byte (24-8*round) of the FP32 sortable key of each candidate.
// key32 = bits ^ (sign ? 0xFFFFFFFF : 0x80000000); sign = bits >> 31.
// collect=false: histogram. collect=true: bin>thr -> out slot prefix+hist[bin+1]++
// (scalar, needs the atomic RMW); bin==thr -> next-round candidate for rounds
// 0..2 (scalar) or, on round 3, direct output while pos < kTopK (scalar, also a
// hist RMW).
inline void round_scan(const float *row, int round, bool collect, int32_t thr,
                       int32_t prefix, int32_t *out, Scratch &sc) {
    (void)row;
    const int r = round & 1;
    const int nr = r ^ 1;
    const uint32_t shift = static_cast<uint32_t>(24 - 8 * round);
    const int32_t count = sc.num[r];
    for (int32_t t = 0; t * kLane < count; ++t) {
        const int32_t vc = min_i32(kLane, count - t * kLane);
        for (int32_t lane = 0; lane < kLane; ++lane) {
            if (lane >= vc) continue;
            const uint32_t k = sc.candkey[r][t * kLane + lane];
            const int32_t b = static_cast<int32_t>((k >> shift) & 0xFFu);
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            if (b > thr) {
                const int32_t pos = prefix + sc.hist[b + 1]++;
                out[pos] = sc.cand[r][t * kLane + lane];
            } else if (b == thr) {
                if (round < 3) {
                    if (sc.num[nr] < kCandCap) {
                        sc.cand[nr][sc.num[nr]] = sc.cand[r][t * kLane + lane];
                        sc.candkey[nr][sc.num[nr]] = k;
                        ++sc.num[nr];
                    } else {
                        sc.error = 1;
                    }
                } else {  // round 3 direct output
                    const int32_t pos = prefix + sc.hist[b + 1]++;
                    if (pos < kTopK) {
                        out[pos] = sc.cand[r][t * kLane + lane];
                    }
                }
            }
        }
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

        tile_zero_i32(sc.hist, 384);
        sc.num[0] = 0;
        sc.num[1] = 0;
        sc.error = 0;
        tile_zero_i32(sc.cand[0], kCandCap);
        tile_zero_i32(sc.cand[1], kCandCap);
        tile_zero_i32(reinterpret_cast<int32_t *>(sc.scan), 2 * kLane);

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
        histogram_cumsum_m32::suffix_cumsum(sc.hist);
        int32_t thr = find_threshold(sc, rem);
        rem -= sc.hist[thr + 1];
        stage1_scan(row, x2, n, true, thr, out, sc);

        for (int round = 0; round < 4 && rem > 0; ++round) {
            const int nr = (round & 1) ^ 1;
            const int32_t prefix = kTopK - rem;
            tile_zero_i32(sc.hist, 384);
            sc.num[nr] = 0;
            round_scan(row, round, false, 0, 0, out, sc);
            histogram_cumsum_m32::suffix_cumsum(sc.hist);
            thr = find_threshold(sc, rem);
            rem -= sc.hist[thr + 1];
            round_scan(row, round, true, thr, prefix, out, sc);
        }
        errors[bx] = sc.error;
    }
}

}  // namespace topk_radix
