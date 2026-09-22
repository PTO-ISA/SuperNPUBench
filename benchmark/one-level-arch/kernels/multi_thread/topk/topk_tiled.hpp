#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp"

// Top-K radix-select, fully tiled transcription of
// topk_scatter_atomic_add_tileop.md.
//
// The whole pipeline (histogram, stage-1 collect, FP32 tail rounds) stays on
// CUBE_M32 tiles; MGATHER / MSCATTER.MASK / the TCMPS CUBE GPR carrier go
// through the TileOP API.  Every helper is always_inline and keeps few tiles
// live, so the backend emits no tile spills.
//
// Remaining (being fixed upstream):
//   * MGATHER.ADD (histogram / slot atom add) is still a hand-written block
//     because, although the GM atom/red wrapper emits B.DATR.Layout since
//     LinxISA/Linx-TileOP-API#185 (PR #209), the LinxV5 backend folds the
//     default LB0/LB2 on every TLSU head while gfrun's gm-atom-red contract
//     requires an explicit LB0. The hand-written block uses literal 1s so the
//     dims survive the fold.
//   * find_threshold uses the TCMPS CUBE GPR carrier, which needs the
//     canonical B.IOR RegDst position fix (ASL: RegDst = bits[11:7]):
//     LinxISA/SuperScalarModel#806.
//
// Depends on LinxISA/Linx-TileOP-API#207 / #208 and LinxISA/SuperScalarModel
// PR #785 (CUBE predicate carriers).

namespace topk_tiled {

using namespace pto;

// Tile geometry and the radix stay compile-time PTO contracts: kLane is the
// M32 cell width / tile-op width, and the 256-bin radix is the algorithm.  The
// outer shape (batch x cols, top-k) is runtime and validated against these
// maxima — same pattern as fa_gmma_dynamic's FaGmmaTilingData.
constexpr int kLane = 32;
constexpr int kBatchMax = 4;
constexpr int kColsMax = 131072;
constexpr int kTopKMax = 1024;
constexpr int kCandCap = 65536;

// Boundary-bucket refinement.  true  -> the four FP32 key bytes give the exact
// FP32 top-k.  false -> the single FP16 key low byte gives the exact top-k of
// the FP16-rounded values (no FP32 pass; ties in the full 16-bit FP16 key are
// broken arbitrarily).  Compile-time so the two dataflows never coexist.
constexpr bool kFp32Refine = true;

struct TopkTilingData {
    int64_t batch;
    int64_t cols;
    int64_t topk;
};

using I32Tile = VecTileM32<int32_t, kLane, 1>;
using U32Tile = VecTileM32<uint32_t, kLane, 1>;
using F32Tile = VecTileM32<float, kLane, 1>;
using F16Tile = VecTileM32<__half, kLane, 2, kLane, 1>;
using U16Tile = VecTileM32<uint16_t, kLane, 2, kLane, 1>;
using U8Tile = VecTileM32<uint8_t, kLane, 4, kLane, 1>;
// CUBE_M32 predicate carrier: a U8 CELL tile with the same logical shape as
// the compared I32 tile (Cols must match for TCMPS).  gfrun decodes CUBE
// TCMPS as a U8 PredicateCell with this geometry.
using PredM32 = VecTileM32<uint8_t, kLane, 1>;

struct Scratch {
    int32_t hist[256];       // 256 bins, flat in GM; H[bin]
    // hist_cumsum's grouped load reads hist[256:384] as its zero boundary, and
    // round_write allocates the top-byte (0xFF) slot through H[256].  Both need
    // hist[256] to be a real, per-round-cleared word, so it must not overlap
    // num[0].  hist_clear zeroes this pad together with hist.
    int32_t hist_pad[256];
    int32_t num[2];          // ping-pong candidate counts, GM
    int32_t error;           // 0 ok, 1 candidate overflow, 2 empty range
    int32_t cand[2][kCandCap];

    uint16_t bin16[kLane];   // FP16 payload staging for the U16 view
    int32_t lane[kLane];     // constant 0..31, loaded as a well-formed <32,1>
    int32_t classOut[kLane]; // tile value exposed to scalar for the RMW
    int32_t slotOut[kLane];  // out slot per lane (0 where not written)
    int32_t eqOut[kLane];    // eq slot / round-3 pos per lane
    uint8_t mskOut[kLane];   // U8 mask staging
};

inline int32_t clamp_lo(int32_t v, int32_t lo) { return v < lo ? lo : v; }
inline int32_t clamp_hi(int32_t v, int32_t hi) { return v > hi ? hi : v; }
inline int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

// ---- section 2: one chunk -> FP16 high-byte bin --------------------------
//
// key16 = (x < 0) ? (~raw16 & 0xffff) : (raw16 | 0x8000); bin = key16 >> 8.
// The FP16 payload is reinterpreted as U16 through a GM round trip so every
// block's declared dtype equals its tile-register tag.
inline __attribute__((always_inline)) I32Tile load_bin(const float *row,
                                                       int32_t base0,
                                                       Scratch &sc) {
    global_tensor<float, RowMajor<kLane, 1>> gsrc(row + base0);
    F32Tile f32;
    TLOAD(f32, gsrc);

    F16Tile f16;
    TCVT(f16, f32);  // RNE; CUBE TCVT keeps Layout=NORM (TileOP #178)

    global_tensor<__half, RowMajor<kLane, 1>> gh(
        reinterpret_cast<__half *>(sc.bin16));
    TSTORE(gh, f16);
    global_tensor<uint16_t, RowMajor<kLane, 1>> gu(sc.bin16);
    U16Tile bits, sign, scaled, mask, key, sh8;
    TLOAD(bits, gu);

    TSHRS(sign, bits, static_cast<uint16_t>(15));
    TMULS(scaled, sign, static_cast<uint16_t>(0x7FFF));
    TADDS(mask, scaled, static_cast<uint16_t>(0x8000));
    TXOR(key, bits, mask);
    TSHRS(sh8, key, static_cast<uint16_t>(8));

    I32Tile bin;
    TCVT(bin, sh8);  // U16 -> S32 numeric bin
    return bin;
}

// ---- section 3.1: clear H ------------------------------------------------

inline __attribute__((always_inline)) void hist_clear(Scratch &sc) {
    using histogram_cumsum_m32::GroupTileS32;
    using histogram_cumsum_m32::group_texpands;
    using histogram_cumsum_m32::group_tstore;
    GroupTileS32 zero;
    group_texpands(zero, 0);
    group_tstore(sc.hist, zero);
    group_tstore(sc.hist_pad, zero);
}

// Forward declarations: the histogram helpers use the lane loader and the
// GM atomic helper defined later in this header.
inline __attribute__((always_inline)) I32Tile load_lane(Scratch &sc);
inline __attribute__((always_inline)) void mgather_add_s32_m32(
    I32Tile &old, int32_t *base, I32Tile &index, I32Tile &value);

// H[bin]++ per valid lane, via MGATHER_ADD old-value semantics.  Padding lanes
// (lane >= vc) get value 0 so they do not touch the histogram.  The index is a
// byte displacement.  Shared by stage-1 hist_add and tail round_hist.
inline __attribute__((always_inline)) void hist_bump(I32Tile &bin, int32_t vc,
                                                     Scratch &sc) {
    I32Tile lane = load_lane(sc);
    // CUBE TCMPS publishes a U8 PredicateCell; TCVT turns it into a numeric
    // 0/1 value (TSELS cannot mix the U8 predicate with an I32 destination).
    PredM32 vmask;
    TCMPS<CmpMode::LT>(vmask, lane, vc);
    I32Tile val, off, old;
    TCVT(val, vmask);
    TMULS(off, bin, 4u);
    mgather_add_s32_m32(old, sc.hist, off, val);
}

// ---- section 3.2: histogram accumulation ---------------------------------
//
// MGATHER_ADD(H[bin*4], value=1) per valid lane; padding lanes get value 0 so
// they do not touch the tail chunk's histogram.  llvm-project#105 is fixed, so
// the named selector is used.
inline __attribute__((always_inline)) void hist_add(I32Tile &bin, int32_t vc,
                                                    Scratch &sc) {
    hist_bump(bin, vc, sc);
}

// ---- section 3.3: 256-bin suffix cumsum ----------------------------------

inline __attribute__((always_inline)) void hist_cumsum(Scratch &sc) {
    histogram_cumsum_m32::suffix_cumsum(sc.hist);
}

// ---- section 3.4: coarse threshold ---------------------------------------
//
// TCMPS<LT> over the suffix-cumsum H with the TileOP CUBE GPR carrier
// (LinxISA/Linx-TileOP-API#200).  One 64-bit carrier covers at most 2 M32
// columns, so H (256 bins) is scanned as 8 [32,1] groups; group g holds bins
// [32g,32g+32) with GPR bit r == bin offset r.  CTZ gives the crossing bin.
inline __attribute__((always_inline)) int32_t find_threshold(Scratch &sc,
                                                             int32_t rem) {
    for (int g = 0; g < 8; ++g) {
        I32Tile H;
        global_tensor<int32_t, RowMajor<kLane, 1>> gh(sc.hist + g * kLane);
        TLOAD(H, gh);
        const uint64_t m = TCMPS<CmpMode::LT>(H, rem);
        if (m != 0) {
            return static_cast<int32_t>(g * kLane + __builtin_ctzll(m) - 1);
        }
    }
    return 255;
}

// ---- shared CUBE_M32 indexed blocks --------------------------------------
//
// GAP(tileop#201): the MGATHER/MSCATTER wrappers hardcode B.DATR NORM, so the
// CUBE layout selector is hand-written.  gfrun's indexed transfers consume a
// byte displacement per index element.

inline __attribute__((always_inline)) void mgather_u32_m32(U32Tile &dst,
                                                           const float *base,
                                                           U32Tile &index) {
    global_tensor<uint32_t, RowMajor<kColsMax, 1>> g(
        reinterpret_cast<const uint32_t *>(base));
    MGATHER(dst, g, index);
}

template <typename MaskT>
inline __attribute__((always_inline)) void mscatter_mask_i32_m32(
    int32_t *base, I32Tile &data, I32Tile &index, MaskT &mask) {
    global_tensor<int32_t, RowMajor<kCandCap, 1>> g(base);
    MSCATTER_MASK(g, data, index, mask);
}

// GM atomic add.  Verified end-to-end by test/kernel/mgather_add_m32.  B.DATR
// must match the CUBE_M32 index/value tiles and LB0 must be explicit for
// gfrun's gm-atom-red descriptor contract.
inline __attribute__((always_inline)) void mgather_add_s32_m32(
    I32Tile &old, int32_t *base, I32Tile &index, I32Tile &value) {
    asm volatile(
        "BSTART.TLSU MGATHER.ADD, %D[DataType]\n"
        "B.DATR CUBE_M32, Zero\n"
        // Literal 1s (not "i" operands): the backend's
        // omitDefaultInlineAsmDims folds a placeholder LB0/LB2 whose value is
        // 1 on any TLSU head, but gfrun's gm-atom-red contract requires an
        // explicit LB0. Hardcoding keeps it in the stream.
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, %c[VROW], ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[Idx], %[Val], mask=1111, last, ->%[Dst]<%Z[DstSize]>\n"
        "B.IOR [%[Base]], []\n"
        : [Dst] "=&Tr"(old.data())
        : [Idx] "Tr"(index.data()), [Val] "Tr"(value.data()),
          [Base] "r"(base),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [VROW] "i"(I32Tile::ValidRow), [DstSize] "i"(I32Tile::TilesizeCode)
        : "memory");
}

// A CUBE TCMPS already publishes the predicate directly (U8 PredicateCell),
// so the mask is used as-is and the 0/1 value comes from a plain TSELS select.
// Predicate AND is expressed as a nested select, e.g.
//   pred_and = vmask ? pred : 0        (TSELS(pred_and, vmask, 0, pred))

inline __attribute__((always_inline)) I32Tile load_lane(Scratch &sc) {
    global_tensor<int32_t, RowMajor<kLane, 1>> g(sc.lane);
    I32Tile lane;
    TLOAD(lane, g);
    return lane;
}

inline __attribute__((always_inline)) U8Tile load_u8(uint8_t *p) {
    global_tensor<uint8_t, RowMajor<kLane, 1>> g(p);
    U8Tile t;
    TLOAD(t, g);
    return t;
}

inline __attribute__((always_inline)) I32Tile load_i32(int32_t *p) {
    global_tensor<int32_t, RowMajor<kLane, 1>> g(p);
    I32Tile t;
    TLOAD(t, g);
    return t;
}

// ---- section 3.5: second scan, output / save candidates ------------------
//
// bin>thr lanes write out[slot] = index; bin==thr lanes append
// cand[0][slot] = index.  Tile version of the old scalar workaround
// (llvm-project#105 is fixed): slot/eq come from MGATHER_ADD old values.
inline __attribute__((always_inline)) void stage1_collect(I32Tile &bin,
                                                          I32Tile &index,
                                                          int32_t vc,
                                                          int32_t thr,
                                                          int32_t *out,
                                                          Scratch &sc) {
    // Tail chunk may be partially valid: gate by lane < vc.  Predicates are
    // CUBE U8 PredicateCells; TCVT to numeric 0/1, TAND to combine.
    I32Tile lane = load_lane(sc);
    PredM32 vmask;
    TCMPS<CmpMode::LT>(vmask, lane, vc);
    I32Tile v01;
    TCVT(v01, vmask);

    // bin+1 as a byte displacement (shared by the GT slot path).
    I32Tile idx, gold;
    TADDS(idx, bin, 1);
    TMULS(idx, idx, 4u);

    // GT lanes: slot = hist[bin+1]++ ; out[slot] = index.
    PredM32 gmask, gamask;
    I32Tile g01, gact;
    TCMPS<CmpMode::GT>(gmask, bin, thr);
    TCVT(g01, gmask);
    TAND(gact, g01, v01);
    TCVT(gamask, gact);
    mgather_add_s32_m32(gold, sc.hist, idx, gact);
    I32Tile goldB;
    TMULS(goldB, gold, 4u);  // MSCATTER.MASK index is a byte displacement
    mscatter_mask_i32_m32(out, index, goldB, gamask);

    // EQ lanes: eq = num[0]++ ; cand[0][eq] = index.
    // The append is gated by eq < kCandCap so the buffer is never overrun; if
    // the window produces more candidates than kCandCap, run() flags
    // errors[bx] = 1 instead of corrupting memory.
    PredM32 emask, eamask, capmask;
    I32Tile e01, eact, zeroIdx, eq, cap01, eactcap;
    TCMPS<CmpMode::EQ>(emask, bin, thr);
    TCVT(e01, emask);
    TAND(eact, e01, v01);
    TCVT(eamask, eact);
    TEXPANDS(zeroIdx, 0);
    mgather_add_s32_m32(eq, sc.num, zeroIdx, eact);
    TCMPS<CmpMode::LT>(capmask, eq, static_cast<int32_t>(kCandCap));
    TCVT(cap01, capmask);
    TAND(eactcap, eact, cap01);
    TCVT(eamask, eactcap);
    I32Tile eqB;
    TMULS(eqB, eq, 4u);
    mscatter_mask_i32_m32(sc.cand[0], index, eqB, eamask);
}

// ---- section 4: boundary refinement -------------------------------------
//
// The tail that turns the bin16 == thr bucket into an exact top-k lives in the
// variant header selected by kFp32Refine, included after round_write below:
//   topk_tail_fp32.hpp  exact FP32, four key bytes
//   topk_tail_fp16.hpp  exact FP16, one key low byte

// tail histogram of one candidate tile: H[byte]++ per valid lane.
// Tile version of the old scalar workaround (llvm-project#105 is fixed).
inline __attribute__((always_inline)) void round_hist(I32Tile &byte, int32_t vc,
                                                      Scratch &sc) {
    hist_bump(byte, vc, sc);
}

// tail write-back of one candidate tile (pseudocode 4.2).
//   bin > thr : pos = H[bin+1]++ + prefix, out[pos] = cand
//   bin == thr: rounds 0..2 append to cand[nr][Num[nr]++];
//               round 3 writes out[pos] while pos < kTopK
inline __attribute__((always_inline)) void round_write(int32_t *out, int round,
                                                       U32Tile &candU,
                                                       I32Tile &byte,
                                                       int32_t vc, int32_t thr,
                                                       int32_t prefix, int32_t topk,
                                                       Scratch &sc) {
    const int nr = (round & 1) ^ 1;

    // Tile version of pseudocode 4.2.  llvm-project#105 is fixed, so the
    // per-lane slot/counter RMW uses MGATHER_ADD old-value semantics instead
    // of the scalar workaround:
    //   bin > thr  : slot = hist[bin+1]++ + prefix ; out[slot] = cand
    //   bin == thr : round<3 append cand[nr][num[nr]++];
    //                round 3 pos = hist[bin+1]++ + prefix, write if pos<topK.
    // The tail candidate block may be partially valid: gate by lane < vc.
    // Predicates are CUBE U8 PredicateCells; TCVT to numeric 0/1, TAND to
    // combine, TCVT back to a U8 mask for MSCATTER.MASK.
    I32Tile lane = load_lane(sc);
    PredM32 vmask;
    TCMPS<CmpMode::LT>(vmask, lane, vc);
    I32Tile v01;
    TCVT(v01, vmask);

    I32Tile candI;
    TCVT(candI, candU);

    // bin+1 as a byte displacement; shared by the GT slot and round-3 EQ pos.
    I32Tile idx, gold, slot;
    TADDS(idx, byte, 1);
    TMULS(idx, idx, 4u);

    // GT lanes: MGATHER_ADD returns the pre-increment hist slot.
    PredM32 gmask, gamask;
    I32Tile g01, gact;
    TCMPS<CmpMode::GT>(gmask, byte, thr);
    TCVT(g01, gmask);
    TAND(gact, g01, v01);
    TCVT(gamask, gact);
    mgather_add_s32_m32(gold, sc.hist, idx, gact);
    TADDS(slot, gold, prefix);
    I32Tile slotB;
    TMULS(slotB, slot, 4u);
    mscatter_mask_i32_m32(out, candI, slotB, gamask);

    // EQ lanes.
    PredM32 emask, eamask;
    I32Tile e01, eact;
    TCMPS<CmpMode::EQ>(emask, byte, thr);
    TCVT(e01, emask);
    TAND(eact, e01, v01);
    TCVT(eamask, eact);

    if (round < 3) {
        // Append: eq = num[nr]++ ; cand[nr][eq] = cand.
        I32Tile zeroIdx, eq;
        TEXPANDS(zeroIdx, 0);
        mgather_add_s32_m32(eq, sc.num + nr, zeroIdx, eact);
        I32Tile eqB;
        TMULS(eqB, eq, 4u);
        mscatter_mask_i32_m32(sc.cand[nr], candI, eqB, eamask);
    } else {
        // round 3: pos = hist[bin+1]++ + prefix ; out[pos] if pos < topk.
        PredM32 posmask, okmask;
        I32Tile gold2, pos, pos01, ok;
        mgather_add_s32_m32(gold2, sc.hist, idx, eact);
        TADDS(pos, gold2, prefix);
        TCMPS<CmpMode::LT>(posmask, pos, topk);
        TCVT(pos01, posmask);
        TAND(ok, pos01, eact);
        TCVT(okmask, ok);
        I32Tile posB;
        TMULS(posB, pos, 4u);
        mscatter_mask_i32_m32(out, candI, posB, okmask);
    }
}

}  // namespace topk_tiled

// The two boundary-refinement tails share the core above; kFp32Refine picks
// which one run() calls.  Both are cheap to compile in; the unused one is
// dropped by if constexpr.
#include "multi_thread/topk/topk_tail_fp32.hpp"
#include "multi_thread/topk/topk_tail_fp16.hpp"

namespace topk_tiled {

// ---- run ----------------------------------------------------------------

inline __attribute__((always_inline)) void run(int32_t *output, int32_t *errors, const float *input,
                const int32_t *starts, const int32_t *ends, Scratch *scratches,
                const TopkTilingData *tiling) {
    const uint32_t tid = get_thread_idx();
    if (tid >= 4) return;

    // Runtime outer shape, validated against the compile-time maxima.
    const int32_t batch = static_cast<int32_t>(tiling->batch);
    const int32_t cols = static_cast<int32_t>(tiling->cols);
    const int32_t topk = static_cast<int32_t>(tiling->topk);
    if (batch <= 0 || batch > kBatchMax || cols <= 0 || cols > kColsMax ||
        topk <= 0 || topk > kTopKMax) {
        return;
    }

    for (int bx = static_cast<int>(tid); bx < batch; bx += 4) {
        Scratch &sc = scratches[bx];
        const float *row = input + bx * cols;
        int32_t *out = output + bx * topk;

        for (int32_t l = 0; l < kLane; ++l) sc.lane[l] = l;
        hist_clear(sc);
        sc.num[0] = 0;
        sc.num[1] = 0;
        sc.error = 0;

        const int32_t x2 = clamp_lo(starts[bx], 0);
        const int32_t x3 = clamp_hi(ends[bx], cols);
        const int32_t n = clamp_lo(x3 - x2, 0);
        if (n < topk) {
            sc.error = 2;
            errors[bx] = sc.error;
            continue;
        }

        // stage 1: histogram
        for (int32_t ch = 0; ch * kLane < n; ++ch) {
            const int32_t vc = min_i32(kLane, n - ch * kLane);
            I32Tile bin = load_bin(row, x2 + ch * kLane, sc);
            hist_add(bin, vc, sc);
        }
        hist_cumsum(sc);

        int32_t rem = topk;
        int32_t thr = find_threshold(sc, rem);
        rem -= sc.hist[thr + 1];

        // stage 1: collect outputs and ==threshold candidates
        for (int32_t ch = 0; ch * kLane < n; ++ch) {
            const int32_t vc = min_i32(kLane, n - ch * kLane);
            const int32_t base = x2 + ch * kLane;
            I32Tile bin = load_bin(row, base, sc);
            I32Tile lane = load_lane(sc);
            I32Tile index;
            TADDS(index, lane, base);
            stage1_collect(bin, index, vc, thr, out, sc);
        }

        // Candidate overflow: stage1_collect gated the append so the buffer is
        // intact, but a window with more than kCandCap ==thr elements cannot be
        // refined.  Flag it (errors.bin != 0) instead of publishing a bad top-k.
        if (sc.num[0] > kCandCap) {
            sc.error = 1;
            errors[bx] = sc.error;
            continue;
        }

        // tail refinement of the bin16 == thr candidates.
        if constexpr (kFp32Refine) {
            tail_fp32(out, row, thr, rem, topk, sc);
        } else {
            tail_fp16(out, row, thr, rem, topk, sc);
        }
        errors[bx] = sc.error;
    }
}

}  // namespace topk_tiled
