#pragma once
//
// FP16-only boundary refinement for topk_tiled: the exact top-k of the
// FP16-rounded values, with no FP32 pass.
//
// Internal header.  It is included by
// kernels/multi_thread/topk/topk_tiled.hpp after the shared core (Scratch,
// loaders, histogram helpers, round_hist/round_write) and is selected by
// kFp32Refine == false.  Do not include it directly.

namespace topk_tiled {

// FP16 key low byte of each candidate: key16 & 0xFF, where key16 is formed
// exactly as load_bin does (RNE round to FP16 then a sign-aware sortable key),
// so the ordering matches stage 1.  The candidate index is gathered as an FP32
// value and the FP16 payload is reinterpreted as U16 through the same GM round
// trip load_bin uses.
inline __attribute__((always_inline)) I32Tile candidate_byte16(const float *row,
                                                               U32Tile &cand,
                                                               Scratch &sc) {
    U32Tile off;
    TMULS(off, cand, 4u);  // byte displacement
    global_tensor<float, RowMajor<kColsMax, 1>> gf(row);
    F32Tile f32;
    MGATHER(f32, gf, off);
    F16Tile f16;
    TCVT(f16, f32);
    global_tensor<__half, RowMajor<kLane, 1>> gh(
        reinterpret_cast<__half *>(sc.bin16));
    TSTORE(gh, f16);
    global_tensor<uint16_t, RowMajor<kLane, 1>> gu(sc.bin16);
    U16Tile bits, sign, scaled, mask, key, l8;
    TLOAD(bits, gu);
    TSHRS(sign, bits, static_cast<uint16_t>(15));
    TMULS(scaled, sign, static_cast<uint16_t>(0x7FFF));
    TADDS(mask, scaled, static_cast<uint16_t>(0x8000));
    TXOR(key, bits, mask);
    TANDS(l8, key, static_cast<uint16_t>(0xFF));
    I32Tile byte;
    TCVT(byte, l8);
    return byte;
}

// One FP16 key low-byte round over the bin16 == thr candidates: stage 1 already
// binned the key high byte, so this completes the 16-bit FP16 ordering.  Uses
// round_write's final pass (round == 3): the bin > lowthr lanes are written to
// out, and the bin == lowthr lanes -- identical in the full 16-bit FP16 key --
// fill the remaining slots in candidate order.  No extra fill pass is needed.
inline __attribute__((always_inline)) void tail_fp16(int32_t *out,
                                                     const float *row,
                                                     int32_t thr, int32_t rem,
                                                     int32_t topk, Scratch &sc) {
    if (rem <= 0) return;
    const int32_t prefix = topk - rem;
    const int32_t count = sc.num[0];
    hist_clear(sc);
    for (int32_t t = 0; t * kLane < count; ++t) {
        const int32_t vc = min_i32(kLane, count - t * kLane);
        global_tensor<uint32_t, RowMajor<kLane, 1>> gc(
            reinterpret_cast<const uint32_t *>(&sc.cand[0][t * kLane]));
        U32Tile candU;
        TLOAD(candU, gc);
        I32Tile byte = candidate_byte16(row, candU, sc);
        round_hist(byte, vc, sc);
    }
    hist_cumsum(sc);
    thr = find_threshold(sc, rem);
    rem -= sc.hist[thr + 1];
    for (int32_t t = 0; t * kLane < count; ++t) {
        const int32_t vc = min_i32(kLane, count - t * kLane);
        global_tensor<uint32_t, RowMajor<kLane, 1>> gc(
            reinterpret_cast<const uint32_t *>(&sc.cand[0][t * kLane]));
        U32Tile candU;
        TLOAD(candU, gc);
        I32Tile byte = candidate_byte16(row, candU, sc);
        round_write(out, 3, candU, byte, vc, thr, prefix, topk, sc);
    }
}

}  // namespace topk_tiled
