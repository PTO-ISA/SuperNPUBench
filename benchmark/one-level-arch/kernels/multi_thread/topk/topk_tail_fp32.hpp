#pragma once
//
// FP32 boundary refinement for topk_tiled: the exact FP32 top-k.
//
// Internal header.  It is included by
// kernels/multi_thread/topk/topk_tiled.hpp after the shared core (Scratch,
// loaders, histogram helpers, round_hist/round_write) and is selected by
// kFp32Refine == true.  Do not include it directly.

namespace topk_tiled {

// key32 = bits ^ (sign ? ~0 : 0x80000000); byte = (key32 >> (24-8*round))&0xFF
inline __attribute__((always_inline)) I32Tile candidate_byte(const float *row,
                                                            U32Tile &cand,
                                                            int round,
                                                            Scratch &sc) {
    U32Tile off, bits, sign, scaled, key, shf, b;
    TMULS(off, cand, 4u);  // byte displacement
    mgather_u32_m32(bits, row, off);
    TSHRS(sign, bits, 31u);
    TMULS(scaled, sign, 0x7FFFFFFFu);
    TADDS(scaled, scaled, 0x80000000u);
    TXOR(key, bits, scaled);
    TSHRS(shf, key, static_cast<uint32_t>(24 - 8 * round));
    TANDS(b, shf, 0xFFu);
    I32Tile byte;
    TCVT(byte, b);
    (void)sc;
    return byte;
}

// Four FP32 key-byte rounds (MSB first) over the bin16 == thr candidates.
// Ping-pong cand[r]/cand[nr]: each round writes its bin > thr lanes to out and
// appends its bin == thr lanes for the next round, whose counter starts empty.
// The last round's bin == thr lanes fill the remaining slots while pos < topk.
inline __attribute__((always_inline)) void tail_fp32(int32_t *out,
                                                     const float *row,
                                                     int32_t thr, int32_t rem,
                                                     int32_t topk, Scratch &sc) {
    for (int round = 0; round < 4 && rem > 0; ++round) {
        const int r = round & 1;
        const int32_t prefix = topk - rem;
        const int32_t count = sc.num[r];
        hist_clear(sc);
        for (int32_t t = 0; t * kLane < count; ++t) {
            const int32_t vc = min_i32(kLane, count - t * kLane);
            global_tensor<uint32_t, RowMajor<kLane, 1>> gc(
                reinterpret_cast<const uint32_t *>(&sc.cand[r][t * kLane]));
            U32Tile candU;
            TLOAD(candU, gc);
            I32Tile byte = candidate_byte(row, candU, round, sc);
            round_hist(byte, vc, sc);
        }
        hist_cumsum(sc);
        thr = find_threshold(sc, rem);
        rem -= sc.hist[thr + 1];

        const int32_t nr = (round & 1) ^ 1;
        sc.num[nr] = 0;
        const int32_t c2 = sc.num[r];
        for (int32_t t = 0; t * kLane < c2; ++t) {
            const int32_t vc = min_i32(kLane, c2 - t * kLane);
            global_tensor<uint32_t, RowMajor<kLane, 1>> gc(
                reinterpret_cast<const uint32_t *>(&sc.cand[r][t * kLane]));
            U32Tile candU;
            TLOAD(candU, gc);
            I32Tile byte = candidate_byte(row, candU, round, sc);
            round_write(out, round, candU, byte, vc, thr, prefix, topk, sc);
        }
    }
}

}  // namespace topk_tiled
