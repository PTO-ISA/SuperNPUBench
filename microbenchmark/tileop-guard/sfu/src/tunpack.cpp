#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TUNPACK (SFU/TEPL layout, PTO 0.58.5) — extract one
// contiguous byte field from each U32 CUBE word, right-align + zero-extend.
// Source: layout-and-rearrangement/layout/TUNPACK.md (v0.58.5).
// Precision: res_check. Golden pins pto-spec rearrangement.asl:332 (logical,
// element-wise): dst = (src >> 8*off) & mask(cnt).
// control[7:0]=byte offset (0..3), control[15:8]=byte count (1..4), off+cnt<=4.
constexpr int M = 32, N = 32, NE = M * N;
static uint32_t ga[NE], gout[NE];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ga, sizeof(ga));
    using T = VecTileM32<uint32_t, M, N>;
    T src, dst;
    global_tensor<uint32_t, RowMajor<M, N>> gA(ga), gD(gout);
    TLOAD_CUBE(src, gA);
    const uint64_t control = 0x0201;   // offset=1, count=2
    BENCHSTART;
    TUNPACK(dst, src, control);
    BENCHEND;
    TSTORE_CUBE(gD, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
