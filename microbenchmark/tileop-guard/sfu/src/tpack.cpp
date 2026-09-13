#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TPACK (SFU/TEPL layout, PTO 0.58.5) — pack two low-order
// byte fields of two U32 CUBE words into one U32; unselected high bits zero.
// Source: layout-and-rearrangement/layout/TPACK.md (v0.58.5).
// Precision: res_check. Golden pins pto-spec rearrangement.asl:296 (logical,
// element-wise): dst = (src0 & mask(w0)) | ((src1 & mask(w1)) << 8*w0).
// control[7:0]=src0 field width, control[15:8]=src1 width (0x0202 -> 2B each).
constexpr int M = 32, N = 32, NE = M * N;
static uint32_t ga[NE], gb[NE], gout[NE];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ga, sizeof(ga));
    guard_read_bin(CHK_DIR "/in_b.bin", gb, sizeof(gb));
    using T = VecTileM32<uint32_t, M, N>;
    T src0, src1, dst;
    global_tensor<uint32_t, RowMajor<M, N>> gA(ga), gB(gb), gD(gout);
    TLOAD_CUBE(src0, gA);
    TLOAD_CUBE(src1, gB);
    const uint64_t control = 0x0202;   // w0=2, w1=2
    BENCHSTART;
    TPACK(dst, src0, src1, control);
    BENCHEND;
    TSTORE_CUBE(gD, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
