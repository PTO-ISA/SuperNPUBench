#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TSHUF (SFU/TEPL layout, PTO 0.58.5) — shuffle 32-bit word
// groups across power-of-two CUBE row segments per GPR control.
// Source: layout-and-rearrangement/layout/TSHUF.md (v0.58.5).
// Precision: res_check, *identity* config. pto-spec rearrangement.asl:222 selects a
// source row per (row,word) from control; a full shuffle golden needs the physical
// layout model. Here mode=UP(0), controls all 0 (b=0) -> candidate_row == row ->
// dst == src (layout-invariant). control scalar = 0. Golden = src (in_a).
constexpr int M = 32, N = 32, NE = M * N;
static uint32_t ga[NE], gctrl[NE], gout[NE];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ga, sizeof(ga));
    guard_read_bin(CHK_DIR "/in_ctrl.bin", gctrl, sizeof(gctrl));
    using T = VecTileM32<uint32_t, M, N>;
    T src, controls, dst;
    global_tensor<uint32_t, RowMajor<M, N>> gA(ga), gC(gctrl), gD(gout);
    TLOAD_CUBE(src, gA);
    TLOAD_CUBE(controls, gC);
    const uint64_t control = 0;   // mode=UP, segment_code=0, boundary=SELF
    BENCHSTART;
    TSHUF(dst, src, controls, control);
    BENCHEND;
    TSTORE_CUBE(gD, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
