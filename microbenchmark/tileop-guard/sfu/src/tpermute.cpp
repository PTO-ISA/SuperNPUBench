#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TPERMUTE (SFU/TEPL layout, PTO 0.58.5) — byte-select from
// two CUBE sources via a U8 index tile; lookup restarts per 128-byte CELL.
// Source: layout-and-rearrangement/layout/TPERMUTE.md (v0.58.5).
// Precision: res_check, *identity* config. pto-spec rearrangement.asl:177 reads
// physical cell bytes; a full permute golden needs the CUBE fractal layout model.
// Here index byte j = j mod 4 (M32 row_bytes=4) -> each dst byte reads src0's own
// byte within its word -> dst == src0 (layout-invariant). Golden = src0 (in_a).
constexpr int M = 32, N = 32, NE = M * N;
static uint32_t ga[NE], gb[NE], gout[NE];
static uint8_t  gidx[NE * 4];               // one U8 index per dst byte (32x128)
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ga, sizeof(ga));
    guard_read_bin(CHK_DIR "/in_b.bin", gb, sizeof(gb));
    guard_read_bin(CHK_DIR "/in_idx.bin", gidx, sizeof(gidx));
    using T = VecTileM32<uint32_t, M, N>;
    using I = VecTileM32<uint8_t, M, N * 4>;
    T src0, src1, dst; I indices;
    global_tensor<uint32_t, RowMajor<M, N>> gA(ga), gB(gb), gD(gout);
    global_tensor<uint8_t, RowMajor<M, N * 4>> gI(gidx);
    TLOAD_CUBE(src0, gA);
    TLOAD_CUBE(src1, gB);
    TLOAD_CUBE(indices, gI);
    BENCHSTART;
    TPERMUTE(dst, src0, src1, indices);
    BENCHEND;
    TSTORE_CUBE(gD, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
