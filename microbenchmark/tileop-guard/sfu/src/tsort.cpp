#include "guard_case.hpp"
// TileOP-API doc guard: TSORT (SFU irregular) — per-row sort, width=32 ascending.
// Source: sort.md full signature TSORT(valueDst, indexDst, source, width, desc).
// Precision: res_check checks sorted VALUES (out.bin); distinct per-row inputs
// make the ordering unambiguous. valueDst/source FP32, indexDst U32.
static float gS[32 * 32], gV[32 * 32];
static uint32_t gI[32 * 32];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", gS, sizeof(gS));
    constexpr int M = 32, N = 32;
    iter_t<float, M, N> ggS(gS), ggV(gV);
    iter_t<uint32_t, M, N> ggI(gI);
    auto s0 = ggS(0, 0);
    auto v0 = ggV(0, 0);
    auto i0 = ggI(0, 0);
    vtile_t<float, M, N> tS, tV;
    vtile_t<uint32_t, M, N> tI;
    TLOAD(tS, s0);
    BENCHSTART; TSORT(tV, tI, tS); BENCHEND;
    TSTORE(v0, tV);
    TSTORE(i0, tI);
    guard_dump_bin(CHK_DIR "/out.bin", gV, sizeof(gV));
    return 0;
}
