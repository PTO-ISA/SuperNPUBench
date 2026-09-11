#include "guard_case.hpp"
// TileOP-API doc guard: TCONCAT (SFU layout) — column concatenation.
// (dst,s0,s1); dst cols = s0.cols + s1.cols (M x N ++ M x N -> M x 2N).
// Precision: res_check, independent golden = horizontal concat [a | b].
static float gA[16 * 8], gB[16 * 8], gC[16 * 16];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA));
    guard_read_bin(CHK_DIR "/in_b.bin", gB, sizeof(gB));
    constexpr int M = 16, N = 8;
    iter_t<float, M, N> ggA(gA), ggB(gB);
    iter_t<float, M, 2 * N> ggC(gC);
    auto a0 = ggA(0, 0);
    auto b0 = ggB(0, 0);
    auto c0 = ggC(0, 0);
    vtile_t<float, M, N> tA, tB;
    vtile_t<float, M, 2 * N> tC;
    TLOAD(tA, a0);
    TLOAD(tB, b0);
    BENCHSTART; TCONCAT(tC, tA, tB); BENCHEND;
    TSTORE(c0, tC);
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC));
    return 0;
}
