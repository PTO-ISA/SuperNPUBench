#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: reinterpret_tile + TCMP — WITNESS of a model-side gap.
// Same both-views bitcast usage as reinterpret_tile.cpp (TANDS), but the
// consuming op is TCMP: bitcast two fp32 tiles to int32 views and compare their
// bit patterns (TCMP<GT> on the int32 views -> predicate), consume with TSEL,
// store the select result. Would-be golden = where(int_bits(a)>int_bits(b), tru, prior).
//
// STATUS (run-fail, genuine model gap): TCMP<tile_out,tile_in> has SEPARATE
// out/in template params, so a plain predicate dst + two int32 VIEW sources
// COMPILES. But the emulator's TCMP handler rejects the view sources at runtime:
//   gfrun: "TCMP requires two compatible Tile sources"
//          (IsCompatibleDataTile on srcs[1]/srcs[2], before any store).
// Contrast: the SAME views feed TANDS fine (reinterpret_tile.cpp PASSES), and
// plain int32 tiles feed TCMP fine (vec/tcmp PASSES) — so the view source is the
// differentiator. This is the one spot the official reinterpret fix missed, at
// the model/emulator layer (compiler/header already accept the view). File as a
// [gfrun][NA] issue; this case flips to PASS once the TCMP handler accepts
// reinterpret-view sources like TANDS does.
constexpr int M = 16, N = 16, NE = M * N;
static float A[NE], B[NE];
static int32_t prior[NE], tru[NE], out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", A, sizeof(A));
    guard_read_bin(CHK_DIR "/in_b.bin", B, sizeof(B));
    guard_read_bin(CHK_DIR "/in_c.bin", prior, sizeof(prior));
    guard_read_bin(CHK_DIR "/in_d.bin", tru, sizeof(tru));
#else
    for (int i = 0; i < NE; ++i) { A[i] = (float)i * 0.5f; B[i] = (float)((i * 3) % 17); prior[i] = -i - 1; tru[i] = i + 100; }
#endif
    iter_t<float, M, N> gA(A), gB(B);
    iter_t<int32_t, M, N> gP(prior), gT(tru), gO(out);
    auto a0 = gA(0, 0), b0 = gB(0, 0);
    auto p0 = gP(0, 0), t0 = gT(0, 0), o0 = gO(0, 0);
    vtile_t<float, M, N> tA, tB;
    vtile_t<int32_t, M, N> tMask, tD, tTru;
    TLOAD(tA, a0);
    TLOAD(tB, b0);
    TLOAD(tD, p0);         // dst prior = false source (TSEL in-place)
    TLOAD(tTru, t0);
    BENCHSTART;
    auto vA = reinterpret_tile<int32_t>(tA);   // fp32 bits as int32
    auto vB = reinterpret_tile<int32_t>(tB);
    TCMP<CmpMode::GT>(tMask, vA, vB);          // <-- model rejects the view sources here
    TSEL(tD, tMask, tTru);                     // dst = mask ? tru : prior
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
