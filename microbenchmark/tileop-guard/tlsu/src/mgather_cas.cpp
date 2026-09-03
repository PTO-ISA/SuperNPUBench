#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: MGATHER_CAS (TLSU memory) — per-element compare-and-swap.
// Source: docs/tileop-usage/tlsu/memory/MGATHER_CAS.md — FULL signature + prose.
//   void MGATHER_CAS(DstTile &observedOld, uint64_t base,
//                    IndexTile &byteDisplacements, ExpectedTile &expected,
//                    ReplacementTile &replacement, uint32_t validCol,
//                    uint32_t validRow = 1);
//   Doc semantics: addr = base + byteDisplacement[i]. observedOld[i] = *addr
//   (value read BEFORE the swap, always). If *addr == expected[i] then
//   *addr = replacement[i], else *addr unchanged.
// Precision: res_check. base = address of a host-owned backing array; golden
//   checks BOTH observedOld (out.bin) AND the post-CAS backing (out_mem.bin).
//   Injective offsets; even lanes HIT (expected==slot) / odd lanes MISS.
constexpr int BM = 8, BN = 1024, BNE = BM * BN;    // backing 8x1024 float (GM)
constexpr int OM = 8, ON = 32, ONE = OM * ON;      // old/exp/rep/off 8x32
static float backing[BNE], expv[ONE], repv[ONE], oldv[ONE];
static uint32_t off[ONE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_base.bin", backing, sizeof(backing));
    guard_read_bin(CHK_DIR "/in_exp.bin", expv, sizeof(expv));
    guard_read_bin(CHK_DIR "/in_rep.bin", repv, sizeof(repv));
    guard_read_bin(CHK_DIR "/in_off.bin", off, sizeof(off));
#else
    for (int i = 0; i < BNE; ++i) backing[i] = (float)i;
    for (int i = 0; i < ONE; ++i) {
        int idx = (i * 101 + 7) % BNE; off[i] = (uint32_t)(idx * 4);
        repv[i] = (float)(-i - 1);
        expv[i] = (i & 1) ? backing[idx] + 1000.0f : backing[idx];  // odd miss / even hit
    }
#endif
    iter_t<float, OM, ON> gE(expv), gR(repv), gO(oldv);
    iter_t<uint32_t, OM, ON> gI(off);
    auto gE0 = gE(0, 0);
    auto gR0 = gR(0, 0);
    auto gO0 = gO(0, 0);
    auto gI0 = gI(0, 0);
    vtile_t<float, OM, ON> tOld, tExp, tRep;
    vtile_t<uint32_t, OM, ON> tIdx;
    TLOAD(tExp, gE0);
    TLOAD(tRep, gR0);
    TLOAD(tIdx, gI0);
    BENCHSTART;
    MGATHER_CAS(tOld, (uint64_t)(uintptr_t)backing, tIdx, tExp, tRep, ON, OM);
    BENCHEND;
    TSTORE(gO0, tOld);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", oldv, sizeof(oldv));            // observedOld
    guard_dump_bin(CHK_DIR "/out_mem.bin", backing, sizeof(backing));  // post-CAS GM
#endif
    return 0;
}
