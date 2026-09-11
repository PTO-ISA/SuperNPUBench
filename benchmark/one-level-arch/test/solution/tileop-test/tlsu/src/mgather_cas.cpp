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
    // Leader-gate (issue #489 idiom): MGATHER_CAS mutates SHARED GM state and is
    // NOT idempotent. gfrun runs 4 SPMD PE-threads (PE0..3), all executing this
    // block against the SAME `backing` base — without a gate the CAS applies 4x,
    // so observedOld on the 2nd..4th pass reads the already-swapped replacement
    // (verified via -t2: lane(0,0) old 4.5->-1.0 across 4 executions). The model's
    // per-execution CAS is correct (old captured before swap); the 4x replay is a
    // pure SPMD-structuring artifact. Confine the atomic + I/O to leader tid 0;
    // workers return and park (no output barrier -> no gfrun hang).
    if (get_thread_idx() != 0) return 0;
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
