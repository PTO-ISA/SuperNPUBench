#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: MGATHER (TLSU memory) — gather from GM by byte offsets.
// Source: docs/tileop-usage/tlsu/memory/MGATHER.md — FULL signature + example:
//     template <tile_out, tile_offset, gm_shape, TmaPadValue Pad=Null>
//     void MGATHER(tile_out &dst, const gm_shape &src, const tile_offset &offset);
//   Example: GM=global_tensor<float,RowMajor<8,1024>>; dst=Tile<Vec,float,8,32>.
//   Doc prose: "offset 中的每个元素是相对于 GM base 的字节位移" -> the addressed
//   element is *(float*)((char*)base + offset[i]); dst collects them.
// NOTE(doc-gap): the MGATHER.md *example* declares offsets as Tile<Vec,uint16_t,...>,
//   but the SAME page's dtype table restricts the index Tile to S32/U32/S64/U64.
//   The uint16 example is self-inconsistent: built verbatim it gfrun-rejects with
//   "illegal MGATHER operand or descriptor contract". We follow the NORMATIVE
//   dtype table (U32) instead of the buggy example. Recorded in REPORT.
// Non-trivial offsets (a strided permutation over the whole base), so the golden
// really exercises the gather addressing rather than an identity copy.
// Precision: res_check; host owns base + offsets; numpy golden = base[off//4].
constexpr int BM = 8, BN = 1024, BNE = BM * BN;    // base 8x1024 float
constexpr int OM = 8, ON = 32, ONE = OM * ON;      // offsets / out 8x32
static float base[BNE], out[ONE];
static uint32_t off[ONE];                          // U32 byte displacement (dtype table)
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_base.bin", base, sizeof(base));
    guard_read_bin(CHK_DIR "/in_off.bin", off, sizeof(off));
#else
    for (int i = 0; i < BNE; ++i) base[i] = (float)i;
    for (int i = 0; i < ONE; ++i) off[i] = (uint32_t)(((i * 101 + 7) % BNE) * 4);
#endif
    gm_t<float, BM, BN> base_gm(base);
    iter_t<uint32_t, OM, ON> gOff(off);
    iter_t<float, OM, ON> gOut(out);
    auto gOff0 = gOff(0, 0);
    auto gOut0 = gOut(0, 0);
    vtile_t<uint32_t, OM, ON> offset;
    vtile_t<float, OM, ON> dst;
    TLOAD(offset, gOff0);
    BENCHSTART;
    MGATHER(dst, base_gm, offset);
    BENCHEND;
    TSTORE(gOut0, dst);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
