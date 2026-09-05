#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: MSCATTER (TLSU memory) — scatter to GM by byte offsets.
// Source: docs/tileop-usage/tlsu/memory/MSCATTER.md — FULL signature + example:
//     template <tile_in, tile_offset, gm_shape>
//     void MSCATTER(gm_shape &dst, const tile_in &src, const tile_offset &offset);
//   Doc prose: "offset 中的每个元素是相对于 GM base 的字节位移" -> the addressed
//   slot base + offset[i] receives src[i]: base[off[i]//elem] = src[i].
// NOTE(doc-gap): same as MGATHER — the example declares offsets uint16_t but the
//   dtype table restricts index Tiles to S32/U32/S64/U64; we use U32 (详述 18).
// Offsets are an INJECTIVE map (no two lanes hit the same slot), so the scatter
//   is order-independent and the numpy golden is deterministic.
// Precision: res_check. Output checked = the base array AFTER the scatter.
constexpr int BM = 8, BN = 1024, BNE = BM * BN;    // base 8x1024 float
constexpr int OM = 8, ON = 32, ONE = OM * ON;      // src / offsets 8x32
static float base[BNE], src[ONE];
static uint32_t off[ONE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_base.bin", base, sizeof(base));
    guard_read_bin(CHK_DIR "/in_src.bin", src, sizeof(src));
    guard_read_bin(CHK_DIR "/in_off.bin", off, sizeof(off));
#else
    for (int i = 0; i < BNE; ++i) base[i] = (float)i;
    for (int i = 0; i < ONE; ++i) { src[i] = (float)(-i - 1); off[i] = (uint32_t)(((i * 101 + 7) % BNE) * 4); }
#endif
    gm_t<float, BM, BN> base_gm(base);
    iter_t<float, OM, ON> gSrc(src);
    iter_t<uint32_t, OM, ON> gOff(off);
    auto gSrc0 = gSrc(0, 0);
    auto gOff0 = gOff(0, 0);
    vtile_t<float, OM, ON> tSrc;
    vtile_t<uint32_t, OM, ON> offset;
    TLOAD(tSrc, gSrc0);
    TLOAD(offset, gOff0);
    BENCHSTART;
    MSCATTER(base_gm, tSrc, offset);
    BENCHEND;
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", base, sizeof(base));
#endif
    return 0;
}
