#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: MGATHER_MASK (TLSU memory) — masked gather from GM.
// Source: docs/tileop-usage/tlsu/memory/MGATHER_MASK.md — FULL signature+example:
//     template <tile_out, tile_offset, tile_mask, gm_shape, TmaPadValue Pad=Null>
//     void MGATHER_MASK(tile_out &dst, const gm_shape &src,
//                       const tile_offset &offset, const tile_mask &mask);
//   Doc prose: "只收集谓词恰为 1 的 lane，并用选定的 padding 值填充禁用的 lane" ->
//   dst[i] = (mask[i]==1) ? base[off[i]//elem] : Pad. Example uses Pad=Zero,
//   mask = Tile<Vec,uint8_t,...>.
// NOTE(doc-gap): example offsets are uint16_t but the dtype table restricts index
//   Tiles to S32/U32/S64/U64; we use U32 (详述 18). Mask stays uint8 per example.
// Precision: res_check; golden = where(mask==1, base[off//4], 0.0).
constexpr int BM = 8, BN = 1024, BNE = BM * BN;
constexpr int OM = 8, ON = 32, ONE = OM * ON;
static float base[BNE], out[ONE];
static uint32_t off[ONE];
static uint8_t msk[ONE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_base.bin", base, sizeof(base));
    guard_read_bin(CHK_DIR "/in_off.bin", off, sizeof(off));
    guard_read_bin(CHK_DIR "/in_mask.bin", msk, sizeof(msk));
#else
    for (int i = 0; i < BNE; ++i) base[i] = (float)i;
    for (int i = 0; i < ONE; ++i) { off[i] = (uint32_t)(((i * 101 + 7) % BNE) * 4); msk[i] = (uint8_t)(i % 3 ? 1 : 0); }
#endif
    using Values = vtile_t<float, OM, ON>;
    using ByteOffsets = vtile_t<uint32_t, OM, ON>;
    using Mask = vtile_t<uint8_t, OM, ON>;
    using GM = gm_t<float, BM, BN>;
    GM base_gm(base);
    iter_t<uint32_t, OM, ON> gOff(off);
    iter_t<uint8_t, OM, ON> gMsk(msk);
    iter_t<float, OM, ON> gOut(out);
    auto gOff0 = gOff(0, 0);
    auto gMsk0 = gMsk(0, 0);
    auto gOut0 = gOut(0, 0);
    Values dst;
    ByteOffsets offset;
    Mask mask;
    TLOAD(offset, gOff0);
    TLOAD(mask, gMsk0);
    BENCHSTART;
    MGATHER_MASK<Values, ByteOffsets, Mask, GM, TmaPadValue::Zero>(dst, base_gm, offset, mask);
    BENCHEND;
    TSTORE(gOut0, dst);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
