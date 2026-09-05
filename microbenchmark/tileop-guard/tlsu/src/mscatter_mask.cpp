#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: MSCATTER_MASK — masked scatter to GM.
// v0.58 header signature (recovered from jcore/template_asm.hpp):
//   MSCATTER_MASK(gm_shape &dst, const src&, const offset&, const mask&)
//   scatters src[i] to base+off[i] where mask[i]==1. Mirrors MGATHER_MASK.
// STATUS: correct 4-arg call; the emitted BSTART.TLSU MSCATTER.MASK bundle is
// not assembled by the current backend ("Match Instruction Error!"), same as
// MGATHER_MASK — a genuine backend gap, now with the correct signature. golden
// (where(mask==1)) is ready in golden.py once the backend supports the encoding.
constexpr int BM = 8, BN = 1024, BNE = BM * BN;
constexpr int OM = 8, ON = 32, ONE = OM * ON;
static float base[BNE], src[ONE];
static uint32_t off[ONE];
static uint8_t msk[ONE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_base.bin", base, sizeof(base));
    guard_read_bin(CHK_DIR "/in_src.bin", src, sizeof(src));
    guard_read_bin(CHK_DIR "/in_off.bin", off, sizeof(off));
    guard_read_bin(CHK_DIR "/in_mask.bin", msk, sizeof(msk));
#else
    for (int i = 0; i < BNE; ++i) base[i] = (float)i;
    for (int i = 0; i < ONE; ++i) { src[i] = -(float)(i + 1); off[i] = (uint32_t)(((i * 101 + 7) % BNE) * 4); msk[i] = (uint8_t)(i % 3 ? 1 : 0); }
#endif
    using Src = vtile_t<float, OM, ON>;
    using ByteOffsets = vtile_t<uint32_t, OM, ON>;
    using Mask = vtile_t<uint8_t, OM, ON>;
    gm_t<float, BM, BN> base_gm(base);
    iter_t<float, OM, ON> gSrc(src);
    iter_t<uint32_t, OM, ON> gOff(off);
    iter_t<uint8_t, OM, ON> gMsk(msk);
    auto gSrc0 = gSrc(0, 0);
    auto gOff0 = gOff(0, 0);
    auto gMsk0 = gMsk(0, 0);
    Src tSrc;
    ByteOffsets offset;
    Mask mask;
    TLOAD(tSrc, gSrc0);
    TLOAD(offset, gOff0);
    TLOAD(mask, gMsk0);
    BENCHSTART;
    MSCATTER_MASK(base_gm, tSrc, offset, mask);
    BENCHEND;
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", base, sizeof(base));
#endif
    return 0;
}
