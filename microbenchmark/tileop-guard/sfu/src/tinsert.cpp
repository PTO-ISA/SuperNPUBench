#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TINSERT — insert a sub-tile at (indexRow, indexCol).
// v0.58 signature: TINSERT(dst, src, int32_t indexRow, int32_t indexCol).
//   dst[indexRow+j, indexCol+i] = src[j,i]; the rest of dst is preserved
//   (in-place patch), so dst must be pre-loaded with a base tile.
// docs/tileop-usage layout.md gives NO C++ signature; 4-arg index-offset form
// recovered from the header contract. Precision: res_check; host owns base +
// patch; numpy golden overwrites the same window.
constexpr int DM = 16, DN = 32, DNE = DM * DN;   // dst base 16x32
constexpr int SM = 8,  SN = 16, SNE = SM * SN;   // patch 8x16
constexpr int OR = 4, OC = 8;                    // insert offset (row,col)
static float base[DNE], patch[SNE], out[DNE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", base, sizeof(base));
    guard_read_bin(CHK_DIR "/in_b.bin", patch, sizeof(patch));
#else
    for (int i = 0; i < DNE; ++i) base[i] = (float)i;
    for (int i = 0; i < SNE; ++i) patch[i] = -(float)(i + 1);
#endif
    iter_t<float, DM, DN> gB(base), gO(out);
    iter_t<float, SM, SN> gP(patch);
    auto gB0 = gB(0, 0);
    auto gP0 = gP(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<float, DM, DN> tD;
    vtile_t<float, SM, SN> tP;
    TLOAD(tD, gB0);        // dst base (preserved outside the window)
    TLOAD(tP, gP0);        // patch
    BENCHSTART;
    TINSERT(tD, tP, OR, OC);
    BENCHEND;
    TSTORE(gO0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
