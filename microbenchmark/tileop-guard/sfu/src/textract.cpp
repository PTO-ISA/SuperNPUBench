#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TEXTRACT — extract a sub-tile at (indexRow, indexCol).
// v0.58 signature: TEXTRACT(dst, src, int32_t indexRow, int32_t indexCol).
//   dst[j,i] = src[indexRow+j, indexCol+i]   (sub-tile copy).
// docs/tileop-usage layout.md gives NO C++ signature; the 4-arg form (and the
// index-offset semantics) is recovered from the header contract. Precision:
// res_check; host owns src; numpy golden extracts the same block.
constexpr int SM = 16, SN = 32, SNE = SM * SN;   // src 16x32
constexpr int DM = 8,  DN = 16, DNE = DM * DN;   // dst 8x16
constexpr int OR = 4, OC = 8;                    // extract offset (row,col)
static float src[SNE], out[DNE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#else
    for (int i = 0; i < SNE; ++i) src[i] = (float)i;
#endif
    iter_t<float, SM, SN> gS(src);
    iter_t<float, DM, DN> gO(out);
    auto gS0 = gS(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<float, SM, SN> tS;
    vtile_t<float, DM, DN> tD;
    TLOAD(tS, gS0);
    BENCHSTART;
    TEXTRACT(tD, tS, OR, OC);
    BENCHEND;
    TSTORE(gO0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
