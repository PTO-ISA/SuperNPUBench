#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: range::subview — source-side range carrier over TSTORE.
// Correct usage per range-modifiers-developer-guide.md: lowercase helper
// range::subview(tile) (SizeCode/Offset/RegSrc auto-derived from the tile type);
// the uppercase range::Subview<...> with hand-filled params is the low-level
// compat interface the guide says NOT to use. subview of a full tile = identity.
constexpr int M = 4, N = 8, NE = M * N;
static float ha[NE], hc[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
#else
    gfill_seq(ha, NE);
#endif
    using Src = vtile_t<float, M, N>;
    Src s;
    iter_t<float, M, N> gA(ha);
    auto gA0 = gA(0, 0);
    global_tensor<float, RowMajor<M, N>> gm(hc);
    TLOAD(s, gA0);
    BENCHSTART;
    auto sv = range::subview(s);       // lowercase helper
    TSTORE(gm, sv);
    BENCHEND;
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
#endif
    return 0;
}
