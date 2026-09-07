#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: range::assemble — destination-side range carrier over TLOAD.
// Correct usage per range-modifiers-developer-guide.md: lowercase lifecycle helper
// (params auto-derived). A standalone single fragment is a complete session ->
// assemble_init_last (INIT=1, LAST=1). The uppercase range::Assemble<...,INIT=1,
// LAST=0,...> hand-filled form (session left open) was a demo misuse.
constexpr int M = 4, N = 8, NE = M * N;
static float ha[NE], hc[NE];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    using Dst = vtile_t<float, M, N>;
    Dst d;
    global_tensor<float, RowMajor<M, N>> gm(ha);
    iter_t<float, M, N> gC(hc);
    auto gC0 = gC(0, 0);
    BENCHSTART;
    auto as = range::assemble_init_last(d, 0);   // lowercase helper, complete session
    TLOAD(as, gm);
    BENCHEND;
    TSTORE(gC0, d);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
