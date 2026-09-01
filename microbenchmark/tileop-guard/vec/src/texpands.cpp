#include "guard_case.hpp"
// TileOP-API doc guard: TEXPANDS (VEC, tile-scalar-and-immediate): dst filled with
// scalar (2 args, no src). Source: engines.md. Precision: res_check, scalar=1.75.
constexpr int GM = 16, GN = 16;
static float gC[GM * GN];
int main() {
    iter_t<float, GM, GN> gc(gC);
    auto gc0 = gc(0, 0);
    vtile_t<float, GM, GN> tC;
    BENCHSTART;
    TEXPANDS(tC, 1.75f);
    BENCHEND;
    TSTORE(gc0, tC);
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC));
    return 0;
}
