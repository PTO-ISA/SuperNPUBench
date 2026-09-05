#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TGPR2T (SFU/TEPL layout, PTO 0.58.5) — re-encode four
// 64-bit GPR predicate plane carriers into a U8 CUBE tile.
// Source: layout-and-rearrangement/layout/TGPR2T.md (v0.58.5).
// Precision: res_check, *sanity* config (no pto-spec ASL: spec still v0.58.4).
// All-zero predicate planes -> all-zero U8 output (zero-in -> zero-out repack).
// dst must be CUBE_M32 32x4 or CUBE_M16 16x8 (static_assert). Golden = zeros.
constexpr int M = 32, N = 4, NE = M * N;
static uint8_t gout[NE];
int main() {
    using T = VecTileM32<uint8_t, M, N>;
    T dst;
    global_tensor<uint8_t, RowMajor<M, N>> gD(gout);
    BENCHSTART;
    TGPR2T(dst, 0ull, 0ull, 0ull, 0ull);
    BENCHEND;
    TSTORE_CUBE(gD, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
