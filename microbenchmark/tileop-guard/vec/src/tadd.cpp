#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TADD (VEC, elementwise-tile-tile, binary)
// Source: docs/tileop-usage/engines.md — VEC | TADD | elementwise-tile-tile.
// NOTE(doc-gap): engines.md lists the op + class but gives NO signature for
// basic arithmetic; the (dst,src0,src1) call shape is inferred from the
// dst-first convention shown in cmp.md / cube.md.
//
// Precision: res_check path. Inputs are host-generated (golden.py gen) and read
// via read() syscall — device-side fills get mangled by the tile backend, so
// host owns the inputs. Output is dumped for an independent numpy golden.
constexpr int GM = 16, GN = 16, GNE = GM * GN;
static float a[GNE], b[GNE], c[GNE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", a, sizeof(a));
    guard_read_bin(CHK_DIR "/in_b.bin", b, sizeof(b));
#else
    guard_fill_seq_f32(a, GNE, 1.0f, 0.1f);
    guard_fill_seq_f32(b, GNE, 2.0f, 0.3f);
#endif
    BENCHSTART;
    g_binary<float, GM, GN>(c, a, b, [](auto& d, auto& s0, auto& s1){ TADD(d, s0, s1); });
    BENCHEND;
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", c, sizeof(c));
#endif
    return 0;
}
