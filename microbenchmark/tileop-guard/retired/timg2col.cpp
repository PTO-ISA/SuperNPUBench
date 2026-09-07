#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TIMG2COL (image-to-column, the convolution primitive).
// Source: docs/tileop-usage/layout-and-rearrangement/layout/TIMG2COL.md — FULL
//   signature + example. Written STRICTLY from that example (a plain Location::Vec
//   RowMajor tile loaded via TLOAD, then TIMG2COL(dst, src, posM, posK)):
//     using GM   = global_tensor<float, RowMajor<8, 256>>;
//     using TileT= Tile<Location::Vec, float, 8, 256, BLayout::RowMajor>;
//     TLOAD(src, src_global); TIMG2COL(dst, src, 3, 5);
//   The doc example does NOT construct a persistent Matrix-location feature-map
//   descriptor source; we deliberately do NOT invent one (that would require
//   reading the ISA/ASL, out of scope for a doc-driven guard).
// STATUS: on the env_test model TIMG2COL is a fail-closed assertion stub
//   (convolution window / repeat / padding contract not implemented), so this
//   doc-faithful demo reaches gfrun and aborts -> run-fail. Recorded in REPORT
//   as "doc example does not run on the current model".
constexpr int GM = 8, GN = 256, GNE = GM * GN;
static float src_data[GNE], out[GNE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src_data, sizeof(src_data));
#else
    guard_fill_seq_f32(src_data, GNE, 1.0f, 0.1f);
#endif
    iter_t<float, GM, GN> gSrc(src_data), gOut(out);
    auto gSrc0 = gSrc(0, 0);
    auto gOut0 = gOut(0, 0);
    vtile_t<float, GM, GN> src, dst;
    TLOAD(src, gSrc0);
    BENCHSTART;
    TIMG2COL(dst, src, /*posM=*/3, /*posK=*/5);
    BENCHEND;
    TSTORE(gOut0, dst);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
