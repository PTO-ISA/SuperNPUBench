#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"

// Minimal RUNTIME probe: does a single [32,32] bf16 tile round-trip
//   TLOAD [32,32] -> TCVT bf16->fp8 -> TSTORE [32,32]
// with NO scale pass and NO TSEL. Purpose: isolate whether the emulator writes
// all 32 rows of a [32,32] data tile, or truncates to 16 (the symptom seen in
// nontail_cublas_fp8's data output). If this probe writes 32 nonzero rows, the
// 16-row truncation is specific to the scale/compute path, not the plain
// [32,32] load/cast/store data path.

using namespace pto;

static __bf16 x[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[32 * 32] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

#ifdef NARROW
    // [32,16] bf16 -> bf16 pure copy. Total 32*16*2=1024 B / 4 threads = 256 B/thread.
    // If per-thread-BYTES is the cap, this writes ALL 32 rows (vs [32,32]=512/thread=16 rows).
    using tile_x = Tile<Location::Vec, __bf16, 32, 16, BLayout::RowMajor>;
    using gm_x = global_tensor<__bf16, RowMajor<32, 16>>;
    using gm_y = global_tensor<__bf16, RowMajor<32, 16>>;
    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_x> y_iter(reinterpret_cast<__bf16*>(y));
    auto gx = x_iter(0, 0);
    auto gy = y_iter(0, 0);
    tile_x xq;
    TLOAD(xq, gx);
    TSTORE(gy, xq);
#else
    using tile_x = Tile<Location::Vec, __bf16,      32, 32, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float,       32, 32, BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, __fp8_e4m3,  32, 32, BLayout::RowMajor>;
    using gm_x = global_tensor<__bf16,   RowMajor<32, 32>>;
    using gm_y = global_tensor<uint8_t,  RowMajor<32, 32>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(y);

    auto gx = x_iter(0, 0);
    auto gy = y_iter(0, 0);
    tile_x xq;
    TLOAD(xq, gx);
    tile_o oq;
#ifdef NOCAST
    TCVT(oq, xq);         // direct bf16 -> fp8, NO fp32 [32,32] intermediate
#else
    tile_f xf;
    TCVT(xf, xq);         // bf16 -> fp32 [32,32] (1024 B/thread) intermediate
    TCVT(oq, xf);
#endif
    TSTORE(gy, oq);
#endif

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
#endif
    return 0;
}
