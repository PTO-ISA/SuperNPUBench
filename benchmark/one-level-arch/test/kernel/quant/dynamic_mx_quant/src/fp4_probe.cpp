#include <common/pto_tileop.hpp>
#include <cstdint>

// Minimal fp4-emission probe: does the toolchain compile+link a single
// fp32 -> packed __fp4_e2m1x2 TCVT + TSTORE when the fp4 output tile is a
// boxed ColMajor fractal (fa_hif4.hpp:85-92 precedent) instead of the
// RowMajor+NoneBox [32,16] that trips pto_tile.hpp's 32B column assert?
using namespace pto;

static __bf16 x[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[32 * 16] __attribute__((aligned(4096))) = {};

int main() {
    using tile_x = Tile<Location::Vec, __bf16, 32, 32, BLayout::ColMajor>;
    using tile_f = Tile<Location::Vec, float, 32, 32, BLayout::ColMajor>;
    // Boxed fractal fp4 output: Rows=32, Cols=16 (packed), Inner=32/16, SLayout::ColMajor.
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, 32, 16, BLayout::ColMajor, 32, 16, SLayout::ColMajor>;

    using gm_x = global_tensor<__bf16,  ColMajor<32, 32>>;
    using gm_y = global_tensor<uint8_t, RowMajor<32, 16>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));

    auto gx = x_iter(0, 0);
    auto gy = y_iter(0, 0);

    tile_x xq;
    TLOAD(xq, gx);
    tile_f xf;
    TCVT(xf, xq);
    tile_o oq;
    TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 single-step (RMode round)
    TSTORE(gy, oq);
    return 0;
}
