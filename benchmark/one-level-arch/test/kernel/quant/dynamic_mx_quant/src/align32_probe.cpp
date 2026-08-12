#include <common/pto_tileop.hpp>
#include <cstdint>

// Minimal reproduction of the 32B (256-bit) contiguous-axis alignment
// static_assert in pto_tile.hpp:649-656.
//
// The assert fires at Tile TYPE INSTANTIATION (not at load/store): for a
// RowMajor + NoneBox tile it requires the -1 (contiguous) axis physical byte
// width to be a multiple of 32 bytes:
//     Cols * type_traits<DType>::bits % (32 * 8) == 0
//
// __fp4_e2m1x2 packs 2 fp4 values per byte -> type_traits<>::bits == 8. A single
// MX block of fp4 output is BlockSize=32 values = 16 packed bytes, i.e. a
// [Rows, 16] RowMajor NoneBox tile:  16 * 8 = 128, 128 % 256 != 0  -> FAILS.
// Widening the contiguous axis to 2 MX blocks (Cols=32 packed bytes) satisfies
// it: 32 * 8 = 256, 256 % 256 == 0 -> OK. This is exactly why the
// dynamic_mx_quant fp4 kernels force TileN % 64 == 0 / pad the physical width.
//
// Build:
//   FAIL (default, reproduces the assert):
//     make TESTCASE=dynamic_mx_quant TYPE=ALIGN32_PROBE
//   OK (contiguous axis widened to 2 MX blocks, compiles clean):
//     make TESTCASE=dynamic_mx_quant TYPE=ALIGN32_PROBE CC_OPTS=-DALIGN_OK

using namespace pto;

int main() {
#ifdef ALIGN_OK
    // Cols = 32 packed bytes (= 2 MX blocks = 64 fp4 values): 32*8 % 256 == 0.
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, 8, 32, BLayout::RowMajor>;
#else
    // Cols = 16 packed bytes (= 1 MX block = 32 fp4 values): 16*8 % 256 == 128 != 0.
    // -> trips pto_tile.hpp:649 "... Rows must be 32 bytes align ...".
    using tile_o = Tile<Location::Vec, __fp4_e2m1x2, 8, 16, BLayout::RowMajor>;
#endif
    tile_o oq;        // instantiate the Tile type -> evaluates the static_assert
    (void)sizeof(oq);
    return 0;
}
