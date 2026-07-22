#include "common/pto_tile.hpp"
#include "jcore/template_asm.hpp"

using namespace pto;

namespace {

using TestShape = Shape<1, 1, 1, 8, 8>;
using TestStride = Stride<1, 1, 64, 8, 1>;
using TestGlobal = GlobalTensor<int32_t, TestShape, TestStride, Layout::ND>;
using TestTile = Tile<Location::Vec, int32_t, 8, 8>;

} // namespace

extern "C" void compile_v057_tma_wrappers(int32_t *base) {
  TestGlobal global(base);
  TestTile dst;
  TestTile data;
  TestTile offset;
  TestTile mask;
  TestTile expected;
  TestTile desired;

  TPREFETCH<TestTile>(global);
  MGATHER(dst, global, offset);
  MSCATTER(global, data, offset);
  MGATHER_MASK(dst, global, offset, mask);
  MSCATTER_MASK(global, data, offset, mask);
  MGATHER_CAS(dst, global, offset, expected, desired);
}
