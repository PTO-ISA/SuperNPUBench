#include <pto/linx/TileOps.hpp>

extern "C" void gemm_tile(int32_t *out, const int32_t *lhs,
                           const int32_t *rhs) {
  constexpr unsigned kTileSize = 8;
  auto t_lhs = pto::linx::tload<kTileSize>(lhs);
  auto t_rhs = pto::linx::tload<kTileSize>(rhs);
  auto t_out = pto::linx::tmatmul<8, 8, 8>(t_lhs, t_rhs);
  pto::linx::tstore<kTileSize>(out, t_out);
}
