#include <cstdint>

#include <pto/linx/TileOps.hpp>

extern "C" void tile_add(std::int32_t *out, const std::int32_t *lhs,
                          const std::int32_t *rhs) {
  constexpr unsigned kTileSize = 8;
  auto t_lhs = pto::linx::tload<kTileSize>(lhs);
  auto t_rhs = pto::linx::tload<kTileSize>(rhs);
  auto t_out = pto::linx::tadd<kTileSize>(t_lhs, t_rhs);
  pto::linx::tstore<kTileSize>(out, t_out);
}
