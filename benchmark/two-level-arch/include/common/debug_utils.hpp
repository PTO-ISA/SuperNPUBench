#ifndef DEBUG_UIILS_HPP
#define DEBUG_UIILS_HPP

#if defined(__linx)
namespace pto {
template <typename tile_shape>
void print_tile(tile_shape &) {}
} // namespace pto
#elif defined(__ARM_FEATURE_SME)
#include "aarch64/utils.hpp"
#elif defined(__cpu_sim__)
#include "cpu_sim/utils.hpp"
#endif

#ifndef __linx
namespace pto {
template <typename tile_shape>
void print_tile(tile_shape &tile) {
  print_tile_Impl(tile);
}

} // namespace pto
#endif

#endif
