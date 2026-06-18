#ifndef DEBUG_UIILS_HPP
#define DEBUG_UIILS_HPP

#ifdef __linx
#include "jcore/utils.hpp"
#elif defined(__ARM_FEATURE_SME)
#include "aarch64/utils.hpp"
#elif defined(__cpu_sim__)
#include "cpu_sim/utils.hpp"
#endif

namespace pto {
template <typename tile_shape>
static inline __attribute__((always_inline))
void print_tile(tile_shape &tile) {
  print_tile_Impl(tile);
}

} // namespace pto

#endif