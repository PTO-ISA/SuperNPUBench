#ifndef PTO_COMMON_RUNTIME_KERNEL_ENV_HPP
#define PTO_COMMON_RUNTIME_KERNEL_ENV_HPP

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

#ifndef PTO_USE_MIXED_TILE_SIMT
#define PTO_USE_MIXED_TILE_SIMT 0
#endif

namespace pto {
namespace kernels {
namespace env {

inline constexpr bool kQemuSmoke = PTO_QEMU_SMOKE != 0;
inline constexpr bool kMixedTileSimt = PTO_USE_MIXED_TILE_SIMT != 0;

template <typename T>
inline constexpr T select(T smoke_value, T full_value) {
  return kQemuSmoke ? smoke_value : full_value;
}

} // namespace env
} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_RUNTIME_KERNEL_ENV_HPP
