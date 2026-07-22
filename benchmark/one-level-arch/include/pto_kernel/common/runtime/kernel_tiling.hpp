#ifndef PTO_COMMON_RUNTIME_KERNEL_TILING_HPP
#define PTO_COMMON_RUNTIME_KERNEL_TILING_HPP

#ifndef PTO_GEMM_TILE_M
#define PTO_GEMM_TILE_M 16
#endif

#ifndef PTO_GEMM_TILE_N
#define PTO_GEMM_TILE_N 16
#endif

#ifndef PTO_GEMM_TILE_K
#define PTO_GEMM_TILE_K 4
#endif

#ifndef PTO_FLASH_TILE_M
#define PTO_FLASH_TILE_M 16
#endif

#ifndef PTO_FLASH_TILE_K
#define PTO_FLASH_TILE_K 4
#endif

#ifndef PTO_FLASH_VEC_TILE_M
#define PTO_FLASH_VEC_TILE_M 8
#endif

#ifndef PTO_FLASH_VEC_TILE_K
#define PTO_FLASH_VEC_TILE_K 4
#endif

#ifndef PTO_FLASH_VEC_YDIM
#define PTO_FLASH_VEC_YDIM 4
#endif

#ifndef PTO_FLASH_CUBE_TILE_M
#define PTO_FLASH_CUBE_TILE_M 16
#endif

#ifndef PTO_FLASH_CUBE_TILE_K
#define PTO_FLASH_CUBE_TILE_K 16
#endif

#ifndef PTO_FLASH_CUBE_YDIM
#define PTO_FLASH_CUBE_YDIM 2
#endif

namespace pto {
namespace kernels {
namespace tiling {

inline constexpr int kGemmTileM = PTO_GEMM_TILE_M;
inline constexpr int kGemmTileN = PTO_GEMM_TILE_N;
inline constexpr int kGemmTileK = PTO_GEMM_TILE_K;

inline constexpr int kFlashTileM = PTO_FLASH_TILE_M;
inline constexpr int kFlashTileK = PTO_FLASH_TILE_K;

inline constexpr int kFlashVecTileM = PTO_FLASH_VEC_TILE_M;
inline constexpr int kFlashVecTileK = PTO_FLASH_VEC_TILE_K;
inline constexpr int kFlashVecYDim = PTO_FLASH_VEC_YDIM;

inline constexpr int kFlashCubeTileM = PTO_FLASH_CUBE_TILE_M;
inline constexpr int kFlashCubeTileK = PTO_FLASH_CUBE_TILE_K;
inline constexpr int kFlashCubeYDim = PTO_FLASH_CUBE_YDIM;

} // namespace tiling
} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_RUNTIME_KERNEL_TILING_HPP
