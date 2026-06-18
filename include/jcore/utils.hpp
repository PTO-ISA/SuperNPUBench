#ifndef JCORES_UTILS_HPP
#define JCORES_UTILS_HPP

#include <iomanip>
#include "common/pto_tileop.hpp"
#include "common/tileop_api.hpp"

namespace pto {
template <typename tile_shape>
static inline __attribute__((always_inline))
void print_tile_Impl(tile_shape &tile) {
  static constexpr size_t tile_size = tile_shape::Rows * tile_shape::Cols;
  typename tile_shape::DType d[tile_size];
  using gm_shape =
      std::conditional_t<tile_shape::isRowMajor || tile_shape::isBoxedLayout,
          global_tensor<typename tile_shape::DType, RowMajor<tile_shape::Rows, tile_shape::Cols>>,
          global_tensor<typename tile_shape::DType, ColMajor<tile_shape::Rows, tile_shape::Cols>>>;
  gm_shape dst(d);
  TCOPYOUT(dst, tile);

  print_tile_info<tile_shape>();
  std::cout << std::fixed << std::setprecision(4);
  if constexpr (!gm_shape::isRowMajor) {
    for (int i = 0; i < tile_shape::Rows; i++) {
      for (int j = 0; j < tile_shape::Cols; j++) {
        int offset = j * tile_shape::Rows + i;
        std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
        if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
          std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
      }
      if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
        std::cout << std::endl;
        for (int k = 0; k < tile_shape::ValidCol; k++)
          std::cout << std::setw(8) << "-" << "\t";
      }
      std::cout << std::endl;
    }
  } else {
      for (int i = 0; i < tile_shape::Rows; i++) {
        for (int j = 0; j < tile_shape::Cols; j++) {
          int offset = i * tile_shape::Cols + j;
          std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
          if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
				    std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
        }
        if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
          std::cout << std::endl;
          for (int k = 0; k < tile_shape::ValidCol; k++)
            std::cout << std::setw(8) << "-" << "\t";
        }
        std::cout << std::endl;
      }
  }
}

} // namspace end

#endif // JCORES_UTILS_HPP
