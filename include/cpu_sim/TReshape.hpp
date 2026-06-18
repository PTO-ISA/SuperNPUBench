#ifndef TRESHAPE_HPP
#define TRESHAPE_HPP

#include "common/pto_tile.hpp"
#include <cstring>

using namespace pto;

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TRESHAPE_Impl(tile_shape_out &tile_out, tile_shape_in &tile_in) {
  std::memcpy(tile_out.data(), tile_in.data(),
              tile_shape_in::Rows * tile_shape_in::Cols *
                  sizeof(typename tile_shape_in::DType));
}

#endif