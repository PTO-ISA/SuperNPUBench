#ifndef TRESHAPE_HPP
#define TRESHAPE_HPP

#include "common/pto_tile.hpp"

using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TRESHAPE_Impl(tile_shape_out &tile_out, tile_shape_in &tile_in) {
  static_assert(tile_shape_in::ValidRow != DYNAMIC &&
                    tile_shape_in::ValidCol != DYNAMIC &&
                    tile_shape_out::ValidRow != DYNAMIC &&
                    tile_shape_out::ValidCol != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::Loc != Location::Acc &&
                    tile_shape_in::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(!tile_shape_out::isBoxedLayout &&
                    !tile_shape_in::isBoxedLayout,
                "TRESHAPE not support Boxed Layout!");
  static_assert(tile_shape_out::Numel == tile_shape_in::Numel,
                "TRESHAPE requires equal tile element counts");

  for (size_t index = 0; index < tile_shape_in::Numel; ++index) {
    tile_out.data()[index] = tile_in.data()[index];
  }
}
#else
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TRESHAPE_Impl(tile_shape_out &tile_out, tile_shape_in &tile_in) {
  static_assert(tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
                tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::Loc != Location::Acc && tile_shape_in::Loc != Location::Acc,
              "Unsupport ACC to be input or output here");
  tile_out.data() = tile_in.data();
}
#endif

#endif
