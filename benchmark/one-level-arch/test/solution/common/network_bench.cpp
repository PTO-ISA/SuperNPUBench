// Host-generated ELF data removes fixture initialization and scalar checking
// from the timed program. Both simulators execute this same kernel ELF.
#include <common/pto_tileop.hpp>
#include <cstdint>
#include "network_config.hpp"
#include "solution/gather_v2/gather_v2.hpp"
#include "solution/view_copy/view_copy.hpp"

extern "C" {
extern DATA_TYPE network_input[];
extern DATA_TYPE network_output[];
extern std::uint32_t network_index[];
}

int main() {
#if NETWORK_GATHER
  const std::uint32_t input_shape[] = {INPUT_SHAPE};
  const std::uint32_t output_shape[] = {OUTPUT_SHAPE};
  supernpu::tile_isa::gather_v2<DATA_TYPE, std::uint32_t, RANK, AXIS,
      INPUT_ELEMENTS, OUTPUT_ELEMENTS, TILE_ELEMENTS>(
      network_input, network_index, network_output + OUTPUT_PREFIX,
      input_shape, output_shape);
#else
  const std::uint32_t shape[] = {SHAPE};
  const std::uint32_t input_stride[] = {INPUT_STRIDE};
  const std::uint32_t output_stride[] = {OUTPUT_STRIDE};
  supernpu::tile_isa::tile_view_copy<DATA_TYPE, RANK, OUTPUT_ELEMENTS,
      TILE_ELEMENTS>(network_input, network_output, shape,
                     INPUT_OFFSET_BYTES, OUTPUT_OFFSET_BYTES,
                     input_stride, output_stride);
#endif
  return 0;
}
