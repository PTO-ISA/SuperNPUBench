#ifndef ELEMENT_ATOMIC_TOPK_HPP
#define ELEMENT_ATOMIC_TOPK_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace element_atomic_topk_kernel {

using namespace pto;

constexpr std::size_t kLanes = 32;
constexpr std::size_t kParts = 4;
constexpr std::size_t kParentLanes = kLanes * kParts;

using ParentTile = CubeTileM32<uint32_t, 32, 4>;
using PartTile = CubeTileM32<uint32_t, 32, 1>;
using LaneTile = PartTile;

#ifdef __linx
using LaneCarrier = uint32_t tile_size(kLanes);

inline void atomic_selected_low8(LaneCarrier &old_values,
                                 const LaneCarrier &high8,
                                 const LaneCarrier &low8,
                                 uint32_t selected_high, uint32_t *histogram,
                                 uint32_t valid_lanes) {
#ifdef LINX_ELEMENTWISE_AVAILABLE
#pragma linx elementwise
#endif
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (lane < valid_lanes && high8[lane] == selected_high) {
      old_values[lane] =
          __atomic_fetch_add(&histogram[low8[lane]], 1u, __ATOMIC_RELAXED);
    } else {
      old_values[lane] = 0;
    }
  }
}
#endif

inline void histogram_high8(const uint32_t *input, std::size_t element_count,
                            uint32_t *histogram, uint32_t *old_values) {
#ifdef __linx
  const std::size_t parent_count =
      (element_count + kParentLanes - 1) / kParentLanes;
  for (std::size_t parent_index = 0; parent_index < parent_count;
       ++parent_index) {
    ParentTile parent;
    global_tensor<uint32_t, RowMajor<32, 4>> source(
        const_cast<uint32_t *>(input + parent_index * kParentLanes));
    TLOAD_CUBE(parent, source);
    auto parts = TPARTVIEW<PartTile, 1, kParts>(parent);

    for (std::size_t part_index = 0; part_index < kParts; ++part_index) {
      auto part = parts[0][part_index];
      LaneTile logical_part;
      LaneTile high8;
      LaneTile always_selected;
      LaneTile old;
      // Materialize the M32 column view without changing its layout.
      TADDS(logical_part, part, 0u);
      TSHRS(high8, logical_part, 8u);
      TANDS(always_selected, logical_part, 0u);
      const std::size_t lane_base = parent_index * kParentLanes + part_index;
      const uint32_t valid_lanes = static_cast<uint32_t>(
          lane_base >= element_count ? 0
              : ((element_count - lane_base + kParts - 1) / kParts < kLanes
                     ? (element_count - lane_base + kParts - 1) / kParts
                     : kLanes));
      atomic_selected_low8(old.data(), always_selected.data(), high8.data(),
                           0u, histogram, valid_lanes);
      global_tensor<uint32_t, RowMajor<32, 1>> destination(
          old_values + parent_index * kParentLanes + part_index * kLanes);
      TSTORE_CUBE(destination, old);
    }
  }
#else
  for (std::size_t i = 0; i < element_count; ++i) {
    const uint32_t bin = (input[i] >> 8) & 0xffu;
    old_values[i] = histogram[bin]++;
  }
#endif
}

inline void histogram_selected_low8(const uint32_t *input,
                                    std::size_t element_count,
                                    uint32_t selected_high, uint32_t *histogram,
                                    uint32_t *old_values) {
#ifdef __linx
  const std::size_t parent_count =
      (element_count + kParentLanes - 1) / kParentLanes;
  for (std::size_t parent_index = 0; parent_index < parent_count;
       ++parent_index) {
    ParentTile parent;
    global_tensor<uint32_t, RowMajor<32, 4>> source(
        const_cast<uint32_t *>(input + parent_index * kParentLanes));
    TLOAD_CUBE(parent, source);
    auto parts = TPARTVIEW<PartTile, 1, kParts>(parent);

    for (std::size_t part_index = 0; part_index < kParts; ++part_index) {
      auto part = parts[0][part_index];
      LaneTile logical_part;
      LaneTile high8;
      LaneTile low8;
      LaneTile old;
      TADDS(logical_part, part, 0u);
      TSHRS(high8, logical_part, 8u);
      TANDS(low8, logical_part, 0xffu);
      const std::size_t lane_base = parent_index * kParentLanes + part_index;
      const uint32_t valid_lanes = static_cast<uint32_t>(
          lane_base >= element_count ? 0
              : ((element_count - lane_base + kParts - 1) / kParts < kLanes
                     ? (element_count - lane_base + kParts - 1) / kParts
                     : kLanes));
      atomic_selected_low8(old.data(), high8.data(), low8.data(), selected_high,
                           histogram, valid_lanes);
      global_tensor<uint32_t, RowMajor<32, 1>> destination(
          old_values + parent_index * kParentLanes + part_index * kLanes);
      TSTORE_CUBE(destination, old);
    }
  }
#else
  for (std::size_t i = 0; i < element_count; ++i) {
    const uint32_t high = (input[i] >> 8) & 0xffu;
    if (high == selected_high) {
      const uint32_t low = input[i] & 0xffu;
      old_values[i] = histogram[low]++;
    } else {
      old_values[i] = 0;
    }
  }
#endif
}

} // namespace element_atomic_topk_kernel

#endif
