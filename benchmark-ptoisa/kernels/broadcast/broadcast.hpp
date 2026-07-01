#ifndef SUPERNPUBENCH_PTOISA_BROADCAST_HPP
#define SUPERNPUBENCH_PTOISA_BROADCAST_HPP

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace pto;

template <typename dtype, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM,
          size_t gIM, size_t gOM, size_t kTileHint>
void broadcast(dtype *in_ptr, dtype *out_ptr, const size_t *in_shape,
               const size_t *out_shape) {
  constexpr size_t kBCast = gOM / gIM;
  constexpr size_t kElementsPerTile = 512;
  constexpr size_t kFullTiles = gOM / kElementsPerTile;
  constexpr size_t kTail = gOM % kElementsPerTile;

  static_assert(MAX_DIM >= IN_DIM && MAX_DIM >= OUT_DIM,
                "MAX_DIM must cover input and output ranks");
  static_assert(gOM % gIM == 0,
                "broadcast output element count must be a multiple of input");
  static_assert(kBCast > 0, "broadcast factor must be positive");
  (void)kTileHint;

  size_t inner = 1;
  size_t in_dim = IN_DIM;
  size_t out_dim = OUT_DIM;
  while (in_dim > 0 && out_dim > 0 &&
         in_shape[in_dim - 1] == out_shape[out_dim - 1]) {
    inner *= out_shape[out_dim - 1];
    --in_dim;
    --out_dim;
  }

  using InputGlobal = global_tensor<dtype, RowMajor<1, gIM>>;
  using OutputGlobal = global_tensor<dtype, RowMajor<1, gOM>>;
  using DataTile = Tile<Location::Vec, dtype, 1, kElementsPerTile,
                        BLayout::RowMajor>;
  using OffsetTile = Tile<Location::Vec, std::uint32_t, 1, kElementsPerTile,
                          BLayout::RowMajor>;
  using OutputIterator = global_iterator<OutputGlobal, DataTile>;

  InputGlobal input_global(in_ptr);
  OutputIterator output_iter(out_ptr);

  auto emit_tile = [&](auto &data_tile, auto &linear, auto &batch,
                       auto &batch_base, auto &inner_q, auto &inner_base,
                       auto &inner_idx, auto &inner_bytes, auto &offset,
                       std::uint32_t base, auto output_addr) {
    using LinearTile = std::remove_reference_t<decltype(linear)>;
    const auto inner_u32 = static_cast<std::uint32_t>(inner);
    const auto group_u32 = static_cast<std::uint32_t>(kBCast * inner);
    TCI<LinearTile, std::uint32_t, 0>(linear, base);

    TDIVS(batch, linear, group_u32);
    TMULS(batch_base, batch,
          static_cast<std::uint32_t>(inner * sizeof(dtype)));

    TDIVS(inner_q, linear, inner_u32);
    TMULS(inner_base, inner_q, inner_u32);
    TSUB(inner_idx, linear, inner_base);
    TMULS(inner_bytes, inner_idx, static_cast<std::uint32_t>(sizeof(dtype)));

    TADD(offset, batch_base, inner_bytes);
    MGATHER(data_tile, input_global, offset);
    TSTORE(output_addr, data_tile);
  };

  for (size_t tile = 0; tile < kFullTiles; ++tile) {
    DataTile data_tile;
    OffsetTile linear;
    OffsetTile batch;
    OffsetTile batch_base;
    OffsetTile inner_q;
    OffsetTile inner_base;
    OffsetTile inner_idx;
    OffsetTile inner_bytes;
    OffsetTile offset;
    emit_tile(data_tile, linear, batch, batch_base, inner_q, inner_base,
              inner_idx, inner_bytes, offset,
              static_cast<std::uint32_t>(tile * kElementsPerTile),
              output_iter(0, static_cast<int>(tile)));
  }

  if constexpr (kTail != 0) {
    using TailDataTile = Tile<Location::Vec, dtype, 1, kElementsPerTile,
                              BLayout::RowMajor, 1, kTail>;
    using TailOffsetTile =
        Tile<Location::Vec, std::uint32_t, 1, kElementsPerTile,
             BLayout::RowMajor, 1, kTail>;
    using TailOutputIterator = global_iterator<OutputGlobal, TailDataTile>;
    TailOutputIterator tail_output_iter(out_ptr);
    TailDataTile data_tile;
    TailOffsetTile linear;
    TailOffsetTile batch;
    TailOffsetTile batch_base;
    TailOffsetTile inner_q;
    TailOffsetTile inner_base;
    TailOffsetTile inner_idx;
    TailOffsetTile inner_bytes;
    TailOffsetTile offset;
    emit_tile(data_tile, linear, batch, batch_base, inner_q, inner_base,
              inner_idx, inner_bytes, offset,
              static_cast<std::uint32_t>(kFullTiles * kElementsPerTile),
              tail_output_iter(0, static_cast<int>(kFullTiles)));
  }
}

#endif // SUPERNPUBENCH_PTOISA_BROADCAST_HPP
