#ifndef SUPERNPUBENCH_PTOISA_BROADCAST_VEC_07_HPP
#define SUPERNPUBENCH_PTOISA_BROADCAST_VEC_07_HPP

#include <common/pto_tileop.hpp>

#include <cstddef>

using namespace pto;

template <typename dtype, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM,
          size_t gIM, size_t gOM, size_t kTileRows>
void broadcast(dtype *in_ptr, dtype *out_ptr, const size_t *, const size_t *) {
  constexpr size_t kN = gIM;
  constexpr size_t kC = gOM / gIM;
  constexpr size_t kTileCols = kC;
  constexpr size_t kMaxRowsByBytes =
      (4096 / (kTileCols * sizeof(dtype))) == 0
          ? 1
          : (4096 / (kTileCols * sizeof(dtype)));
  constexpr size_t kRowsPerTile =
      kTileRows < kMaxRowsByBytes ? kTileRows : kMaxRowsByBytes;

  static_assert(MAX_DIM >= IN_DIM && MAX_DIM >= OUT_DIM,
                "MAX_DIM must cover input and output ranks");
  static_assert(gOM % gIM == 0,
                "broadcast output element count must be a multiple of input");
  static_assert(kRowsPerTile > 0, "broadcast tile must contain at least one row");

  using InputGlobal = global_tensor<dtype, RowMajor<kN, 1>>;
  using OutputGlobal = global_tensor<dtype, RowMajor<kN, kC>>;
  using InputTile = Tile<Location::Vec, dtype, kRowsPerTile, 1,
                         BLayout::RowMajor, kRowsPerTile, 1>;
  using OutputTile = Tile<Location::Vec, dtype, kRowsPerTile, kTileCols,
                          BLayout::RowMajor, kRowsPerTile, kC>;
  using InputIterator = global_iterator<InputGlobal, InputTile>;
  using OutputIterator = global_iterator<OutputGlobal, OutputTile>;

  InputIterator input_iter(in_ptr);
  OutputIterator output_iter(out_ptr);

  constexpr size_t kFullTiles = kN / kRowsPerTile;
  constexpr size_t kTail = kN % kRowsPerTile;

  for (size_t tile = 0; tile < kFullTiles; ++tile) {
    InputTile input_tile;
    OutputTile output_tile;
    TLOAD(input_tile, input_iter(static_cast<int>(tile), 0));
    TROWEXPAND(output_tile, input_tile);
    TSTORE(output_iter(static_cast<int>(tile), 0), output_tile);
  }

  if constexpr (kTail != 0) {
    using TailInputTile = Tile<Location::Vec, dtype, kRowsPerTile, 1,
                               BLayout::RowMajor, kTail, 1>;
    using TailOutputTile = Tile<Location::Vec, dtype, kRowsPerTile, kTileCols,
                                BLayout::RowMajor, kTail, kC>;
    using TailInputIterator = global_iterator<InputGlobal, TailInputTile>;
    using TailOutputIterator = global_iterator<OutputGlobal, TailOutputTile>;
    TailInputIterator tail_input_iter(in_ptr);
    TailOutputIterator tail_output_iter(out_ptr);
    TailInputTile input_tile;
    TailOutputTile output_tile;
    TLOAD(input_tile, tail_input_iter(static_cast<int>(kFullTiles), 0));
    TROWEXPAND(output_tile, input_tile);
    TSTORE(tail_output_iter(static_cast<int>(kFullTiles), 0), output_tile);
  }
}

#endif // SUPERNPUBENCH_PTOISA_BROADCAST_VEC_07_HPP
