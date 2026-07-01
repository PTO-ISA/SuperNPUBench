#ifndef SUPERNPUBENCH_PTOISA_GATHER_HPP
#define SUPERNPUBENCH_PTOISA_GATHER_HPP

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>

using namespace pto;

template <typename dtype = float, typename otype = std::uint32_t, size_t gK,
          size_t gM, size_t gN, size_t tM, size_t tN>
void gather(dtype *in_data_ptr, otype *in_offset_ptr, dtype *out_ptr) {
  constexpr size_t kFullRows = gM / tM;
  constexpr size_t kTailRows = gM % tM;
  constexpr size_t kFullCols = gN / tN;
  constexpr size_t kTailCols = gN % tN;

  using InputGlobal = global_tensor<dtype, RowMajor<gK, gN>>;
  using OffsetGlobal = global_tensor<otype, RowMajor<1, gM>>;
  using OutputGlobal = global_tensor<dtype, RowMajor<gM, gN>>;

  using OffsetTile =
      Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor>;
  using DataTile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
  using OffsetIterator = global_iterator<OffsetGlobal, OffsetTile>;
  using OutputIterator = global_iterator<OutputGlobal, DataTile>;

  InputGlobal input_global(in_data_ptr);
  OffsetIterator offset_iter(in_offset_ptr);
  OutputIterator output_iter(out_ptr);

  for (size_t row_tile = 0; row_tile < kFullRows; ++row_tile) {
    OffsetTile row_index;
    TLOAD(row_index, offset_iter(0, static_cast<int>(row_tile)));
    for (size_t col_tile = 0; col_tile < kFullCols; ++col_tile) {
      DataTile out_tile;
      InputGlobal adjusted_input(in_data_ptr + col_tile * tN);
      MGATHER(out_tile, adjusted_input, row_index);
      TSTORE(output_iter(static_cast<int>(row_tile),
                         static_cast<int>(col_tile)),
             out_tile);
    }
    if constexpr (kTailCols != 0) {
      using TailColTile = Tile<Location::Vec, dtype, tM, tN,
                               BLayout::RowMajor, tM, kTailCols>;
      using TailOutputIterator = global_iterator<OutputGlobal, TailColTile>;
      TailOutputIterator tail_output_iter(out_ptr);
      TailColTile out_tile;
      InputGlobal adjusted_input(in_data_ptr + kFullCols * tN);
      MGATHER(out_tile, adjusted_input, row_index);
      TSTORE(tail_output_iter(static_cast<int>(row_tile),
                              static_cast<int>(kFullCols)),
             out_tile);
    }
  }

  if constexpr (kTailRows != 0) {
    using TailOffsetTile =
        Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor, 1, kTailRows>;
    using TailOffsetIterator = global_iterator<OffsetGlobal, TailOffsetTile>;
    TailOffsetIterator tail_offset_iter(in_offset_ptr);
    TailOffsetTile row_index;
    TLOAD(row_index, tail_offset_iter(0, static_cast<int>(kFullRows)));
    for (size_t col_tile = 0; col_tile < kFullCols; ++col_tile) {
      using TailRowTile = Tile<Location::Vec, dtype, tM, tN,
                               BLayout::RowMajor, kTailRows, tN>;
      using TailOutputIterator = global_iterator<OutputGlobal, TailRowTile>;
      TailOutputIterator tail_output_iter(out_ptr);
      TailRowTile out_tile;
      InputGlobal adjusted_input(in_data_ptr + col_tile * tN);
      MGATHER(out_tile, adjusted_input, row_index);
      TSTORE(tail_output_iter(static_cast<int>(kFullRows),
                              static_cast<int>(col_tile)),
             out_tile);
    }
    if constexpr (kTailCols != 0) {
      using TailTile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,
                            kTailRows, kTailCols>;
      using TailOutputIterator = global_iterator<OutputGlobal, TailTile>;
      TailOutputIterator tail_output_iter(out_ptr);
      TailTile out_tile;
      InputGlobal adjusted_input(in_data_ptr + kFullCols * tN);
      MGATHER(out_tile, adjusted_input, row_index);
      TSTORE(tail_output_iter(static_cast<int>(kFullRows),
                              static_cast<int>(kFullCols)),
             out_tile);
    }
  }
}

#endif // SUPERNPUBENCH_PTOISA_GATHER_HPP
