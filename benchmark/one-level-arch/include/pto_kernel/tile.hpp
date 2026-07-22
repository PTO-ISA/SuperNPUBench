#ifndef PTO_KERNEL_TILE_HPP
#define PTO_KERNEL_TILE_HPP

// Public kernel-authoring surface. Backend headers stay behind this boundary.
#include <pto_kernel/common/pto_tileop.hpp>

namespace pto {

template <typename Element, int Rows, int Cols,
          BLayout Layout = BLayout::RowMajor, int ValidRows = Rows,
          int ValidCols = Cols>
using LocalTile =
    Tile<Location::Vec, Element, Rows, Cols, Layout, ValidRows, ValidCols>;

template <typename Element, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using MatrixLeftTile = TileLeft<Element, Rows, Cols, ValidRows, ValidCols>;

template <typename Element, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using MatrixRightTile = TileRight<Element, Rows, Cols, ValidRows, ValidCols>;

template <typename Element, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using AccumulatorTile = TileAcc<Element, Rows, Cols, ValidRows, ValidCols>;

enum class SharedMove : uint8_t {
  Insert,
  Publish,
  Extract,
  Broadcast,
};

// SharedTile is the source-level block-shared value contract. Operations on
// this type require shared-tile lowering from the selected compiler profile.
template <typename Element_, int Rows_, int Cols_,
          BLayout Layout_ = BLayout::RowMajor, int ValidRows_ = Rows_,
          int ValidCols_ = Cols_>
struct SharedTile {
  using DType = Element_;

  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr int RowValid = ValidRows_;
  static constexpr int ColValid = ValidCols_;
  static constexpr BLayout LayoutTag = Layout_;

  static_assert(ValidRows_ == DYNAMIC ||
                    (ValidRows_ > 0 && ValidRows_ <= Rows_),
                "valid rows must fit the shared tile");
  static_assert(ValidCols_ == DYNAMIC ||
                    (ValidCols_ > 0 && ValidCols_ <= Cols_),
                "valid columns must fit the shared tile");
};

} // namespace pto

#endif // PTO_KERNEL_TILE_HPP
