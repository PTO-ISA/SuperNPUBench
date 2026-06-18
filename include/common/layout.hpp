#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <stdint.h>

#include <iostream>
#include <type_traits>

#include "common/math_utils.hpp"

namespace pto {
enum class Location {
  Vec,
  Mat,
  Left,
  Right,
  Acc,
  Bias,
  Scaling,
};

enum class BLayout {
  RowMajor,
  ColMajor,
};

enum class SLayout {
  NoneBox,
  RowMajor,
  ColMajor,
};

enum class LayoutEnum {
  kNoneBox = 0,
  kRowMajor = 1,
  kColMajor = 2,
};

enum LayoutCvtEnum : uint64_t {
  NORM = 0,
  ND2DN, ND2ZZ, ND2ZN, ND2NZ, ND2NN,
  DN2ND, DN2ZZ, DN2ZN, DN2NZ, DN2NN,
  ZZ2ND, ZZ2DN, ZZ2ZN, ZZ2NZ, ZZ2NN = 15,
  ZN2ND = 17,  ZN2DN, ZN2ZZ, ZN2NZ, ZN2NN,
  NN2ND, NN2DN, NN2ZZ, NN2ZN, NN2NZ,
  NZ2ND, NZ2DN, NZ2ZZ, NZ2ZN, NZ2NN = 31
};

enum PadValueEnum : uint64_t {
  Zero = 0,
  Max = 1,
  Min = 2,
  Null = 3
};

const char *layout_type_to_str(LayoutEnum type) {
  switch (type) {
  case LayoutEnum::kRowMajor:
    return "RowMajor";
  case LayoutEnum::kColMajor:
    return "ColMajor";
  case LayoutEnum::kNoneBox:
    return "kNoneBox";
  }
  return "UnsupportedLayout";
}

class MatrixLayoutPrettyPrinter {
  template <typename Layout>
  static void print(std::ostream &out, const Layout &layout) {
    out << layout_type_to_str(Layout::kType) << "<" << Layout::Rows << ", "
        << Layout::Cols << ">, Strides<" << Layout::RowStride << ", "
        << Layout::ColStride << ">, Numel = " << Layout::Numel;
  }
};

template <const int Rows_, const int Cols_, const int RowStride_,
          const int ColStride_,
          const LayoutEnum kType_ =
              ColStride_ == 1 ? LayoutEnum::kRowMajor : LayoutEnum::kColMajor>
struct MatrixLayout {
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;

  static constexpr int RowStride = RowStride_;
  static constexpr int ColStride = ColStride_;

  static constexpr int Numel = Rows * Cols;

  static constexpr LayoutEnum kType = kType_;

  int operator()(int i, int j) const { return i * RowStride + j * ColStride; }
};

template <const int Row, const int Col, const int kStride = Col>
using RowMajor = MatrixLayout<Row, Col, kStride, 1>;

template <const int Row, const int Col, const int kStride = Row>
using ColMajor = MatrixLayout<Row, Col, 1, kStride>;

template <typename Layout> static constexpr size_t num_rows = Layout::Rows;

template <typename Layout> static constexpr size_t num_cols = Layout::Cols;

template <typename Layout>
static constexpr size_t row_stride = Layout::RowStride;

template <typename Layout>
static constexpr size_t col_stride = Layout::ColStride;

template <typename Layout> static constexpr size_t get_numel = Layout::Numel;

template <typename Layout_>
static constexpr LayoutEnum layout_type = Layout_::kType;

template <typename Layout_> struct is_row_major {
  static constexpr bool value = Layout_::kType == LayoutEnum::kRowMajor;
};

template <typename Layout_> struct is_col_major {
  static constexpr bool value = Layout_::kType == LayoutEnum::kColMajor;
};

template <typename Layout> struct is_contiguous {
  static constexpr bool value =
      is_row_major<Layout>::value
          ? (Layout::RowStride == Layout::Cols && Layout::ColStride == 1)
          : (Layout::ColStride == Layout::Rows && Layout::RowStride == 1);
};

template <typename OuterLayout_, typename InnerLayout_>
struct BlockMatrixLayout {
  using InnerLayout = InnerLayout_;
  using OuterLayout = OuterLayout_;

  static constexpr int Rows = OuterLayout_::Rows;
  static constexpr int Cols = OuterLayout_::Cols;
  static constexpr int Numel = OuterLayout_::Numel;

  static constexpr int kInnerRows = InnerLayout_::Rows;
  static constexpr int kInnerCols = InnerLayout_::Cols;

  static_assert(Rows % kInnerRows == 0,
                "OuterLayout rows must be divisible by InnerLayout rows");
  static_assert(Cols % kInnerCols == 0,
                "OuterLayout cols must be divisible by InnerLayout cols");

  static constexpr int kInnerNumel = InnerLayout_::Numel;
  static constexpr LayoutEnum kType = OuterLayout::kType;

  static constexpr int kTileRows = Rows / kInnerRows;
  static constexpr int kTileCols = Cols / kInnerCols;

  static constexpr bool kIsRowMajor = is_row_major<OuterLayout>::value;
  static constexpr bool kIsContiguous = is_contiguous<OuterLayout>::value;

  static constexpr int RowStride =
      kIsContiguous ? (kIsRowMajor ? kTileCols * kInnerNumel : kInnerNumel)
                    : OuterLayout::RowStride;
  static constexpr int ColStride =
      kIsContiguous ? (kIsRowMajor ? kInnerNumel : kTileRows * kInnerNumel)
                    : OuterLayout::ColStride;

  int operator()(int i, int j) const {
    const int outer_i = RowDivMod::div(i);
    const int outer_j = ColDivMod::div(j);

    const int inner_i = RowDivMod::mod(i);
    const int inner_j = ColDivMod::mod(j);

    return outer_(outer_i, outer_j) + inner_(inner_i, inner_j);
  }

  void dump() const {
    for (int i = 0; i < Rows; ++i) {
      for (int j = 0; j < Cols; ++j) {
        printf("%d, ", operator()(i, j));
      }
      printf("\n");
    }
  }

  auto get_outer_layout() const { return decltype(outer_){}; }

private:
  static constexpr bool kInnerRowsIsPow2 = (kInnerRows & (kInnerRows - 1)) == 0;
  static constexpr bool kInnerColsIsPow2 = (kInnerCols & (kInnerCols - 1)) == 0;

  using RowDivMod = DivModSelector<kInnerRowsIsPow2, kInnerRows>;
  using ColDivMod = DivModSelector<kInnerColsIsPow2, kInnerCols>;

  using BlockOuter = MatrixLayout<kTileRows, kTileCols, RowStride, ColStride,
                                  OuterLayout::kType>;
  BlockOuter outer_;
  InnerLayout inner_;
};

/// @brief Pretty printer for BlockMatrixLayout
template <typename OuterLayout_, typename InnerLayout_>
static std::ostream &
operator<<(std::ostream &out,
           const BlockMatrixLayout<OuterLayout_, InnerLayout_> &layout) {
  out << "BlockMatrixLayout {" << std::endl
      << "    Outer: " << layout.get_outer_layout() << ", " << std::endl
      << "    Inner: " << InnerLayout_{} << std::endl
      << "  }";
  return out;
}

template <typename OuterLayout_, typename InnerLayout_>
concept BlockRowMajorLayout =
    is_row_major<OuterLayout_>::value && is_row_major<InnerLayout_>::value;
 
template <typename OuterLayout_, typename InnerLayout_>
concept BlockColMajorLayout =
    is_col_major<OuterLayout_>::value && is_col_major<InnerLayout_>::value;
 
template <typename OuterLayout_, typename InnerLayout_>
concept BlockMixedLayout = (is_row_major<OuterLayout_>::value &&
                            is_col_major<InnerLayout_>::value) ||
                           (is_col_major<OuterLayout_>::value &&
                            is_row_major<InnerLayout_>::value);
 
template <typename OuterLayout_, typename InnerLayout_>
  requires BlockRowMajorLayout<OuterLayout_, InnerLayout_>
using BlockRowMajor = BlockMatrixLayout<OuterLayout_, InnerLayout_>;
 
template <typename OuterLayout_, typename InnerLayout_>
  requires BlockColMajorLayout<OuterLayout_, InnerLayout_>
using BlockColMajor = BlockMatrixLayout<OuterLayout_, InnerLayout_>;

template <typename OuterLayout_, typename InnerLayout_>
  requires BlockMixedLayout<OuterLayout_, InnerLayout_>
using BlockMixed = BlockMatrixLayout<OuterLayout_, InnerLayout_>;
} // namespace pto

#endif