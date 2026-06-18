#ifndef __GLOBAL_ITERATOR_HPP__
#define __GLOBAL_ITERATOR_HPP__

#include "common/pto_tile.hpp"

using namespace pto;

template <typename glb_tensor, is_tile_data_v tl_tensor>
struct global_iterator {

  using DType = typename glb_tensor::DType;
  global_iterator(DType *data) : data_(data) {}

  /*2D*/
  auto operator()(int i, int j) {

    // assert(i >= 0 && i < row_dim && "i out of the range!");
    // assert(j >= 0 && j < col_dim && "j out of the range!");
    using tile_layout =
        MatrixLayout<tl_tensor::Rows, tl_tensor::Cols, glb_tensor::kRowStride,
                     glb_tensor::kColStride>;

    using new_tile = global_tensor<DType, tile_layout>;

    int offset =
        glb_tensor::isRowMajor
            ? i * glb_tensor::kCols * tl_tensor::Rows + j * tl_tensor::Cols
            : i * tl_tensor::Rows + j * glb_tensor::kRows * tl_tensor::Cols;

    new_tile tile(data_ + offset);
    return tile;
  };

private:
  DType *data_;
};

#endif
