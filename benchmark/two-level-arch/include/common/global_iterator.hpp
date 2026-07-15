#ifndef __GLOBAL_ITERATOR_HPP__
#define __GLOBAL_ITERATOR_HPP__

#include "common/pto_tile.hpp"

using namespace pto;

template <typename glb_tensor, is_tile_data_v tl_tensor>
struct global_iterator {

  using DType = typename glb_tensor::DType;
  global_iterator(DType *data) : data_(data) {}

  /*2D*/
  static_assert(glb_tensor::staticStride[0] == 1 && glb_tensor::staticStride[1] == 1 &&
                "global_iterator can only be use in 2D global_tensor");
  auto operator()(int i, int j) {
    using shape = Shape<1, 1, 1, 1, 1>;
    using stride = std::conditional_t<glb_tensor::isRowMajor,
          Stride<1, 1, glb_tensor::RowStride * glb_tensor::ColStride, glb_tensor::RowStride, 1>,
          Stride<1, 1, glb_tensor::RowStride * glb_tensor::ColStride, 1, glb_tensor::ColStride>>;

    using new_tile =
      std::conditional_t<glb_tensor::isRowMajor,
          GlobalTensor<DType, shape, stride, Layout::ND>,
          GlobalTensor<DType, shape, stride, Layout::DN>>;

    int offset =
        glb_tensor::isRowMajor
            ? i * glb_tensor::RowStride * tl_tensor::Rows + j * tl_tensor::Cols
            : i * tl_tensor::Rows + j * glb_tensor::Rows * tl_tensor::Cols;

    new_tile tile(data_ + offset);
    return tile;
  };

private:
  DType *data_;
};

#endif
