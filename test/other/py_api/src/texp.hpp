#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col>
void texp_py(float* dst, float* src) {
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            int offset = i * (tile_row * gm_col) + j * tile_col;
            gm_shape s0(src + offset);
            gm_shape res(dst + offset);

            tile_shape d0, d1;

            TCOPYIN(d0, s0);
            TEXP(d1, d0);
            TCOPYOUT(res, d1);
        }
    }
}

#ifdef __cpu_sim__
void bind_texp(py::module_& m) {
    m.def("texp", [](py::array_t<float> dst_py, py::array_t<float> src_py){
        float* dst = static_cast<float*>(dst_py.request().ptr);
        float* src = static_cast<float*>(src_py.request().ptr);
        texp_py<16, 16, 8, 8>(dst, src);}
    );
    return;
}
#endif