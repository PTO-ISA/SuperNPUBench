#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tmax_py(float* dst, float* src0, float* src1){
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = tile_tensor<float, RowMajor<tile_row, tile_col>>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            int offset = i * (tile_row * gm_col) + j * tile_col;
            gm_shape s0(src0 + offset);
            gm_shape s1(src1 + offset);
            gm_shape res(dst + offset);

            tile_shape d0, d1, d2;

            TCOPYIN(d0, s0);
            TCOPYIN(d1, s1);
            TMAX(d2, d1, d0);
            TCOPYOUT(res, d2);
        }
    }
}

#ifdef __cpu_sim__
void bind_tmax(py::module_& m) {
    m.def("tmax", [](py::array_t<float> dst_py, py::array_t<float> src0_py, py::array_t<float> src1_py){
        float* dst = static_cast<float*>(dst_py.request().ptr);
        float* src0 = static_cast<float*>(src0_py.request().ptr);
        float* src1 = static_cast<float*>(src1_py.request().ptr);
        tmax_py<16, 16, 8, 8>(dst, src0, src1);}
    );
    return;
}
#endif