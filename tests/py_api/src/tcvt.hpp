#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <uint16_t K>
void tcvtnz2zn(float* dst, float* src) {
    using gm_shape_A = global_tensor<float, RowMajor<K, K>>;
    using gm_shape_B = global_tensor<float, RowMajor<K, K>>;
    using tile_shape_in = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_nz = tile_tensor_out<float, K, K>;
    using tile_shape_zn = tile_tensor_right<float, K, K>;
    using tile_shape_out = tile_tensor<float, RowMajor<K, K>>;

    gm_shape_A s(src);
    gm_shape_B res(dst);

    tile_shape_in d0;
    tile_shape_nz d1;
    tile_shape_zn d2;
    tile_shape_out d3;
    

    TCOPYIN(d0, s);
    TRESHAPE(d1, d0);
    TCVT(d2, d1);
    TRESHAPE(d3, d2);
    TCOPYOUT(res, d3);
}


template <uint16_t K>
void tcvtzn2nz(float* dst, float* src) {
    using gm_shape_A = global_tensor<float, RowMajor<K, K>>;
    using gm_shape_B = global_tensor<float, RowMajor<K, K>>;
    using tile_shape_in = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_nz = tile_tensor_out<float, K, K>;
    using tile_shape_zn = tile_tensor_right<float, K, K>;
    using tile_shape_out = tile_tensor<float, RowMajor<K, K>>;

    gm_shape_A s(src);
    gm_shape_B res(dst);

    tile_shape_in d0;
    tile_shape_zn d1;
    tile_shape_nz d2;
    tile_shape_out d3;
    

    TCOPYIN(d0, s);
    TRESHAPE(d1, d0);
    TCVT(d2, d1);
    TRESHAPE(d3, d2);
    TCOPYOUT(res, d3);
    
}
template <uint16_t K>
void tcvtnz2rowmajor(float* dst, float* src) {
    using gm_shape_A = global_tensor<float, RowMajor<K, K>>;
    using gm_shape_B = global_tensor<float, RowMajor<K, K>>;
    using tile_shape_in = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_nz = tile_tensor_left<float, K, K>;
    using tile_shape_rowmajor = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_out = tile_tensor<float, RowMajor<K, K>>;

    gm_shape_A s(src);
    gm_shape_B res(dst);

    tile_shape_in d0;
    tile_shape_nz d1;
    tile_shape_rowmajor d2;
    tile_shape_out d3;
    

    TCOPYIN(d0, s);
    TRESHAPE(d1, d0);
    TCVT(d2, d1);
    TRESHAPE(d3, d2);
    TCOPYOUT(res, d3);    
}



template <uint16_t K>
void tcvtrowmajor2nz(float* dst, float* src) {
    using gm_shape_A = global_tensor<float, RowMajor<K, K>>;
    using gm_shape_B = global_tensor<float, RowMajor<K, K>>;
    using tile_shape_in = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_nz = tile_tensor_left<float, K, K>;
    using tile_shape_rowmajor = tile_tensor<float, RowMajor<K, K>>;
    using tile_shape_out = tile_tensor<float, RowMajor<K, K>>;

    gm_shape_A s(src);
    gm_shape_B res(dst);

    tile_shape_in d0;
    tile_shape_rowmajor d1;
    tile_shape_nz d2;
    tile_shape_out d3;
    

    TCOPYIN(d0, s);
    TRESHAPE(d1, d0);
    TCVT(d2, d1);
    TRESHAPE(d3, d2);
    TCOPYOUT(res, d3);      
}

// Python 接口绑定
#ifdef __cpu_sim__
    void bind_tcvt(py::module_& m) {
        m.def("tcvtnz2zn", [](py::array_t<float> dst_py, py::array_t<float> src_py) {
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src = static_cast<float*>(src_py.request().ptr);
            tcvtnz2zn<32>(dst, src);
        });
        m.def("tcvtzn2nz", [](py::array_t<float> dst_py, py::array_t<float> src_py) {
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src = static_cast<float*>(src_py.request().ptr);
            tcvtzn2nz<32>(dst, src);
        });
        m.def("tcvtnz2rowmajor", [](py::array_t<float> dst_py, py::array_t<float> src_py) {
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src = static_cast<float*>(src_py.request().ptr);
            tcvtnz2rowmajor<32>(dst, src);
        });
        m.def("tcvtrowmajor2nz", [](py::array_t<float> dst_py, py::array_t<float> src_py) {
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src = static_cast<float*>(src_py.request().ptr);
            tcvtrowmajor2nz<32>(dst, src);
        });
        return;
    }
#endif