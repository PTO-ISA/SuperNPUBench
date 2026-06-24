#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "softmax.hpp"

#ifdef __cpu_sim__ 
    void bind_softmax(py::module_& m) {
        m.def("softmax",[](py::array_t<float> dst_py, py::array_t<float> src_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src = static_cast<float*>(src_py.request().ptr);
            softmax<32,32,8,8>(dst, src);}
            );
        return;
    }
#endif