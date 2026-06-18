#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "flash_attention.hpp"

#ifdef __cpu_sim__ 
    void bind_flash_attention(py::module_& m) {
        m.def("flash_attention",[](py::array_t<float> out_ptr_py, 
                                      py::array_t<float> q_ptr_py, 
                                      py::array_t<float> k_ptr_py,
                                      py::array_t<float> v_ptr_py){
            float* out_ptr = static_cast<float*>(out_ptr_py.request().ptr);
            float* q_ptr = static_cast<float*>(q_ptr_py.request().ptr);
            float* k_ptr = static_cast<float*>(k_ptr_py.request().ptr);
            float* v_ptr = static_cast<float*>(v_ptr_py.request().ptr);                      
            flash_attention<64, 256, 64, 64>(out_ptr, q_ptr, k_ptr, v_ptr);}
        );
        return;
    }
#endif