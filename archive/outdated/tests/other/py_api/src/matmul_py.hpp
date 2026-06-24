#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.hpp"

// M N K
#ifdef __cpu_sim__ 
    void bind_matmul(py::module_& m) {
        m.def("matmul_mask",[](py::array_t<float> dst_py, 
                                    py::array_t<float> src0_py, 
                                    py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);                     
            matmul_mask<117, 131, 45, 16, 16, 16>(dst, src0, src1);}
        );
        m.def("matmul_frac",[](py::array_t<float> dst_py, 
                                    py::array_t<float> src0_py, 
                                    py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);                     
            matmul_frac<128, 128, 256, 64, 64, 64>(dst, src0, src1);}
        );
        m.def("matmul_rm",[](py::array_t<float> dst_py, 
                               py::array_t<float> src0_py, 
                               py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);                     
            matmul_rm<32, 32, 32, 8, 8, 8>(dst, src0, src1);}
        );
        m.def("matmul_tile_rm",[](py::array_t<float> dst_py, 
                                      py::array_t<float> src0_py, 
                                      py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);                     
            matmul_tile_rm<32, 32, 32>(dst, src0, src1);}
        );
        m.def("matmul_tile_frac",[](py::array_t<float> dst_py, 
                                    py::array_t<float> src0_py, 
                                    py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);                     
            matmul_tile_frac<32, 32, 32>(dst, src0, src1);}
        );
        return;
    }
#endif