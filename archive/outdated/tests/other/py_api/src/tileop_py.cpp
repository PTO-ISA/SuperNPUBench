#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//including test function here
#include "tadd.hpp"
#include "tsub.hpp"
#include "texp.hpp"
#include "tmax.hpp"
#include "matmul_py.hpp"
#include "softmax_py.hpp"
#include "flash_attention_py.hpp"

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

namespace py = pybind11;

//bind corresponding module
PYBIND11_MODULE(tileop_py, m) {
    m.doc() = "tildeop test";
    py::module_ tadd_api = m.def_submodule("tadd_api", "TADD Function");
    bind_tadd(tadd_api);
    py::module_ tsub_api = m.def_submodule("tsub_api", "TSUB Function");
    bind_tsub(tsub_api);
    py::module_ matmul_api = m.def_submodule("matmul_api", "MATMUL Function");
    bind_matmul(matmul_api);
    py::module_ texp_api = m.def_submodule("texp_api", "TEXP Function");
    bind_texp(texp_api);
    py::module_ tmax_api = m.def_submodule("tmax_api", "TMAX Function");
    bind_tmax(tmax_api);
    py::module_ softmax_api = m.def_submodule("softmax_api", "SoftMax Function");
    bind_softmax(softmax_api);
    py::module_ fa_api = m.def_submodule("fa_api", "FlashAttention Function");
    bind_flash_attention(fa_api);
}