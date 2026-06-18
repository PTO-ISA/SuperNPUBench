#include "ds_config.h"
#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "mla.hpp"

#ifndef ROW
#define ROW 128
#endif

#ifndef DIM
#define DIM 64
#endif

#ifndef EXT_DIM
#define EXT_DIM 64
#endif

int main(){
    Tensor<dtype, ROW, DIM> x;
    Tensor<dtype, ROW, EXT_DIM, DIM> out;
    expand<dtype, ROW, DIM, EXT_DIM>(out.data(), x.data());
    return 0;
}