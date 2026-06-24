#include "ds_config.h"
#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "mla.hpp"

#ifndef ROW
#define ROW 64
#endif

#ifndef DIM1
#define DIM1 64
#endif

#ifndef DIM2
#define DIM2 64
#endif

int main(){

    Tensor<dtype, ROW, DIM1+DIM2> in(0);
    Tensor<dtype, ROW, DIM1> out1;
    Tensor<dtype, ROW, DIM2> out2;
    split<dtype, ROW, DIM1, DIM2>(out1.data(), out2.data(), in.data());

    return 0;
}