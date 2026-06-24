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

    Tensor<dtype, ROW, DIM1+DIM2> out;
    Tensor<dtype, ROW, DIM1> in1(1);
    Tensor<dtype, ROW, DIM2> in2(2);
    concat<dtype, ROW, DIM1, DIM2>(out.data(), in1.data(), in2.data());

    return 0;
}