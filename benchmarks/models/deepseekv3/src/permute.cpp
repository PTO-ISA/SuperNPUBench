#include "ds_config.h"
#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "mla.hpp"

#ifndef DIM0
#define DIM0 64
#endif

#ifndef DIM1
#define DIM1 64
#endif

#ifndef DIM2
#define DIM2 64
#endif

#ifndef DIM3
#define DIM3 64
#endif

int main(){

    Tensor<dtype, DIM0, DIM1, DIM2, DIM3> in(1);
    Tensor<dtype, DIM0, DIM2, DIM1, DIM3> out;
    permute<dtype, DIM0, DIM1, DIM2, DIM3>(out.data(), in.data());
    
    return 0;
}