#include <common/pto_tileop.hpp>
#include "params_mx_A8W8.h"
#include <benchmark_support/npu/npu_cube.h>

int main(){
    float a[M*K];
    float b[N*K];
    float c[M*N];
    float dequant[N*1];

    float amx[M*K/32];
    float bmx[N*K/32];
    
    matmul_mask<M,N,K,64,64,64>(c, a, b);
}