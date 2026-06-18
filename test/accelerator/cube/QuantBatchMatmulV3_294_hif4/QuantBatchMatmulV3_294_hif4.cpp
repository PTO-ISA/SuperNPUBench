#include <common/pto_tileop.hpp>
#include "params_mx_A4W4.h"
#include "../../include/accelerator_cube.h"

int main(){
    float a[M*K];
    float b[N*K];
    float c[M*N];
    float dequant[N*1];
    matmul_mask<M,N,K,64,64,64>(c, a, b);
}