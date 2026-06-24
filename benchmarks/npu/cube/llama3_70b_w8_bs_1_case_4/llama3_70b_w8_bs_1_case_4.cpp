#include <common/pto_tileop.hpp>
#include "params_A16W8.h"
#include <benchmark_support/npu/npu_cube.h>

int main(){
    __half a[MPC*KPC];
    __fp8_e4m3 b[NPC*KPC];
    __half c[MPC*NPC];
    float dequant[N*1];
    matmul_kernel_a16w8<M,N,K,64,64,64>(c, a, b, dequant);
}