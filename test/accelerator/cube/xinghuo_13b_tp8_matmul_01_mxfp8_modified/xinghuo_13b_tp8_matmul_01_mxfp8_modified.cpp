#include <common/pto_tileop.hpp>
#include "params_mx_A8W8.h"
#include "../../include/accelerator_cube.h"

int main(){
    __fp8_e4m3 a[MPC*KPC];
    __fp8_e5m2 b[NPC*KPC];
    __fp8_e4m3 c[MPC*NPC];
    float dequant[N*1];
    matmul_kernel_a8w8<M,N,K,64,64,64>(c, a, b, dequant);
}