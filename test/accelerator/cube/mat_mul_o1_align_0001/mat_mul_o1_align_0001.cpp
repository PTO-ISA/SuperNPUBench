#include <common/pto_tileop.hpp>
#include "params_A16W8.h"

template <const int kM, const int kN, const int kK, const int kTM,
          const int kTN, const int kTK>
void matmul_kernel(__bf16 *c_ptr, __bf16 *a_ptr, __fp8_e4m3 *b_ptr, int8_t *dequant) {

}

int main(){
    __bf16 a[M*K];
    __fp8_e4m3 b[N*K];
    __bf16 c[M*N];

    int8_t *dequant;
    matmul_kernel<M,N,K,64,64,64>(c, a, b, dequant);
}