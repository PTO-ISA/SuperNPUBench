#include "pto_kernels/matmul/gemm.cpp"

int main() {
  constexpr int m = pto::kernels::shapes::kMatmulM;
  constexpr int n = pto::kernels::shapes::kMatmulN;
  constexpr int k = pto::kernels::shapes::kMatmulK;
  alignas(64) static int lhs[m * k]{};
  alignas(64) static int rhs[k * n]{};
  alignas(64) static int dst[m * n]{};
  gemm_i32(lhs, rhs, dst);
  return 0;
}
