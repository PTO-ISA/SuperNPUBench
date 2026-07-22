#include "pto_kernels/matmul/gemm_performance.cpp"

int main() {
  constexpr int m = pto::kernels::shapes::kMatmulM;
  constexpr int n = pto::kernels::shapes::kMatmulN;
  constexpr int k = pto::kernels::shapes::kMatmulK;
  alignas(64) static float lhs[m * k]{};
  alignas(64) static float rhs[k * n]{};
  alignas(64) static float dst[m * n]{};
  gemm_performance_f32(lhs, rhs, dst, 2);
  return 0;
}
