#include "pto_kernels/matmul/gemm_demo.cpp"

int main() {
  constexpr int m = pto::kernels::shapes::kMatmulM;
  constexpr int n = pto::kernels::shapes::kMatmulN;
  constexpr int k = pto::kernels::shapes::kMatmulK;
  alignas(64) static float lhs[m * k]{};
  alignas(64) static float rhs[k * n]{};
  alignas(64) static float dst[m * n]{};
  gemm_demo_f32(dst, lhs, rhs);
  return 0;
}
