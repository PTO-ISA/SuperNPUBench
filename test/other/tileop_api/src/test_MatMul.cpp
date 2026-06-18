#include <common/pto_tileop.hpp>

#include "../../../kernels/matmul.hpp"
#include "../data.hpp"

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

int main() {
  const uint16_t M = 160;
  const uint16_t K = 80;
  const uint16_t N = 320;
  const uint16_t TM = 32;
  const uint16_t TK = 32;
  const uint16_t TN = 32;

  size_t size_A = M * K;
  size_t size_B = K * N;
  size_t size_C = M * N;

  float *dst = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst);
  init_src_fp(dst, size_C);

  float *src0 = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_A);
  float *src1 = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_B);

#ifdef LINX_PMC
  PMC_START();
#endif

  matmul<M, N, K, TM, TN, TK>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_C);

  free(dst);
  free(src0);
  free(src1);

  return 0;
}
