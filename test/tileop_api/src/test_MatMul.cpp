#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t M, uint16_t N, uint16_t K>
void test(float *dst, float *src0, float *src1) {
  using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A = TileLeft<float, M, K>;
  using tile_shape_B = TileRight<float, K, N>;
  using tile_shape_C = TileAcc<float, M, N>;
  using tile_shape_O = TileLeft<float, M, K>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;
  tile_shape_O d3;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  MATMUL(d2, d0, d1);
  TCVT(d3, d2);
  TCOPYOUT(res, d3);
}

int main() {
  const uint16_t M = 16;
  const uint16_t K = 8;
  const uint16_t N = 32;

  size_t size_A = M * K;
  size_t size_B = K * N;
  size_t size_C = M * N;

  float *dst = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_C);

  float *src0 = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_A);
  float *src1 = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_B);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<M, N, K>(dst, src0, src1);

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
