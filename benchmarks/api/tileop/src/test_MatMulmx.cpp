#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

// MATMULMX: A + AX, B + BX
template <uint16_t M, uint16_t N, uint16_t K>
void test_mx(float *dst, float *src0, float *src0x, float *src1, float *src1x) {
  using gm_shape_A  = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_AX = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B  = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_BX = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C  = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A  = TileLeft<float, M, K>;
  using tile_shape_AX = TileLeft<float, M, K>;
  using tile_shape_B  = TileRight<float, K, N>;
  using tile_shape_BX = TileRight<float, K, N>;
  using tile_shape_C  = TileAcc<float, M, N>;
  using tile_shape_O  = TileLeft<float, M, N>;

  gm_shape_A  s0(src0);
  gm_shape_AX s0x(src0x);
  gm_shape_B  s1(src1);
  gm_shape_BX s1x(src1x);
  gm_shape_C  res(dst);

  tile_shape_A  d0;
  tile_shape_AX d0x;
  tile_shape_B  d1;
  tile_shape_BX d1x;
  tile_shape_C  d2;
  tile_shape_O  d3;

  TCOPYIN(d0,  s0);
  TCOPYIN(d0x, s0x);
  TCOPYIN(d1,  s1);
  TCOPYIN(d1x, s1x);

  MATMULMX(d2, d0, d0x, d1, d1x);
  TCVT(d3, d2);
  TCOPYOUT(res, d3);
}

// MATMULMXB: A, B + BX
template <uint16_t M, uint16_t N, uint16_t K>
void test_mxb(float *dst, float *src0, float *src1, float *src1x) {
  using gm_shape_A  = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B  = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_BX = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C  = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A  = TileLeft<float, M, K>;
  using tile_shape_B  = TileRight<float, K, N>;
  using tile_shape_BX = TileRight<float, K, N>;
  using tile_shape_C  = TileAcc<float, M, N>;
  using tile_shape_O  = TileLeft<float, M, N>;

  gm_shape_A  s0(src0);
  gm_shape_B  s1(src1);
  gm_shape_BX s1x(src1x);
  gm_shape_C  res(dst);

  tile_shape_A  d0;
  tile_shape_B  d1;
  tile_shape_BX d1x;
  tile_shape_C  d2;
  tile_shape_O  d3;

  TCOPYIN(d0,  s0);
  TCOPYIN(d1,  s1);
  TCOPYIN(d1x, s1x);

  MATMULMXB(d2, d0, d1, d1x);
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

  float *dst_mx = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst_mx);
  init_dst(dst_mx, size_C);

  float *dst_mxb = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst_mxb);
  init_dst(dst_mxb, size_C);

  float *src0 = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_A);

  float *src0x = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0x);
  init_src_fp(src0x, size_A);

  float *src1 = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_B);

  float *src1x = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1x);
  init_src_fp(src1x, size_B);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_mx<M, N, K>(dst_mx, src0, src0x, src1, src1x);
  test_mxb<M, N, K>(dst_mxb, src0, src1, src1x);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result MX:\n");
  OutArray(dst_mx, size_C);

  printf("Result MXB:\n");
  OutArray(dst_mxb, size_C);

  free(dst_mx);
  free(dst_mxb);
  free(src0);
  free(src0x);
  free(src1);
  free(src1x);

  return 0;
}
