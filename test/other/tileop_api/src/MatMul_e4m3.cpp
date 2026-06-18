#include <common/pto_tileop.hpp>
#include "../data.hpp"
#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <typename TA, typename TB>
void __vec__ test_cvt(typename TA::TileDType __out__ a,
                      typename TB::TileDType __in__ b) {
  using AType = typename TA::DType;
  using BType = typename TB::DType;
  BType *pb = blkv_get_tile_ptr(b);
  AType *pa = blkv_get_tile_ptr(a);
  int x = blkv_get_index_x();
  int y = blkv_get_index_y();
  int idx = index<TA>(y, x);
  AType o = (AType)(pb[idx]);
  pa[idx] = o;
}

template <uint16_t M, uint16_t N, uint16_t K>
void test(float *dst, float *src0, float *src1) {
  using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A = TileLeft<float, M, K>;
  using tile_shape_B = TileRight<float, K, N>;
  using tile_shape_C = TileAcc<float, M, N>;
  using tile_shape_LA = TileLeft<__fp8_e4m3, M, K>;
  using tile_shape_LB = TileRight<__fp8_e4m3, K, N>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;
  tile_shape_LA lda;
  tile_shape_LB ldb;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  test_cvt<tile_shape_LA, tile_shape_A><<<M, K, 1>>>(lda.data(), d0.data());
  test_cvt<tile_shape_LB, tile_shape_B><<<K, N, 1>>>(ldb.data(), d1.data());
  MATMUL(d2, lda, ldb);
  TCOPYOUT(res, d2);
}

int main() {
  const uint16_t M = 64;
  const uint16_t K = 32;
  const uint16_t N = 64;

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
