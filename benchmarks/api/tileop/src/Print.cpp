#include "../data.hpp"
#include <common/debug_utils.hpp>
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif
template <uint16_t M, uint16_t N, uint16_t K>
void test_ACC(float *dst, float *src0, float *src1) {
  using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A = TileLeft<float, M, K>;
  using tile_shape_B = TileRight<float, K, N>;
  using tile_shape_C = TileAcc<float, M, N>;
  using tile_shape_O = Tile<Location::Vec, float, M, N>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;
  tile_shape_O d3;

  TLOAD(d0, s0);
  TLOAD(d1, s1);
  MATMUL(d2, d0, d1);
  TCVT(d3, d2);
  print_tile(d3);
  TSTORE(res, d3);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_Nz(T *dst) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileLeft<T, tile_row, tile_col, 16, 16>;
  gm_shape g(dst);
  tile_shape t;
  TLOAD(t, g);
  print_tile(t);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_Zn(T *dst) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileRight<T, tile_row, tile_col, 16, 16>;
  gm_shape g(dst);
  tile_shape t;
  TLOAD(t, g);
  print_tile(t);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_RowMajor(T *dst) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
  gm_shape g(dst);
  tile_shape t;
  TLOAD(t, g);
  print_tile(t);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_ColMajor(T *dst) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
  gm_shape g(dst);
  tile_shape t;
  TLOAD(t, g);
  print_tile(t);
}

int main() {
  const uint16_t gm_row = 16;
  const uint16_t gm_col = 16;

  size_t gm_size = gm_row * gm_col;

  float *data = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(data);
  init_rows_fp(data, gm_row, gm_col);

  __half *data_hf = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(data_hf);
  init_rows_fp(data_hf, gm_row, gm_col);

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

  test_ACC<M, N, K>(dst, src0, src1);

  test_Nz<gm_row, gm_col, gm_row, gm_col, float>(data);
  test_Zn<gm_row, gm_col, gm_row, gm_col, float>(data);
  test_ColMajor<gm_row, gm_col, gm_row, gm_col, float>(data);
  test_RowMajor<gm_row, gm_col, gm_row, gm_col, float>(data);

  test_Nz<gm_row, gm_col, gm_row, gm_col, __half>(data_hf);
  test_Zn<gm_row, gm_col, gm_row, gm_col, __half>(data_hf);
  test_ColMajor<gm_row, gm_col, gm_row, gm_col, __half>(data_hf);
  test_RowMajor<gm_row, gm_col, gm_row, gm_col, __half>(data_hf);

#ifdef LINX_PMC
  PMC_END();
#endif

  free(dst);
  free(src0);
  free(src1);
  free(data);
  free(data_hf);
  return 0;
}
