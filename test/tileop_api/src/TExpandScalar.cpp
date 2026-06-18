#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col,typename T>
void test_rm(T *dst, T s) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col>;
  gm_shape res(dst);

  tile_shape d0;
  TEXPANDSCALAR(d0, s);
  TCOPYOUT(res, d0);
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col,typename T>
void test_rm_dynamic(T *dst, T s) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, 2*tile_row, 2*tile_col, BLayout::RowMajor, -1, -1>;

  volatile size_t tile_valid_row = tile_row;
  volatile size_t tile_valid_col = tile_col;

  gm_shape res(dst);
  tile_shape d0(tile_valid_row, tile_valid_col);

  TEXPANDSCALAR(d0, s);
  TCOPYOUT(res, d0);
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col,typename T>
void test_cm(T *dst, T s) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
  gm_shape res(dst);

  tile_shape d0;
  TEXPANDSCALAR(d0, s);
  TCOPYOUT(res, d0);
}

int main() {
  const uint16_t gm_row = 16;
  const uint16_t gm_col = 32;
  const uint16_t tile_row = 16;
  const uint16_t tile_col = 32;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;
  // float32
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);
  // float16
  __half *dst1 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst1);
  init_dst(dst1, gm_size);
  // int8
  int8_t *dst2 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst2);
  init_dst(dst2, gm_size);
  // int16
  int16_t *dst3 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst3);
  init_dst(dst3, gm_size);
  // int32
  int32_t *dst4 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst4);
  init_dst(dst4, gm_size);
  // int64
  int64_t *dst5 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst5);
  init_dst(dst5, gm_size);
  // int32 dynamic
  int32_t *dst6 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst6);
  init_dst(dst6, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<gm_row, gm_col, tile_row, tile_col, float>(dst, s_fp32);
  test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst1, s_fp16);
  test_rm<gm_row, gm_col, tile_row, tile_col, int8_t>(dst2, s_i8);
  test_rm<gm_row, gm_col, tile_row, tile_col, int16_t>(dst3, s_i16);
  test_rm<gm_row, gm_col, tile_row, tile_col, int32_t>(dst4, s_i32);
  test_rm<gm_row, gm_col, tile_row, tile_col, int64_t>(dst5, s_i64);
  test_rm_dynamic<gm_row, gm_col, tile_row, tile_col, int32_t>(dst6, s_i32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst1, gm_size);
  OutArray(dst2, gm_size);
  OutArray(dst3, gm_size);
  OutArray(dst4, gm_size);
  OutArray(dst5, gm_size);
  OutArray(dst6, gm_size);

  free(dst);
  free(dst1);
  free(dst2);
  free(dst3);
  free(dst4);
  free(dst5);
  free(dst6);

  return 0;
}