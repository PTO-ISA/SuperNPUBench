#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <size_t row, size_t col, typename T> void test_rm(T *dst, T *src) {
  using gm_shape_in = global_tensor<T, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<T, RowMajor<col, row>>;

  using tile_shape_in = Tile<Location::Vec, T, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, T, row, col, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TTRANS(d1, d0);
  TCOPYOUT(res, d1);
}

template <size_t row, size_t col, typename T> void test_cm(T *dst, T *src) {
  using gm_shape_in = global_tensor<T, ColMajor<row, col>>;
  using gm_shape_out = global_tensor<T, ColMajor<col, row>>;

  using tile_shape_in = Tile<Location::Vec, T, row, col, BLayout::ColMajor>;
  using tile_shape_out = Tile<Location::Vec, T, row, col, BLayout::ColMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TTRANS(d1, d0);
  TCOPYOUT(res, d1);
}

int main() {
  const size_t row = 32;
  const size_t col = 32;

  size_t size_in = row * col;
  size_t size_out = col * row;

  // int8
  int8_t *dst_int8 = (int8_t *)malloc(size_out * sizeof(int8_t));
  check_mem_alloc(dst_int8);
  init_dst(dst_int8, size_out);

  int8_t *src_int8 = (int8_t *)malloc(size_in * sizeof(int8_t));
  check_mem_alloc(src_int8);
  init_src_int(src_int8, size_in);

  // int16
  int16_t *dst_int16 = (int16_t *)malloc(size_out * sizeof(int16_t));
  check_mem_alloc(dst_int16);
  init_dst(dst_int16, size_out);

  int16_t *src_int16 = (int16_t *)malloc(size_in * sizeof(int16_t));
  check_mem_alloc(src_int16);
  init_src_int(src_int16, size_in);

  // int32
  int32_t *dst_int32 = (int32_t *)malloc(size_out * sizeof(int32_t));
  check_mem_alloc(dst_int32);
  init_dst(dst_int32, size_out);

  int32_t *src_int32 = (int32_t *)malloc(size_in * sizeof(int32_t));
  check_mem_alloc(src_int32);
  init_src_int(src_int32, size_in);

  // int 64
  int64_t *dst_int64 = (int64_t *)malloc(size_out * sizeof(int64_t));
  check_mem_alloc(dst_int64);
  init_dst(dst_int64, size_out);

  int64_t *src_int64 = (int64_t *)malloc(size_in * sizeof(int64_t));
  check_mem_alloc(src_int64);
  init_src_int(src_int64, size_in);

  // __half
  __half *dst_f16 = (__half *)malloc(size_out * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, size_out);

  __half *src_f16 = (__half *)malloc(size_in * sizeof(__half));
  check_mem_alloc(src_f16);
  init_src_fp(src_f16, size_in);

  // __fp32
  __fp32 *dst_f32 = (__fp32 *)malloc(size_out * sizeof(__fp32));
  check_mem_alloc(dst_f32);
  init_dst(dst_f32, size_out);

  __fp32 *src_f32 = (__fp32 *)malloc(size_in * sizeof(__fp32));
  check_mem_alloc(src_f32);
  init_src_fp(src_f32, size_in);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<row, col, int8_t>(dst_int8, src_int8);
  test_rm<row, col, int16_t>(dst_int16, src_int16);
  test_rm<row, col, int32_t>(dst_int32, src_int32);
  test_rm<row, col, int64_t>(dst_int64, src_int64);
  test_cm<row, col, __half>(dst_f16, src_f16);
  test_cm<row, col, __fp32>(dst_f32, src_f32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_int8, size_out);
  OutArray(dst_int16, size_out);
  OutArray(dst_int32, size_out);
  OutArray(dst_int64, size_out);
  OutArray(dst_f16, size_out);
  OutArray(dst_f32, size_out);

  free(dst_int8);
  free(src_int8);
  free(dst_int16);
  free(src_int16);
  free(dst_int32);
  free(src_int32);
  free(dst_int64);
  free(src_int64);
  free(dst_f16);
  free(src_f16);
  free(dst_f32);
  free(src_f32);

  return 0;
}