#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_rm(T *dst, T *src0, T *src1) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape s1(src1 + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1, d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TMAX(d2, d1, d0);
      TCOPYOUT(res, d2);
    }
  }
}
template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_cm(T *dst, T *src0, T *src1) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape s1(src1 + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1, d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TMAX(d2, d1, d0);
      TCOPYOUT(res, d2);
    }
  }
}

int main() {
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 64;
  const uint16_t tile_row = 32;
  const uint16_t tile_col = 32;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;
  // float32
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  float *src0 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);
  float *src1 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, gm_size);
  // float16
  __half *dst1 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst1);
  init_dst(dst1, gm_size);

  __half *src2 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src2);
  init_src_fp(src2, gm_size);
  __half *src3 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src3);
  init_src_fp(src3, gm_size);
  // int8
  int8_t *dst2 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst2);
  init_dst(dst2, gm_size);

  int8_t *src4 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src4);
  init_src_int(src4, gm_size);
  int8_t *src5 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src5);
  init_src_int(src5, gm_size);
  // int16
  int16_t *dst3 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst3);
  init_dst(dst3, gm_size);

  int16_t *src6 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src6);
  init_src_int(src6, gm_size);
  int16_t *src7 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src7);
  init_src_int(src7, gm_size);
  // int32
  int32_t *dst4 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst4);
  init_dst(dst4, gm_size);

  int32_t *src8 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src8);
  init_src_int(src8, gm_size);
  int32_t *src9 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src9);
  init_src_int(src9, gm_size);
  // int64
  int64_t *dst5 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst5);
  init_dst(dst5, gm_size);

  int64_t *src10 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src10);
  init_src_int(src10, gm_size);
  int64_t *src11 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src11);
  init_src_int(src11, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<gm_row, gm_col, tile_row, tile_col,float>(dst, src0, src1);
  test_rm<gm_row, gm_col, tile_row, tile_col,__half>(dst1, src2, src3);
  test_rm<gm_row, gm_col, tile_row, tile_col,int8_t>(dst2, src4, src5);
  test_rm<gm_row, gm_col, tile_row, tile_col,int16_t>(dst3, src6, src7);
  test_rm<gm_row, gm_col, tile_row, tile_col,int32_t>(dst4, src8, src9);
  test_rm<gm_row, gm_col, tile_row, tile_col,int64_t>(dst5, src10, src11);

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

  free(dst);
  free(src0);
  free(src1);
  free(dst1);
  free(src2);
  free(src3);
  free(dst2);
  free(src4);
  free(src5);
  free(dst3);
  free(src6);
  free(src7);
  free(dst4);
  free(src8);
  free(src9);
  free(dst5);
  free(src10);
  free(src11);

  return 0;
}