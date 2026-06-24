#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_RowMajor(T *dst, T *src0, T *src1) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_row; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape s1(src1 + offset);
      gm_shape res(dst + offset);
  
      tile_shape d0, d1, d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TADD(d2, d1, d0);
      TCOPYOUT(res, d2);
    }
  }
}
 
template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_ColMajor(T *dst, T *src0, T *src1) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_col; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_row; ++j) {
      int offset = i * (tile_col * gm_row) + j * tile_row;
      gm_shape s0(src0 + offset);
      gm_shape s1(src1 + offset);
      gm_shape res(dst + offset);
  
      tile_shape d0, d1, d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TADD(d2, d1, d0);
      TCOPYOUT(res, d2);
    }
  }
}

int main() {
  const uint16_t gm_row = 128;
  const uint16_t gm_col = 128;
  const uint16_t tile_row = 32;
  const uint16_t tile_col = 32;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  float *src0 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);
  float *src1 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, gm_size);

  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);
 
  __half *src0_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src0_f16);
  init_src_fp(src0_f16, gm_size);
  __half *src1_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src1_f16);
  init_src_fp(src1_f16, gm_size);
 
  int8_t *dst_i8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst_i8);
  init_dst(dst_i8, gm_size);
 
  int8_t *src0_i8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src0_i8);
  init_src_int(src0_i8, gm_size);
  int8_t *src1_i8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src1_i8);
  init_src_int(src1_i8, gm_size);
 
  int16_t *dst_i16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst_i16);
  init_dst(dst_i16, gm_size);
 
  int16_t *src0_i16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src0_i16);
  init_src_int(src0_i16, gm_size);
  int16_t *src1_i16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src1_i16);
  init_src_int(src1_i16, gm_size);
  
  int32_t *dst_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst_i32);
  init_dst(dst_i32, gm_size);
 
  int32_t *src0_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src0_i32);
  init_src_int(src0_i32, gm_size);
  int32_t *src1_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src1_i32);
  init_src_int(src1_i32, gm_size);
 
  int64_t *dst_i64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst_i64);
  init_dst(dst_i64, gm_size);
 
  int64_t *src0_i64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src0_i64);
  init_src_int(src0_i64, gm_size);
  int64_t *src1_i64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src1_i64);
  init_src_int(src1_i64, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, float>(dst, src0, src1);
 
  test_RowMajor<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src0_f16, src1_f16);
 
  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int8_t>(dst_i8, src0_i8, src1_i8);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int16_t>(dst_i16, src0_i16, src1_i16);
 
  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int32_t>(dst_i32, src0_i32, src1_i32);
 
  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int64_t>(dst_i64, src0_i64, src1_i64);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  //OutArray(dst_f16, gm_size);
  OutArray(dst_i8, gm_size);
  OutArray(dst_i16, gm_size);
  OutArray(dst_i32, gm_size);
  OutArray(dst_i64, gm_size);
 
  free(dst);
  free(src0);
  free(src1);
 
  free(dst_f16);
  free(src0_f16);
  free(src1_f16);
 
  free(dst_i8);
  free(src0_i8);
  free(src1_i8);
 
  free(dst_i16);
  free(src0_i16);
  free(src1_i16);
 
  free(dst_i32);
  free(src0_i32);
  free(src1_i32);
 
  free(dst_i64);
  free(src0_i64);
  free(src1_i64);

  return 0;
}