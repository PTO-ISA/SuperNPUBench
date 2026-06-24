#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_RowMajor(T *dst, T *src0, T *src1) {
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
      TLOAD(d0, s0);
      TLOAD(d1, s1);
      TMIN(d2, d1, d0);
      TSTORE(res, d2);
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
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape s1(src1 + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1, d2;
      TLOAD(d0, s0);
      TLOAD(d1, s1);
      TMIN(d2, d1, d0);
      TSTORE(res, d2);
    }
  }
}

int main() {
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 64;
  const uint16_t tile_row = 16;
  const uint16_t tile_col = 16;

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

  float *dst_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst_col);
  init_dst(dst_col, gm_size);

  float *src0_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0_col);
  init_src_fp(src0_col, gm_size);
  float *src1_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src1_col);
  init_src_fp(src1_col, gm_size);

  float *dst_Nz = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst_Nz);
  init_dst(dst, gm_size);

  float *src0_Nz = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0_Nz);
  init_src_fp(src0_Nz, gm_size);
  float *src1_Nz = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src1_Nz);
  init_src_fp(src1_Nz, gm_size);

  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);

  __half *src0_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src0_f16);
  init_src_fp(src0_f16, gm_size);
  __half *src1_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src1_f16);
  init_src_fp(src1_f16, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, float>(dst, src0, src1);

  test_ColMajor<gm_row, gm_col, tile_row, tile_col, float>(dst_col, src0_col, src1);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src0_f16, src1_f16);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst_col, gm_size);
  OutArray(dst_Nz, gm_size);
  OutArray(dst_f16, gm_size);

  free(dst);
  free(src0);
  free(src1);

  free(dst_col);
  free(src0_col);
  free(src1_col);

  free(dst_Nz);
  free(src0_Nz);
  free(src1_Nz);

  free(dst_f16);
  free(src0_f16);
  free(src1_f16);

  return 0;
}