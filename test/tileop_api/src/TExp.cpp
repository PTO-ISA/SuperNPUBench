#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test_rm(T *dst, T *src) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TCOPYIN(d0, s0);
      TEXP(d1, d0);
      TCOPYOUT(res, d1);
    }
  }
}
template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test_cm(T *dst, T *src) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TCOPYIN(d0, s0);
      TEXP(d1, d0);
      TCOPYOUT(res, d1);
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
  // float32
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);
  float *src = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, gm_size);
  // float16
  __half *dst2 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst2);
  init_dst(dst2, gm_size);
  __half *src2 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src2);
  init_src_fp(src2, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  // TExp只支持float32和16
  test_rm<gm_row, gm_col, tile_row, tile_col,float>(dst, src);
  // half编译通过，运行出错
  // test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst2,src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst2, gm_size);


  free(dst);
  free(src);
  free(dst2);
  free(src2);

  return 0;
}