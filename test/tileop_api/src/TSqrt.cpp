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
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  glb_iterator gSIter(src);
  glb_iterator gDIter(dst);

  size_t block_row = gm_row / tile_row;
  size_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      auto s0 = gSIter(i, j);
      auto res = gDIter(i, j);

      tile_shape t0, t1;
      TCOPYIN(t0, s0);
      TSQRT(t1, t0);
      TCOPYOUT(res, t1);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test_cm(T *dst, T *src) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  glb_iterator gSIter(src);
  glb_iterator gDIter(dst);

  size_t block_row = gm_row / tile_row;
  size_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_col; ++i) {
    for (int j = 0; j < block_row; ++j) {
      auto s0 = gSIter(j, i);
      auto res = gDIter(j, i);

      tile_shape t0, t1;
      TCOPYIN(t0, s0);
      TSQRT(t1, t0);
      TCOPYOUT(res, t1);
    }
  }
}

int main() {

  const size_t gm_row = 32;
  const size_t gm_col = 32;
  const size_t tile_row = 16;
  const size_t tile_col = 16;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

  // __half
  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);

  __half *src_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src_f16);
  init_rows_fp(src_f16, gm_row, gm_col);

  // __fp32
  __fp32 *dst_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(dst_f32);
  init_dst(dst_f32, gm_size);

  __fp32 *src_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(src_f32);
  init_rows_fp(src_f32, gm_row, gm_col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src_f16);
  test_cm<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src_f16);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_f16, gm_size);
  OutArray(dst_f32, gm_size);

  free(dst_f16);
  free(src_f16);
  free(dst_f32);
  free(src_f32);

  return 0;
}