#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col>
void test_RowMajor(float *dst, float *src, float s) {
  using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TLOAD(d0, s0);
      TMINS(d1, d0, s);
      TSTORE(res, d1);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col>
void test_ColMajor(float *dst, float *src, float s) {
  using gm_shape = global_tensor<float, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::ColMajor>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TLOAD(d0, s0);
      TMINS(d1, d0, s);
      TSTORE(res, d1);
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

  float *src = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, gm_size);

  float *dst_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst_col);
  init_dst(dst_col, gm_size);

  float *src_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src_col);
  init_src_fp(src_col, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor<gm_row, gm_col, tile_row, tile_col>(dst, src, s_fp32);

  test_ColMajor<gm_row, gm_col, tile_row, tile_col>(dst_col, src_col, s_fp32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst_col, gm_size);

  free(dst);
  free(src);

  free(dst_col);
  free(src_col);

  return 0;
}