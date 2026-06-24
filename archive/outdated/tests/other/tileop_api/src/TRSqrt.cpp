#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col>
void test(float *dst, float *src) {
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
      TCOPYIN(d0, s0);
      TRSQRT(d1, d0);
      TCOPYOUT(res, d1);
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

  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  float *src = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<gm_row, gm_col, tile_row, tile_col>(dst, src);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);

  free(dst);
  free(src);

  return 0;
}