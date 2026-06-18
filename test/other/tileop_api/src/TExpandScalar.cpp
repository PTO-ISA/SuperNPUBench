#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col>
void test(float *dst, float s) {
  using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;
  gm_shape res(dst);

  tile_shape d0;
  TEXPANDSCALAR(d0, s);
  TCOPYOUT(res, d0);
}

int main() {
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 128;
  const uint16_t tile_row = 64;
  const uint16_t tile_col = 128;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<gm_row, gm_col, tile_row, tile_col>(dst, s_fp32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);

  free(dst);

  return 0;
}