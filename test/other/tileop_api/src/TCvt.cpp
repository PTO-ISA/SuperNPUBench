#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t row, uint16_t col> void Test(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_out = TileLeft<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

int main() {
  const uint16_t row = 64;
  const uint16_t col = 128;

  size_t size = row * col;

  float *dst = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size);

  float *src = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size);

#ifdef LINX_PMC
  PMC_START();
#endif

  Test<row, col>(dst, src);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size);

  free(dst);
  free(src);

  return 0;
}