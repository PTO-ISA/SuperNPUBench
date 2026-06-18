#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t row, uint16_t col> void test(float *dst, float *src) {
  using gm_shape_in = global_tensor<float, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<float, RowMajor<row, 1>>;

  using tile_shape_in = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, row, 1, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TROWMAX(d1, d0);
  TCOPYOUT(res, d1);
}

int main() {
  const uint16_t row = 128;
  const uint16_t col = 64;

  size_t size_in = row * col;
  size_t size_out = row;

  float *dst = (float *)malloc(size_out * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_out);

  float *src = (float *)malloc(size_in * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size_in);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<row, col>(dst, src);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_out);

  free(dst);
  free(src);

  return 0;
}