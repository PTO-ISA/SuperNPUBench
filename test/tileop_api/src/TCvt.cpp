#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t row, uint16_t col> void testRow2Nz(float *dst, float *src) {
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

template <uint16_t row, uint16_t col> void testNz2Col(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
  using tile_shape_out = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testNz2Zn(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
  using tile_shape_out = TileRight<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testZn2Nz(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileRight<float, row, col>;
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

template <uint16_t row, uint16_t col> void testNz2Nz(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
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
  const uint16_t row = 16;
  const uint16_t col = 32;

  size_t size = row * col;

  float *dst = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size);

  float *src = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size);

  float *dst1 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst1);
  init_dst(dst1, size);

  float *src1 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src1);
  init_rows_fp(src1, row, col);

  float *dst2 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst2);
  init_dst(dst2, size);

  float *src2 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src2);
  init_rows_fp(src2, row, col);

#ifdef LINX_PMC
  PMC_START();
#endif

  testRow2Nz<row, col>(dst, src);
  testNz2Col<row, col>(dst1, src1);
  testNz2Zn<row, col>(dst2, src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size);
  OutArray(dst1, size);
  OutArray(dst2, size);

  free(dst);
  free(src);
  free(dst1);
  free(src1);
  free(dst2);
  free(src2);

  return 0;
}