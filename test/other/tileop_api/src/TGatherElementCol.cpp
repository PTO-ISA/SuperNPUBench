#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t src0_row, uint16_t src0_col, uint16_t row, uint16_t col>
void test(float *dst, float *src0, uint16_t *src1) {
  using gm_shape_src0 = global_tensor<float, RowMajor<src0_row, src0_col>>;
  using gm_shape_src1 = global_tensor<uint16_t, RowMajor<row, col>>;
  using gm_shape_dst = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_src0 = Tile<Location::Vec, float, src0_row, src0_col, BLayout::RowMajor>;
  using tile_shape_src1 = Tile<Location::Vec, uint16_t, row, col, BLayout::RowMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;

  gm_shape_src0 s0(src0);
  gm_shape_src1 s1(src1);
  gm_shape_dst res(dst);

  tile_shape_src0 d0;
  tile_shape_src1 d1;
  tile_shape_dst d2;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TGATHERELEMENTCOL(d2, d0, d1);
  TCOPYOUT(res, d2);
}

int main() {
  const uint16_t src0_row = 128;
  const uint16_t src0_col = 64;
  const uint16_t row = 32;
  const uint16_t col = 32;

  size_t size_src0 = src0_row * src0_col;
  size_t size_src1_dst = row * col;

  float *dst = (float *)malloc(size_src1_dst * sizeof(float));
  check_mem_alloc(dst);
  init_src_fp(dst, size_src1_dst);

  float *src0 = (float *)malloc(size_src0 * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_src0);
  uint16_t *src1 = (uint16_t *)malloc(size_src1_dst * sizeof(uint16_t));
  check_mem_alloc(src1);
  init_index(src1, row, col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<src0_row, src0_col, row, col>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_src1_dst);

  free(dst);
  free(src0);
  free(src1);

  return 0;
}