#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t row, uint16_t col>
void test(float *dst, float *src0, float *src1, uint16_t *src2) {
  using gm_shape_fp32 = global_tensor<float, RowMajor<row, col>>;
  using gm_shape_uint16 = global_tensor<uint16_t, RowMajor<row, col>>;

  using tile_shape_fp32 = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_uint16 = Tile<Location::Vec, uint16_t, row, col, BLayout::RowMajor>;

  gm_shape_fp32 s0(src0);
  gm_shape_fp32 s1(src1);
  gm_shape_uint16 s2(src2);
  gm_shape_fp32 res(dst);

  tile_shape_fp32 d0;
  tile_shape_fp32 d1;
  tile_shape_uint16 d2;
  tile_shape_fp32 d3;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TCOPYIN(d2, s2);
  TSELECT(d3, d0, d1, d2);
  TCOPYOUT(res, d3);
}

int main() {
  const uint16_t row = 64;
  const uint16_t col = 128;

  size_t size = row * col;

  float *dst = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst);
  init_src_fp(dst, size);

  float *src0 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size);
  float *src1 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size);
  uint16_t *src2 = (uint16_t *)malloc(size * sizeof(uint16_t));
  check_mem_alloc(src2);
  init_01(src2, row, col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<row, col>(dst, src0, src1, src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size);

  free(dst);
  free(src0);
  free(src1);
  free(src2);

  return 0;
}