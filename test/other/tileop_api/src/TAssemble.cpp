#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t dst_row, uint16_t dst_col, uint16_t src0_row,
          uint16_t src0_col, uint16_t src1_row, uint16_t src1_col,
          uint16_t src2_row, uint16_t src2_col>
void test(float *dst, float *src0, float *src1, float *src2) {
  using gm_shape_dst = global_tensor<float, RowMajor<dst_row, dst_col>>;
  using gm_shape_src0 = global_tensor<float, RowMajor<src0_row, src0_col>>;
  using gm_shape_src1 = global_tensor<float, RowMajor<src1_row, src1_col>>;
  using gm_shape_src2 = global_tensor<float, RowMajor<src2_row, src2_col>>;

  using tile_shape_dst = Tile<Location::Vec, float, dst_row, dst_col, BLayout::RowMajor>;
  using tile_shape_src0 = Tile<Location::Vec, float, src0_row, src0_col, BLayout::RowMajor>;
  using tile_shape_src1 = Tile<Location::Vec, float, src1_row, src1_col, BLayout::RowMajor>;
  using tile_shape_src2 = Tile<Location::Vec, float, src2_row, src2_col, BLayout::RowMajor>;

  gm_shape_src0 s0(src0);
  gm_shape_src1 s1(src1);
  gm_shape_src2 s2(src2);
  gm_shape_dst res(dst);

  tile_shape_src0 d0;
  tile_shape_src1 d1;
  tile_shape_src2 d2;
  tile_shape_dst d3;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TCOPYIN(d2, s2);
  TASSEMBLE(d3, d0, d1, d2);
  TCOPYOUT(res, d3);
}

int main() {
  const uint16_t dst_row = 64;
  const uint16_t dst_col = 128;
  const uint16_t src0_row = 64;
  const uint16_t src0_col = 32;
  const uint16_t src1_row = 64;
  const uint16_t src1_col = 32;
  const uint16_t src2_row = 64;
  const uint16_t src2_col = 64;

  size_t size_dst = dst_row * dst_col;
  size_t size_src0 = src0_row * src0_col;
  size_t size_src1 = src1_row * src1_col;
  size_t size_src2 = src2_row * src2_col;

  float *dst = (float *)malloc(size_dst * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_dst);
  float *src0 = (float *)malloc(size_src0 * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_src0);
  float *src1 = (float *)malloc(size_src1 * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_src1);
  float *src2 = (float *)malloc(size_src2 * sizeof(float));
  check_mem_alloc(src2);
  init_src_fp(src2, size_src2);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<dst_row, dst_col, src0_row, src0_col, src1_row, src1_col, src2_row,
       src2_col>(dst, src0, src1, src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_dst);

  free(dst);
  free(src0);
  free(src1);
  free(src2);

  return 0;
}