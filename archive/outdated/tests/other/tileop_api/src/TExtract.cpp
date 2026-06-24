#include "../data.hpp"
#include <cstdint>
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t src_row, uint16_t src_col, uint16_t dst_row,
          uint16_t dst_col>
void test(float *dst, float *src, uint16_t offest) {
  using gm_shape_in = global_tensor<float, RowMajor<src_row, src_col>>;
  using gm_shape_out = global_tensor<float, RowMajor<dst_col, dst_row>>;

  using tile_shape_in = Tile<Location::Vec, float, src_row, src_col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, dst_col, dst_row, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TEXTRACT(d1, d0, offest);
  TCOPYOUT(res, d1);
}

int main() {
  const uint16_t src_row = 128;
  const uint16_t src_col = 64;
  const uint16_t dst_row = 32;
  const uint16_t dst_col = 32;

  const uint16_t offest = 5;

  size_t size_in = src_row * src_col;
  size_t size_out = dst_col * dst_row;

  float *dst = (float *)malloc(size_out * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_out);

  float *src = (float *)malloc(size_in * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size_in);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<src_row, src_col, dst_row, dst_col>(dst, src, offest);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_out);

  free(dst);
  free(src);

  return 0;
}