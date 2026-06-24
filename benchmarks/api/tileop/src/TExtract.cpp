#include "../data.hpp"
#include <common/pto_tileop.hpp>
#include <cstdint>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <size_t src_row, size_t src_col, size_t dst_row,
          size_t dst_col>
void test_rm(float *dst, float *src, size_t offset_i, size_t offset_j) {
  using gm_shape_in = global_tensor<float, RowMajor<src_row, src_col>>;
  using gm_shape_out = global_tensor<float, RowMajor<dst_row, dst_col>>;

  using tile_shape_in = Tile<Location::Vec, float, src_row, src_col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, dst_row, dst_col, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TLOAD(d0, s0);
  TEXTRACT(d1, d0, offset_i, offset_j);
  TSTORE(res, d1);
}

template <size_t src_row, size_t src_col, size_t dst_row, size_t dst_col>
void test_rm_dynamic(float *dst, float *src, size_t offset_i, size_t offset_j) {
  using gm_shape_in = global_tensor<float, RowMajor<src_row, src_col>>;
  using gm_shape_out = global_tensor<float, RowMajor<dst_row, dst_col>>;

  using tile_shape_in = Tile<Location::Vec, float, 2*src_row, 2*src_col, BLayout::RowMajor, -1, -1>;
  using tile_shape_out = Tile<Location::Vec, float, 2*dst_row, 2*dst_col, BLayout::RowMajor, -1, -1>;

  volatile size_t src_valid_row = src_row;
  volatile size_t src_valid_col = src_col;
  volatile size_t dst_valid_row = dst_row;
  volatile size_t dst_valid_col = dst_col;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0(src_valid_row, src_valid_col);
  tile_shape_out d1(dst_valid_row, dst_valid_col);

  TLOAD(d0, s0);
  TEXTRACT(d1, d0, offset_i, offset_j);
  TSTORE(res, d1);
}

template <size_t src_row, size_t src_col, size_t dst_row, size_t dst_col>
void test_cm(float *dst, float *src, size_t offset_i, size_t offset_j) {
  using gm_shape_in = global_tensor<float, ColMajor<src_row, src_col>>;
  using gm_shape_out = global_tensor<float, ColMajor<dst_row, dst_col>>;

  using tile_shape_in = Tile<Location::Vec, float, src_row, src_col, BLayout::ColMajor>;
  using tile_shape_out = Tile<Location::Vec, float, dst_row, dst_col, BLayout::ColMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TLOAD(d0, s0);
  TEXTRACT(d1, d0, offset_i, offset_j);
  TSTORE(res, d1);
}

int main() {
  const size_t src_row = 32;
  const size_t src_col = 16;
  const size_t dst_row = 16;
  const size_t dst_col = 8;

  const size_t offset_i = 16;
  const size_t offset_j = 8;

  size_t size_in = src_row * src_col;
  size_t size_out = dst_col * dst_row;

  float *dst1 = (float *)malloc(size_out * sizeof(float));
  check_mem_alloc(dst1);
  init_dst(dst1, size_out);
  float *dst2 = (float *)malloc(size_out * sizeof(float));
  check_mem_alloc(dst2);
  init_dst(dst2, size_out);
  float *dst3 = (float *)malloc(size_out * sizeof(float));
  check_mem_alloc(dst3);
  init_dst(dst3, size_out);

  float *src = (float *)malloc(size_in * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size_in);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_cm<src_row, src_col, dst_row, dst_col>(dst1, src, offset_i, offset_j);
  test_rm<src_row, src_col, dst_row, dst_col>(dst2, src, offset_i, offset_j);
  test_rm_dynamic<src_row, src_col, dst_row, dst_col>(dst3, src, offset_i, offset_j);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst1, size_out);
  OutArray(dst2, size_out);
  OutArray(dst3, size_out);

  free(dst1);
  free(dst2);
  free(dst3);
  free(src);

  return 0;
}