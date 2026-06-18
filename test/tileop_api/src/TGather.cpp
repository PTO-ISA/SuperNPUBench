#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t src_row, uint16_t src_col, uint16_t row, uint16_t col>
void test_RowMajor(float *dst, float *src, uint16_t *indices) {
  using gm_shape_src = global_tensor<float, RowMajor<src_row, src_col>>;
  using gm_shape_indices = global_tensor<uint16_t, RowMajor<row, col>>;
  using gm_shape_dst = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_src = Tile<Location::Vec, float, src_row, src_col, BLayout::RowMajor>;
  using tile_shape_indices = Tile<Location::Vec, uint16_t, row, col, BLayout::RowMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;

  gm_shape_src s0(src);
  gm_shape_indices s1(indices);
  gm_shape_dst res(dst);

  tile_shape_src d0;
  tile_shape_indices d1;
  tile_shape_dst d2;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TGATHER(d2, d0, d1);
  TCOPYOUT(res, d2);
}

template <uint16_t src_row, uint16_t src_col, uint16_t row, uint16_t col>
void test_ColMajor(float *dst, float *src, uint16_t *indices) {
  using gm_shape_src = global_tensor<float, ColMajor<src_row, src_col>>;
  using gm_shape_indices = global_tensor<uint16_t, ColMajor<row, col>>;
  using gm_shape_dst = global_tensor<float, ColMajor<row, col>>;

  using tile_shape_src = Tile<Location::Vec, float, src_row, src_col, BLayout::ColMajor>;
  using tile_shape_indices = Tile<Location::Vec, uint16_t, row, col, BLayout::ColMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, row, col, BLayout::ColMajor>;

  gm_shape_src s0(src);
  gm_shape_indices s1(indices);
  gm_shape_dst res(dst);

  tile_shape_src d0;
  tile_shape_indices d1;
  tile_shape_dst d2;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TGATHER(d2, d0, d1);
  TCOPYOUT(res, d2);
}

int main() {
  const uint16_t src_row = 32;
  const uint16_t src_col = 64;
  const uint16_t row = 16;
  const uint16_t col = 32;

  size_t size_src = src_row * src_col;
  size_t size_indices_dst = row * col;

  float *dst = (float *)malloc(size_indices_dst * sizeof(float));
  check_mem_alloc(dst);
  init_src_fp(dst, size_indices_dst);

  float *src = (float *)malloc(size_src * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size_src);
  uint16_t *indices = (uint16_t *)malloc(size_indices_dst * sizeof(uint16_t));
  check_mem_alloc(indices);
  init_index(indices, row, col);

  float *dst_colmajor = (float *)malloc(size_indices_dst * sizeof(float));
  check_mem_alloc(dst_colmajor);
  init_src_fp(dst_colmajor, size_indices_dst);

  float *src_colmajor = (float *)malloc(size_src * sizeof(float));
  check_mem_alloc(src_colmajor);
  init_src_fp(src_colmajor, size_src);
  uint16_t *indices_colmajor = (uint16_t *)malloc(size_indices_dst * sizeof(uint16_t));
  check_mem_alloc(indices_colmajor);
  init_index(indices_colmajor, row, col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor<src_row, src_col, row, col>(dst, src, indices);

  test_ColMajor<src_row, src_col, row, col>(dst_colmajor, src_colmajor, indices_colmajor);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_indices_dst);
  OutArray(dst_colmajor, size_indices_dst);

  free(dst);
  free(src);
  free(indices);

  free(dst_colmajor);
  free(src_colmajor);
  free(indices_colmajor);

  return 0;
}