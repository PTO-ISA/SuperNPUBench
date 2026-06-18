#include "../data.hpp"
#include <common/pto_tileop.hpp>
#include <cstdint>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <size_t tile_row, size_t tile_col, size_t valid_row, size_t valid_col>
void test_rm(int32_t *dst, int32_t *src) {
  using gm_shape_in = global_tensor<int32_t, RowMajor<tile_row, tile_col>>;
  using gm_shape_out = global_tensor<int32_t, RowMajor<tile_row, tile_col>>;

  using tile_shape_in = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::RowMajor, valid_row, valid_col>;
  using tile_shape_out = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::RowMajor, tile_row, tile_col,
                              SLayout::NoneBox, 512, PadValue::Zero>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TFILLPAD(d1, d0);
  TCOPYOUT(res, d1);
}

template <size_t tile_row, size_t tile_col, size_t valid_row, size_t valid_col>
void test_rm_dynamic(int32_t *dst, int32_t *src) {
  using gm_shape_in = global_tensor<int32_t, RowMajor<tile_row, tile_col>>;
  using gm_shape_out = global_tensor<int32_t, RowMajor<tile_row, tile_col>>;

  using tile_shape_in = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::RowMajor, -1, -1>;
  using tile_shape_out = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::RowMajor, -1, -1,
                              SLayout::NoneBox, 512, PadValue::Zero>;

  volatile size_t src_valid_row = valid_row;
  volatile size_t src_valid_col = valid_col;

  volatile size_t dst_valid_row = tile_row;
  volatile size_t dst_valid_col = tile_col;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0(src_valid_row, src_valid_col);
  tile_shape_out d1(dst_valid_row, dst_valid_col);

  TCOPYIN(d0, s0);
  TFILLPAD(d1, d0);
  TCOPYOUT(res, d1);
}

template <size_t tile_row, size_t tile_col, size_t valid_row, size_t valid_col>
void test_cm(int32_t *dst, int32_t *src) {
  using gm_shape_in = global_tensor<int32_t, ColMajor<tile_row, tile_col>>;
  using gm_shape_out = global_tensor<int32_t, ColMajor<tile_row, tile_col>>;

  using tile_shape_in = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::ColMajor, valid_row, valid_col>;
  using tile_shape_out = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::ColMajor, tile_row, tile_col,
                              SLayout::NoneBox, 512, PadValue::Zero>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TFILLPAD(d1, d0);
  TCOPYOUT(res, d1);
}


int main() {
  const size_t tile_row = 16;
  const size_t tile_col = 16;
  const size_t valid_row = 9;
  const size_t valid_col = 9;

  size_t size = tile_row * tile_col;

  int32_t *dst1 = (int32_t *)malloc(size * sizeof(int32_t));
  check_mem_alloc(dst1);
  init_dst(dst1, size);

  int32_t *dst2 = (int32_t *)malloc(size * sizeof(int32_t));
  check_mem_alloc(dst2);
  init_dst(dst2, size);

  int32_t *dst3 = (int32_t *)malloc(size * sizeof(int32_t));
  check_mem_alloc(dst3);
  init_dst(dst3, size);
 
  int32_t *src = (int32_t *)malloc(size * sizeof(int32_t));
  check_mem_alloc(src);
  init_src_int(src, size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<tile_row, tile_col, valid_row, valid_col>(dst1, src);
  test_cm<tile_row, tile_col, valid_row, valid_col>(dst2, src);
  test_rm_dynamic<tile_row, tile_col, valid_row, valid_col>(dst3, src);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst1, size);
  OutArray(dst2, size);
  OutArray(dst3, size);

  free(dst1);
  free(dst2);
  free(dst3);
  free(src);

  return 0;
}