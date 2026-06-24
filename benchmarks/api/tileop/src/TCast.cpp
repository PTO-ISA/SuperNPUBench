#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <size_t row, size_t col, typename T1, typename T2>
void test_rm(T2 *dst, T1 *src) {
  using gm_shape_in = global_tensor<T1, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<T2, RowMajor<row, col>>;
  using tile_shape_in = Tile<Location::Vec, T1, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, T2, row, col, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TLOAD(d0, s0);
  TCAST(d1, d0);
  TSTORE(res, d1);
}

template <size_t row, size_t col, typename T1, typename T2>
void test_cm(T2 *dst, T1 *src) {
  using gm_shape_in = global_tensor<T1, ColMajor<row, col>>;
  using gm_shape_out = global_tensor<T2, ColMajor<row, col>>;
  using tile_shape_in = Tile<Location::Vec, T1, row, col, BLayout::ColMajor>;
  using tile_shape_out = Tile<Location::Vec, T2, row, col, BLayout::ColMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TLOAD(d0, s0);
  TCAST(d1, d0);
  TSTORE(res, d1);
}

template <size_t row, size_t col, typename T1, typename T2>
void test_Nz(T2 *dst, T1 *src) {
  using gm_shape_in = global_tensor<T1, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<T2, RowMajor<row, col>>;
  using tile_shape_in = TileLeft<T1, row, col>;
  using tile_shape_out = TileLeft<T2, row, col>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TLOAD(d0, s0);
  TCAST(d1, d0);
  TSTORE(res, d1);
}

int main() {
  const size_t row = 16;
  const size_t col = 16;

  size_t size = row * col;

  float *dst_fp32 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst_fp32);
  init_dst(dst_fp32, size);
  __half *dst_fp16 = (__half *)malloc(size * sizeof(__half));
  check_mem_alloc(dst_fp16);
  init_dst(dst_fp16, size);

  float *src_fp32 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src_fp32);
  init_src_fp(src_fp32, size);
  __half *src_fp16 = (__half *)malloc(size * sizeof(__half));
  check_mem_alloc(src_fp16);
  init_src_fp(src_fp16, size);

#ifdef LINX_PMC
  PMC_START();
#endif

  // 编译通过, 但qemu运行会崩掉
  test_rm<row, col, float, __half>(dst_fp16, src_fp32);
  test_cm<row, col, __half, float>(dst_fp32, src_fp16);

  // 编译报错
  // test_Nz<row, col, float, __half>(dst_fp16, src_fp32);
  // test_Nz<row, col, __half, float>(dst_fp32, src_fp16);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_fp16, size);
  OutArray(dst_fp32, size);

  free(dst_fp32);
  free(dst_fp16);
  free(src_fp32);
  free(src_fp16);

  return 0;
}