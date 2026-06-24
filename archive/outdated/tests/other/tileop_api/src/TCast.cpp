#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef __linx
template <uint16_t row, uint16_t col, typename T1, typename T2>
void test(T2 *dst, T1 *src) {
  using gm_shape_in = global_tensor<T1, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<T2, RowMajor<row, col>>;
  using tile_shape_in = Tile<Location::Vec, T1, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, T2, row, col, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCAST(d1, d0);
  TCOPYOUT(res, d1);
}

int main() {
  const uint16_t row = 64;
  const uint16_t col = 128;

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

  test<row, col>(dst_fp16, src_fp32);
  test<row, col>(dst_fp32, src_fp16);

#ifdef LINX_PMC
  PMC_END();
#endif

  free(dst_fp32);
  free(dst_fp16);
  free(src_fp32);
  free(src_fp16);

  return 0;
}
#else
int main() { return 0; }
#endif