#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef __linx
int main();

static inline __attribute__((noreturn)) void linx_supernpu_exit(uint32_t code) {
  if (code == 0) {
    __asm__ volatile(
        "BSTART.STD\n"
        "lui 65545, ->u\n"
        "lui 5, ->t\n"
        "addi t#1, 1365, ->t\n"
        "c.swi t#1, [u#1, 0]\n"
        "BSTOP\n"
        ::: "memory");
  } else {
    __asm__ volatile(
        "BSTART.STD\n"
        "lui 65545, ->u\n"
        "lui 19, ->t\n"
        "addi t#1, 819, ->t\n"
        "c.swi t#1, [u#1, 0]\n"
        "BSTOP\n"
        ::: "memory");
  }
  while (1) {
  }
}

extern "C" __attribute__((noreturn, section(".text._start"))) void
_start(void) {
  linx_supernpu_exit(static_cast<uint32_t>(main()));
}
#endif

template <uint16_t row, uint16_t col,typename T>
void test_rm(T *dst, T *src) {
  using gm_shape_in = global_tensor<T, RowMajor<row, col>>;
  using gm_shape_out = global_tensor<T, RowMajor<row, col>>;

  using tile_shape_in = Tile<Location::Vec, T, row, col, BLayout::RowMajor, 1, col>;
  using tile_shape_out = Tile<Location::Vec, T, row, col, BLayout::RowMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);

  tile_shape_in d0;
  tile_shape_out d1;
  TLOAD(d0, s0);
  TEXPANDROW(d1, d0);
  TSTORE(res, d1);
}
template <uint16_t row, uint16_t col,typename T>
void test_cm(T *dst, T *src) {
  using gm_shape_in = global_tensor<T, ColMajor<row, col>>;
  using gm_shape_out = global_tensor<T, ColMajor<row, col>>;

  using tile_shape_in = Tile<Location::Vec, T, row, col, BLayout::ColMajor, 1, col>;
  using tile_shape_out = Tile<Location::Vec, T, row, col, BLayout::ColMajor>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);

  tile_shape_in d0;
  tile_shape_out d1;
  TLOAD(d0, s0);
  TEXPANDROW(d1, d0);
  TSTORE(res, d1);
}

int main() {
#ifdef __linx
  constexpr uint16_t row = 4;
  constexpr uint16_t col = 8;
  constexpr uint16_t size = row * col;

  static int64_t dst_rm[size];
  static int64_t dst_cm[size];
  static int64_t src_rm[size];
  static int64_t src_cm[size];
  init_dst(dst_rm, size);
  init_dst(dst_cm, size);
  init_src_int(src_rm, size);
  init_src_int(src_cm, size);

  test_rm<row, col, int64_t>(dst_rm, src_rm);
  test_cm<row, col, int64_t>(dst_cm, src_cm);
  return 0;
#else
  const uint16_t row = 32;
  const uint16_t col = 32;
  size_t size_in = col;
  size_t size_out = row * col;

  const uint16_t row1 = 64;
  const uint16_t col1 = 64;
  size_t size_in1 = col1;
  size_t size_out1 = row1 * col1;

//float32
  float *dst = (float *)malloc(size_out1 * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_out1);

  float *src = (float *)malloc(size_in1 * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size_in1);
  //float16
  __half *dst1 = (__half *)malloc(size_out * sizeof(__half));
  check_mem_alloc(dst1);
  init_dst(dst1, size_out);

  __half *src1 = (__half *)malloc(size_in * sizeof(__half));
  check_mem_alloc(src1);
  init_src_fp(src1, size_in);
  //int8
  int8_t *dst2 = (int8_t *)malloc(size_out * sizeof(int8_t));
  check_mem_alloc(dst2);
  init_dst(dst2, size_out);

  int8_t *src2 = (int8_t *)malloc(size_in * sizeof(int8_t));
  check_mem_alloc(src2);
  init_src_int(src2, size_in);
//int16
  int16_t *dst3 = (int16_t *)malloc(size_out * sizeof(int16_t));
  check_mem_alloc(dst3);
  init_dst(dst3, size_out);

  int16_t *src3 = (int16_t *)malloc(size_in * sizeof(int16_t));
  check_mem_alloc(src3);
  init_src_int(src3, size_in);
  //int32
  int32_t *dst4 = (int32_t *)malloc(size_out * sizeof(int32_t));
  check_mem_alloc(dst4);
  init_dst(dst4, size_out);

  int32_t *src4 = (int32_t *)malloc(size_in * sizeof(int32_t));
  check_mem_alloc(src4);
  init_src_int(src4, size_in);
    //int64
  int64_t *dst5 = (int64_t *)malloc(size_out * sizeof(int64_t));
  check_mem_alloc(dst5);
  init_dst(dst5, size_out);

  int64_t *src5 = (int64_t *)malloc(size_in * sizeof(int64_t));
  check_mem_alloc(src5);
  init_src_int(src5, size_in);
#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<row, col, float>(dst, src);
  test_rm<row, col, __half>(dst1, src1);
  test_rm<row, col, int8_t>(dst2, src2);
  test_rm<row, col, int16_t>(dst3, src3);
  test_rm<row, col, int32_t>(dst4, src4);
  test_rm<row, col, int64_t>(dst5, src5);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_out);
  OutArray(dst1, size_out);
  OutArray(dst2, size_out);
  OutArray(dst3, size_out);
  OutArray(dst4, size_out);
  OutArray(dst5, size_out);

  free(dst);
  free(src);
  free(dst1);
  free(src1);
  free(dst2);
  free(src2);
  free(dst3);
  free(src3);
  free(dst4);
  free(src4);
  free(dst5);
  free(src5);
  return 0;
#endif
}
