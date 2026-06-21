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

extern "C" __attribute__((noreturn, section(".text._start"))) void _start(void) {
  linx_supernpu_exit(static_cast<uint32_t>(main()));
}
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test(T *dst, T *src) {
  using gm_shape_in = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using gm_shape_out = global_tensor<T, RowMajor<gm_row * 2, gm_col / 2>>;

  using tile_shape_in = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, T, tile_row * 2, tile_col / 2, BLayout::RowMajor>;
  gm_shape_in s0(src);
  gm_shape_out res(dst);

  tile_shape_in d0;
  tile_shape_out d1;
  TCOPYIN(d0, s0);
  TRESHAPE(d1, d0);
  TCOPYOUT(res, d1);
}

int main() {
#ifdef __linx
  constexpr size_t gm_row = 4;
  constexpr size_t gm_col = 8;
  constexpr size_t tile_row = 4;
  constexpr size_t tile_col = 8;
#else
  constexpr size_t gm_row = 64;
  constexpr size_t gm_col = 64;
  constexpr size_t tile_row = 64;
  constexpr size_t tile_col = 64;
#endif

  constexpr size_t gm_size = gm_row * gm_col;
  constexpr size_t tile_size = tile_row * tile_col;
  (void)tile_size;

#ifdef __linx
  static int64_t dst[gm_size];
  static int64_t src[gm_size];
  init_dst(dst, gm_size);
  init_src_uint(src, gm_size);

  test<gm_row, gm_col, tile_row, tile_col, int64_t>(dst, src);

  return 0;
#else
  // int8_t
  int8_t *dst_int8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst_int8);
  init_dst(dst_int8, gm_size);

  int8_t *src_int8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src_int8);
  init_src_uint(src_int8, gm_size);

  // int16_t
  int16_t *dst_int16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst_int16);
  init_dst(dst_int16, gm_size);

  int16_t *src_int16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src_int16);
  init_src_uint(src_int16, gm_size);

  // int32_t
  int32_t *dst_int32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst_int32);
  init_dst(dst_int32, gm_size);

  int32_t *src_int32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src_int32);
  init_src_uint(src_int32, gm_size);

  // int64_t
  int64_t *dst_int64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst_int64);
  init_dst(dst_int64, gm_size);

  int64_t *src_int64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src_int64);
  init_src_uint(src_int64, gm_size);

  // __half
  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);

  __half *src_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src_f16);
  init_src_fp(src_f16, gm_size);

  // __fp32
  __fp32 *dst_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(dst_f32);
  init_dst(dst_f32, gm_size);

  __fp32 *src_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(src_f32);
  init_src_fp(src_f32, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<gm_row, gm_col, tile_row, tile_col, int8_t>(dst_int8, src_int8);
  test<gm_row, gm_col, tile_row, tile_col, int16_t>(dst_int16, src_int16);
  test<gm_row, gm_col, tile_row, tile_col, int32_t>(dst_int32, src_int32);
  test<gm_row, gm_col, tile_row, tile_col, int64_t>(dst_int64, src_int64);
  test<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src_f16);
  test<gm_row, gm_col, tile_row, tile_col, __fp32>(dst_f32, src_f32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_int8, gm_size);
  OutArray(dst_int16, gm_size);
  OutArray(dst_int32, gm_size);
  OutArray(dst_int64, gm_size);
  OutArray(dst_f16, gm_size);
  OutArray(dst_f32, gm_size);
  

  free(dst_int8);
  free(src_int8);
  free(dst_int16);
  free(src_int16);
  free(dst_int32);
  free(src_int32);
  free(dst_int64);
  free(src_int64);
  free(dst_f16);
  free(src_f16);
  free(dst_f32);
  free(src_f32);
  return 0;
#endif
}
