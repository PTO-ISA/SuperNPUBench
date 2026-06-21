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

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_RowMajor(T *dst, T *src0) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_row; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape res(dst + offset);
  
      tile_shape d0, d1;
      TCOPYIN(d0, s0);
      TABS(d1, d0);
      TCOPYOUT(res, d1);
    }
  }
}
 
template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_ColMajor(T *dst, T *src0) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_col; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_row; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src0 + offset);
      gm_shape res(dst + offset);
  
      tile_shape d0, d1;
      TCOPYIN(d0, s0);
      TABS(d1, d0);
      TCOPYOUT(res, d1);
    }
  }
}

int main() {
#ifdef __linx
  constexpr uint16_t gm_row = 4;
  constexpr uint16_t gm_col = 4;
  constexpr uint16_t tile_row = 4;
  constexpr uint16_t tile_col = 4;
#else
  constexpr uint16_t gm_row = 64;
  constexpr uint16_t gm_col = 64;
  constexpr uint16_t tile_row = 16;
  constexpr uint16_t tile_col = 16;
#endif

  constexpr size_t gm_size = gm_row * gm_col;
  constexpr size_t tile_size = tile_row * tile_col;
  (void)tile_size;

#ifdef __linx
  static int64_t dst[gm_size];
  static int64_t src[gm_size];
  init_dst(dst, gm_size);
  init_src_int(src, gm_size);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int64_t>(dst, src);

  return 0;
#else
  float *dst_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst_col);
  init_dst(dst_col, gm_size);

  float *src0_col = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0_col);
  init_src_fp(src0_col, gm_size);

  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);
 
  __half *src0_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src0_f16);
  init_src_fp(src0_f16, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_ColMajor<gm_row, gm_col, tile_row, tile_col, float>(dst_col, src0_col);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src0_f16);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_col, gm_size);
  OutArray(dst_f16, gm_size);

  free(dst_col);
  free(src0_col);
 
  free(dst_f16);
  free(src0_f16);

  return 0;
#endif
}
