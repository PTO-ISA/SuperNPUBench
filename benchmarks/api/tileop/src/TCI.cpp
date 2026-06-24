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

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col,typename T>
void test_rm(T *dst, T s) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor, 1, tile_col>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape res(dst + offset);

      tile_shape d1;
      TCI<tile_shape, T, 0>(d1, s);
      TSTORE(res, d1);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col,typename T>
void test_cm(T *dst, T s) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor, tile_row, 1>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape res(dst + offset);

      tile_shape d1;
      TCI<tile_shape, T, 0>(d1, s);
      TSTORE(res, d1);
    }
  }
}

int main() {
#ifdef __linx
  constexpr uint16_t gm_row = 8;
  constexpr uint16_t gm_col = 8;
  constexpr uint16_t tile_row = 8;
  constexpr uint16_t tile_col = 8;
  constexpr uint16_t gm_size = gm_row * gm_col;

  static int32_t dst_rm[gm_size];
  static int32_t dst_cm[gm_size];
  init_dst(dst_rm, gm_size);
  init_dst(dst_cm, gm_size);

  test_rm<gm_row, gm_col, tile_row, tile_col, int32_t>(dst_rm, s_i32);
  test_cm<gm_row, gm_col, tile_row, tile_col, int32_t>(dst_cm, s_i32);
  return 0;
#else
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 32;
  const uint16_t tile_row = 64;
  const uint16_t tile_col = 32;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;
  // float32
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  // float16
  __half *dst1 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst1);
  init_dst(dst1, gm_size);

  // int8
  int8_t *dst2 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst2);
  init_dst(dst2, gm_size);

  // int16
  int16_t *dst3 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst3);
  init_dst(dst3, gm_size);
  int16_t *dst3_col = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst3_col);
  init_dst(dst3_col, gm_size);

  // int32
  int32_t *dst4 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst4);
  init_dst(dst4, gm_size);
  int32_t *dst4_col = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst4_col);
  init_dst(dst4_col, gm_size);

  // int64
  int64_t *dst5 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst5);
  init_dst(dst5, gm_size);


#ifdef LINX_PMC
  PMC_START();
#endif

  // test_rm<gm_row, gm_col, tile_row, tile_col, float>(dst, s_fp32);
  // test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst1, s_fp16);
  // test_rm<gm_row, gm_col, tile_row, tile_col, int8_t>(dst2, s_i8);
  test_rm<gm_row, gm_col, tile_row, tile_col, int16_t>(dst3, s_i16);
  test_cm<gm_row, gm_col, tile_row, tile_col, int16_t>(dst3_col, s_i16);
  test_rm<gm_row, gm_col, tile_row, tile_col, int32_t>(dst4, s_i32);
  test_cm<gm_row, gm_col, tile_row, tile_col, int32_t>(dst4_col, s_i32);
  // test_rm<gm_row, gm_col, tile_row, tile_col, int64_t>(dst5, s_i64);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  // OutArray(dst, gm_size);
  // OutArray(dst1, gm_size);
  // OutArray(dst2, gm_size);
  OutArray(dst3, gm_size);
  OutArray(dst3_col, gm_size);
  OutArray(dst4, gm_size);
  OutArray(dst4_col, gm_size);
  // OutArray(dst5, gm_size);

  free(dst);
  free(dst1);
  free(dst2);
  free(dst3);
  free(dst3_col);
  free(dst4);
  free(dst4_col);
  free(dst5);

  return 0;
#endif
}
