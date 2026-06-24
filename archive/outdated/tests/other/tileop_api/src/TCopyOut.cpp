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
  using shape = Shape<1, 1, 1, tile_row, tile_col>;
  using stride = Stride<1, 1, gm_row * gm_col, gm_col, 1>;
  using gm_shape = GlobalTensor<T, shape, stride, Layout::ND>;
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

      tile_shape d0;
      TCOPYIN(d0, s0);
      TCOPYOUT(res, d0);
    }
  }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void test_RowMajor_Dynamic(T *dst, T *src0) {
  using gm_shape = global_tensor<T, RowMajor<-1, -1>>;
  using tile_shape = Tile<Location::Vec, T, 2*tile_row, 2*tile_col, BLayout::RowMajor, -1, -1>;

  volatile size_t tile_valid_row = tile_row - 2;
  volatile size_t tile_valid_col = tile_col - 2;

  volatile size_t gm_valid_row = gm_row;
  volatile size_t gm_valid_col = gm_col;

  uint16_t block_row = (gm_row + tile_valid_row - 1) / tile_valid_row;
  uint16_t block_col = (gm_col + tile_valid_col - 1) / tile_valid_col;

  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      uint16_t remainder_row = gm_row - i * tile_valid_row;
      uint16_t remainder_col = gm_col - j * tile_valid_col;

      uint16_t active_row = remainder_row < tile_valid_row ? remainder_row : tile_valid_row;
      uint16_t active_col = remainder_col < tile_valid_col ? remainder_col : tile_valid_col;

      int offset = i * (tile_valid_row * gm_valid_col) + j * tile_valid_col;
      gm_shape s0(src0 + offset, gm_valid_row, gm_valid_col);
      gm_shape res(dst + offset, gm_valid_row, gm_valid_col);

      tile_shape d0(active_row, active_col);
      TCOPYIN(d0, s0);
      TCOPYOUT(res, d0);
    }
  }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_ColMajor(T *dst, T *src0) {
  using shape = Shape<1, 1, 1, tile_row, tile_col>;
  using stride = Stride<1, 1, gm_row * gm_col, 1, gm_row>;
  using gm_shape = GlobalTensor<T, shape, stride, Layout::DN>;
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

      tile_shape d0;
      TCOPYIN(d0, s0);
      TCOPYOUT(res, d0);
    }
  }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void test_Nz_Dynamic(T *dst, T *src0) {
  using gm_shape = global_tensor<T, RowMajor<-1, -1>>;
  using tile_shape = TileLeft<T, tile_row, tile_col, -1, -1>;

  volatile size_t tile_valid_row = tile_row - 2;
  volatile size_t tile_valid_col = tile_col - 2;

  volatile size_t gm_valid_row = gm_row;
  volatile size_t gm_valid_col = gm_col;

  uint16_t block_row = (gm_row + tile_valid_row - 1) / tile_valid_row;
  uint16_t block_col = (gm_col + tile_valid_col - 1) / tile_valid_col;

  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      uint16_t remainder_row = gm_row - i * tile_valid_row;
      uint16_t remainder_col = gm_col - j * tile_valid_col;

      uint16_t active_row = remainder_row < tile_valid_row ? remainder_row : tile_valid_row;
      uint16_t active_col = remainder_col < tile_valid_col ? remainder_col : tile_valid_col;

      int offset = i * (tile_valid_row * gm_valid_col) + j * tile_valid_col;
      gm_shape s0(src0 + offset, gm_valid_row, gm_valid_col);
      gm_shape res(dst + offset, gm_valid_row, gm_valid_col);

      tile_shape d0(active_row, active_col);
      tile_shape d1(active_row, active_col);
      TCOPYIN(d0, s0);
      TCOPYOUT(res, d0);
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
  constexpr uint16_t tile_row = 32;
  constexpr uint16_t tile_col = 32;
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
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  float *src0 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);

  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);

  __half *src0_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src0_f16);
  init_src_fp(src0_f16, gm_size);

  int8_t *dst_i8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(dst_i8);
  init_dst(dst_i8, gm_size);

  int8_t *src0_i8 = (int8_t *)malloc(gm_size * sizeof(int8_t));
  check_mem_alloc(src0_i8);
  init_src_int(src0_i8, gm_size);

  int16_t *dst_i16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(dst_i16);
  init_dst(dst_i16, gm_size);

  int16_t *src0_i16 = (int16_t *)malloc(gm_size * sizeof(int16_t));
  check_mem_alloc(src0_i16);
  init_src_int(src0_i16, gm_size);

  int32_t *dst_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst_i32);
  init_dst(dst_i32, gm_size);

  int32_t *src0_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src0_i32);
  init_src_int(src0_i32, gm_size);

  int64_t *dst_i64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(dst_i64);
  init_dst(dst_i64, gm_size);

  int64_t *src0_i64 = (int64_t *)malloc(gm_size * sizeof(int64_t));
  check_mem_alloc(src0_i64);
  init_src_int(src0_i64, gm_size);

  int32_t *dst1_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst1_i32);
  init_dst(dst1_i32, gm_size);

  int32_t *src1_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src1_i32);
  init_src_int(src1_i32, gm_size);

  int32_t *dst_nz_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst_nz_i32);
  init_dst(dst_nz_i32, gm_size);

  int32_t *src_nz_i32 = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(src_nz_i32);
  init_src_int(src_nz_i32, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, float>(dst, src0);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src0_f16);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int8_t>(dst_i8, src0_i8);

  test_RowMajor<gm_row, gm_col, tile_row, tile_col, int16_t>(dst_i16, src0_i16);

  test_ColMajor<gm_row, gm_col, tile_row, tile_col, int32_t>(dst_i32, src0_i32);

  test_ColMajor<gm_row, gm_col, tile_row, tile_col, int64_t>(dst_i64, src0_i64);

  test_RowMajor_Dynamic<gm_row + 1, gm_col + 1, tile_row, tile_col, int32_t>(dst1_i32, src1_i32);

  test_Nz_Dynamic<gm_row + 1, gm_col + 1, tile_row, tile_col, int32_t>(dst_nz_i32, src_nz_i32);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst_f16, gm_size);
  OutArray(dst_i8, gm_size);
  OutArray(dst_i16, gm_size);
  OutArray(dst_i32, gm_size);
  OutArray(dst_i64, gm_size);
  OutArray(dst1_i32, gm_size);
  OutArray(dst_nz_i32, gm_size);

  free(dst);
  free(src0);

  free(dst_f16);
  free(src0_f16);

  free(dst_i8);
  free(src0_i8);

  free(dst_i16);
  free(src0_i16);

  free(dst_i32);
  free(src0_i32);

  free(dst_i64);
  free(src0_i64);

  free(dst1_i32);
  free(src1_i32);

  return 0;
#endif
}
