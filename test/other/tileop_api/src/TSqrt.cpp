#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef __linx
int main();

extern "C" void *memcpy(void *dst, const void *src, size_t n) {
  volatile uint8_t *d = static_cast<volatile uint8_t *>(dst);
  const volatile uint8_t *s = static_cast<const volatile uint8_t *>(src);
  for (size_t i = 0; i < n; ++i) {
    d[i] = s[i];
  }
  return dst;
}

extern "C" void *memset(void *dst, int value, size_t n) {
  volatile uint8_t *d = static_cast<volatile uint8_t *>(dst);
  const uint8_t byte = static_cast<uint8_t>(value);
  for (size_t i = 0; i < n; ++i) {
    d[i] = byte;
  }
  return dst;
}

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
          uint64_t tile_col, typename T>
void test_rm(T *dst, T *src) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col>;
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  glb_iterator gSIter(src);
  glb_iterator gDIter(dst);

  size_t block_row = gm_row / tile_row;
  size_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      auto s0 = gSIter(i, j);
      auto res = gDIter(i, j);

      tile_shape t0, t1;
      TCOPYIN(t0, s0);
      TSQRT(t1, t0);
      TCOPYOUT(res, t1);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test_cm(T *dst, T *src) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  glb_iterator gSIter(src);
  glb_iterator gDIter(dst);

  size_t block_row = gm_row / tile_row;
  size_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_col; ++i) {
    for (int j = 0; j < block_row; ++j) {
      auto s0 = gSIter(j, i);
      auto res = gDIter(j, i);

      tile_shape t0, t1;
      TCOPYIN(t0, s0);
      TSQRT(t1, t0);
      TCOPYOUT(res, t1);
    }
  }
}

int main() {

#ifdef __linx
  constexpr size_t gm_row = 4;
  constexpr size_t gm_col = 4;
  constexpr size_t tile_row = 4;
  constexpr size_t tile_col = 4;
#else
  const size_t gm_row = 32;
  const size_t gm_col = 32;
  const size_t tile_row = 16;
  const size_t tile_col = 16;
#endif

  constexpr size_t gm_size = gm_row * gm_col;
  constexpr size_t tile_size = tile_row * tile_col;
  (void)gm_size;
  (void)tile_size;

#ifdef __linx
  using row_tile = Tile<Location::Vec, int64_t, tile_row, tile_col>;
  using col_tile =
      Tile<Location::Vec, int64_t, tile_row, tile_col, BLayout::ColMajor>;
  row_tile src_rm, dst_rm;
  col_tile src_cm, dst_cm;

  for (size_t i = 0; i < tile_row; ++i) {
    for (size_t j = 0; j < tile_col; ++j) {
      int64_t expected = static_cast<int64_t>(i * tile_col + j);
      size_t row_index = index<row_tile>(i, j);
      size_t col_index = index<col_tile>(i, j);
      src_rm.data()[row_index] = expected * expected;
      src_cm.data()[col_index] = expected * expected;
      dst_rm.data()[row_index] = 0;
      dst_cm.data()[col_index] = 0;
    }
  }

  TSQRT(dst_rm, src_rm);
  TSQRT(dst_cm, src_cm);

  for (size_t i = 0; i < tile_row; ++i) {
    for (size_t j = 0; j < tile_col; ++j) {
      int64_t expected = static_cast<int64_t>(i * tile_col + j);
      if (dst_rm.data()[index<row_tile>(i, j)] != expected) {
        return 1;
      }
      if (dst_cm.data()[index<col_tile>(i, j)] != expected) {
        return 2;
      }
    }
  }

  return 0;
#else
  // __half
  __half *dst_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst(dst_f16, gm_size);

  __half *src_f16 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src_f16);
  init_rows_fp(src_f16, gm_row, gm_col);

  // __fp32
  __fp32 *dst_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(dst_f32);
  init_dst(dst_f32, gm_size);

  __fp32 *src_f32 = (__fp32 *)malloc(gm_size * sizeof(__fp32));
  check_mem_alloc(src_f32);
  init_rows_fp(src_f32, gm_row, gm_col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src_f16);
  test_cm<gm_row, gm_col, tile_row, tile_col, __half>(dst_f16, src_f16);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst_f16, gm_size);
  OutArray(dst_f32, gm_size);

  free(dst_f16);
  free(src_f16);
  free(dst_f32);
  free(src_f32);

  return 0;
#endif
}
