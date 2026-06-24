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

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TLOAD(d0, s0);
      TEXP(d1, d0);
      TSTORE(res, d1);
    }
  }
}
template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row,
          uint64_t tile_col, typename T>
void test_cm(T *dst, T *src) {
  using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      gm_shape res(dst + offset);

      tile_shape d0, d1;
      TLOAD(d0, s0);
      TEXP(d1, d0);
      TSTORE(res, d1);
    }
  }
}

int main() {
#ifdef __linx
  constexpr size_t tile_row = 4;
  constexpr size_t tile_col = 4;
  using row_tile = Tile<Location::Vec, int64_t, tile_row, tile_col>;
  using col_tile =
      Tile<Location::Vec, int64_t, tile_row, tile_col, BLayout::ColMajor>;

  row_tile src_rm, dst_rm;
  col_tile src_cm, dst_cm;

  for (size_t i = 0; i < tile_row; ++i) {
    for (size_t j = 0; j < tile_col; ++j) {
      int64_t value = static_cast<int64_t>((i + j) % 6);
      size_t row_index = index<row_tile>(i, j);
      size_t col_index = index<col_tile>(i, j);
      src_rm.data()[row_index] = value;
      src_cm.data()[col_index] = value;
      dst_rm.data()[row_index] = 0;
      dst_cm.data()[col_index] = 0;
    }
  }

  TEXP(dst_rm, src_rm);
  TEXP(dst_cm, src_cm);

  for (size_t i = 0; i < tile_row; ++i) {
    for (size_t j = 0; j < tile_col; ++j) {
      int64_t value = static_cast<int64_t>((i + j) % 6);
      int64_t expected = linx_tile_iexp(value);
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
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 64;
  const uint16_t tile_row = 16;
  const uint16_t tile_col = 16;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;
  // float32
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);
  float *src = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, gm_size);
  // float16
  __half *dst2 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(dst2);
  init_dst(dst2, gm_size);
  __half *src2 = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(src2);
  init_src_fp(src2, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  // TExp只支持float32和16
  test_rm<gm_row, gm_col, tile_row, tile_col,float>(dst, src);
  // half编译通过，运行出错
  // test_rm<gm_row, gm_col, tile_row, tile_col, __half>(dst2,src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);
  OutArray(dst2, gm_size);


  free(dst);
  free(src);
  free(dst2);
  free(src2);

  return 0;
#endif
}
