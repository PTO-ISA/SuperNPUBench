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

using namespace pto;

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test(T *c_ptr, T *a_ptr, T *b_ptr) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  static constexpr int block_row = gm_row / tile_row;
  static constexpr int block_col = gm_col / tile_col;
  static constexpr int remainder_row = gm_row % tile_row;
  static constexpr int remainder_col = gm_col % tile_col;

  using trailing_rows_shape =
      Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor, tile_row, remainder_col>;
  using trailing_cols_shape =
      Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor, remainder_row, tile_col>;
  using trailing_corner_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor,
                                            remainder_row, remainder_col>;

  glb_iterator gAIter(a_ptr);
  glb_iterator gBIter(b_ptr);
  glb_iterator gCIter(c_ptr);
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      auto gA = gAIter(i, j);
      auto gB = gBIter(i, j);
      auto gC = gCIter(i, j);

      tile_shape tA, tB, tC;
      TLOAD(tA, gA);
      TLOAD(tB, gB);
      TADD(tC, tA, tB);
      TSTORE(gC, tC);
    }
    if constexpr (remainder_col) {
      auto gA = gAIter(i, block_col);
      auto gB = gBIter(i, block_col);
      auto gC = gCIter(i, block_col);

      trailing_rows_shape tA, tB, tC;
      TLOAD(tA, gA);
      TLOAD(tB, gB);
      TADD(tC, tA, tB);
      TSTORE(gC, tC);
    }
  }
  if constexpr (remainder_row) {
    for (int j = 0; j < block_col; ++j) {
      auto gA = gAIter(block_row, j);
      auto gB = gBIter(block_row, j);
      auto gC = gCIter(block_row, j);

      trailing_cols_shape tA, tB, tC;
      TLOAD(tA, gA);
      TLOAD(tB, gB);
      TADD(tC, tA, tB);
      TSTORE(gC, tC);
    }
    if constexpr (remainder_col) {
      auto gA = gAIter(block_row, block_col);
      auto gB = gBIter(block_row, block_col);
      auto gC = gCIter(block_row, block_col);

      trailing_corner_shape tA, tB, tC;
      TLOAD(tA, gA);
      TLOAD(tB, gB);
      TADD(tC, tA, tB);
      TSTORE(gC, tC);
    }
  }
}

int main() {
#ifdef __linx
  constexpr uint16_t gm_row = 6;
  constexpr uint16_t gm_col = 6;
  constexpr uint16_t tile_row = 4;
  constexpr uint16_t tile_col = 4;
#else
  const uint16_t gm_row = 66;
  const uint16_t gm_col = 66;
  const uint16_t tile_row = 16;
  const uint16_t tile_col = 16;
#endif

  constexpr size_t gm_size = gm_row * gm_col;
  constexpr size_t tile_size = tile_row * tile_col;
  (void)tile_size;

#ifdef __linx
  static int64_t dst[gm_size];
  static int64_t src0[gm_size];
  static int64_t src1[gm_size];
  init_dst(dst, gm_size);
  init_src_int(src0, gm_size);
  init_src_uint(src1, gm_size);

  test<gm_row, gm_col, tile_row, tile_col, int64_t>(dst, src0, src1);
  return 0;
#else
  float *dst = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);

  float *src0 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);
  float *src1 = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, gm_size);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, gm_size);

  free(dst);
  free(src0);
  free(src1);

  return 0;
#endif
}
