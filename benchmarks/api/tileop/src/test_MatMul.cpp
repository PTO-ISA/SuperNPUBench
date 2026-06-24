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

template <uint16_t M, uint16_t N, uint16_t K, typename T>
void test_linx_row_major(T *dst, T *src0, T *src1) {
  using gm_shape_A = global_tensor<T, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<T, RowMajor<K, N>>;
  using gm_shape_C = global_tensor<T, RowMajor<M, N>>;

  using tile_shape_A = Tile<Location::Vec, T, M, K, BLayout::RowMajor>;
  using tile_shape_B = Tile<Location::Vec, T, K, N, BLayout::RowMajor>;
  using tile_shape_C = Tile<Location::Vec, T, M, N, BLayout::RowMajor>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;

  TLOAD(d0, s0);
  TLOAD(d1, s1);
  MATMUL(d2, d0, d1);
  TSTORE(res, d2);
}
#endif

template <uint16_t M, uint16_t N, uint16_t K>
void test(float *dst, float *src0, float *src1) {
  using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A = TileLeft<float, M, K>;
  using tile_shape_B = TileRight<float, K, N>;
  using tile_shape_C = TileAcc<float, M, N>;
  using tile_shape_O = TileLeft<float, M, K>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;
  tile_shape_O d3;

  TLOAD(d0, s0);
  TLOAD(d1, s1);
  MATMUL(d2, d0, d1);
  TCVT(d3, d2);
  TSTORE(res, d3);
}

int main() {
#ifdef __linx
  constexpr uint16_t M = 4;
  constexpr uint16_t K = 4;
  constexpr uint16_t N = 4;
  constexpr size_t size_A = M * K;
  constexpr size_t size_B = K * N;
  constexpr size_t size_C = M * N;

  static int64_t dst_i64[size_C];
  static int64_t src0_i64[size_A];
  static int64_t src1_i64[size_B];

  for (size_t row = 0; row < M; ++row) {
    for (size_t k = 0; k < K; ++k) {
      src0_i64[row * K + k] = static_cast<int64_t>((row + 1) * (k + 2));
    }
  }
  for (size_t k = 0; k < K; ++k) {
    for (size_t col = 0; col < N; ++col) {
      src1_i64[k * N + col] = static_cast<int64_t>((k + 1) + (col + 1));
    }
  }
  for (size_t i = 0; i < size_C; ++i) {
    dst_i64[i] = 0;
  }

  test_linx_row_major<M, N, K, int64_t>(dst_i64, src0_i64, src1_i64);

  for (size_t row = 0; row < M; ++row) {
    for (size_t col = 0; col < N; ++col) {
      int64_t expected = 0;
      for (size_t k = 0; k < K; ++k) {
        expected += src0_i64[row * K + k] * src1_i64[k * N + col];
      }
      if (dst_i64[row * N + col] != expected) {
        return 1;
      }
    }
  }

  return 0;
#else
  const uint16_t M = 16;
  const uint16_t K = 8;
  const uint16_t N = 32;

  size_t size_A = M * K;
  size_t size_B = K * N;
  size_t size_C = M * N;

  float *dst = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_C);

  float *src0 = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_A);
  float *src1 = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_B);

#ifdef LINX_PMC
  PMC_START();
#endif

  test<M, N, K>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_C);

  free(dst);
  free(src0);
  free(src1);

  return 0;
#endif
}
