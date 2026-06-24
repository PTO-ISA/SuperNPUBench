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

template <uint16_t M, uint16_t N, uint16_t K, typename T>
void test_RowMajor(T *dst, T *src0, T *src1) {
  using gm_shape_A = global_tensor<T, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<T, RowMajor<K, N>>;
  using gm_shape_C = global_tensor<T, RowMajor<M, N>>;

  using tile_shape_A = Tile<Location::Vec, T, M, K, BLayout::RowMajor>;
  using tile_shape_B = Tile<Location::Vec, T, K, N, BLayout::RowMajor>;;
  using tile_shape_C = Tile<Location::Vec, T, M, N, BLayout::RowMajor>;;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TCOPYIN(d2, res);
  MATMACC(d2, d0, d1);
  TCOPYOUT(res, d2);
}

template <uint16_t M, uint16_t N, uint16_t K, typename T>
void test_ColMajor(T *dst, T *src0, T *src1) {
  using gm_shape_A = global_tensor<T, ColMajor<M, K>>;
  using gm_shape_B = global_tensor<T, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<T, ColMajor<M, N>>;

  using tile_shape_A = Tile<Location::Vec, T, M, K, BLayout::ColMajor>;
  using tile_shape_B = Tile<Location::Vec, T, K, N, BLayout::ColMajor>;
  using tile_shape_C = Tile<Location::Vec, T, M, N, BLayout::ColMajor>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;

  TCOPYIN(d0, s0);
  TCOPYIN(d1, s1);
  TCOPYIN(d2, res);
  MATMACC(d2, d0, d1);
  TCOPYOUT(res, d2);
}

int main() {
#ifdef __linx
  constexpr uint16_t M = 4;
  constexpr uint16_t K = 4;
  constexpr uint16_t N = 4;
#else
  const uint16_t M = 16;
  const uint16_t K = 8;
  const uint16_t N = 32;
#endif

  constexpr size_t size_A = M * K;
  constexpr size_t size_B = K * N;
  constexpr size_t size_C = M * N;

#ifdef __linx
  static int64_t dst_rm[size_C];
  static int64_t src0_rm[size_A];
  static int64_t src1_rm[size_B];
  static int64_t base_rm[size_C];

  for (size_t row = 0; row < M; ++row) {
    for (size_t k = 0; k < K; ++k) {
      const int64_t value = static_cast<int64_t>((row + 1) * (k + 2));
      src0_rm[row * K + k] = value;
    }
  }
  for (size_t k = 0; k < K; ++k) {
    for (size_t col = 0; col < N; ++col) {
      const int64_t value = static_cast<int64_t>((k + 1) + (col + 1));
      src1_rm[k * N + col] = value;
    }
  }
  for (size_t row = 0; row < M; ++row) {
    for (size_t col = 0; col < N; ++col) {
      const int64_t value = static_cast<int64_t>(10 + row * N + col);
      dst_rm[row * N + col] = value;
      base_rm[row * N + col] = value;
    }
  }

  test_RowMajor<M, N, K, int64_t>(dst_rm, src0_rm, src1_rm);

  for (size_t row = 0; row < M; ++row) {
    for (size_t col = 0; col < N; ++col) {
      int64_t expected = base_rm[row * N + col];
      for (size_t k = 0; k < K; ++k) {
        expected += src0_rm[row * K + k] * src1_rm[k * N + col];
      }
      if (dst_rm[row * N + col] != expected) {
        return 1;
      }
    }
  }

  return 0;
#else

  float *dst = (float *)malloc(size_C * sizeof(float));
  check_mem_alloc(dst);
  init_dst_no_zero(dst, size_C);

  float *src0 = (float *)malloc(size_A * sizeof(float));
  check_mem_alloc(src0);
  init_src_fp(src0, size_A);
  float *src1 = (float *)malloc(size_B * sizeof(float));
  check_mem_alloc(src1);
  init_src_fp(src1, size_B);

  __half *dst_f16 = (__half *)malloc(size_C * sizeof(__half));
  check_mem_alloc(dst_f16);
  init_dst_no_zero(dst_f16, size_C);

  __half *src0_f16 = (__half *)malloc(size_A * sizeof(__half));
  check_mem_alloc(src0_f16);
  init_src_fp(src0_f16, size_A);
  __half *src1_f16 = (__half *)malloc(size_B * sizeof(__half));
  check_mem_alloc(src1_f16);
  init_src_fp(src1_f16, size_B);

  int8_t *dst_i8 = (int8_t *)malloc(size_C * sizeof(int8_t));
  check_mem_alloc(dst_i8);
  init_dst_no_zero(dst_i8, size_C);

  int8_t *src0_i8 = (int8_t *)malloc(size_A * sizeof(int8_t));
  check_mem_alloc(src0_i8);
  init_src_int(src0_i8, size_A);
  int8_t *src1_i8 = (int8_t *)malloc(size_B * sizeof(int8_t));
  check_mem_alloc(src1_i8);
  init_src_int(src1_i8, size_B);

  int16_t *dst_i16 = (int16_t *)malloc(size_C * sizeof(int16_t));
  check_mem_alloc(dst_i16);
  init_dst_no_zero(dst_i16, size_C);

  int16_t *src0_i16 = (int16_t *)malloc(size_A * sizeof(int16_t));
  check_mem_alloc(src0_i16);
  init_src_int(src0_i16, size_A);
  int16_t *src1_i16 = (int16_t *)malloc(size_B * sizeof(int16_t));
  check_mem_alloc(src1_i16);
  init_src_int(src1_i16, size_B);

  int32_t *dst_i32 = (int32_t *)malloc(size_C * sizeof(int32_t));
  check_mem_alloc(dst_i32);
  init_dst_no_zero(dst_i32, size_C);

  int32_t *src0_i32 = (int32_t *)malloc(size_A * sizeof(int32_t));
  check_mem_alloc(src0_i32);
  init_src_int(src0_i32, size_A);
  int32_t *src1_i32 = (int32_t *)malloc(size_B * sizeof(int32_t));
  check_mem_alloc(src1_i32);
  init_src_int(src1_i32, size_B);

  int64_t *dst_i64 = (int64_t *)malloc(size_C * sizeof(int64_t));
  check_mem_alloc(dst_i64);
  init_dst_no_zero(dst_i64, size_C);

  int64_t *src0_i64 = (int64_t *)malloc(size_A * sizeof(int64_t));
  check_mem_alloc(src0_i64);
  init_src_int(src0_i64, size_A);
  int64_t *src1_i64 = (int64_t *)malloc(size_B * sizeof(int64_t));
  check_mem_alloc(src1_i64);
  init_src_int(src1_i64, size_B);

#ifdef LINX_PMC
  PMC_START();
#endif

  //test_RowMajor<M, N, K, float>(dst, src0, src1);

  //test_RowMajor<M, N, K, __half>(dst_f16, src0_f16, src1_f16);

  //test_RowMajor<M, N, K, int8_t>(dst_i8, src0_i8, src1_i8);

  //test_RowMajor<M, N, K, int16_t>(dst_i16, src0_i16, src1_i16);

  test_RowMajor<M, N, K, int32_t>(dst_i32, src0_i32, src1_i32);

  test_RowMajor<M, N, K, int64_t>(dst_i64, src0_i64, src1_i64);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_C);
  OutArray(dst_f16, size_C);
  OutArray(dst_i8, size_C);
  OutArray(dst_i16, size_C);
  OutArray(dst_i32, size_C);
  OutArray(dst_i64, size_C);

  free(dst);
  free(src0);
  free(src1);

  free(dst_f16);
  free(src0_f16);
  free(src1_f16);

  free(dst_i8);
  free(src0_i8);
  free(src1_i8);

  free(dst_i16);
  free(src0_i16);
  free(src1_i16);

  free(dst_i32);
  free(src0_i32);
  free(src1_i32);

  free(dst_i64);
  free(src0_i64);
  free(src1_i64);

  return 0;
#endif
}
