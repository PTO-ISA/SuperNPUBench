#include <common/pto_tileop.hpp>
#include "../data.hpp"
#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef __linx
int main();

extern "C" void *memcpy(void *dst, const void *src, size_t n) {
  auto *d = static_cast<unsigned char *>(dst);
  const auto *s = static_cast<const unsigned char *>(src);
  for (size_t i = 0; i < n; ++i) {
    d[i] = s[i];
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
    __asm__ volatile("" ::: "memory");
  }
}

extern "C" __attribute__((noreturn, section(".text._start"))) void _start(void) {
  linx_supernpu_exit(static_cast<uint32_t>(main()));
}

template <uint16_t M, uint16_t N, uint16_t K>
void test(int64_t *dst, int64_t *src0, int64_t *src1) {
  using gm_shape_A = global_tensor<int64_t, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<int64_t, RowMajor<K, N>>;
  using gm_shape_C = global_tensor<int64_t, RowMajor<M, N>>;

  using tile_shape_A = Tile<Location::Vec, int64_t, M, K, BLayout::RowMajor>;
  using tile_shape_B = Tile<Location::Vec, int64_t, K, N, BLayout::RowMajor>;
  using tile_shape_C = Tile<Location::Vec, int64_t, M, N, BLayout::RowMajor>;

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
#else
template <typename TA, typename TB>
void __vec__ test_cvt(typename TA::TileDType __out__ a,
                      typename TB::TileDType __in__ b) {
  using AType = typename TA::DType;
  using BType = typename TB::DType;
  BType *pb = blkv_get_tile_ptr(b);
  AType *pa = blkv_get_tile_ptr(a);
  int x = blkv_get_index_x();
  int y = blkv_get_index_y();
  int idx = index<TA>(y, x);
  AType o = (AType)(pb[idx]);
  pa[idx] = o;
}

template <uint16_t M, uint16_t N, uint16_t K>
void test(float *dst, float *src0, float *src1) {
  using gm_shape_A = global_tensor<float, RowMajor<M, K>>;
  using gm_shape_B = global_tensor<float, ColMajor<K, N>>;
  using gm_shape_C = global_tensor<float, RowMajor<M, N>>;

  using tile_shape_A = TileLeft<float, M, K>;
  using tile_shape_B = TileRight<float, K, N>;
  using tile_shape_C = TileAcc<float, M, N>;
  using tile_shape_LA = TileLeft<__fp8_e4m3, M, K>;
  using tile_shape_LB = TileRight<__fp8_e4m3, K, N>;

  gm_shape_A s0(src0);
  gm_shape_B s1(src1);
  gm_shape_C res(dst);

  tile_shape_A d0;
  tile_shape_B d1;
  tile_shape_C d2;
  tile_shape_LA lda;
  tile_shape_LB ldb;

  TLOAD(d0, s0);
  TLOAD(d1, s1);
  test_cvt<tile_shape_LA, tile_shape_A><<<M, K, 1>>>(lda.data(), d0.data());
  test_cvt<tile_shape_LB, tile_shape_B><<<K, N, 1>>>(ldb.data(), d1.data());
  MATMUL(d2, lda, ldb);
  TSTORE(res, d2);
}
#endif

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

  init_dst(dst_i64, size_C);
  init_src_int(src0_i64, size_A);
  init_src_int(src1_i64, size_B);

  test<M, N, K>(dst_i64, src0_i64, src1_i64);

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
  const uint16_t M = 64;
  const uint16_t K = 32;
  const uint16_t N = 64;

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
