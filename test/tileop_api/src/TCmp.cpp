#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T, CmpMode Mode>
void test_RowMajor_CmpMode(int32_t *dst, T *src0, T *src1) {
  using gm_shape_in = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using gm_shape_out = global_tensor<int32_t, RowMajor<gm_row, gm_col>>;
  using tile_shape_in = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::RowMajor>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_row; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape_in s0(src0 + offset);
      gm_shape_in s1(src1 + offset);
      gm_shape_out res(dst + offset);
  
      tile_shape_in d0, d1;
      tile_shape_out d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TCMP(d2, d1, d0, Mode);  // 使用模板参数Mode
      TCOPYOUT(res, d2);
    }
  }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T, CmpMode Mode>
void test_ColMajor_CmpMode(int32_t *dst, T *src0, T *src1) {
  using gm_shape_in = global_tensor<T, ColMajor<gm_row, gm_col>>;
  using gm_shape_out = global_tensor<int32_t, ColMajor<gm_row, gm_col>>;
  using tile_shape_in = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
  using tile_shape_out = Tile<Location::Vec, int32_t, tile_row, tile_col, BLayout::ColMajor>;
 
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  #pragma clang loop unroll(full)
  for (int i = 0; i < block_col; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < block_row; ++j) {
      int offset = i * (tile_col * gm_row) + j * tile_row;
      gm_shape_in s0(src0 + offset);
      gm_shape_in s1(src1 + offset);
      gm_shape_out res(dst + offset);
  
      tile_shape_in d0, d1;
      tile_shape_out d2;
      TCOPYIN(d0, s0);
      TCOPYIN(d1, s1);
      TCMP(d2, d1, d0, Mode);  // 使用模板参数Mode
      TCOPYOUT(res, d2);
    }
  }
}

// 拆分为单个比较模式的测试函数
template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T, CmpMode Mode>
void test_SingleCmpMode_RowMajor() {
  size_t gm_size = gm_row * gm_col;
  
  int32_t *dst = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);
  T *src0 = (T *)malloc(gm_size * sizeof(T));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);
  T *src1 = (T *)malloc(gm_size * sizeof(T));
  check_mem_alloc(src1);
  init_src_fp(src1, gm_size);
#ifdef LINX_PMC
  PMC_START();
#endif

  test_RowMajor_CmpMode<gm_row, gm_col, tile_row, tile_col, T, Mode>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  OutArray(dst, gm_size);
  
  free(dst);
  free(src0);
  free(src1);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T, CmpMode Mode>
void test_SingleCmpMode_ColMajor() {
  size_t gm_size = gm_row * gm_col;
  
  int32_t *dst = (int32_t *)malloc(gm_size * sizeof(int32_t));
  check_mem_alloc(dst);
  init_dst(dst, gm_size);
  T *src0 = (T *)malloc(gm_size * sizeof(T));
  check_mem_alloc(src0);
  init_src_fp(src0, gm_size);
  T *src1 = (T *)malloc(gm_size * sizeof(T));
  check_mem_alloc(src1);
  init_src_fp(src1, gm_size);
#ifdef LINX_PMC
  PMC_START();
#endif

  test_ColMajor_CmpMode<gm_row, gm_col, tile_row, tile_col, T, Mode>(dst, src0, src1);

#ifdef LINX_PMC
  PMC_END();
#endif

  OutArray(dst, gm_size);
  
  free(dst);
  free(src0);
  free(src1);
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_AllCmpModes_RowMajor() {
  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::GE>();
  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::GT>();
  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::LE>();
  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::LT>();
  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::NE>();
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_AllCmpModes_ColMajor() {
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::GE>();
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::GT>();
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::LE>();
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::LT>();
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, T, CmpMode::NE>();
}

int main() {
  const uint16_t gm_row = 64;
  const uint16_t gm_col = 32;
  const uint16_t tile_row = 64;
  const uint16_t tile_col = 32;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

  printf("Result:\n");
  //  测试float类型的所有比较模式
  test_AllCmpModes_RowMajor<gm_row, gm_col, tile_row, tile_col, float>();
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, float>();

  // 测试__half类型的所有比较模式
  test_AllCmpModes_RowMajor<gm_row, gm_col, tile_row, tile_col, __half>();
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, __half>();

  // 测试int32_t类型的所有比较模式
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, int32_t>();
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, int32_t>();

  // 测试int64_t类型的所有比较模式
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, int64_t>();
  test_AllCmpModes_ColMajor<gm_row, gm_col, tile_row, tile_col, int64_t>();

  test_SingleCmpMode_RowMajor<gm_row, gm_col, tile_row, tile_col, int32_t, CmpMode::EQ>();
  test_SingleCmpMode_ColMajor<gm_row, gm_col, tile_row, tile_col, int32_t, CmpMode::EQ>();
  return 0;
}