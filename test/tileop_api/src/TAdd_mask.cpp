#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

using namespace pto;

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col>
void test(float *c_ptr, float *a_ptr, float *b_ptr) {
  using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;
  using glb_iterator = global_iterator<gm_shape, tile_shape>;

  static constexpr int block_row = gm_row / tile_row;
  static constexpr int block_col = gm_col / tile_col;
  static constexpr int remainder_row = gm_row % tile_row;
  static constexpr int remainder_col = gm_col % tile_col;

  using trailing_rows_shape =
      Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor, tile_row, remainder_col>;
  using trailing_cols_shape =
      Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor, remainder_row, tile_col>;
  using trailing_corner_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor,
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
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
    if constexpr (remainder_col) {
      auto gA = gAIter(i, block_col);
      auto gB = gBIter(i, block_col);
      auto gC = gCIter(i, block_col);

      trailing_rows_shape tA, tB, tC;
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
  }
  if constexpr (remainder_row) {
    for (int j = 0; j < block_col; ++j) {
      auto gA = gAIter(block_row, j);
      auto gB = gBIter(block_row, j);
      auto gC = gCIter(block_row, j);

      trailing_cols_shape tA, tB, tC;
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
    if constexpr (remainder_col) {
      auto gA = gAIter(block_row, block_col);
      auto gB = gBIter(block_row, block_col);
      auto gC = gCIter(block_row, block_col);

      trailing_corner_shape tA, tB, tC;
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
  }
}

int main() {
  const uint16_t gm_row = 66;
  const uint16_t gm_col = 66;
  const uint16_t tile_row = 16;
  const uint16_t tile_col = 16;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

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
}
