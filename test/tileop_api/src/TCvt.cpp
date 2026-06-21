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

template <uint16_t row, uint16_t col> void testRow2Nz(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_out = TileLeft<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testNz2Col(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
  using tile_shape_out = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testNz2Zn(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
  using tile_shape_out = TileRight<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testZn2Nz(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileRight<float, row, col>;
  using tile_shape_out = TileLeft<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

template <uint16_t row, uint16_t col> void testNz2Nz(float *dst, float *src) {
  using gm_shape = global_tensor<float, RowMajor<row, col>>;

  using tile_shape_in = TileLeft<float, row, col>;
  using tile_shape_out = TileLeft<float, row, col>;

  gm_shape s0(src);
  gm_shape res(dst);

  tile_shape_in d0;
  tile_shape_out d1;

  TCOPYIN(d0, s0);
  TCVT(d1, d0);
  TCVT(d0, d1);
  TCOPYOUT(res, d0);
}

int main() {
#ifdef __linx
  constexpr uint16_t row = 16;
  constexpr uint16_t col = 16;
  using row_tile = Tile<Location::Vec, int64_t, row, col>;
  using col_tile = Tile<Location::Vec, int64_t, row, col, BLayout::ColMajor>;
  using nz_tile = TileLeft<int64_t, row, col>;
  using zn_tile = TileRight<int64_t, row, col>;

  row_tile row_src;
  row_tile row_round;
  col_tile col_src;
  col_tile col_round;
  nz_tile nz_a;
  nz_tile nz_b;
  zn_tile zn;

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      row_src.data()[index<row_tile>(i, j)] =
          static_cast<int64_t>((i + 1) * 100 + j);
      col_src.data()[index<col_tile>(i, j)] =
          static_cast<int64_t>((i + 1) * 1000 + j);
    }
  }

  TCVT(nz_a, row_src);
  TCVT(row_round, nz_a);
  TCVT(zn, nz_a);
  TCVT(nz_b, zn);

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      if (row_round.data()[index<row_tile>(i, j)] !=
          row_src.data()[index<row_tile>(i, j)]) {
        return 1;
      }
      if (nz_b.data()[index<nz_tile>(i, j)] !=
          nz_a.data()[index<nz_tile>(i, j)]) {
        return 2;
      }
    }
  }

  TCVT(nz_a, col_src);
  TCVT(col_round, nz_a);
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      if (col_round.data()[index<col_tile>(i, j)] !=
          col_src.data()[index<col_tile>(i, j)]) {
        return 3;
      }
    }
  }

  return 0;
#else
  const uint16_t row = 16;
  const uint16_t col = 32;

  size_t size = row * col;

  float *dst = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size);

  float *src = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src);
  init_src_fp(src, size);

  float *dst1 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst1);
  init_dst(dst1, size);

  float *src1 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src1);
  init_rows_fp(src1, row, col);

  float *dst2 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(dst2);
  init_dst(dst2, size);

  float *src2 = (float *)malloc(size * sizeof(float));
  check_mem_alloc(src2);
  init_rows_fp(src2, row, col);

#ifdef LINX_PMC
  PMC_START();
#endif

  testRow2Nz<row, col>(dst, src);
  testNz2Col<row, col>(dst1, src1);
  testNz2Zn<row, col>(dst2, src2);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size);
  OutArray(dst1, size);
  OutArray(dst2, size);

  free(dst);
  free(src);
  free(dst1);
  free(src1);
  free(dst2);
  free(src2);

  return 0;
#endif
}
