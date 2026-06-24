#include <common/pto_tileop.hpp>

#ifndef ROW
#define ROW 32
#endif

#ifndef COL
#define COL 32
#endif

#ifndef RROW
#define RROW 16
#endif

#ifndef RCOL
#define RCOL 64
#endif

template <uint64_t row, uint64_t col, uint64_t rrow,
          uint64_t rcol>
void test_reshape() {
  using tile_shape_in = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, rrow, rcol, BLayout::RowMajor>;

  tile_shape_in d0(0);
  tile_shape_out d1;
  TRESHAPE(d1, d0);
}

int main() {
  test_reshape<ROW, COL, RROW, RCOL>();
  return 0;
}