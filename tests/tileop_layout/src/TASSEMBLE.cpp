#include <common/pto_tileop.hpp>

#ifndef ROW
#define ROW 32
#endif

#ifndef COL1
#define COL1 32
#endif

#ifndef COL2
#define COL2 32
#endif

#ifndef COL3
#define COL3 32
#endif


template <uint64_t row, uint64_t col1, uint64_t col2,
          uint64_t col3>
void test_assemble() {
  using tile_shape_in1 = Tile<Location::Vec, float, row, col1, BLayout::RowMajor>;
  using tile_shape_in2 = Tile<Location::Vec, float, row, col2, BLayout::RowMajor>;
  using tile_shape_in3 = Tile<Location::Vec, float, row, col3, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, row, col1+col2+col3, BLayout::RowMajor>;

  tile_shape_in1 d1(0);
  tile_shape_in2 d2(0);
  tile_shape_in3 d3(0);
  tile_shape_out o1;
  TASSEMBLE(o1, d1, d2, d3);
}

int main() {
  test_assemble<ROW, COL1, COL2, COL3>();
  return 0;
}