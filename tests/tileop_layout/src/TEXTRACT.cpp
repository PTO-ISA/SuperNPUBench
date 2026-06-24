#include <common/pto_tileop.hpp>

#ifndef ROW
#define ROW 32
#endif

#ifndef COL
#define COL 32
#endif

#ifndef EROW
#define EROW 16
#endif

#ifndef ECOL
#define ECOL 16
#endif

#ifndef RIDX
#define RIDX 16
#endif

#ifndef CIDX
#define CIDX 16
#endif

template <uint64_t row, uint64_t col, uint64_t erow,
          uint64_t ecol, uint64_t ridx, uint64_t cidx>
void test_extract() {
  using tile_shape_in = Tile<Location::Vec, float, row, col, BLayout::RowMajor>;
  using tile_shape_out = Tile<Location::Vec, float, erow, ecol, BLayout::RowMajor>;

  tile_shape_in d0(0);
  tile_shape_out d1;
  TEXTRACT(d1, d0, ridx, cidx);
}

int main() {
  test_extract<ROW, COL, EROW, ECOL, RIDX, CIDX>();
  return 0;
}