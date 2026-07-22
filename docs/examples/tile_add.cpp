#include <pto_kernel/tile.hpp>

using pto::LocalTile;
using pto::RowMajor;
using pto::TADD;
using pto::TLOAD;
using pto::TSTORE;
using pto::global_iterator;
using pto::global_tensor;

namespace {

constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kTileRows = 32;
constexpr int kTileCols = 32;

using ValueTile = LocalTile<int, kTileRows, kTileCols>;
using Matrix = global_tensor<int, RowMajor<kRows, kCols>>;
using MatrixTiles = global_iterator<Matrix, ValueTile>;

} // namespace

extern "C" void tile_add(int *out, int *lhs, int *rhs) {
  MatrixTiles out_tiles(out);
  MatrixTiles lhs_tiles(lhs);
  MatrixTiles rhs_tiles(rhs);

  for (int tile_row = 0; tile_row < kRows / kTileRows; ++tile_row) {
    for (int tile_col = 0; tile_col < kCols / kTileCols; ++tile_col) {
      ValueTile left;
      ValueTile right;
      ValueTile sum;

      TLOAD(left, lhs_tiles(tile_row, tile_col));
      TLOAD(right, rhs_tiles(tile_row, tile_col));
      TADD(sum, left, right);
      TSTORE(out_tiles(tile_row, tile_col), sum);
    }
  }
}
