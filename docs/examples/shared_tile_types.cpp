#include <pto_kernel/tile.hpp>

using pto::LocalTile;
using pto::MatrixLeftTile;
using pto::SharedMove;
using pto::SharedTile;

using LocalRows = LocalTile<float, 8, 32>;
using LocalLeft = MatrixLeftTile<float, 8, 16>;
using SharedMatrix = SharedTile<float, 32, 32>;
using SharedRight = SharedTile<float, 16, 32>;

static_assert(LocalRows::Rows == 8);
static_assert(LocalLeft::Cols == 16);
static_assert(SharedMatrix::Rows == 32);
static_assert(SharedRight::Cols == 32);
static_assert(SharedMove::Insert != SharedMove::Broadcast);

extern "C" int shared_tile_type_probe() {
  return SharedMatrix::Rows * SharedMatrix::Cols;
}
