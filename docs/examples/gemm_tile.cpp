#include <pto_kernel/tile.hpp>

using namespace pto;

namespace {

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kTK = 8;

using LhsTile = MatrixLeftTile<int, kTM, kTK>;
using RhsTile = MatrixRightTile<int, kTK, kTN>;
using AccTile = AccumulatorTile<int, kTM, kTN>;
using OutTile = LocalTile<int, kTM, kTN>;

using Lhs = global_tensor<int, RowMajor<kM, kK>>;
using Rhs = global_tensor<int, ColMajor<kK, kN>>;
using Out = global_tensor<int, RowMajor<kM, kN>>;

using LhsTiles = global_iterator<Lhs, LhsTile>;
using RhsTiles = global_iterator<Rhs, RhsTile>;
using OutTiles = global_iterator<Out, OutTile>;

} // namespace

extern "C" void gemm_tile(int *out, int *lhs, int *rhs) {
  LhsTiles lhs_tiles(lhs);
  RhsTiles rhs_tiles(rhs);
  OutTiles out_tiles(out);

  for (int m = 0; m < kM / kTM; ++m) {
    for (int n = 0; n < kN / kTN; ++n) {
      LhsTile left;
      RhsTile right;
      AccTile accum;

      TLOAD(left, lhs_tiles(m, 0));
      TLOAD(right, rhs_tiles(0, n));
      TMATMUL(accum, left, right);

      for (int k = 1; k < kK / kTK; ++k) {
        TLOAD(left, lhs_tiles(m, k));
        TLOAD(right, rhs_tiles(k, n));
        TMATMUL_ACC(accum, accum, left, right);
      }

      OutTile result;
      TCVT(result, accum);
      TSTORE(out_tiles(m, n), result);
    }
  }
}
