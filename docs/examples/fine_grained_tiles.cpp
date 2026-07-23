#include <pto_kernel/tile.hpp>

using pto::BLayout;
using pto::DYNAMIC;
using pto::LocalTile;
using pto::RowMajor;
using pto::TADD;
using pto::TADDS;
using pto::TLOAD;
using pto::TMAXS;
using pto::TMINS;
using pto::TROWSUM;
using pto::TSTORE;
using pto::global_iterator;
using pto::global_tensor;

namespace {

constexpr int kFineTileBytes = 128;

using F32VectorTile = LocalTile<float, 1, 32>;
using F32MatrixTile = LocalTile<float, 4, 8>;
using F32RowSums =
    LocalTile<float, 4, 8, BLayout::RowMajor, 4, 1>;
using F32TailTile =
    LocalTile<float, 1, 32, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

template <typename Tile>
constexpr int logical_tile_bytes() {
  return Tile::Rows * Tile::Cols * sizeof(typename Tile::DType);
}

static_assert(logical_tile_bytes<F32VectorTile>() == kFineTileBytes);
static_assert(logical_tile_bytes<F32MatrixTile>() == kFineTileBytes);
static_assert(logical_tile_bytes<F32RowSums>() == kFineTileBytes);
static_assert(logical_tile_bytes<F32TailTile>() == kFineTileBytes);

using Vector = global_tensor<float, RowMajor<1, 256>>;
using VectorTiles = global_iterator<Vector, F32VectorTile>;

using Matrix = global_tensor<float, RowMajor<16, 16>>;
using MatrixTiles = global_iterator<Matrix, F32MatrixTile>;

using FourVectors = global_tensor<float, RowMajor<1, 128>>;
using FourVectorTiles = global_iterator<FourVectors, F32VectorTile>;

using TailBuffer = global_tensor<float, RowMajor<1, 32>>;
using TailTiles = global_iterator<TailBuffer, F32TailTile>;

using ReductionInput = global_tensor<float, RowMajor<4, 8>>;
using ReductionInputTiles = global_iterator<ReductionInput, F32MatrixTile>;
using ReductionOutput = global_tensor<float, RowMajor<4, 1>>;
using ReductionOutputTiles = global_iterator<ReductionOutput, F32RowSums>;

} // namespace

// Eight 128-byte tiles cover 256 FP32 elements, like a wide SIMD loop.
extern "C" void fine_vector_add(float *out, float *lhs, float *rhs) {
  VectorTiles out_tiles(out);
  VectorTiles lhs_tiles(lhs);
  VectorTiles rhs_tiles(rhs);

  for (int tile_col = 0; tile_col < 8; ++tile_col) {
    F32VectorTile left;
    F32VectorTile right;
    F32VectorTile sum;
    TLOAD(left, lhs_tiles(0, tile_col));
    TLOAD(right, rhs_tiles(0, tile_col));
    TADD(sum, left, right);
    TSTORE(out_tiles(0, tile_col), sum);
  }
}

// A 4 x 8 tile keeps row and column meaning while retaining a 128-byte payload.
extern "C" void fine_matrix_clamp(float *out, float *input, float low,
                                   float high) {
  MatrixTiles out_tiles(out);
  MatrixTiles input_tiles(input);

  for (int tile_row = 0; tile_row < 4; ++tile_row) {
    for (int tile_col = 0; tile_col < 2; ++tile_col) {
      F32MatrixTile values;
      F32MatrixTile lower_bounded;
      F32MatrixTile clamped;
      TLOAD(values, input_tiles(tile_row, tile_col));
      TMAXS(lower_bounded, values, low);
      TMINS(clamped, lower_bounded, high);
      TSTORE(out_tiles(tile_row, tile_col), clamped);
    }
  }
}

// Independent tile chains expose instruction-level parallelism to the scheduler.
extern "C" void fine_independent_chains(float *out, float *input,
                                         float bias) {
  FourVectorTiles out_tiles(out);
  FourVectorTiles input_tiles(input);

  F32VectorTile x0, x1, x2, x3;
  F32VectorTile y0, y1, y2, y3;
  TLOAD(x0, input_tiles(0, 0));
  TLOAD(x1, input_tiles(0, 1));
  TLOAD(x2, input_tiles(0, 2));
  TLOAD(x3, input_tiles(0, 3));
  TADDS(y0, x0, bias);
  TADDS(y1, x1, bias);
  TADDS(y2, x2, bias);
  TADDS(y3, x3, bias);
  TSTORE(out_tiles(0, 0), y0);
  TSTORE(out_tiles(0, 1), y1);
  TSTORE(out_tiles(0, 2), y2);
  TSTORE(out_tiles(0, 3), y3);
}

// The physical payload remains 128 bytes; only the first 19 lanes are valid.
extern "C" void fine_tail_add(float *out, float *input, float bias) {
  TailTiles out_tiles(out);
  TailTiles input_tiles(input);

  F32TailTile values(1, 19);
  F32TailTile result(1, 19);
  TLOAD(values, input_tiles(0, 0));
  TADDS(result, values, bias);
  TSTORE(out_tiles(0, 0), result);
}

// Reducing each 8-element row produces a 4 x 1 valid region in a 128-byte tile.
extern "C" void fine_row_sum(float *out, float *input) {
  ReductionInputTiles input_tiles(input);
  ReductionOutputTiles output_tiles(out);

  F32MatrixTile values;
  F32RowSums sums;
  TLOAD(values, input_tiles(0, 0));
  TROWSUM(sums, values);
  TSTORE(output_tiles(0, 0), sums);
}
