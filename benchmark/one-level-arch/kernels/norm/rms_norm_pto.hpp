#ifndef SUPERNPU_RMS_NORM_PTO_HPP
#define SUPERNPU_RMS_NORM_PTO_HPP

#include <common/pto_tileop.hpp>

namespace supernpu::norm {

// Computes one RMSNorm layer:
//
//   output[m, n] =
//       input[m, n] * rsqrt(sum(input[m, :] ^ 2) / N + epsilon) * weight[n]
//
// The feature dimension is processed in TileN-wide chunks. Each chunk first
// contributes a partial row sum, then a second pass normalizes the input and
// applies the per-column weight.
template <int M, int N, int TileM, int TileN>
void rms_norm(float *output, const float *input, const float *weight,
              float epsilon = 1.0e-6f) {
    using namespace pto;

    constexpr int kTileByteLimit = 4 * 1024;
    static_assert(M > 0 && N > 0, "matrix dimensions must be positive");
    static_assert(TileM > 0 && TileN > 0,
                  "tile dimensions must be positive");
    static_assert(M % TileM == 0, "M must be divisible by TileM");
    static_assert(N % TileN == 0, "N must be divisible by TileN");
    static_assert(TileN % 8 == 0,
                  "TileN must satisfy the TileOP alignment requirement");
    static_assert(TileM * TileN * sizeof(float) <= kTileByteLimit,
                  "data tile must not exceed 4 KiB");
    static_assert(TileM * 8 * sizeof(float) <= kTileByteLimit,
                  "row-reduction tile must not exceed 4 KiB");
    static_assert(TileN * sizeof(float) <= kTileByteLimit,
                  "weight tile must not exceed 4 KiB");

    using GlobalMatrix = global_tensor<float, RowMajor<M, N>>;
    using GlobalWeight = global_tensor<float, RowMajor<1, N>>;
    using MatrixTile =
        Tile<Location::Vec, float, TileM, TileN, BLayout::RowMajor>;
    // TROWSUM exposes one valid value per row while retaining eight physical
    // columns to satisfy the Tile storage/alignment rules.
    using RowTile =
        Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;
    using WeightTile =
        Tile<Location::Vec, float, 1, TileN, BLayout::RowMajor>;

    using MatrixIterator = global_iterator<GlobalMatrix, MatrixTile>;
    using WeightIterator = global_iterator<GlobalWeight, WeightTile>;

    MatrixIterator input_iter(const_cast<float *>(input));
    MatrixIterator output_iter(output);
    WeightIterator weight_iter(const_cast<float *>(weight));

    constexpr int kRowBlocks = M / TileM;
    constexpr int kColumnBlocks = N / TileN;
    constexpr float kInvN = 1.0f / static_cast<float>(N);

    for (int row_block = 0; row_block < kRowBlocks; ++row_block) {
        RowTile row_square_sum;

        // Seed the reduction with the first feature chunk so no explicit
        // tile-zero operation is needed.
        {
            MatrixTile input_tile;
            MatrixTile squared_tile;
            auto input_global = input_iter(row_block, 0);
            TLOAD(input_tile, input_global);
            TMUL(squared_tile, input_tile, input_tile);
            TROWSUM(row_square_sum, squared_tile);
        }

        for (int column_block = 1; column_block < kColumnBlocks;
             ++column_block) {
            MatrixTile input_tile;
            MatrixTile squared_tile;
            RowTile partial_square_sum;

            auto input_global = input_iter(row_block, column_block);
            TLOAD(input_tile, input_global);
            TMUL(squared_tile, input_tile, input_tile);
            TROWSUM(partial_square_sum, squared_tile);
            TADD(row_square_sum, row_square_sum, partial_square_sum);
        }

        RowTile row_mean;
        RowTile denominator;
        RowTile inverse_rms;
        TMULS(row_mean, row_square_sum, kInvN);
        TADDS(denominator, row_mean, epsilon);
        TRSQRT(inverse_rms, denominator);

        for (int column_block = 0; column_block < kColumnBlocks;
             ++column_block) {
            MatrixTile input_tile;
            MatrixTile normalized_tile;
            MatrixTile output_tile;
            WeightTile weight_tile;

            auto input_global = input_iter(row_block, column_block);
            auto weight_global = weight_iter(0, column_block);
            auto output_global = output_iter(row_block, column_block);

            TLOAD(input_tile, input_global);
            TLOAD(weight_tile, weight_global);
            TROWEXPANDMUL(normalized_tile, input_tile, inverse_rms);
            TCOLEXPANDMUL(output_tile, normalized_tile, weight_tile);
            TSTORE(output_global, output_tile);
        }
    }
}

} // namespace supernpu::norm

#endif
