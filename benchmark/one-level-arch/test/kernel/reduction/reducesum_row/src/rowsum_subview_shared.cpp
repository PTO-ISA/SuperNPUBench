#include <common/pto_tileop.hpp>

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

using namespace pto;

// rowsum_subview_shared: shared L1 tiles with RowMajor (ND) layout.
// The input (128×16) is processed in 32-row chunks.  Each 32×16 block is
// loaded from GM into a SharedTile (L1) via TLOAD<Matrix, 1>, partitioned
// into four 32×4 column sub-views via TPARTVIEW, then TROWSUM reduces each
// sub-view to a 32×1 local partial sum, and TADD combines the four
// partials into the final row-sum stored back to GM.

#ifndef ROWSUM_ROWS
#define ROWSUM_ROWS 128
#endif

#ifndef ROWSUM_COLS
#define ROWSUM_COLS 16
#endif

#ifndef ROWSUM_PARTS
#define ROWSUM_PARTS 4
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

template <int Rows, int Cols, int Parts>
void rowsum_subview_shared(float *out_ptr, float *in_ptr) {
    static_assert(Cols % Parts == 0,
                  "Columns must be divisible by the number of subviews");
    static_assert(Parts == 4,
                  "This example spells out four column views");
    static_assert(Rows % 32 == 0,
                  "Row subspace must be a multiple of 32 rows");
    static_assert(32 * (Cols / Parts) * sizeof(float) >= 128,
                  "Each subview must contain at least one 128-byte CELL");

    constexpr int kCubeRows = 32;
    constexpr int kSubCols = Cols / Parts;
    constexpr int kNumCubes = Rows / kCubeRows;

    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, 1>>;

    // Shared L1 tiles.  SharedMatrixLeft uses RowMajor (ND) layout — the
    // correct storage format for shared tiles.  The parent and sub-tile
    // are both SharedTile so locations match for TPARTVIEW.
    using tileInMatrix = SharedMatrixLeft<float, kCubeRows, Cols>;
    using tileInPartMatrix = SharedMatrixLeft<float, kCubeRows, kSubCols>;
    using tileInShared = SharedTile<tileInMatrix>;
    using tileInPartShared = SharedTile<tileInPartMatrix>;

    // TROWSUM output is a local L0 Vec tile (single column).
    using tilePartSum = Tile<Location::Vec, float, kCubeRows, 1,
                             BLayout::RowMajor>;

    static_assert(tileInMatrix::LogicalTileBytes ==
                      Parts * tileInPartMatrix::LogicalTileBytes,
                  "Subviews must exactly cover the input tile");

    using itIn = global_iterator<gmIn, tileInMatrix>;
    using itOut = global_iterator<gmOut, tilePartSum>;

    itIn input_iter(in_ptr);
    itOut output_iter(out_ptr);

    for (int c = 0; c < kNumCubes; ++c) {
        auto gIn = input_iter(c, 0);
        tileInShared input_tile;
        TLOAD<tileInMatrix, 1>(input_tile, gIn);

        auto input_parts = TPARTVIEW<tileInPartShared, 1, Parts>(input_tile);
        auto input_part0 = input_parts[0][0];
        auto input_part1 = input_parts[0][1];
        auto input_part2 = input_parts[0][2];
        auto input_part3 = input_parts[0][3];

        tilePartSum partial_sum0;
        tilePartSum partial_sum1;
        tilePartSum partial_sum2;
        tilePartSum partial_sum3;
        TROWSUM(partial_sum0, input_part0);
        TROWSUM(partial_sum1, input_part1);
        TROWSUM(partial_sum2, input_part2);
        TROWSUM(partial_sum3, input_part3);

        tilePartSum partial_sum01;
        tilePartSum partial_sum23;
        tilePartSum output_tile;
        TADD(partial_sum01, partial_sum0, partial_sum1);
        TADD(partial_sum23, partial_sum2, partial_sum3);
        TADD(output_tile, partial_sum01, partial_sum23);

        auto gOut = output_iter(c, 0);
        TSTORE(gOut, output_tile);
    }
}

int main() {
    static float input_buffer[ROWSUM_ROWS * ROWSUM_COLS + 2 * ALIGN];
    static float output_buffer[ROWSUM_ROWS + 2 * ALIGN];

    float *input = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(input_buffer) & ALIGN_MASK) + ALIGN);
    float *output = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(output_buffer) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                   ROWSUM_ROWS * ROWSUM_COLS * sizeof(float));
#else
    for (int i = 0; i < ROWSUM_ROWS * ROWSUM_COLS; ++i) {
        input[i] = static_cast<float>((i % 23) - 11) * 0.25f;
    }
#endif

    BENCHSTART;
    rowsum_subview_shared<ROWSUM_ROWS, ROWSUM_COLS, ROWSUM_PARTS>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin",
                    reinterpret_cast<uint8_t *>(output),
                    ROWSUM_ROWS * sizeof(float));
#endif

    return 0;
}
