#include <common/pto_tileop.hpp>

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

using namespace pto;

#ifndef ROWSUM_ROWS
#define ROWSUM_ROWS 32
#endif

#ifndef ROWSUM_COLS
#define ROWSUM_COLS 32
#endif

#ifndef ROWSUM_PARTS
#define ROWSUM_PARTS 4
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

template <int Rows, int Cols, int Parts>
void rowsum_subview(float *out_ptr, float *in_ptr) {
    static_assert(Cols % Parts == 0,
                  "Columns must be divisible by the number of subviews");
    static_assert(Parts == 4,
                  "This example spells out four CUBE_M32 column views");
    static_assert(Rows == 32,
                  "CUBE_M32 row subspace must contain 32 rows");
    static_assert(Rows * (Cols / Parts) * sizeof(float) >= 128,
                  "Each subview must contain at least one 128-byte CELL");

    constexpr int kSubCols = Cols / Parts;

    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, 1>>;
    // In CUBE_M32 CELL order, consecutive ranges partition columns while
    // keeping all 32 rows, so four ranges represent 32x8 views.
    using tileIn = CubeTileM32<float, Rows, Cols>;
    using tileInPart = CubeTileM32<float, Rows, kSubCols>;

    // Each column-range reduction produces one partial value per input row.
    using tilePartSum = Tile<Location::Vec, float, Rows, 1,
                             BLayout::RowMajor>;

    static_assert(tileIn::LogicalTileBytes ==
                      Parts * tileInPart::LogicalTileBytes,
                  "Subviews must exactly cover the input tile");
    gmIn input_gm(in_ptr);
    gmOut output_gm(out_ptr);
    tileIn input_tile;
    TLOAD(input_tile, input_gm);

    auto input_parts = TPARTVIEW<tileInPart, 1, Parts>(input_tile);
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
    TSTORE(output_gm, output_tile);
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
    rowsum_subview<ROWSUM_ROWS, ROWSUM_COLS, ROWSUM_PARTS>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin",
                    reinterpret_cast<uint8_t *>(output),
                    ROWSUM_ROWS * sizeof(float));
#endif

    return 0;
}
