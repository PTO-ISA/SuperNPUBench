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
    static_assert(Rows % Parts == 0,
                  "Rows must be divisible by the number of subviews");
    static_assert(Parts == 4,
                  "This example spells out the four assemble phases");
    static_assert((Rows / Parts) * Cols * sizeof(float) >= 128,
                  "Each subview must contain at least one 128-byte CELL");

    constexpr int kSubRows = Rows / Parts;

    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, 1>>;
    using tileIn = Tile<Location::Vec, float, Rows, Cols,
                        BLayout::RowMajor>;
    using tileInPart = Tile<Location::Vec, float, kSubRows, Cols,
                            BLayout::RowMajor>;

    // TROWSUM has one valid value per row. Four physical columns keep each
    // 8-row fragment at the minimum 128-byte range size.
    using tilePartSum = Tile<Location::Vec, float, kSubRows, 4,
                             BLayout::RowMajor, kSubRows, 1>;
    using tileOut = Tile<Location::Vec, float, Rows, 4,
                         BLayout::RowMajor, Rows, 1>;
    using outputParts = TileArray<tilePartSum, Parts, 1>;

    static_assert(tileIn::LogicalTileBytes ==
                      Parts * tileInPart::LogicalTileBytes,
                  "Subviews must exactly cover the input tile");
    static_assert(tileInPart::LogicalTileBytes %
                          range::RangeAddressUnitBytes ==
                      0,
                  "Subview offsets must use complete 128-byte range units");
    static_assert(tileOut::LogicalTileBytes ==
                      Parts * tilePartSum::LogicalTileBytes,
                  "Assembled fragments must exactly cover the output tile");
    static_assert(outputParts::ParentSizeCode == tileOut::TilesizeCode,
                  "Assembly parent capacity must match the output tile");

    gmIn input_gm(in_ptr);
    gmOut output_gm(out_ptr);
    tileIn input_tile;
    TLOAD(input_tile, input_gm);

    // TPARTVIEW emits B.SUBVIEW when a fragment is consumed by TROWSUM.
    auto input_parts = TPARTVIEW<tileInPart, Parts, 1>(input_tile);
    outputParts output_parts;

    auto input_part0 = input_parts[0][0];
    auto input_part1 = input_parts[1][0];
    auto input_part2 = input_parts[2][0];
    auto input_part3 = input_parts[3][0];
    tilePartSum partial_sum0;
    tilePartSum partial_sum1;
    tilePartSum partial_sum2;
    tilePartSum partial_sum3;
    TROWSUM(partial_sum0, input_part0);
    TROWSUM(partial_sum1, input_part1);
    TROWSUM(partial_sum2, input_part2);
    TROWSUM(partial_sum3, input_part3);

    // Public TileArray output references encode INIT/MIDDLE/MIDDLE/LAST from
    // their slot ordinals; no internal assembly helper is needed.
    TCVT(output_parts[0][0], partial_sum0);
    TCVT(output_parts[1][0], partial_sum1);
    TCVT(output_parts[2][0], partial_sum2);
    TCVT(output_parts[3][0], partial_sum3);

    tileOut output_tile =
        TASSEMBLY<tileOut, tilePartSum, Parts, 1>(
            static_cast<outputParts &&>(output_parts));
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
