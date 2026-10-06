#pragma once

#include <common/pto_tileop.hpp>

#include <cstdint>

using namespace pto;

// Reduce Rows x Columns along axis 0.  Output column blocks are assigned to
// PEs round-robin, so every output element has exactly one writer and no
// cross-PE reduction or barrier is required.
template <bool IsMax, typename DType, int Rows, int Columns,
          int TileRows, int TileColumns, std::uint32_t PeCount>
void reduce_colvec_pto(DType *input, DType *output,
                       std::uint32_t pe_id, std::uint32_t pe_count) {
    static_assert(Rows > 0 && Columns > 0);
    static_assert(TileRows > 0 && TileColumns > 0);
    static_assert(PeCount > 0);

    constexpr int kRowBlocks = Rows / TileRows;
    constexpr int kColumnBlocks = Columns / TileColumns;
    constexpr int kTailRows = Rows % TileRows;
    constexpr int kTailColumns = Columns % TileColumns;

    using InputGm = global_tensor<DType, RowMajor<Rows, Columns>>;
    using OutputGm = global_tensor<DType, RowMajor<1, Columns>>;

    using DataTile = Tile<Location::Vec, DType, TileRows, TileColumns,
                          BLayout::RowMajor>;
    using DataTileTailRows = Tile<Location::Vec, DType, TileRows, TileColumns,
                                  BLayout::RowMajor, kTailRows, TileColumns>;
    using ResultTile = Tile<Location::Vec, DType, 1, TileColumns,
                            BLayout::RowMajor, 1, TileColumns>;

    using DataTileTailColumns =
        Tile<Location::Vec, DType, TileRows, TileColumns,
             BLayout::RowMajor, TileRows, kTailColumns>;
    using DataTileCorner =
        Tile<Location::Vec, DType, TileRows, TileColumns,
             BLayout::RowMajor, kTailRows, kTailColumns>;
    using ResultTileTailColumns =
        Tile<Location::Vec, DType, 1, TileColumns,
             BLayout::RowMajor, 1, kTailColumns>;

    using InputIterator = global_iterator<InputGm, DataTile>;
    using OutputIterator = global_iterator<OutputGm, ResultTile>;
    InputIterator input_iter(input);
    OutputIterator output_iter(output);

    DataTile data;
    DataTileTailRows data_tail_rows;
    ResultTile partial;
    ResultTile result;
    DataTileTailColumns data_tail_columns;
    DataTileCorner data_corner;
    ResultTileTailColumns partial_tail_columns;
    ResultTileTailColumns result_tail_columns;

    for (int column_block = static_cast<int>(pe_id);
         column_block < kColumnBlocks;
         column_block += static_cast<int>(pe_count)) {
        auto out = output_iter(0, column_block);

        if constexpr (IsMax) {
            if constexpr (kRowBlocks > 0) {
                auto in0 = input_iter(0, column_block);
                TLOAD(data, in0);
                TCOLMAX(result, data);
            } else {
                auto in0 = input_iter(0, column_block);
                TLOAD(data_tail_rows, in0);
                TCOLMAX(result, data_tail_rows);
            }
        } else {
            TEXPANDS(result, static_cast<DType>(0));
        }

        #pragma clang loop unroll(full)
        for (int row_block = IsMax ? 1 : 0;
             row_block < kRowBlocks; ++row_block) {
            auto in = input_iter(row_block, column_block);
            TLOAD(data, in);
            if constexpr (IsMax) {
                TCOLMAX(partial, data);
                TMAX(result, result, partial);
            } else {
                TCOLSUM(partial, data);
                TADD(result, result, partial);
            }
        }

        if constexpr (kTailRows > 0 && (!IsMax || kRowBlocks > 0)) {
            auto in = input_iter(kRowBlocks, column_block);
            TLOAD(data_tail_rows, in);
            if constexpr (IsMax) {
                TCOLMAX(partial, data_tail_rows);
                TMAX(result, result, partial);
            } else {
                TCOLSUM(partial, data_tail_rows);
                TADD(result, result, partial);
            }
        }
        TSTORE(out, result);
    }

    if constexpr (kTailColumns > 0) {
        if (pe_id == static_cast<std::uint32_t>(kColumnBlocks) % pe_count) {
            auto out = output_iter(0, kColumnBlocks);

            if constexpr (IsMax) {
                if constexpr (kRowBlocks > 0) {
                    auto in0 = input_iter(0, kColumnBlocks);
                    TLOAD(data_tail_columns, in0);
                    TCOLMAX(result_tail_columns, data_tail_columns);
                } else {
                    auto in0 = input_iter(0, kColumnBlocks);
                    TLOAD(data_corner, in0);
                    TCOLMAX(result_tail_columns, data_corner);
                }
            } else {
                TEXPANDS(result_tail_columns, static_cast<DType>(0));
            }

            #pragma clang loop unroll(full)
            for (int row_block = IsMax ? 1 : 0;
                 row_block < kRowBlocks; ++row_block) {
                auto in = input_iter(row_block, kColumnBlocks);
                TLOAD(data_tail_columns, in);
                if constexpr (IsMax) {
                    TCOLMAX(partial_tail_columns, data_tail_columns);
                    TMAX(result_tail_columns, result_tail_columns,
                         partial_tail_columns);
                } else {
                    TCOLSUM(partial_tail_columns, data_tail_columns);
                    TADD(result_tail_columns, result_tail_columns,
                         partial_tail_columns);
                }
            }

            if constexpr (kTailRows > 0 && (!IsMax || kRowBlocks > 0)) {
                auto in = input_iter(kRowBlocks, kColumnBlocks);
                TLOAD(data_corner, in);
                if constexpr (IsMax) {
                    TCOLMAX(partial_tail_columns, data_corner);
                    TMAX(result_tail_columns, result_tail_columns,
                         partial_tail_columns);
                } else {
                    TCOLSUM(partial_tail_columns, data_corner);
                    TADD(result_tail_columns, result_tail_columns,
                         partial_tail_columns);
                }
            }
            TSTORE(out, result_tail_columns);
        }
    }
}
