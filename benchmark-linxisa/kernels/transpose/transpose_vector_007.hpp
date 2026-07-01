/**
 * @file transpose_vector_007_tile.hpp
 * @brief 4096×3 矩阵转置为 3×4096 矩阵（Tile ISA 实现，使用 TTRANS）
 *
 * 输入：4096×3 矩阵
 * 输出：3×4096 矩阵
 *
 * 算法：分块 2D 转置
 * - 将 4096×3 分成多个 TileRows×TileCols 的小块
 * - 对每个小块用 TTRANS 转置
 * - 存储到输出的对应位置
 */

#ifndef TRANSPOSE_VECTOR_007_TILE_HPP
#define TRANSPOSE_VECTOR_007_TILE_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

/**
 * @brief 4096×3 → 3×4096 转置（使用 TTRANS）
 *
 * 分块策略：
 * - TileRows = 16, TileCols = 3
 * - 输入块：16×3
 * - 输出块：3×16（TTRANS 后）
 * - 共 256 个块（4096/16 = 256）
 *
 * @tparam dtype 数据类型
 * @param out_ptr 输出指针（3×4096）
 * @param in_ptr 输入指针（4096×3）
 */
template<typename dtype>
void transpose_007_tile(
    dtype *out_ptr,
    dtype *in_ptr
)
{
    constexpr int INPUT_ROWS = 4096;
    constexpr int INPUT_COLS = 3;
    constexpr int OUTPUT_ROWS = 3;
    constexpr int OUTPUT_COLS = 4096;

    // 分块参数
    constexpr int TILE_ROWS = 16;  // 输入块行数
    constexpr int TILE_COLS = 3;   // 输入块列数
    constexpr int NUM_ROW_TILES = INPUT_ROWS / TILE_ROWS;  // 256
    constexpr int NUM_COL_TILES = INPUT_COLS / TILE_COLS;  // 1

    // 全局张量定义
    using InputGlobal = global_tensor<dtype, RowMajor<INPUT_ROWS, INPUT_COLS>>;
    using OutputGlobal = global_tensor<dtype, RowMajor<OUTPUT_ROWS, OUTPUT_COLS>>;

    // Tile 定义
    using SrcTile = Tile<Location::Vec, dtype, TILE_ROWS, TILE_COLS, BLayout::RowMajor>;
    using DstTile = Tile<Location::Vec, dtype, TILE_COLS, TILE_ROWS, BLayout::RowMajor>;

    // 迭代器
    using InputIterator = global_iterator<InputGlobal, SrcTile>;
    using OutputIterator = global_iterator<OutputGlobal, DstTile>;

    InputIterator input_iter(in_ptr);
    OutputIterator output_iter(out_ptr);

    // 遍历所有块
    for (int row_tile = 0; row_tile < NUM_ROW_TILES; ++row_tile) {
        for (int col_tile = 0; col_tile < NUM_COL_TILES; ++col_tile) {
            // 输入位置：(row_tile, col_tile) -> 对应输入矩阵的 [row_tile*16 : (row_tile+1)*16, col_tile*3 : (col_tile+1)*3]
            auto input_global = input_iter(row_tile, col_tile);

            // 输出位置：(col_tile, row_tile) -> 对应输出矩阵的 [col_tile*3 : (col_tile+1)*3, row_tile*16 : (row_tile+1)*16]
            auto output_global = output_iter(col_tile, row_tile);

            SrcTile src_tile;
            DstTile dst_tile;

            // 加载输入块（16×3）
            TLOAD(src_tile, input_global);

            // 转置：16×3 -> 3×16
            TTRANS(dst_tile, src_tile);

            // 存储到输出
            TSTORE(output_global, dst_tile);
        }
    }
}

#endif  // TRANSPOSE_VECTOR_007_TILE_HPP
