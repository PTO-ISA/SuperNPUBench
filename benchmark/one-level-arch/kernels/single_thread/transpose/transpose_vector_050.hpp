/**
 * @file transpose_vector_050_tile.hpp
 * @brief 8×64×64 张量转置为 64×8×64 张量（Tile ISA 实现，逐行处理）
 * 
 * 输入：8×64×64 张量（展平为 512×64）
 * 输出：64×8×64 张量（展平为 512×64）
 * 
 * 转置关系：交换第 0 维和第 1 维
 * - input[a][b][c] → output[b][a][c]
 * - 其中 a∈[0,8), b∈[0,64), c∈[0,64)
 * 
 * 行置换关系：
 * - 输入行号 = a*64 + b
 * - 输出行号 = b*8 + a
 * - 每行 64 个元素
 * 
 * 算法：逐行复制
 * - 对每个 (a, b) 对，复制 input 行 (a*64+b) 到 output 行 (b*8+a)
 * - 共 8×64 = 512 行，每行用 TLOAD + TSTORE 处理
 */

#ifndef TRANSPOSE_VECTOR_050_TILE_HPP
#define TRANSPOSE_VECTOR_050_TILE_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

/**
 * @brief 8×64×64 → 64×8×64 转置（逐行处理）
 * 
 * 实现方式：
 * - 外层循环遍历 a ∈ [0, 8)
 * - 内层循环遍历 b ∈ [0, 64)
 * - 对每个 (a, b)，将 input 行 (a*64+b) 复制到 output 行 (b*8+a)
 * 
 * 每行处理：
 * - 用 1×64 的 tile 加载一行
 * - 用 TSTORE 存储到目标位置
 * 
 * @tparam dtype 数据类型
 * @param out_ptr 输出指针（64×8×64 = 512×64）
 * @param in_ptr 输入指针（8×64×64 = 512×64）
 */
template<typename dtype>
void transpose_050_tile(
    dtype *out_ptr,
    dtype *in_ptr
)
{
    constexpr int DIM0 = 8;    // 第 0 维大小
    constexpr int DIM1 = 64;   // 第 1 维大小
    constexpr int DIM2 = 64;   // 第 2 维大小（每行元素数）
    constexpr int TOTAL_ROWS = DIM0 * DIM1;  // 512

    // 全局张量定义
    using InputGlobal = global_tensor<dtype, RowMajor<TOTAL_ROWS, DIM2>>;
    using OutputGlobal = global_tensor<dtype, RowMajor<TOTAL_ROWS, DIM2>>;
    
    // 行 tile：1×64
    using RowTile = Tile<Location::Vec, dtype, 1, DIM2, BLayout::RowMajor>;
    
    // 行迭代器
    using RowIterator = global_iterator<InputGlobal, RowTile>;

    InputGlobal input_global(in_ptr);
    OutputGlobal output_global(out_ptr);
    RowIterator input_row_iter(in_ptr);
    RowIterator output_row_iter(out_ptr);

    RowTile row_tile;

    // 逐行处理
    for (int a = 0; a < DIM0; ++a) {
        for (int b = 0; b < DIM1; ++b) {
            // 源行号：a*64 + b
            int src_row = a * DIM1 + b;
            
            // 目标行号：b*8 + a
            int dst_row = b * DIM0 + a;
            
            // 加载源行
            auto src_global = input_row_iter(src_row, 0);
            TLOAD(row_tile, src_global);
            
            // 存储到目标行
            auto dst_global = output_row_iter(dst_row, 0);
            TSTORE(dst_global, row_tile);
        }
    }
}

#endif  // TRANSPOSE_VECTOR_050_TILE_HPP
