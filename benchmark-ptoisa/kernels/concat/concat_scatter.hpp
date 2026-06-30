#ifndef CONCAT_SCATTER_KERNEL_HPP
#define CONCAT_SCATTER_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

/**
 * @brief 通用 N 维 concat scatter（Tile ISA 实现）
 *
 * 算法：对输入张量的每个元素，计算其多维坐标，用输出 stride 计算输出 byte offset，
 * 用 MSCATTER 将数据写入输出。
 *
 * 注意：当前实现中 n（来自哪个输入张量）始终为 0，即输入被放置在输出的起始位置。
 * 这与原始 __vec__ 版本行为一致。
 *
 * @tparam DType 数据类型
 * @tparam MAX_DIM 最大维度数
 * @tparam gIM 输入总元素数
 * @tparam gOM 输出总元素数
 * @tparam tM 每个 tile 处理的元素数
 * @tparam DATA_DIM 数据维度数（编译期常量）
 * @tparam CONCAT_DIM 拼接维度索引（编译期常量）
 */
template<typename DType, size_t MAX_DIM = 8, const int gIM, const int gOM,
         const int tM, size_t DATA_DIM, size_t CONCAT_DIM>
void concat_scatter(
    DType *in_ptr,
    DType *out_ptr,
    size_t *in_shape,
    size_t *out_shape
)
{
    constexpr int kFullTiles = gIM / tM;
    constexpr int kTailElements = gIM % tM;

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, gIM>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, gOM>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, tM, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, uint16_t, 1, tM, pto::BLayout::RowMajor>;
    using InputIterator = global_iterator<InputGlobal, DataTile>;

    InputGlobal input_global(in_ptr);
    OutputGlobal output_global(out_ptr);
    InputIterator input_iter(in_ptr);

    uint32_t input_base = 0;
    for (int tile_index = 0; tile_index < kFullTiles; ++tile_index) {
        auto input_tile_global = input_iter(0, tile_index);
        DataTile data_tile;
        OffsetTile offset_tile;

        OffsetTile linear_index;
        OffsetTile quotient;
        OffsetTile cycle;
        OffsetTile cycle_base;
        OffsetTile coordinate;
        OffsetTile contribution;

        TCI<OffsetTile, uint16_t, 0>(linear_index, static_cast<uint16_t>(input_base));
        TEXPANDS(offset_tile, static_cast<uint16_t>(0));

        // 计算输出 stride（标量核心，运行时计算）
        uint16_t output_stride[DATA_DIM];
        output_stride[DATA_DIM - 1] = 1;
        for (int i = static_cast<int>(DATA_DIM) - 2; i >= 0; --i) {
            output_stride[i] = output_stride[i + 1] * static_cast<uint16_t>(out_shape[i + 1]);
        }

        // 对每个维度提取输入坐标并计算输出 byte offset
        for (int d = static_cast<int>(DATA_DIM) - 1; d >= 0; --d) {
            uint32_t input_stride_for_dim = 1;
            for (int dd = d + 1; dd < static_cast<int>(DATA_DIM); ++dd) {
                input_stride_for_dim *= static_cast<uint32_t>(in_shape[dd]);
            }

            if (input_stride_for_dim == 1) {
                TMOV(quotient, linear_index);
            } else {
                TDIVS(quotient, linear_index, static_cast<uint16_t>(input_stride_for_dim));
            }

            uint16_t in_dim_size = static_cast<uint16_t>(in_shape[d]);
            TDIVS(cycle, quotient, in_dim_size);
            TMULS(cycle_base, cycle, in_dim_size);
            TSUB(coordinate, quotient, cycle_base);

            TMULS(contribution, coordinate,
                  static_cast<uint16_t>(output_stride[d] * sizeof(DType)));
            TADD(offset_tile, offset_tile, contribution);
        }

        TLOAD(data_tile, input_tile_global);
        MSCATTER(output_global, data_tile, offset_tile);

        input_base += tM;
    }

    if constexpr (kTailElements != 0) {
        using TailDataTile = pto::Tile<pto::Location::Vec, DType, 1, tM,
                                       pto::BLayout::RowMajor, 1, kTailElements>;
        using TailOffsetTile = pto::Tile<pto::Location::Vec, uint16_t, 1, tM,
                                         pto::BLayout::RowMajor, 1, kTailElements>;

        auto input_tile_global = input_iter(0, kFullTiles);
        TailDataTile data_tile;
        TailOffsetTile offset_tile;

        TailOffsetTile linear_index;
        TailOffsetTile quotient;
        TailOffsetTile cycle;
        TailOffsetTile cycle_base;
        TailOffsetTile coordinate;
        TailOffsetTile contribution;

        TCI<TailOffsetTile, uint16_t, 0>(linear_index, static_cast<uint16_t>(input_base));
        TEXPANDS(offset_tile, static_cast<uint16_t>(0));

        uint16_t output_stride[DATA_DIM];
        output_stride[DATA_DIM - 1] = 1;
        for (int i = static_cast<int>(DATA_DIM) - 2; i >= 0; --i) {
            output_stride[i] = output_stride[i + 1] * static_cast<uint16_t>(out_shape[i + 1]);
        }

        for (int d = static_cast<int>(DATA_DIM) - 1; d >= 0; --d) {
            uint32_t input_stride_for_dim = 1;
            for (int dd = d + 1; dd < static_cast<int>(DATA_DIM); ++dd) {
                input_stride_for_dim *= static_cast<uint32_t>(in_shape[dd]);
            }

            if (input_stride_for_dim == 1) {
                TMOV(quotient, linear_index);
            } else {
                TDIVS(quotient, linear_index, static_cast<uint16_t>(input_stride_for_dim));
            }

            uint16_t in_dim_size = static_cast<uint16_t>(in_shape[d]);
            TDIVS(cycle, quotient, in_dim_size);
            TMULS(cycle_base, cycle, in_dim_size);
            TSUB(coordinate, quotient, cycle_base);

            TMULS(contribution, coordinate,
                  static_cast<uint16_t>(output_stride[d] * sizeof(DType)));
            TADD(offset_tile, offset_tile, contribution);
        }

        TLOAD(data_tile, input_tile_global);
        MSCATTER(output_global, data_tile, offset_tile);
    }
}

#endif
