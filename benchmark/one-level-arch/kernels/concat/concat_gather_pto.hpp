#ifndef CONCAT_GATHER_KERNEL_HPP
#define CONCAT_GATHER_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

/**
 * @brief 通用 N 维 concat gather（Tile ISA 实现）
 *
 * 算法：对输出张量的每个元素，计算其多维坐标，确定来自哪个输入张量（n）
 * 以及在输入张量内的偏移，用 MGATHER 从输入中 gather 数据。
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
void concat_gather(
    DType *in_ptr,
    DType *out_ptr,
    size_t *in_shape,
    size_t *out_shape
)
{
    constexpr int kFullTiles = gOM / tM;
    constexpr int kTailElements = gOM % tM;

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, gIM>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, gOM>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, tM, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, uint32_t, 1, tM, pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(in_ptr);
    OutputIterator output_iter(out_ptr);

    uint32_t output_base = 0;
    for (int tile_index = 0; tile_index < kFullTiles; ++tile_index) {
        auto output_global = output_iter(0, tile_index);
        DataTile output_tile;
        OffsetTile offset_tile;

        OffsetTile linear_index;
        OffsetTile quotient;
        OffsetTile cycle;
        OffsetTile cycle_base;
        OffsetTile coordinate;
        OffsetTile contribution;

        TCI(linear_index, output_base);
        TEXPANDS(offset_tile, static_cast<uint32_t>(0));

        // 计算输入 stride（标量核心，运行时计算）
        uint32_t input_stride[DATA_DIM];
        input_stride[DATA_DIM - 1] = 1;
        for (int i = static_cast<int>(DATA_DIM) - 2; i >= 0; --i) {
            input_stride[i] = input_stride[i + 1] * static_cast<uint32_t>(in_shape[i + 1]);
        }
        uint32_t input_size = input_stride[0] * static_cast<uint32_t>(in_shape[0]);

        // 对每个维度提取坐标并计算 byte offset
        for (int d = static_cast<int>(DATA_DIM) - 1; d >= 0; --d) {
            uint32_t output_stride = 1;
            for (int dd = d + 1; dd < static_cast<int>(DATA_DIM); ++dd) {
                output_stride *= static_cast<uint32_t>(out_shape[dd]);
            }

            if (output_stride == 1) {
                TCVT(quotient, linear_index);
            } else {
                TDIVS(quotient, linear_index, output_stride);
            }

            uint32_t out_dim_size = static_cast<uint32_t>(out_shape[d]);
            TDIVS(cycle, quotient, out_dim_size);
            TMULS(cycle_base, cycle, out_dim_size);
            TSUB(coordinate, quotient, cycle_base);

            if (static_cast<size_t>(d) == CONCAT_DIM) {
                uint32_t in_dim_size = static_cast<uint32_t>(in_shape[d]);
                OffsetTile n_tile;
                OffsetTile offset_in;
                OffsetTile n_times_shape;

                TDIVS(n_tile, coordinate, in_dim_size);
                TMULS(n_times_shape, n_tile, in_dim_size);
                TSUB(offset_in, coordinate, n_times_shape);

                TMULS(contribution, offset_in,
                      input_stride[d] * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);

                TMULS(contribution, n_tile,
                      input_size * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);
            } else {
                TMULS(contribution, coordinate,
                      input_stride[d] * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);
            }
        }

        MGATHER(output_tile, input_global, offset_tile);
        TSTORE(output_global, output_tile);

        output_base += tM;
    }

    if constexpr (kTailElements != 0) {
        using TailDataTile = pto::Tile<pto::Location::Vec, DType, 1, tM,
                                       pto::BLayout::RowMajor, 1, kTailElements>;
        using TailOffsetTile = pto::Tile<pto::Location::Vec, uint32_t, 1, tM,
                                         pto::BLayout::RowMajor, 1, kTailElements>;

        auto output_global = output_iter(0, kFullTiles);
        TailDataTile output_tile;
        TailOffsetTile offset_tile;

        TailOffsetTile linear_index;
        TailOffsetTile quotient;
        TailOffsetTile cycle;
        TailOffsetTile cycle_base;
        TailOffsetTile coordinate;
        TailOffsetTile contribution;

        TCI(linear_index, output_base);
        TEXPANDS(offset_tile, static_cast<uint32_t>(0));

        uint32_t input_stride[DATA_DIM];
        input_stride[DATA_DIM - 1] = 1;
        for (int i = static_cast<int>(DATA_DIM) - 2; i >= 0; --i) {
            input_stride[i] = input_stride[i + 1] * static_cast<uint32_t>(in_shape[i + 1]);
        }
        uint32_t input_size = input_stride[0] * static_cast<uint32_t>(in_shape[0]);

        for (int d = static_cast<int>(DATA_DIM) - 1; d >= 0; --d) {
            uint32_t output_stride = 1;
            for (int dd = d + 1; dd < static_cast<int>(DATA_DIM); ++dd) {
                output_stride *= static_cast<uint32_t>(out_shape[dd]);
            }

            if (output_stride == 1) {
                TCVT(quotient, linear_index);
            } else {
                TDIVS(quotient, linear_index, output_stride);
            }

            uint32_t out_dim_size = static_cast<uint32_t>(out_shape[d]);
            TDIVS(cycle, quotient, out_dim_size);
            TMULS(cycle_base, cycle, out_dim_size);
            TSUB(coordinate, quotient, cycle_base);

            if (static_cast<size_t>(d) == CONCAT_DIM) {
                uint32_t in_dim_size = static_cast<uint32_t>(in_shape[d]);
                TailOffsetTile n_tile;
                TailOffsetTile offset_in;
                TailOffsetTile n_times_shape;

                TDIVS(n_tile, coordinate, in_dim_size);
                TMULS(n_times_shape, n_tile, in_dim_size);
                TSUB(offset_in, coordinate, n_times_shape);

                TMULS(contribution, offset_in,
                      input_stride[d] * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);

                TMULS(contribution, n_tile,
                      input_size * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);
            } else {
                TMULS(contribution, coordinate,
                      input_stride[d] * static_cast<uint32_t>(sizeof(DType)));
                TADD(offset_tile, offset_tile, contribution);
            }
        }

        MGATHER(output_tile, input_global, offset_tile);
        TSTORE(output_global, output_tile);
    }
}

#endif
