#ifndef CONCAT_SCATTER_KERNEL_HPP
#define CONCAT_SCATTER_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>
#include <type_traits>

using namespace pto;

/**
 * @brief 通用 N 维 concat scatter（Tile ISA 实现）
 *
 * 算法：原始 2D concat profile 把连续的 N 个 [rows, cols]
 * 输入 scatter 为 [rows, N, cols]。每个原始 Tile 输入块先整体
 * snapshot 到 ElementTile，再用普通标量 C++ 下标在 element region 中写回。
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
    size_t *out_shape,
    // in_ptr[0] 对应的全局展平下标。
    uint32_t input_base_start = 0,
    // 单 PE 原始用例处理 gOM 个元素；多 PE wrapper 显式传入本 PE
    // 的整 tile 元素数。
    uint32_t processed_elements = static_cast<uint32_t>(gOM)
)
{
    if constexpr (DATA_DIM == 2 && CONCAT_DIM == 1 &&
                  (((std::is_same_v<DType, int32_t> ||
                     std::is_same_v<DType, __half>) && tM == 512) ||
                   (std::is_same_v<DType, float> && tM == 128))) {
        static_assert(gIM > 0 && gOM > 0,
                      "concat scatter requires non-empty allocations");
        static_assert(gIM <= static_cast<int>(__INT32_MAX__) &&
                          gOM <= static_cast<int>(__INT32_MAX__),
                      "concat scatter requires B32-representable indices");

        constexpr uint32_t kTileElements = tM;
        constexpr uint32_t kPartElements =
            std::is_same_v<DType, __half> ? 64u : 32u;

        if (processed_elements == 0u)
            return;
        if (processed_elements % kTileElements != 0u ||
            processed_elements > static_cast<uint32_t>(gIM) ||
            input_base_start > static_cast<uint32_t>(gIM) -
                                   processed_elements ||
            processed_elements > static_cast<uint32_t>(gOM) ||
            input_base_start > static_cast<uint32_t>(gOM) -
                                   processed_elements)
            __builtin_trap();

        if (in_shape[0] > __UINT32_MAX__ ||
            in_shape[1] > __UINT32_MAX__ ||
            out_shape[0] > __UINT32_MAX__ ||
            out_shape[1] > __UINT32_MAX__)
            __builtin_trap();
        const uint32_t in_rows = static_cast<uint32_t>(in_shape[0]);
        const uint32_t in_cols = static_cast<uint32_t>(in_shape[1]);
        const uint32_t out_rows = static_cast<uint32_t>(out_shape[0]);
        const uint32_t out_cols = static_cast<uint32_t>(out_shape[1]);
        if (in_rows == 0u || in_cols == 0u || out_rows != in_rows ||
            out_cols == 0u || out_cols % in_cols != 0u ||
            in_rows > __UINT32_MAX__ / in_cols ||
            out_rows > __UINT32_MAX__ / out_cols ||
            out_rows * out_cols != static_cast<uint32_t>(gOM))
            __builtin_trap();

        const uint32_t input_elements = in_rows * in_cols;
        for (uint32_t batch_begin = 0; batch_begin < processed_elements;
             batch_begin += kTileElements) {
            ElementTile<DType, kTileElements> input_tile;

            // 整块 TLOAD 必须位于 scatter region 之前。输入输出别名时，
            // 本块保留完整 snapshot，下一块则观察到之前块的写入。
            TLOAD(input_tile, in_ptr + batch_begin, kTileElements);
            auto input_parts =
                TPARTVIEW<kPartElements>(input_tile, kTileElements);

            for (uint32_t part = 0; part < input_parts.size(); ++part) {
                ElementTile<DType, kPartElements> values;
                auto input_part = input_parts.part(part);
                TMOV(values, input_part);
                auto &value_elements = TPARTELEMENT(values);

#pragma pto element for
                for (uint32_t element = 0; element < kPartElements;
                     ++element) {
                    const uint32_t local =
                        batch_begin +
                        input_parts.logical_index(part, element);
                    const uint32_t global = input_base_start + local;
                    const uint32_t input_no = global / input_elements;
                    const uint32_t within_input = global % input_elements;
                    const uint32_t row = within_input / in_cols;
                    const uint32_t col = within_input % in_cols;
                    const uint32_t output_index =
                        row * out_cols + input_no * in_cols + col;
                    out_ptr[output_index] = value_elements[element];
                }
            }
        }
    } else {
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

    uint32_t input_base = input_base_start;
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

        TCI(linear_index, static_cast<uint16_t>(input_base));
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
                TCVT(quotient, linear_index);
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

        TCI(linear_index, static_cast<uint16_t>(input_base));
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
                TCVT(quotient, linear_index);
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
}

#endif
