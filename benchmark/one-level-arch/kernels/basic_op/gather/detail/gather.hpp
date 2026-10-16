// Gather 算子 — PTO 一层编程模型
//
// 原始 FP32/U32 配置按 MxN tile 工作。每个 tile 都重新快照行索引，
// 再用普通标量 C++ 下标在 `#pragma pto element for` 中 gather。所有结果先
// 经 32 -> 128 -> 2048 -> 16384 的公开 ElementTile 层级关闭，之后才写 GM。
// 这个顺序保留 input/index/output 可能别名时的原始逐 tile 可观察语义。
// 其他模板实例继续使用既有 MGATHER 路径。

#include <common/pto_tileop.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <utility>

using namespace pto;

// ----------------------------------------------------------------------------
// gather: 按行索引从数据表中 gather 行
//
// 输入:
//   in_data_ptr   — 数据表, shape (gK, gN), row-major
//   in_offset_ptr — 行索引数组, shape (gM,), 每个元素是数据表的行号
//   out_ptr       — 输出, shape (gM, gN), out[j,:] = in_data[in_offset[j],:]
//
// 模板参数 (与原 gather.hpp 一致):
//   dtype  — 数据类型 (float, half, ...)
//   otype  — 索引类型 (uint32_t, int32_t)
//   gK     — 数据表行数
//   gM     — 输出行数 (= 索引数组长度)
//   gN     — 数据表列数 (= 每行元素数)
//   tM     — tile 行数
//   tN     — tile 列数
// ----------------------------------------------------------------------------
template<typename dtype = float, typename otype = uint32_t,
         size_t gK, size_t gM, size_t gN, size_t tM, size_t tN>
void gather(
    dtype *in_data_ptr,
    otype *in_offset_ptr,
    dtype *out_ptr
    ) {
    constexpr bool kElementHierarchySupported =
        std::is_same_v<dtype, float> &&
        (std::is_same_v<otype, uint32_t> || std::is_same_v<otype, int32_t>) &&
        (tM == 32 || tM == 128) &&
        (tN == 2048 / tM || tN == 16384 / tM) &&
        gK > 0 && gN > 0 && gM > 0 &&
        gK <= __UINT32_MAX__ / gN && gM <= __UINT32_MAX__ / gN;

    if constexpr (kElementHierarchySupported) {
        static_assert(gK * gN <= __UINT32_MAX__,
                      "gather data offsets must fit the B32 index domain");
        static_assert(tM == 32 || tM == 128,
                      "original gather uses 32- or 128-row tiles");
        static_assert(tM * tN == 2048 || tM * tN == 16384,
                      "original gather requires a public F32 hierarchy");

        constexpr uint32_t kIndexElements = tM;
        constexpr uint32_t kLeafElements = 32;
        constexpr uint32_t kTileElements = tM * tN;
        constexpr uint32_t kMiddleElements = 128;
        constexpr uint32_t kMiddleCount = kTileElements / kMiddleElements;

        for (uint32_t m_tile = 0; m_tile < (gM + tM - 1) / tM; ++m_tile) {
            const uint32_t m_base = m_tile * static_cast<uint32_t>(tM);
            uint32_t valid_rows = static_cast<uint32_t>(tM);
            if constexpr (gM % tM != 0) {
                const uint32_t remaining_rows =
                    static_cast<uint32_t>(gM) - m_base;
                if (remaining_rows < valid_rows)
                    valid_rows = remaining_rows;
            }

            for (uint32_t n_tile = 0; n_tile < (gN + tN - 1) / tN; ++n_tile) {
                const uint32_t n_base = n_tile * static_cast<uint32_t>(tN);
                uint32_t valid_cols = static_cast<uint32_t>(tN);
                if constexpr (gN % tN != 0) {
                    const uint32_t remaining_cols =
                        static_cast<uint32_t>(gN) - n_base;
                    if (remaining_cols < valid_cols)
                        valid_cols = remaining_cols;
                }

                // 索引必须在每个原始 MxN tile 内重新快照。这样若 output
                // 与 index 别名，前一个 N 块的写入会被下一个 N 块观察到。
                ElementTile<otype, kIndexElements> index_snapshot;
                if constexpr (tM == kLeafElements) {
                    auto &index_elements = TPARTELEMENT(index_snapshot);
#pragma pto element for
                    for (uint32_t element = 0; element < kLeafElements; ++element)
                        if (element < valid_rows)
                            index_elements[element] = in_offset_ptr[m_base + element];
                } else {
                    ElementTile<otype, kIndexElements> index_layout;
                    auto layout_parts = TPARTVIEW<kLeafElements>(index_layout, valid_rows);
                    TileArray<ElementTile<otype, kLeafElements>, 1,
                              kIndexElements / kLeafElements> index_leaves;
                    for (uint32_t part = 0; part < kIndexElements / kLeafElements; ++part) {
                        ElementTile<otype, kLeafElements> loaded;
                        auto &elements = TPARTELEMENT(loaded);
                        const uint32_t valid = layout_parts.valid_size(part);
#pragma pto element for
                        for (uint32_t element = 0; element < kLeafElements; ++element)
                            if (element < valid)
                                elements[element] = in_offset_ptr[m_base + static_cast<uint32_t>(layout_parts.logical_index(part, element))];
                        TMOV(index_leaves[0][part], loaded);
                    }
                    auto closed_indices = TASSEMBLY<ElementTile<otype, kIndexElements>>(
                        std::move(index_leaves));
                    TMOV(index_snapshot, closed_indices);
                }
                const dtype *data_block = in_data_ptr + n_base;
                dtype *output_block =
                    out_ptr + m_base * static_cast<uint32_t>(gN) + n_base;

                if constexpr (kTileElements == 2048) {
                    TileArray<ElementTile<float, kMiddleElements>, 1,
                              kMiddleCount>
                        middle;

                    for (uint32_t group = 0; group < kMiddleCount; ++group) {
                        TileArray<ElementTile<float, kLeafElements>, 1, 4>
                            leaves;
                        for (uint32_t leaf = 0; leaf < 4; ++leaf) {
                            const uint32_t part = group * 4 + leaf;
                            const uint32_t index_part_number = part / tN;
                            const uint32_t column = part % tN;

                            ElementTile<otype, kLeafElements> indices;
                            uint32_t valid_index_elements;
                            if constexpr (tM == kLeafElements) {
                                TMOV(indices, index_snapshot);
                                valid_index_elements = valid_rows;
                            } else {
                                auto index_parts = TPARTVIEW<kLeafElements>(index_snapshot, valid_rows);
                                auto index_part = index_parts.part(index_part_number);
                                TMOV(indices, index_part);
                                valid_index_elements = index_parts.valid_size(index_part_number);
                            }

                            ElementTile<float, kLeafElements> gathered;
                            auto &index_elements = TPARTELEMENT(indices);
                            auto &gathered_elements = TPARTELEMENT(gathered);

#pragma pto element for
                            for (uint32_t element = 0;
                                 element < kLeafElements; ++element) {
                                if (element < valid_index_elements &&
                                    column < valid_cols) {
                                    const uint32_t data_index =
                                        index_elements[element] *
                                            static_cast<uint32_t>(gN) +
                                        column;
                                    gathered_elements[element] =
                                        data_block[data_index];
                                }
                            }
                            TMOV(leaves[0][leaf], gathered);
                        }
                        auto closed =
                            TASSEMBLY<ElementTile<float, kMiddleElements>>(
                                std::move(leaves));
                        TMOV(middle[0][group], closed);
                    }

                    // 完整 2048 元素快照关闭后才开始任何 GM 写回。
                    auto snapshot = TASSEMBLY<ElementTile<float, kTileElements>>(
                        std::move(middle));
                    auto output_parts =
                        TPARTVIEW<kLeafElements>(snapshot, kTileElements);
                    for (uint32_t row_group = 0;
                         row_group < tM / kLeafElements; ++row_group) {
                        for (uint32_t column = 0; column < valid_cols;
                             ++column) {
                            const uint32_t part = row_group * tN + column;
                            ElementTile<float, kLeafElements> values;
                            auto output_part = output_parts.part(part);
                            TMOV(values, output_part);
                            auto &value_elements = TPARTELEMENT(values);
                            dtype *output_column =
                                output_block +
                                row_group * static_cast<uint32_t>(gN) +
                                column;

#pragma pto element for
                            for (uint32_t element = 0;
                                 element < kLeafElements; ++element) {
                                // TPARTVIEW<32> 的公开映射为
                                // flat = element * part_count + part；按 MxN
                                // 展开后 row = element*(tM/32)+part/tN。
                                const uint32_t row =
                                    element * static_cast<uint32_t>(tM / kLeafElements) + row_group;
                                const uint32_t output_index =
                                    element * static_cast<uint32_t>(tM / kLeafElements) *
                                    static_cast<uint32_t>(gN);
                                if constexpr (gM % tM == 0) {
                                    output_column[output_index] =
                                        value_elements[element];
                                } else if (row < valid_rows) {
                                    output_column[output_index] =
                                        value_elements[element];
                                }
                            }
                        }
                    }
                } else {
                    TileArray<ElementTile<float, 2048>, 1, 8> outer;

                    for (uint32_t outer_index = 0; outer_index < 8;
                         ++outer_index) {
                        TileArray<ElementTile<float, kMiddleElements>, 1, 16>
                            middle;
                        for (uint32_t group = 0; group < 16; ++group) {
                            TileArray<ElementTile<float, kLeafElements>, 1, 4>
                                leaves;
                            for (uint32_t leaf = 0; leaf < 4; ++leaf) {
                                const uint32_t part =
                                    (outer_index * 16 + group) * 4 + leaf;
                                const uint32_t index_part_number = part / tN;
                                const uint32_t column = part % tN;

                                ElementTile<otype, kLeafElements> indices;
                                uint32_t valid_index_elements;
                                if constexpr (tM == kLeafElements) {
                                    TMOV(indices, index_snapshot);
                                    valid_index_elements = valid_rows;
                                } else {
                                    auto index_parts = TPARTVIEW<kLeafElements>(index_snapshot, valid_rows);
                                    auto index_part = index_parts.part(index_part_number);
                                    TMOV(indices, index_part);
                                    valid_index_elements = index_parts.valid_size(index_part_number);
                                }

                                ElementTile<float, kLeafElements> gathered;
                                auto &index_elements = TPARTELEMENT(indices);
                                auto &gathered_elements =
                                    TPARTELEMENT(gathered);

#pragma pto element for
                                for (uint32_t element = 0;
                                     element < kLeafElements; ++element) {
                                    if (element < valid_index_elements &&
                                        column < valid_cols) {
                                        const uint32_t data_index =
                                            index_elements[element] *
                                                static_cast<uint32_t>(gN) +
                                            column;
                                        gathered_elements[element] =
                                            data_block[data_index];
                                    }
                                }
                                TMOV(leaves[0][leaf], gathered);
                            }
                            auto closed128 =
                                TASSEMBLY<ElementTile<float, kMiddleElements>>(
                                    std::move(leaves));
                            TMOV(middle[0][group], closed128);
                        }
                        auto closed2048 =
                            TASSEMBLY<ElementTile<float, 2048>>(
                                std::move(middle));
                        TMOV(outer[0][outer_index], closed2048);
                    }

                    // 所有 gather load 完成且层级关闭后，才允许逐元素写回。
                    auto snapshot = TASSEMBLY<ElementTile<float, kTileElements>>(
                        std::move(outer));
                    auto output_parts =
                        TPARTVIEW<kLeafElements>(snapshot, kTileElements);
                    for (uint32_t row_group = 0;
                         row_group < tM / kLeafElements; ++row_group) {
                        for (uint32_t column = 0; column < valid_cols;
                             ++column) {
                            const uint32_t part = row_group * tN + column;
                            ElementTile<float, kLeafElements> values;
                            auto output_part = output_parts.part(part);
                            TMOV(values, output_part);
                            auto &value_elements = TPARTELEMENT(values);
                            dtype *output_column =
                                output_block +
                                row_group * static_cast<uint32_t>(gN) +
                                column;

#pragma pto element for
                            for (uint32_t element = 0;
                                 element < kLeafElements; ++element) {
                                // TPARTVIEW<32> 的公开映射为
                                // flat = element * part_count + part；按 MxN
                                // 展开后 row = element*(tM/32)+part/tN。
                                const uint32_t row =
                                    element * static_cast<uint32_t>(tM / kLeafElements) + row_group;
                                const uint32_t output_index =
                                    element * static_cast<uint32_t>(tM / kLeafElements) *
                                    static_cast<uint32_t>(gN);
                                if constexpr (gM % tM == 0) {
                                    output_column[output_index] =
                                        value_elements[element];
                                } else if (row < valid_rows) {
                                    output_column[output_index] =
                                        value_elements[element];
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
    const size_t Mb = gM / tM;
    const size_t Nb = gN / tN;
    const size_t rmd_M = gM % tM;
    const size_t rmd_N = gN % tN;

    using gm_shapeInOffset = global_tensor<otype, RowMajor<1, gM>>;
    using gm_shapeIn       = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOut      = global_tensor<dtype, RowMajor<gM, gN>>;

    using tile_shapeInOffset     = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData         = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor>;
    using tile_shapeInOffset_rmd_n   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData_rmd_n       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeInOffset_rmd_mn  = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_mn      = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeInOffset_rmd_m   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_m       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, tN>;

    tile_shapeInOffset inOffsetTile;
    tile_shapeData outTile;
    tile_shapeInOffset_rmd_n inOffsetTile_rmd_n;
    tile_shapeData_rmd_n outTile_rmd_n;
    tile_shapeInOffset_rmd_mn inOffsetTile_rmd_mn;
    tile_shapeData_rmd_mn outTile_rmd_mn;
    tile_shapeInOffset_rmd_m inOffsetTile_rmd_m;
    tile_shapeData_rmd_m outTile_rmd_m;

    using itInOffset = global_iterator<gm_shapeInOffset, tile_shapeInOffset>;
    using itOut      = global_iterator<gm_shapeOut, tile_shapeData>;

    itInOffset gInOffsetIter(in_offset_ptr);
    itOut      gOIter(out_ptr);

    // ---- 主循环: Mb × Nb 个完整 tile ----
    for (int j = 0; j < Mb; ++j) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, i);
            size_t n_base  = i * tN;

            // TLOAD: 加载行索引 tile (1, tM) from GM
            // [当前编译器] 名为 TCOPYIN
            TLOAD(inOffsetTile, gInOffset);

            // MGATHER<Coalesce::Row>: 按行索引从数据表取数
            //   dst[r,:] = table[idx[r], :]
            //   table 指针偏移 n_base 个元素, 使取数起始列为 n_base
            //   (tablePtr + idx * gN + n_base 定位到 row idx, col n_base)
            // [当前编译器] template_asm.h 的 MGATHER 无 Coalesce 模板参数;
            //             且按字节偏移取数, 非行索引
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile, adjustedGm, inOffsetTile);

            // TSTORE: 写回输出 tile (tM, tN) to GM
            // [当前编译器] 名为 TCOPYOUT
            TSTORE(gO, outTile);
        }

        // ---- rmd_N: 最后一个列块不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_n, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_n, adjustedGm, inOffsetTile_rmd_n);
            TSTORE(gO, outTile_rmd_n);
        }
    }

    // ---- rmd_M: 最后一个行块不完整 ----
    if constexpr (rmd_M) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, i);
            size_t n_base  = i * tN;

            TLOAD(inOffsetTile_rmd_m, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_m, adjustedGm, inOffsetTile_rmd_m);
            TSTORE(gO, outTile_rmd_m);
        }

        // ---- rmd_M + rmd_N: 右下角不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_mn, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_mn, adjustedGm, inOffsetTile_rmd_mn);
            TSTORE(gO, outTile_rmd_mn);
        }
    }
    }
}
