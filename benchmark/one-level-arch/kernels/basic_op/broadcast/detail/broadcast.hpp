// Broadcast 的逐元素示例：用普通 C++ 坐标计算表达读取，用公共 TileOp
// 组装完整快照，再统一写回。FP16/2048 路径为64->128->2048层级，保留
// 尾部谓词和逐 Tile 的输入/输出重叠语义。其他 profile 保留既有兼容路径。

#include <common/pto_tileop.hpp>
#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <utility>

// ============================================================================
// 维度规则：从后面对齐，前面自动补 1，维度=1 可广播
// ============================================================================

// 既有 TileOp 偏移生成路径，供尚未迁移的 profile 使用。
template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void gen_offset_pto(
    tile_shapeOffset &out,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base,
    const size_t total_elements)
{
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");

    using off_t = typename tile_shapeOffset::DType;   // uint32_t

    tile_shapeOffset idxTile;     // 当前正在分解的线性索引
    tile_shapeOffset coordTile;   // 当前输出维坐标
    tile_shapeOffset tmpTile;     // TREMS 需要 (A2A3 要求 tmp >= 1 行, 列数 >= dst)

    // ---- Step 1: TCI 生成索引序列 [base, base+1, ..., base+N-1] ----
    TCI(idxTile, (off_t)base);

    // ---- Step 2: TEXPANDS 初始化 out = 0 ----
    TEXPANDS(out, (off_t)0);

    size_t stride = 1;

    // ---- Step 3: 从最内维到最外维逐维 divmod + stride 累加 ----
    #pragma clang loop unroll(full)
    for (int d = (int)OUT_DIM - 1; d >= 0; d--) {
        off_t out_d = (off_t)out_shape[d];

        // TREMS: coord = idx % out_d
        TREMS(coordTile, idxTile, out_d);

        // TDIVS: idx = idx / out_d (推进到下一维)
        TDIVS(idxTile, idxTile, out_d);

        // 该输出维是否对应输入维
        int i = d - (int)(OUT_DIM - IN_DIM);
        if (i >= 0) {
            if (in_shape[i] != 1) {            // 非广播维才累加
                // TMULS: tmp = coord * stride
                TMULS(tmpTile, coordTile, (off_t)stride);
                // TADD: out += tmp
                TADD(out, out, tmpTile);
            }
            stride *= in_shape[i];             // stride 更新 (广播维 ==1 不变)
        }
    }

}


template <size_t IN_DIM, size_t OUT_DIM, uint32_t LogicalCount>
void broadcast_f16_element_tile(
    const __half *in_ptr,
    __half *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape,
    uint32_t output_base,
    uint32_t batch_begin)
{
    constexpr uint32_t kTileElements = 2048;
    constexpr uint32_t kLeafElements = 64;
    constexpr uint32_t kLeavesPerGroup = 2;
    constexpr uint32_t kGroupCount = 16;
    constexpr uint32_t kPartCount = kLeavesPerGroup * kGroupCount;
    static_assert(LogicalCount > 0 && LogicalCount <= kTileElements);

    uint32_t input_extents[IN_DIM];
    uint32_t input_strides[IN_DIM];
    uint32_t output_extents[OUT_DIM];
    uint32_t output_strides[OUT_DIM];

    uint32_t stride = 1;
    for (int dimension = static_cast<int>(OUT_DIM) - 1;
         dimension >= 0; --dimension) {
        output_extents[dimension] =
            static_cast<uint32_t>(out_shape[dimension]);
        output_strides[dimension] = stride;
        stride *= output_extents[dimension];
    }

    stride = 1;
    for (int dimension = static_cast<int>(IN_DIM) - 1;
         dimension >= 0; --dimension) {
        input_extents[dimension] =
            static_cast<uint32_t>(in_shape[dimension]);
        input_strides[dimension] = stride;
        stride *= input_extents[dimension];
    }

    // 逻辑分片和完整快照均由公共 API 管理；所有输入先读取，再统一写回，
    // 保留 input/output 重叠时原有的逐 Tile 快照语义。
    pto::ElementTile<__half, kTileElements> logical_layout;
    auto logical_parts =
        pto::TPARTVIEW<kLeafElements>(logical_layout, LogicalCount);
    pto::TileArray<pto::ElementTile<__half, 128>, 1, kGroupCount> groups;

    for (uint32_t group = 0; group < kGroupCount; ++group) {
        pto::TileArray<pto::ElementTile<__half, kLeafElements>, 1,
                       kLeavesPerGroup>
            leaves;

        for (uint32_t leaf_in_group = 0;
             leaf_in_group < kLeavesPerGroup; ++leaf_in_group) {
            const uint32_t part =
                group * kLeavesPerGroup + leaf_in_group;
            const uint32_t part_count =
                static_cast<uint32_t>(logical_parts.valid_size(part));
            pto::ElementTile<__half, kLeafElements> gathered;
            auto &elements = pto::TPARTELEMENT(gathered);

#pragma pto element for
            for (uint32_t element = 0; element < kLeafElements; ++element) {
                if (element < part_count) {
                    const uint32_t local_output =
                        batch_begin +
                        logical_parts.logical_index(part, element);
                    const uint32_t global_output =
                        output_base + local_output;
                    uint32_t input_index = 0;

                    for (uint32_t dimension = 0; dimension < IN_DIM;
                         ++dimension) {
                        const uint32_t output_dimension =
                            dimension + static_cast<uint32_t>(OUT_DIM -
                                                              IN_DIM);
                        const uint32_t coordinate =
                            (global_output /
                             output_strides[output_dimension]) %
                            output_extents[output_dimension];
                        if (input_extents[dimension] != 1)
                            input_index +=
                                coordinate * input_strides[dimension];
                    }

                    elements[element] = in_ptr[input_index];
                }
            }

            TMOV(leaves[0][leaf_in_group], gathered);
        }

        auto closed_group =
            pto::TASSEMBLY<pto::ElementTile<__half, 128>>(
                std::move(leaves));
        TMOV(groups[0][group], closed_group);
    }

    auto output_tile =
        pto::TASSEMBLY<pto::ElementTile<__half, kTileElements>>(
            std::move(groups));
    auto output_parts =
        pto::TPARTVIEW<kLeafElements>(output_tile, LogicalCount);

    for (uint32_t part = 0; part < kPartCount; ++part) {
        const uint32_t part_count =
            static_cast<uint32_t>(output_parts.valid_size(part));
        pto::ElementTile<__half, kLeafElements> stored;
        auto borrowed = output_parts.part(part);
        TMOV(stored, borrowed);
        TSTORE(out_ptr + batch_begin, stored, output_parts, part, part_count);
    }
}

// ----------------------------------------------------------------------------
// broadcast: 接口与原 broadcast.hpp 一致
// ----------------------------------------------------------------------------
template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t gIM, size_t gOM, size_t tM>
void broadcast(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape,
    // Global flattened index represented by out_ptr[0]. The default keeps
    // the original full-tensor behavior; multi-PE wrappers use a nonzero
    // value for disjoint output slices.
    size_t output_base = 0
    ) {
    if constexpr (std::is_same_v<dtype, __half> && tM == 2048 &&
                  IN_DIM <= OUT_DIM && gIM > 0 && gOM > 0 &&
                  gIM <= UINT32_MAX && gOM <= UINT32_MAX) {
        static_assert(IN_DIM > 0 && OUT_DIM <= MAX_DIM,
                      "broadcast ranks must fit the shape arrays");
        uint32_t input_elements = 1;
        uint32_t output_elements = 1;
        for (uint32_t d = 0; d < OUT_DIM; ++d) {
            if (out_shape[d] == 0 || out_shape[d] > UINT32_MAX ||
                output_elements > UINT32_MAX / out_shape[d])
                __builtin_trap();
            output_elements *= static_cast<uint32_t>(out_shape[d]);
        }
        for (uint32_t d = 0; d < IN_DIM; ++d) {
            if (in_shape[d] == 0 || in_shape[d] > UINT32_MAX ||
                input_elements > UINT32_MAX / in_shape[d] ||
                (in_shape[d] != 1 && in_shape[d] != out_shape[d + OUT_DIM - IN_DIM]))
                __builtin_trap();
            input_elements *= static_cast<uint32_t>(in_shape[d]);
        }
        if (input_elements > gIM || gOM > output_elements ||
            output_base > output_elements - gOM)
            __builtin_trap();
        constexpr uint32_t kTileElements = 2048;
        constexpr uint32_t kFullTiles =
            static_cast<uint32_t>(gOM) / kTileElements;
        constexpr uint32_t kTailElements =
            static_cast<uint32_t>(gOM) % kTileElements;

        for (uint32_t tile = 0; tile < kFullTiles; ++tile) {
            const uint32_t batch_begin = tile * kTileElements;
            broadcast_f16_element_tile<IN_DIM, OUT_DIM, kTileElements>(
                in_ptr, out_ptr, in_shape, out_shape,
                static_cast<uint32_t>(output_base), batch_begin);
        }
        if constexpr (kTailElements != 0)
            broadcast_f16_element_tile<IN_DIM, OUT_DIM, kTailElements>(
                in_ptr, out_ptr, in_shape, out_shape,
                static_cast<uint32_t>(output_base),
                kFullTiles * kTileElements);
        return;
    }

    const size_t Mb = gOM / tM;
    const size_t rmd_M = gOM % tM;

    using gm_shapeIn  = global_tensor<dtype, RowMajor<1, gIM>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using tile_shapeData      = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor>;
    using tile_shapeOffset    = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd  = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeOffset_rmd= Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor, 1, rmd_M>;

    gm_shapeIn inGm(in_ptr);
    tile_shapeData outTile;
    tile_shapeOffset offsetTile;
    tile_shapeData_rmd outTile_rmd;
    tile_shapeOffset_rmd offsetTile_rmd;
    size_t base = output_base;

    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
    itOut gOIter(out_ptr);

    size_t total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);

        // 计算偏移 tile (元素索引)
        gen_offset_pto<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>(
            offsetTile, in_shape, out_shape, base, total_elements);
        base += total_elements;

        // MGATHER: 按 offsetTile 中的元素索引从 inGm 取数
        MGATHER(outTile, inGm, offsetTile);

        // TSTORE: 将 outTile 写回 global memory
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;
        gen_offset_pto<dtype, tile_shapeOffset_rmd, MAX_DIM, IN_DIM, OUT_DIM>(
            offsetTile_rmd, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(outTile_rmd, inGm, offsetTile_rmd);
        TSTORE(gO, outTile_rmd);
    }
}
