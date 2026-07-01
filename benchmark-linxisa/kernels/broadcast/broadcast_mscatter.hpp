#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include <cstdint>
#include <cstdio>

#define MAX_DIMS 8

// ==============================================
// Scatter Vec核：输入坐标 + 广播各维偏移 → 生成输出字节偏移
// 兼容任意广播维度，和原gen_offset接口风格一致
// ==============================================
template<typename dtype, typename tile_shape, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void __vec__ gen_scatter_offset(
    typename tile_shape::TileDType __out__ out_addr,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base_in,
    const size_t total_elements,
    // 广播维度当前步进值 通用传入
    const size_t broadcast_step[MAX_DIM],
    // 标记哪些维度是广播维
    const bool is_broadcast_dim[MAX_DIM]
) {
    size_t index = blkv_get_index_x();
    if (index >= total_elements) return;
    size_t in_idx = index + base_in;

    // 1. 输入一维索引 → 输入多维坐标
    size_t in_coord[MAX_DIM] = {0};
    size_t tmp = in_idx;
    for (int d = IN_DIM - 1; d >= 0; d--) {
        in_coord[d] = tmp % in_shape[d];
        tmp /= in_shape[d];
    }

    // 2. 映射到输出坐标 后对齐规则
    size_t out_coord[MAX_DIM] = {0};
    // 先拷贝非广播维度
    for (int i = 0; i < IN_DIM; i++) {
        size_t o_dim = OUT_DIM - IN_DIM + i;
        if(!is_broadcast_dim[o_dim]){
            out_coord[o_dim] = in_coord[i];
        }
    }
    // 填充广播维度的当前步进值
    for(int d = 0; d < OUT_DIM; d++){
        if(is_broadcast_dim[d]){
            out_coord[d] = broadcast_step[d];
        }
    }

    // 3. 输出坐标 → 字节偏移 行主序
    size_t out_offset = 0;
    size_t data_width = sizeof(dtype);
    size_t stride = 1;
    for (int d = OUT_DIM - 1; d >= 0; d--) {
        out_offset += out_coord[d] * stride;
        stride *= out_shape[d];
    }
    out_offset *= data_width;

    blkv_get_tile_ptr(out_addr)[index] = out_offset;
}

template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void gen_scatter_offset_impl(
    tile_shapeOffset &out_addr,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base_in,
    const size_t total_elements,
    const size_t broadcast_step[MAX_DIM],
    const bool is_broadcast_dim[MAX_DIM]
) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");
    gen_scatter_offset<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>
        <<<tile_shapeOffset::ValidCol, 1, 1>>>(
            out_addr.data(), in_shape, out_shape,
            base_in, total_elements,
            broadcast_step, is_broadcast_dim
        );
}

// ==============================================
// 通用MSCATTER版 Broadcast
// 自动识别任意广播维度
// ==============================================
template<typename dtype, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM, size_t gIM, size_t gOM, size_t tM>
void broadcast_mscatter(
    dtype *in_ptr,
    dtype *out_ptr,
    dtype *god_ptr,
    uint32_t *idx_ptr,
    const size_t *in_shape,
    const size_t *out_shape
) {
    const size_t input_tiles = gIM / tM;
    const size_t rmd_input = gIM % tM;

    // Tile/全局张量定义
    using gm_shapeIn = global_tensor<dtype, RowMajor<1, gIM>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using gm_shapeOffset = global_tensor<uint32_t, RowMajor<1, gOM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
    using tile_shapeOffset = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, 512>;
    using tile_shapeOffset_rmd = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor, 1, 512>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);
    gm_shapeOffset idxGm(idx_ptr);

    tile_shapeData inDataTile;
    tile_shapeOffset offsetTile;
    tile_shapeData_rmd inDataTile_rmd;
    tile_shapeOffset_rmd offsetTile_rmd;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOffset = global_iterator<gm_shapeOffset, tile_shapeOffset>;
    itIn gInIter(in_ptr);
    itOffset gOffsetIter(in_ptr);

    // ======================
    // 1. 预处理：自动识别广播维度 + 记录各广播维度长度
    // ======================
    bool is_broadcast_dim[MAX_DIM] = {false};
    size_t bcast_dim_len[MAX_DIM] = {0};
    int bcast_cnt = 0;

    // 找到待广播维度 (后对齐规则)
    for(int d = 0; d < OUT_DIM; d++){
        int in_d = d - (OUT_DIM - IN_DIM);
        // 输入对应维为1 且 输出维更大 → 是广播维
        if(in_d >= 0 && in_shape[in_d] == 1 && out_shape[d] > 1){
            is_broadcast_dim[d] = true;
            bcast_dim_len[d] = out_shape[d];
            bcast_cnt ++;
        }
    }

    // ======================
    // 2. 通用N维广播遍历：进位法，适配任意广播维度数量
    // ======================
    size_t bcast_step[MAX_DIM] = {0};
    bool done = false;

    auto next_broadcast_step = [&]() -> bool {
        // 进位迭代所有广播组合，在所有待广播维度遍历
        int pos = OUT_DIM - 1;
        for(; pos >= 0; pos--){
            if(!is_broadcast_dim[pos]) continue;
            bcast_step[pos] ++;
            if(bcast_step[pos] < bcast_dim_len[pos]){
                return true;
            }
            bcast_step[pos] = 0;
        }
        return false;
    };

    size_t base_in = 0;
    size_t total_elements = tM;

    // ======================
    // 3. 遍历输入整Tile
    // ======================
    for (int i = 0; i < input_tiles; ++i) {
        auto gIn = gInIter(0, i);
        TCOPYIN(inDataTile, gIn);

        // 重置广播步进
        memset(bcast_step, 0, sizeof(bcast_step));
        done = false;

        // 遍历所有广播维度组合
        int offset_idx = 0;
        while(!done){
            // Vec核计算当前广播位置的输出地址
            gen_scatter_offset_impl<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>(
                offsetTile, in_shape, out_shape,
                base_in, total_elements,
                bcast_step, is_broadcast_dim
            );
            // 散射写入
            MSCATTER(outGm, inDataTile, offsetTile);
            auto gOffset = gOffsetIter(0, offset_idx);
            TCOPYOUT(gOffset, offsetTile);
            offset_idx ++;

            // 下一组广播坐标
            done = !next_broadcast_step();
        }
        base_in += total_elements;
    }

    // ======================
    // 4. 处理尾部余数Tile
    // ======================
    if constexpr (rmd_input > 0) {
        auto gIn = gInIter(0, input_tiles);
        total_elements = rmd_input;
        TCOPYIN(inDataTile_rmd, gIn);

        memset(bcast_step, 0, sizeof(bcast_step));
        done = false;
        while(!done){
            gen_scatter_offset_impl<dtype, tile_shapeOffset_rmd, MAX_DIM, IN_DIM, OUT_DIM>(
                offsetTile_rmd, in_shape, out_shape,
                base_in, total_elements,
                bcast_step, is_broadcast_dim
            );
            MSCATTER(outGm, inDataTile_rmd, offsetTile_rmd);
            auto gOffset = gOffsetIter(0, offset_idx);
            TCOPYOUT(gOffset, offsetTile_rmd);
            offset_idx ++;

            done = !next_broadcast_step();
        }
    }
}