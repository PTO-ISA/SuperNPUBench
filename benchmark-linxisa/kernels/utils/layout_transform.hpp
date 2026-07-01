#include <common/block_vector_compat.hpp>

using namespace pto::blkv;
using pto::is_global_data_v;
using pto::is_tile_data_v;

template<typename tile_shape>
void __vec__ gen_offset_ND2ZZ(
    typename tile_shape::TileDType __out__ out,
    const uint32_t in_dim_y,  // rows
    const uint32_t in_dim_x, // cols
    const uint32_t out_dim_y, // rows
    const uint32_t out_dim_x, // cols
    const uint32_t tile_i, // row_idx
    const uint32_t tile_j  // col_idx
) {
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    uint32_t i = blkv_get_index_x();
    uint32_t sub_tile_i = i / inner_cols; // frac_col_idx
    uint32_t idx_i = i % inner_cols;

    #pragma clang loop unroll(full)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
        uint32_t sub_tile_j = j / inner_rows;
        uint32_t idx_j = j % inner_rows;

        // size_t idx_src = tile_shape_in::RowStride * j + i;
        uint32_t idx_src = (tile_i*out_dim_y + j) * in_dim_x + tile_j * out_dim_x + i;
        // size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
        //                 sub_tile_j * tile_shape_out::InnerNumel +
        //                 idx_j * inner_cols + idx_i;
        uint32_t idx_dst = sub_tile_j * inner_rows * tile_shape::ValidCol + 
                           sub_tile_i * tile_shape::InnerNumel +
                           idx_j * inner_cols + idx_i;
        blkv_get_tile_ptr(out)[idx_dst] = idx_src;
    }
}

template<typename tile_shape>
void __vec__ gen_offset_ND2NN_new(
    typename tile_shape::TileDType __out__ out,
    const uint32_t in_dim_y,
    const uint32_t in_dim_x,
    const uint32_t out_dim_y,
    const uint32_t out_dim_x,
    const uint32_t tile_i,
    const uint32_t tile_j
) {
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    uint16_t x = blkv_get_index_x();
    uint16_t sub_tile_i = x / inner_rows; // frac_row_idx
    uint16_t idx_i = x % inner_rows;

    #pragma clang loop unroll(full)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
        uint32_t sub_tile_j = j / inner_cols;
        uint32_t idx_j = j % inner_cols;

        uint32_t idx_src = (tile_i*out_dim_y + x) * in_dim_x + tile_j * out_dim_x + j;
        uint32_t idx_dst = sub_tile_j * inner_cols * tile_shape::ValidRow + 
                           sub_tile_i * tile_shape::InnerNumel +
                           idx_j * inner_cols + idx_i;
        blkv_get_tile_ptr(out)[idx_dst] = idx_src;
    }
    // uint32_t idx_dst = sub_tile_i * inner_cols * tile_shape::ValidRow + 
    //                     sub_tile_j * tile_shape::InnerNumel +
    //                     idx_i * inner_cols + idx_j;
    // blkv_get_tile_ptr(out)[idx_dst] = (i*out_dim_y + y) * in_dim_x + j * out_dim_x + x;
}

template<typename tile_shape>
void __vec__ gen_offset_ND2NN(
    typename tile_shape::TileDType __out__ out,
    const uint32_t in_dim_y,
    const uint32_t in_dim_x,
    const uint32_t out_dim_y,
    const uint32_t out_dim_x,
    const uint32_t i,
    const uint32_t j
) {
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    uint16_t x = blkv_get_index_x();
    uint16_t y = blkv_get_index_y();
    uint16_t sub_tile_i = y / inner_cols; // frac_col_idx
    uint16_t idx_i = y % inner_cols;
    uint16_t sub_tile_j = x / inner_rows;
    uint16_t idx_j = x % inner_rows;

    uint32_t idx_dst = sub_tile_i * inner_cols * tile_shape::ValidRow + 
                        sub_tile_j * tile_shape::InnerNumel +
                        idx_i * inner_cols + idx_j;
    blkv_get_tile_ptr(out)[idx_dst] = (i*out_dim_y + y) * in_dim_x + j * out_dim_x + x;
}

template<is_global_data_v glb_tensor, is_tile_data_v tl_tensor, typename tile_shapeOffset>
void gen_ND2ZZ_offset_Impl(
    glb_tensor src,
    tl_tensor dst_tensor,
    tile_shapeOffset &offset,
    const uint32_t i, 
    const uint32_t j) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");
    pto::blkv::blkv_for_1d(tl_tensor::ValidCol, [&] {
        gen_offset_ND2ZZ<tile_shapeOffset>(offset.data(), glb_tensor::ColStride,
                                           glb_tensor::RowStride,
                                           tl_tensor::ValidRow,
                                           tl_tensor::ValidCol, i, j);
    });
}

template<is_global_data_v glb_tensor, is_tile_data_v tl_tensor, typename tile_shapeOffset>
void gen_ND2NN_offset_Impl(
    glb_tensor src,
    tl_tensor dst_tensor,
    tile_shapeOffset &offset,
    const uint32_t i, 
    const uint32_t j) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");
    pto::blkv::blkv_for_1d(tl_tensor::ValidRow, [&] {
        gen_offset_ND2NN_new<tile_shapeOffset>(
            offset.data(), glb_tensor::ColStride, glb_tensor::RowStride,
            tl_tensor::ValidRow, tl_tensor::ValidCol, i, j);
    });
}
