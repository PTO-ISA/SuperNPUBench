#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include "multi_tile.hpp"

// template <typename, typename = void>
// struct blkc_has_data_member : std::false_type {};

// template <typename T>
// struct blkc_has_data_member<T, std::void_t<decltype(std::declval<T>().data)>>
//     : std::true_type {};

// template <typename T>
// inline constexpr bool blkc_has_data_member_v = blkc_has_data_member<T>::value;


// template <typename T, typename IndexT>
// inline void blkc_assign_elem(__vbuf__ T* ptr, IndexT idx, T value) {
//   if constexpr (blkc_has_data_member<T>::value) {
//     ptr[idx].data = value.data;
//   } else {
//     ptr[idx] = value;
//   }
// }

// template <typename T, typename IndexT, typename U>
// inline void blkc_assign_cast(__vbuf__ T* ptr, IndexT idx, U value) {
//   T tmp = static_cast<T>(value);
//   blkc_assign_elem(ptr, idx, tmp);
// }

// #define BLKC_ASSIGN_CAST(tile, idx, value) \
//   blkc_assign_cast(blkv_get_tile_ptr(tile), idx, value)

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_new_max(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSrc::DType src_scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        #ifndef RES_CHECK
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        #else
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0] * src_scale, src_ptr[src_idx_1] * src_scale);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2] * src_scale, src_ptr[src_idx_3] * src_scale);
        #endif
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    #ifndef RES_CHECK
    upd_max = upd_max * src_scale;
    #endif
    new_max_ptr[max_idx] = upd_max; 

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max); 
}

template<typename tileMax, typename tileScale>
void __vec__ flashsoftmax_scale(
                        typename tileScale::TileDType __out__ scale,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileMax::TileDType __in__ new_max
                    ){
    size_t i = blkv_get_index_x();

    size_t idx = i * tileMax::RowStride;
    blkv_get_tile_ptr(scale)[idx] = blkv_fexp(blkv_get_tile_ptr(old_max)[idx] - blkv_get_tile_ptr(new_max)[idx]);
}

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_src_exp_nd(
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::RowStride + i;
    size_t idx_max = j * tileMax::RowStride;
    blkv_get_tile_ptr(src_exp)[idx] = blkv_fexp(blkv_get_tile_ptr(src)[idx]*src_scale - blkv_get_tile_ptr(new_max)[idx_max]);
}

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_src_exp_dn(
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::ColStride + i;
    size_t idx_max = i * tileMax::RowStride;
    blkv_get_tile_ptr(src_exp)[idx] = blkv_fexp(blkv_get_tile_ptr(src)[idx]*src_scale - blkv_get_tile_ptr(new_max)[idx_max]);
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_new_sum(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType   *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t sum_idx = i*tileSum::RowStride;

    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale_ptr[sum_idx];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;

        typename tileSrc::DType exp_src_0 = src_ptr[src_idx_0];
        typename tileSrc::DType exp_src_1 = src_ptr[src_idx_1];
        typename tileSrc::DType exp_src_2 = src_ptr[src_idx_2];
        typename tileSrc::DType exp_src_3 = src_ptr[src_idx_3];
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;        
        upd_sum += exp_src_0123;
    }
    blkv_get_tile_ptr(new_sum)[sum_idx] = upd_sum;
}

template<typename tileOut, typename tileScale>
void __vec__ flashsoftmax_new_out(
                        typename tileOut::TileDType __out__ new_out,
                        const typename tileOut::TileDType __in__ old_out,
                        const typename tileScale::TileDType __in__ scale,
                        const typename tileOut::TileDType __in__ rescale_out
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileOut::RowStride + i;
    size_t idx_scale = j * tileScale::RowStride;

    blkv_get_tile_ptr(new_out)[idx] = (blkv_get_tile_ptr(rescale_out)[idx] + blkv_get_tile_ptr(old_out)[idx]) * blkv_get_tile_ptr(scale)[idx_scale];
}

template <typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void flashsoftmax_dn_sout(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::DType &src_scale
                        ){
    tileScale scale;
    flashsoftmax_new_max<tileSrc, tileMax><<<tileMax::ValidRow, 1, 1>>>(scale.data(), new_max.data(), src.data(), old_max.data(), src_scale);
    if constexpr(tileSrc::isRowMajor){
        flashsoftmax_src_exp_nd<tileSrc, tileMax><<<tileSrc::ValidCol, tileSrc::ValidRow, 1>>>(src_exp.data(), src.data(), new_max.data(), src_scale);
    }else{
        flashsoftmax_src_exp_dn<tileSrc, tileMax><<<tileSrc::ValidRow, tileSrc::ValidCol, 1>>>(src_exp.data(), src.data(), new_max.data(), src_scale);
    }
    flashsoftmax_new_sum<tileSrc, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(new_sum.data(), src_exp.data(), old_sum.data(), scale.data());
    flashsoftmax_new_out<tileOut, tileScale><<<tileOut::ValidCol, tileOut::ValidRow, 1>>>(rescale_out.data(), old_out.data(), scale.data(), rescale_out.data());
}

template<typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_kernel(
                        typename tileOut::TileDType __out__ rescale_out,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileOut::TileDType __in__ old_out,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileOut::TileDType __in__ rescale_old_out,
                        const typename tileSrc::DType src_scale
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileOut::DType *rescale_out_ptr = blkv_get_tile_ptr(rescale_out);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileOut::DType *rescale_old_out_ptr = blkv_get_tile_ptr(rescale_old_out);
    __vbuf__ typename tileOut::DType *old_out_ptr = blkv_get_tile_ptr(old_out);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = old_max_ptr[max_idx];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        #ifndef RES_CHECK
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        #else
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0] * src_scale, src_ptr[src_idx_1] * src_scale);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2] * src_scale, src_ptr[src_idx_3] * src_scale);
        #endif
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    #ifndef RES_CHECK
    upd_max = upd_max * src_scale;
    #endif
    new_max_ptr[max_idx] = upd_max;

    typename tileMax::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);

    size_t sum_idx = i*tileSum::RowStride;
    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = exp_src_0;
        src_exp_ptr[src_idx_1] = exp_src_1;
        src_exp_ptr[src_idx_2] = exp_src_2;
        src_exp_ptr[src_idx_3] = exp_src_3;
    }
    new_sum_ptr[sum_idx] = upd_sum;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileOut::ValidCol;j++){
        size_t out_idx = i * tileOut::RowStride + j * tileOut::ColStride;
        rescale_out_ptr[out_idx] = (rescale_old_out_ptr[out_idx] + old_out_ptr[out_idx]) * scale;
    }
}

template <typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void flashsoftmax_dn_mout(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::DType &src_scale
                        ){
    flashsoftmax_dn_mout_kernel<tileSrc, tileOut, tileMax, tileSum, tileScale><<<tileSrc::ValidRow, 1, 1>>>(rescale_out.data(), new_max.data(), new_sum.data(), src_exp.data(), src.data(), old_out.data(), old_max.data(), old_sum.data(), rescale_out.data(), src_scale);
}

template <is_multi_tile tileSrc, is_multi_tile tileOut, is_multi_tile tileMax, is_multi_tile tileSum, is_multi_tile tileScale>
void flashsoftmax_dn_mout_multi(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::TileType::DType &src_scale
                        ){
    #pragma clang loop unroll(full)
    for (int i = 0; i < tileOut::NumTiles; ++i) {
    flashsoftmax_dn_mout_kernel<typename tileSrc::TileType, typename tileOut::TileType, typename tileMax::TileType, typename tileSum::TileType, typename tileScale::TileType>
        <<<tileSrc::TileType::ValidRow, 1, 1>>>(rescale_out.Tiles[i].data(),
                                                new_max.Tiles[i].data(),
                                                new_sum.Tiles[i].data(),
                                                src_exp.Tiles[i].data(),
                                                src.Tiles[i].data(),
                                                old_out.Tiles[i].data(),
                                                old_max.Tiles[i].data(),
                                                old_sum.Tiles[i].data(),
                                                rescale_out.Tiles[i].data(),
                                                src_scale);
    }
}

template<typename tileSrc, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_norescale_kernel(
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = old_max_ptr[max_idx];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        #ifndef RES_CHECK
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        #else
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0] * src_scale, src_ptr[src_idx_1] * src_scale);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2] * src_scale, src_ptr[src_idx_3] * src_scale);
        #endif
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    #ifndef RES_CHECK
    upd_max = upd_max * src_scale;
    #endif
    new_max_ptr[max_idx] = upd_max;

    typename tileMax::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);

    size_t sum_idx = i*tileSum::RowStride;
    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = exp_src_0;
        src_exp_ptr[src_idx_1] = exp_src_1;
        src_exp_ptr[src_idx_2] = exp_src_2;
        src_exp_ptr[src_idx_3] = exp_src_3;
    }
    new_sum_ptr[sum_idx] = upd_sum;
}

template <typename tileSrc, typename tileMax, typename tileSum, typename tileScale>
void flashsoftmax_dn_mout_norescale(
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::DType &src_scale
                        ){
    flashsoftmax_dn_mout_norescale_kernel<tileSrc, tileMax, tileSum, tileScale><<<tileSrc::ValidRow, 1, 1>>>(new_max.data(), new_sum.data(), src_exp.data(), src.data(), old_max.data(), old_sum.data(), src_scale);
}

template <is_multi_tile tileSrc, is_multi_tile tileMax, is_multi_tile tileSum, is_multi_tile tileScale>
void flashsoftmax_dn_mout_norescale_multi(
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::TileType::DType &src_scale
                        ){
    #pragma clang loop unroll(full)
    for (int i = 0; i < tileSrc::NumTiles; ++i) {
    flashsoftmax_dn_mout_norescale_kernel<typename tileSrc::TileType, typename tileMax::TileType, typename tileSum::TileType, typename tileScale::TileType>
        <<<tileSrc::TileType::ValidRow, 1, 1>>>(new_max.Tiles[i].data(),
                                                new_sum.Tiles[i].data(),
                                                src_exp.Tiles[i].data(),
                                                src.Tiles[i].data(),
                                                old_max.Tiles[i].data(),
                                                old_sum.Tiles[i].data(),
                                                src_scale);
    }
}

template<typename tileO_cast, typename tileO, typename tileSum>
void __vec__ normalize(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ out,
                        const typename tileO::TileDType __in__ rescale_out,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;

    blkv_get_tile_ptr(out_cast)[idx] = static_cast<typename tileO_cast::DType>( (blkv_get_tile_ptr(rescale_out)[idx] + blkv_get_tile_ptr(out)[idx]) /  blkv_get_tile_ptr(sum)[idx_sum] );
}

#include "fa_fp4.h"
#include "fa_opt1.h"
#include "fa_opt2.h"
#include "fa_opt3.h"
#include "fa_opt4.h"
#include "fa_dcore.h"
#include "fa_2d_unroll.h"
#ifdef _2D_UNROLL_PTO
#include "fa_2d_unroll_pto.h"
#endif
#include "fa_template_2d_unroll.h"
#include "fa_dynamic.h"
#include "fa_manual.h"

template <int S, int D, int tM, int tK>
void flashsoftmax(float *input, float *max, float *sum, float *input_scale, uint16_t *bitmask_gm, __half *output) {

    using gmIn = global_tensor<float, ColMajor<S, S>>;
    using gmMax = global_tensor<float, RowMajor<S, 1>>;
    using gmSum = global_tensor<float, RowMajor<S, 1>>;
    using gmOut = global_tensor<__half, RowMajor<S, D>>;

    using tileIn = Tile<Location::Vec, float, tM, tK, BLayout::ColMajor>;
    using tileMax = Tile<Location::Vec, float, tM, 16, BLayout::RowMajor, tM, 1>;
    using tileSum = Tile<Location::Vec, float, tM, 16, BLayout::RowMajor, tM, 1>;
    using tileScale = Tile<Location::Vec, float, tM, 16, BLayout::RowMajor, tM, 1>;
    
    using tileO = Tile<Location::Vec, float, tM, D, BLayout::RowMajor>;
    using tileO_cast = Tile<Location::Vec, __half, tM, D, BLayout::RowMajor>;

    const int Bm = S/tM; 
    const int Bk = S/tK;

    for(int i=0;i<Bm;i++){
        tileMax tMax(-20000);
        tileSum tSum(0);
        tileO tO(0), tRescaleO(0);

        #pragma clang loop unroll(full)
        for(int j=0;j<Bk;j++){
            uint32_t offset = i * tM * S + j * tK;  
            gmIn gIn(input+offset);
            tileIn tIn;
            TCOPYIN(tIn, gIn);

            tileMax tNewMax;
            tileSum tNewSum;
            tileO tNewO;
            tileIn tExpW;
            flashsoftmax_dn_sout<tileIn, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tIn, tO, tMax, tSum, input_scale[0]);

            //更新全局Max/Sum/Out
            tMax = tNewMax;
            tSum = tNewSum;
        }

        TADD(tO, tO, tRescaleO);

        uint32_t offset = i * tM;
        gmMax gMax(max+offset);
        TCOPYOUT(gMax, tMax);

        gmSum gSum(sum+offset);
        TCOPYOUT(gSum, tSum);

        offset = i * tM * D;
        gmOut gOut(output + offset);
        tileO_cast tO_cast;
        TCAST(tO_cast, tO);
        TCOPYOUT(gOut, tO_cast);
    }
}
