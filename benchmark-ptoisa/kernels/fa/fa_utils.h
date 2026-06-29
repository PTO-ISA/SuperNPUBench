// FA utility functions shared across different implementations

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_cast_kernel(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
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
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    upd_max = upd_max * src_scale;
    new_max_ptr[max_idx] = upd_max;

    typename tileScale::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);

    size_t scale_idx = i*tileScale::RowStride;
    scale_ptr[scale_idx] = scale;

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
        src_exp_ptr[src_idx_0] = static_cast<typename tileSrc_cast::DType>(exp_src_0);
        src_exp_ptr[src_idx_1] = static_cast<typename tileSrc_cast::DType>(exp_src_1);
        src_exp_ptr[src_idx_2] = static_cast<typename tileSrc_cast::DType>(exp_src_2);
        src_exp_ptr[src_idx_3] = static_cast<typename tileSrc_cast::DType>(exp_src_3);
    }
    new_sum_ptr[sum_idx] = upd_sum;
}

template<typename tileO, typename tileScale>
void __vec__ global_update(
                        typename tileO::TileDType __out__ update_out,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ pv,
                        const typename tileScale::TileDType __in__ scale
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_scale = i * tileScale::RowStride;

    typename tileScale::DType local_scale = blkv_get_tile_ptr(scale)[idx_scale];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileO::ValidCol;j++){
        size_t idx = j * tileO::ColStride + i * tileO::RowStride;
        blkv_get_tile_ptr(update_out)[idx] = blkv_get_tile_ptr(old_out)[idx] * local_scale + blkv_get_tile_ptr(pv)[idx] ;
    }
}

template<typename tileO_cast, typename tileO, typename tileSum, typename tileScale>
void __vec__ normalize_with_last_update(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ last_pv,
                        const typename tileScale::TileDType __in__ last_scale,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;
    size_t idx_scale = j * tileScale::RowStride;

    blkv_get_tile_ptr(out_cast)[idx] = 
        (blkv_get_tile_ptr(last_pv)[idx] +
         blkv_get_tile_ptr(old_out)[idx] * blkv_get_tile_ptr(last_scale)[idx_scale]) /
        blkv_get_tile_ptr(sum)[idx_sum];
}

template<typename tileO_cast, typename tileO, typename tileSum, typename tileScale>
void __vec__ normalize_with_last_update_nocast(
                        typename tileO::TileDType __out__ out,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ last_pv,
                        const typename tileScale::TileDType __in__ last_scale,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;
    size_t idx_scale = j * tileScale::RowStride;

    blkv_get_tile_ptr(out)[idx] = 
        (blkv_get_tile_ptr(last_pv)[idx] +
         blkv_get_tile_ptr(old_out)[idx] * blkv_get_tile_ptr(last_scale)[idx_scale]) /
        blkv_get_tile_ptr(sum)[idx_sum];
}
