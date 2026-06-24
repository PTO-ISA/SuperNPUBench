template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_cast_kernel_template(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ local_max,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *local_max_ptr = blkv_get_tile_ptr(local_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = blkv_max(old_max_ptr[max_idx], local_max_ptr[max_idx]);

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
        typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] - upd_max);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] - upd_max);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] - upd_max);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] - upd_max);
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

// ------- Ydim==2 --------
template<typename tileSrc, typename tileMax>
void __vec__ new_max_2src_template(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileMax::TileDType __in__ local_max0,
                        const typename tileMax::TileDType __in__ local_max1,
                        const typename tileMax::TileDType __in__ old_max
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *local_max0_ptr = blkv_get_tile_ptr(local_max0);
    __vbuf__ typename tileMax::DType *local_max1_ptr = blkv_get_tile_ptr(local_max1);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    upd_max = blkv_max(upd_max, local_max0_ptr[max_idx]);
    upd_max = blkv_max(upd_max, local_max1_ptr[max_idx]);

    new_max_ptr[max_idx] = upd_max;

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max);
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax>
void __vec__ src_exp_2src_template(
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::ColStride + i;
    size_t idx_max = i * tileMax::RowStride;
    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    blkv_get_tile_ptr(src_exp0)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src0)[idx] - new_max_val));
    blkv_get_tile_ptr(src_exp1)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src1)[idx] - new_max_val));
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ new_sum_2src_template(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType   *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType   *src1_ptr = blkv_get_tile_ptr(src1);
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

        typename tileSrc::DType s0_exp_src_0 = src0_ptr[src_idx_0];
        typename tileSrc::DType s0_exp_src_1 = src0_ptr[src_idx_1];
        typename tileSrc::DType s0_exp_src_2 = src0_ptr[src_idx_2];
        typename tileSrc::DType s0_exp_src_3 = src0_ptr[src_idx_3];
        typename tileSrc::DType s0_exp_src_01 = s0_exp_src_0 + s0_exp_src_1;
        typename tileSrc::DType s0_exp_src_23 = s0_exp_src_2 + s0_exp_src_3;
        typename tileSrc::DType s0_exp_src_0123 = s0_exp_src_01 + s0_exp_src_23;

        typename tileSrc::DType s1_exp_src_0 = src1_ptr[src_idx_0];
        typename tileSrc::DType s1_exp_src_1 = src1_ptr[src_idx_1];
        typename tileSrc::DType s1_exp_src_2 = src1_ptr[src_idx_2];
        typename tileSrc::DType s1_exp_src_3 = src1_ptr[src_idx_3];
        typename tileSrc::DType s1_exp_src_01 = s1_exp_src_0 + s1_exp_src_1;
        typename tileSrc::DType s1_exp_src_23 = s1_exp_src_2 + s1_exp_src_3;
        typename tileSrc::DType s1_exp_src_0123 = s1_exp_src_01 + s1_exp_src_23;

        upd_sum += (s0_exp_src_0123 + s1_exp_src_0123);
    }
    blkv_get_tile_ptr(new_sum)[sum_idx] = upd_sum;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ src_exp_2src_with_new_sum_template(
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_max = i * tileMax::RowStride;
    size_t idx_sum = i * tileSum::RowStride;

    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr   = blkv_get_tile_ptr(scale);

    typename tileSum::DType upd_sum = old_sum_ptr[idx_sum] * scale_ptr[idx_sum];

    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t idx_0 = j * tileSrc::ColStride + i;
        size_t idx_1 = (j+1) * tileSrc::ColStride + i;
        size_t idx_2 = (j+2) * tileSrc::ColStride + i;
        size_t idx_3 = (j+3) * tileSrc::ColStride + i;
        typename tileSrc::DType src0_exp0 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_0] - new_max_val);
        typename tileSrc::DType src0_exp1 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_1] - new_max_val);
        typename tileSrc::DType src0_exp2 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_2] - new_max_val);
        typename tileSrc::DType src0_exp3 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_3] - new_max_val);

        BLKC_ASSIGN_CAST(src_exp0, idx_0, src0_exp0);
        BLKC_ASSIGN_CAST(src_exp0, idx_1, src0_exp1);
        BLKC_ASSIGN_CAST(src_exp0, idx_2, src0_exp2);
        BLKC_ASSIGN_CAST(src_exp0, idx_3, src0_exp3);
        typename tileSum::DType src0_exp_sum = src0_exp0 + src0_exp1 + src0_exp2 + src0_exp3;

        typename tileSrc::DType src1_exp0 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_0] - new_max_val);
        typename tileSrc::DType src1_exp1 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_1] - new_max_val);
        typename tileSrc::DType src1_exp2 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_2] - new_max_val);
        typename tileSrc::DType src1_exp3 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_3] - new_max_val);

        BLKC_ASSIGN_CAST(src_exp1, idx_0, src1_exp0);
        BLKC_ASSIGN_CAST(src_exp1, idx_1, src1_exp1);
        BLKC_ASSIGN_CAST(src_exp1, idx_2, src1_exp2);
        BLKC_ASSIGN_CAST(src_exp1, idx_3, src1_exp3);
        typename tileSum::DType src1_exp_sum = src1_exp0 + src1_exp1 + src1_exp2 + src1_exp3;

        upd_sum += src0_exp_sum + src1_exp_sum;
    }
    blkv_get_tile_ptr(new_sum)[idx_sum] = upd_sum;
}
// ------- Ydim==2 --------

// ------- Ydim==4 --------
template<typename tileSrc, typename tileMax>
void __vec__ new_max_4src_template(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileMax::TileDType __in__ local_max0,
                        const typename tileMax::TileDType __in__ local_max1,
                        const typename tileMax::TileDType __in__ local_max2,
                        const typename tileMax::TileDType __in__ local_max3,
                        const typename tileMax::TileDType __in__ old_max
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *local_max0_ptr = blkv_get_tile_ptr(local_max0);
    __vbuf__ typename tileMax::DType *local_max1_ptr = blkv_get_tile_ptr(local_max1);
    __vbuf__ typename tileMax::DType *local_max2_ptr = blkv_get_tile_ptr(local_max2);
    __vbuf__ typename tileMax::DType *local_max3_ptr = blkv_get_tile_ptr(local_max3);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    typename tileMax::DType local_max_01 = blkv_max(local_max0_ptr[max_idx], local_max1_ptr[max_idx]);
    typename tileMax::DType local_max_23 = blkv_max(local_max2_ptr[max_idx], local_max3_ptr[max_idx]);

    typename tileMax::DType local_max_0123 = blkv_max(local_max_01, local_max_23);
    upd_max = blkv_max(upd_max, local_max_0123);
    new_max_ptr[max_idx] = upd_max;

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max);
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax>
void __vec__ src_exp_4src_template(
                        typename tileSrc::TileDType __out__ src_exp0,
                        typename tileSrc::TileDType __out__ src_exp1,
                        typename tileSrc::TileDType __out__ src_exp2,
                        typename tileSrc::TileDType __out__ src_exp3,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileMax::TileDType __in__ new_max
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::ColStride + i;
    size_t idx_max = i * tileMax::RowStride;
    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    blkv_get_tile_ptr(src_exp0)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src0)[idx] - new_max_val));
    blkv_get_tile_ptr(src_exp1)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src1)[idx] - new_max_val));
    blkv_get_tile_ptr(src_exp2)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src2)[idx] - new_max_val));
    blkv_get_tile_ptr(src_exp3)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src3)[idx] - new_max_val));
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum>
void __vec__ src_exp_2src_with_local_sum_template(
                        typename tileSum::TileDType __out__ local_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_max = i * tileMax::RowStride;

    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    typename tileSum::DType upd_sum = 0;
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t idx_0 = j * tileSrc::ColStride + i;
        size_t idx_1 = (j+1) * tileSrc::ColStride + i;
        size_t idx_2 = (j+2) * tileSrc::ColStride + i;
        size_t idx_3 = (j+3) * tileSrc::ColStride + i;
        typename tileSrc::DType src0_exp0 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_0] - new_max_val);
        typename tileSrc::DType src0_exp1 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_1] - new_max_val);
        typename tileSrc::DType src0_exp2 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_2] - new_max_val);
        typename tileSrc::DType src0_exp3 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_3] - new_max_val);

        BLKC_ASSIGN_CAST(src_exp0, idx_0, src0_exp0);
        BLKC_ASSIGN_CAST(src_exp0, idx_1, src0_exp1);
        BLKC_ASSIGN_CAST(src_exp0, idx_2, src0_exp2);
        BLKC_ASSIGN_CAST(src_exp0, idx_3, src0_exp3);
        typename tileSum::DType src0_exp_sum = src0_exp0 + src0_exp1 + src0_exp2 + src0_exp3;

        typename tileSrc::DType src1_exp0 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_0] - new_max_val);
        typename tileSrc::DType src1_exp1 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_1] - new_max_val);
        typename tileSrc::DType src1_exp2 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_2] - new_max_val);
        typename tileSrc::DType src1_exp3 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_3] - new_max_val);

        BLKC_ASSIGN_CAST(src_exp1, idx_0, src1_exp0);
        BLKC_ASSIGN_CAST(src_exp1, idx_1, src1_exp1);
        BLKC_ASSIGN_CAST(src_exp1, idx_2, src1_exp2);
        BLKC_ASSIGN_CAST(src_exp1, idx_3, src1_exp3);
        typename tileSum::DType src1_exp_sum = src1_exp0 + src1_exp1 + src1_exp2 + src1_exp3;

        upd_sum += src0_exp_sum + src1_exp_sum;
    }
    size_t idx_sum = i * tileSum::RowStride;
    blkv_get_tile_ptr(local_sum)[idx_sum] = upd_sum;
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ new_sum_4src_template(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType   *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType   *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType   *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType   *src3_ptr = blkv_get_tile_ptr(src3);
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

        typename tileSrc::DType s0_exp_src_0 = src0_ptr[src_idx_0];
        typename tileSrc::DType s0_exp_src_1 = src0_ptr[src_idx_1];
        typename tileSrc::DType s0_exp_src_2 = src0_ptr[src_idx_2];
        typename tileSrc::DType s0_exp_src_3 = src0_ptr[src_idx_3];
        typename tileSrc::DType s0_exp_src_01 = s0_exp_src_0 + s0_exp_src_1;
        typename tileSrc::DType s0_exp_src_23 = s0_exp_src_2 + s0_exp_src_3;
        typename tileSrc::DType s0_exp_src_0123 = s0_exp_src_01 + s0_exp_src_23;

        typename tileSrc::DType s1_exp_src_0 = src1_ptr[src_idx_0];
        typename tileSrc::DType s1_exp_src_1 = src1_ptr[src_idx_1];
        typename tileSrc::DType s1_exp_src_2 = src1_ptr[src_idx_2];
        typename tileSrc::DType s1_exp_src_3 = src1_ptr[src_idx_3];
        typename tileSrc::DType s1_exp_src_01 = s1_exp_src_0 + s1_exp_src_1;
        typename tileSrc::DType s1_exp_src_23 = s1_exp_src_2 + s1_exp_src_3;
        typename tileSrc::DType s1_exp_src_0123 = s1_exp_src_01 + s1_exp_src_23;

        typename tileSrc::DType s2_exp_src_0 = src2_ptr[src_idx_0];
        typename tileSrc::DType s2_exp_src_1 = src2_ptr[src_idx_1];
        typename tileSrc::DType s2_exp_src_2 = src2_ptr[src_idx_2];
        typename tileSrc::DType s2_exp_src_3 = src2_ptr[src_idx_3];
        typename tileSrc::DType s2_exp_src_01 = s2_exp_src_0 + s2_exp_src_1;
        typename tileSrc::DType s2_exp_src_23 = s2_exp_src_2 + s2_exp_src_3;
        typename tileSrc::DType s2_exp_src_0123 = s2_exp_src_01 + s2_exp_src_23;

        typename tileSrc::DType s3_exp_src_0 = src3_ptr[src_idx_0];
        typename tileSrc::DType s3_exp_src_1 = src3_ptr[src_idx_1];
        typename tileSrc::DType s3_exp_src_2 = src3_ptr[src_idx_2];
        typename tileSrc::DType s3_exp_src_3 = src3_ptr[src_idx_3];
        typename tileSrc::DType s3_exp_src_01 = s3_exp_src_0 + s3_exp_src_1;
        typename tileSrc::DType s3_exp_src_23 = s3_exp_src_2 + s3_exp_src_3;
        typename tileSrc::DType s3_exp_src_0123 = s3_exp_src_01 + s3_exp_src_23;

        upd_sum += (s0_exp_src_0123 + s1_exp_src_0123 + s2_exp_src_0123 + s3_exp_src_0123);
    }
    blkv_get_tile_ptr(new_sum)[sum_idx] = upd_sum;
}
// ------- Ydim==4 --------

// ------- Ydim==8 --------
template<typename tileSrc, typename tileMax>
void __vec__ local_max_4src_template(
                        typename tileMax::TileDType __out__ local_max,
                        const typename tileMax::TileDType __in__ local_max0,
                        const typename tileMax::TileDType __in__ local_max1,
                        const typename tileMax::TileDType __in__ local_max2,
                        const typename tileMax::TileDType __in__ local_max3
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *local_max0_ptr = blkv_get_tile_ptr(local_max0);
    __vbuf__ typename tileMax::DType *local_max1_ptr = blkv_get_tile_ptr(local_max1);
    __vbuf__ typename tileMax::DType *local_max2_ptr = blkv_get_tile_ptr(local_max2);
    __vbuf__ typename tileMax::DType *local_max3_ptr = blkv_get_tile_ptr(local_max3);
    __vbuf__ typename tileMax::DType *local_max_ptr = blkv_get_tile_ptr(local_max);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = -1e30f;

    typename tileMax::DType local_max_01 = blkv_max(local_max0_ptr[max_idx], local_max1_ptr[max_idx]);
    typename tileMax::DType local_max_23 = blkv_max(local_max2_ptr[max_idx], local_max3_ptr[max_idx]);

    typename tileMax::DType local_max_0123 = blkv_max(local_max_01, local_max_23);

    upd_max = blkv_max(upd_max, local_max_0123);
    local_max_ptr[max_idx] = upd_max;
}

template<typename tileSrc, typename tileSum>
void __vec__ local_sum_4src_template(
                        typename tileSum::TileDType __out__ local_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSum::DType   *local_sum_ptr = blkv_get_tile_ptr(local_sum);
    __vbuf__ typename tileSrc::DType   *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType   *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType   *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType   *src3_ptr = blkv_get_tile_ptr(src3);

    size_t sum_idx = i*tileSum::RowStride;

    typename tileSum::DType upd_sum = 0;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;

        typename tileSrc::DType s0_exp_src_0 = src0_ptr[src_idx_0];
        typename tileSrc::DType s0_exp_src_1 = src0_ptr[src_idx_1];
        typename tileSrc::DType s0_exp_src_2 = src0_ptr[src_idx_2];
        typename tileSrc::DType s0_exp_src_3 = src0_ptr[src_idx_3];
        typename tileSrc::DType s0_exp_src_01 = s0_exp_src_0 + s0_exp_src_1;
        typename tileSrc::DType s0_exp_src_23 = s0_exp_src_2 + s0_exp_src_3;
        typename tileSrc::DType s0_exp_src_0123 = s0_exp_src_01 + s0_exp_src_23;

        typename tileSrc::DType s1_exp_src_0 = src1_ptr[src_idx_0];
        typename tileSrc::DType s1_exp_src_1 = src1_ptr[src_idx_1];
        typename tileSrc::DType s1_exp_src_2 = src1_ptr[src_idx_2];
        typename tileSrc::DType s1_exp_src_3 = src1_ptr[src_idx_3];
        typename tileSrc::DType s1_exp_src_01 = s1_exp_src_0 + s1_exp_src_1;
        typename tileSrc::DType s1_exp_src_23 = s1_exp_src_2 + s1_exp_src_3;
        typename tileSrc::DType s1_exp_src_0123 = s1_exp_src_01 + s1_exp_src_23;

        typename tileSrc::DType s2_exp_src_0 = src2_ptr[src_idx_0];
        typename tileSrc::DType s2_exp_src_1 = src2_ptr[src_idx_1];
        typename tileSrc::DType s2_exp_src_2 = src2_ptr[src_idx_2];
        typename tileSrc::DType s2_exp_src_3 = src2_ptr[src_idx_3];
        typename tileSrc::DType s2_exp_src_01 = s2_exp_src_0 + s2_exp_src_1;
        typename tileSrc::DType s2_exp_src_23 = s2_exp_src_2 + s2_exp_src_3;
        typename tileSrc::DType s2_exp_src_0123 = s2_exp_src_01 + s2_exp_src_23;

        typename tileSrc::DType s3_exp_src_0 = src3_ptr[src_idx_0];
        typename tileSrc::DType s3_exp_src_1 = src3_ptr[src_idx_1];
        typename tileSrc::DType s3_exp_src_2 = src3_ptr[src_idx_2];
        typename tileSrc::DType s3_exp_src_3 = src3_ptr[src_idx_3];
        typename tileSrc::DType s3_exp_src_01 = s3_exp_src_0 + s3_exp_src_1;
        typename tileSrc::DType s3_exp_src_23 = s3_exp_src_2 + s3_exp_src_3;
        typename tileSrc::DType s3_exp_src_0123 = s3_exp_src_01 + s3_exp_src_23;

        upd_sum += (s0_exp_src_0123 + s1_exp_src_0123 + s2_exp_src_0123 + s3_exp_src_0123);
    }
    blkv_get_tile_ptr(local_sum)[sum_idx] = upd_sum;
}

template<typename tileScale, typename tileMax>
void __vec__ new_max_of_2_loc_max_template(
                        typename tileScale::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileMax::TileDType __in__ local_max_0,
                        const typename tileMax::TileDType __in__ local_max_1,
                        const typename tileMax::TileDType __in__ old_max
                                ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *local_max_0_ptr = blkv_get_tile_ptr(local_max_0);
    __vbuf__ typename tileMax::DType *local_max_1_ptr = blkv_get_tile_ptr(local_max_1);
    __vbuf__ typename tileMax::DType *new_max_ptr     = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr     = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;
    typename tileMax::DType local_max_01 = blkv_max(local_max_0_ptr[max_idx], local_max_1_ptr[max_idx]);
    upd_max = blkv_max(upd_max, local_max_01);
    new_max_ptr[max_idx] = upd_max;
    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max);
}
template<typename tileScale, typename tileSum>
void __vec__ new_sum_of_2_loc_sum_template(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSum::TileDType __in__ local_sum_0,
                        const typename tileSum::TileDType __in__ local_sum_1,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){

    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSum::DType   *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSum::DType   *local_sum_0_ptr = blkv_get_tile_ptr(local_sum_0);
    __vbuf__ typename tileSum::DType   *local_sum_1_ptr = blkv_get_tile_ptr(local_sum_1);
    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t sum_idx = i*tileSum::RowStride;

    new_sum_ptr[sum_idx] = old_sum_ptr[sum_idx] * scale_ptr[sum_idx] + local_sum_0_ptr[sum_idx] + local_sum_1_ptr[sum_idx];
}
// ------- Ydim==8 --------

// ------- Ydim==16 --------
template<typename tileScale, typename tileMax>
void __vec__ new_max_of_4_loc_max_template(
                        typename tileScale::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileMax::TileDType __in__ local_max_0,
                        const typename tileMax::TileDType __in__ local_max_1,
                        const typename tileMax::TileDType __in__ local_max_2,
                        const typename tileMax::TileDType __in__ local_max_3,
                        const typename tileMax::TileDType __in__ old_max
                                ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *local_max_0_ptr = blkv_get_tile_ptr(local_max_0);
    __vbuf__ typename tileMax::DType *local_max_1_ptr = blkv_get_tile_ptr(local_max_1);
    __vbuf__ typename tileMax::DType *local_max_2_ptr = blkv_get_tile_ptr(local_max_2);
    __vbuf__ typename tileMax::DType *local_max_3_ptr = blkv_get_tile_ptr(local_max_3);
    __vbuf__ typename tileMax::DType *new_max_ptr     = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr     = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;
    typename tileMax::DType local_max_01 = blkv_max(local_max_0_ptr[max_idx], local_max_1_ptr[max_idx]);
    typename tileMax::DType local_max_23 = blkv_max(local_max_2_ptr[max_idx], local_max_3_ptr[max_idx]);
    typename tileMax::DType local_max_0123 = blkv_max(local_max_01, local_max_23);
    upd_max = blkv_max(upd_max, local_max_0123);
    new_max_ptr[max_idx] = upd_max;
    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max);
}
template<typename tileScale, typename tileSum>
void __vec__ new_sum_of_4_loc_sum_template(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSum::TileDType __in__ local_sum_0,
                        const typename tileSum::TileDType __in__ local_sum_1,
                        const typename tileSum::TileDType __in__ local_sum_2,
                        const typename tileSum::TileDType __in__ local_sum_3,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){

    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSum::DType   *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSum::DType   *local_sum_0_ptr = blkv_get_tile_ptr(local_sum_0);
    __vbuf__ typename tileSum::DType   *local_sum_1_ptr = blkv_get_tile_ptr(local_sum_1);
    __vbuf__ typename tileSum::DType   *local_sum_2_ptr = blkv_get_tile_ptr(local_sum_2);
    __vbuf__ typename tileSum::DType   *local_sum_3_ptr = blkv_get_tile_ptr(local_sum_3);
    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t sum_idx = i*tileSum::RowStride;

    new_sum_ptr[sum_idx] = old_sum_ptr[sum_idx] * scale_ptr[sum_idx] +
                           local_sum_0_ptr[sum_idx] + local_sum_1_ptr[sum_idx] +
                           local_sum_2_ptr[sum_idx] + local_sum_3_ptr[sum_idx];
}
// ------- Ydim==16 --------

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_template_2d_unroll(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_cast = Tile<Location::Vec, typename tileW_type<dtype>::DType, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];

        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                TLOAD(tQ[x], gQ);
            }
        #endif

        tileMax tMax[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tSum[x], 0);
        }

        tileO_out tPV_out;
        tileO tO[Xdim], tPV[Xdim];
        tileScale tScale[Xdim];

        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j+=Ydim) {

            tileK tK[Ydim];

            #ifdef MULTI_LDST
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y+=4){
                    auto gK = gIterK(0, j+y);
                    TLOAD4_DN2ZN(tK[y+3], tK[y+2], tK[y+1], tK[y], gK);
                }
            #else
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    auto gK = gIterK(0, j+y);
                    TLOAD(tK[y], gK);
                }
            #endif

            tileMax tLocalMax[Xdim][Ydim];
            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    MATMUL(tW_out, tQ[x], tK[y]);
                    // Nz -> ColMajor
                    ACCCVT_RMAX_SCALE_NZ2DN(tLocalMax[x][y], tW[x][y], tW_out, scale);
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];

            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    flashsoftmax_dn_mout_cast_kernel_template<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                    tScale[x].data(),
                                                    tNewMax[x].data(),
                                                    tNewSum[x].data(),
                                                    tExpW[x][0].data(),
                                                    tW[x][0].data(),
                                                    tLocalMax[x][0].data(),
                                                    tMax[x].data(),
                                                    tSum[x].data()
                                                    );
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src_template<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tLocalMax[x][0].data(), tLocalMax[x][1].data(),
                                                                tMax[x].data());
                    // src_exp_2src_template<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tW[x][0].data(), tW[x][1].data(),
                    //                                             tNewMax[x].data());
                    // new_sum_2src_template<tileW, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                    //                                             tNewSum[x].data(),
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tSum[x].data(),
                    //                                             tScale[x].data()
                    //                                             );
                    src_exp_2src_with_new_sum_template<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data());

                }
            #elif Ydim == 4
                tileSum tLocalSum[Xdim][2];
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_4src_template<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(),
                                                                tMax[x].data());
                    // src_exp_4src_template<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                    //                                             tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                    //                                             tNewMax[x].data());
                    src_exp_2src_with_local_sum_template<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                                                   tW[x][0].data(), tW[x][1].data(), tNewMax[x].data());
                    src_exp_2src_with_local_sum_template<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                                                   tW[x][2].data(), tW[x][3].data(), tNewMax[x].data());
                    // new_sum_4src_template<tileW, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                    //                                             tNewSum[x].data(),
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                    //                                             tSum[x].data(),
                    //                                             tScale[x].data()
                    //                                             );
                    new_sum_of_2_loc_sum_template<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 8
                tileMax tLocalMax4[Xdim][2];
                tileSum tLocalSum[Xdim][2];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src_template<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax4[x][k].data(), tLocalMax[x][4*k].data(), tLocalMax[x][4*k+1].data(), tLocalMax[x][4*k+2].data(), tLocalMax[x][4*k+3].data());
                    }
                    new_max_of_2_loc_max_template<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax4[x][0].data(), tLocalMax4[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        src_exp_4src_template<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data());
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_sum_4src_template<tileW, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_2_loc_sum_template<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax4[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    for(int k=0;k<4;k++){
                        local_max_4src_template<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax4[x][k].data(), tLocalMax[x][4*k].data(), tLocalMax[x][4*k+1].data(), tLocalMax[x][4*k+2].data(), tLocalMax[x][4*k+3].data());
                    }
                    new_max_of_4_loc_max_template<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax4[x][0].data(), tLocalMax4[x][1].data(), tLocalMax4[x][2].data(), tLocalMax4[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src_template<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data());
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src_template<tileW, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum_template<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
            #endif

            tileV tV[Ydim];

            #ifdef MULTI_LDST
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y+=4){
                    auto gV = gIterV(j+y, 0);
                    TLOAD4_ND2ZN(tV[y+3], tV[y+2], tV[y+1], tV[y], gV);
                }
            #else
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    auto gV = gIterV(j+y, 0);
                    TLOAD(tV[y], gV);
                }
            #endif

            // ColMajor -> Nz
            // 计算当前块的加权输出: O_j = W * V
            tileW_left tW_left[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    if constexpr (type_traits<typename tileO_cast::DType>::bits == 4) {
                        tileW_cast tW_left_tmp;
                        TMOV_DN2NZ(tW_left_tmp, tExpW[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		    } else {
                        TMOV_DN2NZ(tW_left[x][y], tExpW[x][y]);
		    }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMUL(tPV_out, tW_left[x][y], tV[y]);
                    }else{
                        MATMACC(tPV_out, tW_left[x][y], tV[y]);
                    }
                }
                TCVT(tPV[x], tPV_out);
            }

            // update
            if(j==0){
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    tO[x] = tPV[x];
                }
            }else if(j<(Kb-Ydim)){
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
                }
            }
            // 更新最大值状态
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        tileO_cast tO_cast[Xdim];
        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            if constexpr (type_traits<typename tileO_cast::DType>::bits == 4) {
            normalize_with_last_update_nocast<tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
            TMOV_NORM(tO_cast[x], tO[x]);
            } else {
            normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
            }
        }
        // 写回全局内存

        #ifdef MULTI_LDST
            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; x+=2) {
                auto dstO = gIterO(i+x, 0);
                TSTORE2_DN2DN(dstO, tO_cast[x+1], tO_cast[x]);
            }
        #else
            #pragma clang loop unroll(full)
            for (int x = 0; x < Xdim; ++x) {
                auto dstO = gIterO(i+x, 0);
                TSTORE(dstO, tO_cast[x]);
            }
        #endif

    }
}
