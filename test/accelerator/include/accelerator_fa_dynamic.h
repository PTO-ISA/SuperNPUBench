#ifndef __vbuf__
#define __vbuf__
#endif

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale, const int SrcCols>
void __vec__ flashsoftmax_dn_mout_cast_kernel_dynamic(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
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

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<SrcCols;j+=4){
    // //for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<SrcCols;j+=4){
    // //for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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

// ------- Ydim==2 --------
template<typename tileSrc, typename tileMax>
void __vec__ new_max_2src_dynamic(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileMax::DType s0_max_01 = blkv_max(src0_ptr[src_idx_0], src0_ptr[src_idx_1] );
        typename tileMax::DType s0_max_23 = blkv_max(src0_ptr[src_idx_2], src0_ptr[src_idx_3] );
        typename tileMax::DType s0_max_0123 = blkv_max(s0_max_01, s0_max_23);

        typename tileMax::DType s1_max_01 = blkv_max(src1_ptr[src_idx_0], src1_ptr[src_idx_1] );
        typename tileMax::DType s1_max_23 = blkv_max(src1_ptr[src_idx_2], src1_ptr[src_idx_3] );
        typename tileMax::DType s1_max_0123 = blkv_max(s1_max_01, s1_max_23);

        upd_max = blkv_max(upd_max, s0_max_0123);
        upd_max = blkv_max(upd_max, s1_max_0123);
    }
    upd_max = upd_max * src_scale;
    new_max_ptr[max_idx] = upd_max;

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max);
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ new_sum_2src_dynamic(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale,
                        const size_t src_valid_col
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType   *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType   *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t sum_idx = i*tileSum::RowStride;

    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale_ptr[sum_idx];

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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
// ------- Ydim==2 --------

// ------- Ydim==4 --------
template<typename tileSrc, typename tileMax>
void __vec__ new_max_4src_dynamic(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType *src3_ptr = blkv_get_tile_ptr(src3);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileMax::DType s0_max_01 = blkv_max(src0_ptr[src_idx_0], src0_ptr[src_idx_1] );
        typename tileMax::DType s0_max_23 = blkv_max(src0_ptr[src_idx_2], src0_ptr[src_idx_3] );
        typename tileMax::DType s0_max_0123 = blkv_max(s0_max_01, s0_max_23);

        typename tileMax::DType s1_max_01 = blkv_max(src1_ptr[src_idx_0], src1_ptr[src_idx_1] );
        typename tileMax::DType s1_max_23 = blkv_max(src1_ptr[src_idx_2], src1_ptr[src_idx_3] );
        typename tileMax::DType s1_max_0123 = blkv_max(s1_max_01, s1_max_23);

        typename tileMax::DType s2_max_01 = blkv_max(src2_ptr[src_idx_0], src2_ptr[src_idx_1] );
        typename tileMax::DType s2_max_23 = blkv_max(src2_ptr[src_idx_2], src2_ptr[src_idx_3] );
        typename tileMax::DType s2_max_0123 = blkv_max(s2_max_01, s2_max_23);

        typename tileMax::DType s3_max_01 = blkv_max(src3_ptr[src_idx_0], src3_ptr[src_idx_1] );
        typename tileMax::DType s3_max_23 = blkv_max(src3_ptr[src_idx_2], src3_ptr[src_idx_3] );
        typename tileMax::DType s3_max_0123 = blkv_max(s3_max_01, s3_max_23);

        upd_max = blkv_max(upd_max, s0_max_0123);
        upd_max = blkv_max(upd_max, s1_max_0123);
        upd_max = blkv_max(upd_max, s2_max_0123);
        upd_max = blkv_max(upd_max, s3_max_0123);
    }
    upd_max = upd_max * src_scale;
    new_max_ptr[max_idx] = upd_max; 

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max); 
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ new_sum_4src_dynamic(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale,
                        const size_t src_valid_col
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

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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
void __vec__ local_max_4src_dynamic(
                        typename tileMax::TileDType __out__ local_max,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType *src3_ptr = blkv_get_tile_ptr(src3);
    __vbuf__ typename tileMax::DType *local_max_ptr = blkv_get_tile_ptr(local_max);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = -1e30f;

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileMax::DType s0_max_01 = blkv_max(src0_ptr[src_idx_0], src0_ptr[src_idx_1] );
        typename tileMax::DType s0_max_23 = blkv_max(src0_ptr[src_idx_2], src0_ptr[src_idx_3] );
        typename tileMax::DType s0_max_0123 = blkv_max(s0_max_01, s0_max_23);

        typename tileMax::DType s1_max_01 = blkv_max(src1_ptr[src_idx_0], src1_ptr[src_idx_1] );
        typename tileMax::DType s1_max_23 = blkv_max(src1_ptr[src_idx_2], src1_ptr[src_idx_3] );
        typename tileMax::DType s1_max_0123 = blkv_max(s1_max_01, s1_max_23);

        typename tileMax::DType s2_max_01 = blkv_max(src2_ptr[src_idx_0], src2_ptr[src_idx_1] );
        typename tileMax::DType s2_max_23 = blkv_max(src2_ptr[src_idx_2], src2_ptr[src_idx_3] );
        typename tileMax::DType s2_max_0123 = blkv_max(s2_max_01, s2_max_23);

        typename tileMax::DType s3_max_01 = blkv_max(src3_ptr[src_idx_0], src3_ptr[src_idx_1] );
        typename tileMax::DType s3_max_23 = blkv_max(src3_ptr[src_idx_2], src3_ptr[src_idx_3] );
        typename tileMax::DType s3_max_0123 = blkv_max(s3_max_01, s3_max_23);

        typename tileMax::DType s01_max = blkv_max(s0_max_0123, s1_max_0123);
        typename tileMax::DType s23_max = blkv_max(s2_max_0123, s3_max_0123);
        typename tileMax::DType s0123_max = blkv_max(s01_max, s23_max);
        upd_max = blkv_max(upd_max, s0123_max);
    }
    upd_max = upd_max * src_scale;
    local_max_ptr[max_idx] = upd_max;  
}

template<typename tileSrc, typename tileSum>
void __vec__ local_sum_4src_dynamic(
                        typename tileSum::TileDType __out__ local_sum,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const size_t src_valid_col
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSum::DType   *local_sum_ptr = blkv_get_tile_ptr(local_sum);
    __vbuf__ typename tileSrc::DType   *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType   *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType   *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType   *src3_ptr = blkv_get_tile_ptr(src3);

    size_t sum_idx = i*tileSum::RowStride;

    typename tileSum::DType upd_sum = 0;

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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
// ------- Ydim==8 --------

template<typename tileO, typename tileScale>
void __vec__ global_update_dynamic(
                        typename tileO::TileDType __out__ update_out,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ pv,
                        const typename tileScale::TileDType __in__ scale,
                        const size_t src_valid_col
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_scale = i * tileScale::RowStride;

    typename tileScale::DType local_scale = blkv_get_tile_ptr(scale)[idx_scale];

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileO::ValidCol;j++){
    for(size_t j=0;j<src_valid_col;j++){
        size_t idx = j * tileO::ColStride + i * tileO::RowStride;
        blkv_get_tile_ptr(update_out)[idx] = blkv_get_tile_ptr(old_out)[idx] * local_scale + blkv_get_tile_ptr(pv)[idx] ;
    }
}

template <typename dtype, int qD, int vD, int kTm, int kTk>
__attribute__((noinline)) void flash_attention_dynamic(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, int Sq, int Skv) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<-1, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, -1>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<-1, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<-1, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), -1, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, -1>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk, -1, -1>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_left = TileLeft<dtype, kTm, kTk, -1, -1>; 

    using tileO_out  = TileAcc<float, kTm, vD, -1, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor, -1, vD>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor, -1, vD>;

    using tileV      = TileRight<dtype, kTk, vD, -1, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]


    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    const int rQ = Sq % kTm;
    const int rK = Skv % kTk;

    #ifdef FA_DYNAMIC
    static_assert(Xdim==1 || Xdim==2, "Only Support Xdim=1/2 in fa dynamic version");
    static_assert(Ydim==1 || Ydim==2 || Ydim==4, "Only Support Ydim=1/2/4 in fa dynamic version");
    #endif

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ) {
        int dyn_m = (i+1) * kTm > Sq ? rQ:kTm;
        tileQ tQ[Xdim]; for (auto& x : tQ) { x = tileQ(dyn_m);}

        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                size_t offset_Q = (i+x) * tileQ::Rows * qD;
                gmQ gQ(q_ptr+offset_Q, Sq);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                size_t offset_Q = (i+x) * tileQ::Rows * qD;
                gmQ gQ(q_ptr+offset_Q, Sq);
                TCOPYIN(tQ[x], gQ);
            }
        #endif

        tileMax tMax[Xdim]; for (auto& x : tMax) { x = tileMax(dyn_m);}
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim]; for (auto& x : tSum) { x = tileSum(dyn_m);}
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tSum[x], 0);
        }

        tileO_out tPV_out(dyn_m);
        tileO tO[Xdim]; for (auto& x : tO) { x = tileO(dyn_m);}
        tileO tPV[Xdim]; for (auto& x : tPV) { x = tileO(dyn_m);}
        tileScale tScale[Xdim]; for (auto& x : tScale) { x = tileScale(dyn_m);}

        for (int j = 0; j < Kb; ) {
            int dyn_k = (j+1) * kTk > Skv ? rK:kTk;

            tileK tK[Ydim];for (auto& x : tK) { x = tileK(dyn_k);}

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                size_t offset_K = (j+y) * tileK::Cols * qD;
                gmK gK(k_ptr+offset_K, Skv);
                TCOPYIN(tK[y], gK);
            }

            tileW tW[Xdim][Ydim];for (auto& row : tW) for (auto& x : row) { x = tileW(dyn_m, dyn_k);}
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out(dyn_m, dyn_k);
                    MATMUL(tW_out, tQ[x], tK[y]);
                    // Nz -> ColMajor
                    TCVT(tW[x][y], tW_out);
                }
            }

            tileMax tNewMax[Xdim];for (auto& x : tNewMax) { x = tileMax(dyn_m);}
            tileSum tNewSum[Xdim];for (auto& x : tNewSum) { x = tileSum(dyn_m);}

            #if Ydim == 1
            tileW_cast tExpW[Xdim][Ydim];for (auto& row : tExpW) for (auto& x : row) { x = tileW_cast(dyn_m, dyn_k);}
            #else
            tileW tExpW[Xdim][Ydim];for (auto& row : tExpW) for (auto& x : row) { x = tileW(dyn_m, dyn_k);}
            #endif

            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    if(dyn_k == 64){
                    flashsoftmax_dn_mout_cast_kernel_dynamic<tileW, tileW_cast, tileMax, tileSum, tileScale, 64><<<tW[x][0].GetValidRow(), 1, 1>>>(
                                                    tScale[x].data(),
                                                    tNewMax[x].data(),
                                                    tNewSum[x].data(),
                                                    tExpW[x][0].data(),
                                                    tW[x][0].data(),
                                                    tMax[x].data(),
                                                    tSum[x].data(),
                                                    scale, tW[0][0].GetValidCol());
                    }else{
                    flashsoftmax_dn_mout_cast_kernel_dynamic<tileW, tileW_cast, tileMax, tileSum, tileScale, kTk><<<tW[x][0].GetValidRow(), 1, 1>>>(
                                                    tScale[x].data(),
                                                    tNewMax[x].data(),
                                                    tNewSum[x].data(),
                                                    tExpW[x][0].data(),
                                                    tW[x][0].data(),
                                                    tMax[x].data(),
                                                    tSum[x].data(),
                                                    scale, tW[0][0].GetValidCol());
                    }
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale, tW[0][0].GetValidCol());
                    src_exp_2src<tileW, tileMax><<<tW[0][0].GetValidRow(), tW[0][0].GetValidCol(), 1>>>(
                                                                tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(),
                                                                scale);
                    new_sum_2src_dynamic<tileW, tileSum, tileScale><<<tSum[0].GetValidRow(), 1, 1>>>(
                                                                tNewSum[x].data(),
                                                                tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tSum[x].data(),
                                                                tScale[x].data(), tExpW[0][0].GetValidCol()
                                                                );
                }
            #elif Ydim == 4
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_4src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data(),
                                                                scale, tW[0][0].GetValidCol());
                    src_exp_4src<tileW, tileMax><<<tW[0][0].GetValidRow(), tW[0][0].GetValidCol(), 1>>>(
                                                                tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tNewMax[x].data(),
                                                                scale);
                    new_sum_4src_dynamic<tileW, tileSum, tileScale><<<tSum[0].GetValidRow(), 1, 1>>>(
                                                                tNewSum[x].data(),
                                                                tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                tSum[x].data(),
                                                                tScale[x].data(), tExpW[0][0].GetValidCol()
                                                                );
                }
            #else
                #ifdef FA_DYNAMIC
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4, "Not Supprot Ydim != 1 or 2 or 4 for dynamic fa.");
                #endif
            #endif

            tileV tV[Ydim];for (auto& x : tV) { x = tileV(dyn_k);}

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                size_t offset_V = (j+y) * tileV::Rows * vD;
                gmV gV(v_ptr+offset_V, Skv);
                TCOPYIN(tV[y], gV);
            }

            // ColMajor -> Nz
            // 计算当前块的加权输出: O_j = W * V
            tileW_left tW_left[Xdim][Ydim];for (auto& row : tW_left) for (auto& x : row) { x = tileW_left(dyn_m, dyn_k);}
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TMOV_DN2NZ_DYN(tW_left[x][y], tExpW[x][y]);
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
                    global_update_dynamic<tileO, tileScale><<<tO[0].GetValidRow(), 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tO[0].GetValidCol());
                }
            }
            // 更新最大值状态
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }

            j+=Ydim;
        }

        tileO_cast tO_cast[Xdim];for (auto& x : tO_cast) { x = tileO_cast(dyn_m);}
        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tO[0].GetValidRow(), tO[0].GetValidCol(), 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
        }
        // 写回全局内存

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            size_t offset_O = (i+x) * tileO_cast::Rows * vD;
            gmO dstO(out_ptr+offset_O, Sq);
            TCOPYOUT(dstO, tO_cast[x]);
        }

        i+=Xdim;
    }
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ __attribute__((nospill)) flashsoftmax_dn_mout_cast_kernel_dynamic_unroll(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
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

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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

    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
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

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ src_exp_2src_with_new_sum_dynamic(
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_max = i * tileMax::RowStride;
    size_t idx_sum = i * tileSum::RowStride;

    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr   = blkv_get_tile_ptr(scale);

    typename tileSum::DType upd_sum = old_sum_ptr[idx_sum] * scale_ptr[idx_sum];

    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
        size_t idx_0 = j * tileSrc::ColStride + i;
        size_t idx_1 = (j+1) * tileSrc::ColStride + i;
        size_t idx_2 = (j+2) * tileSrc::ColStride + i;
        size_t idx_3 = (j+3) * tileSrc::ColStride + i;
        typename tileSrc::DType src0_exp0 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_0]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp1 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_1]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp2 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_2]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp3 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_3]*src_scale - new_max_val);

        BLKC_ASSIGN_CAST(src_exp0, idx_0, src0_exp0);
        BLKC_ASSIGN_CAST(src_exp0, idx_1, src0_exp1);
        BLKC_ASSIGN_CAST(src_exp0, idx_2, src0_exp2);
        BLKC_ASSIGN_CAST(src_exp0, idx_3, src0_exp3);
        typename tileSum::DType src0_exp_sum = src0_exp0 + src0_exp1 + src0_exp2 + src0_exp3;

        typename tileSrc::DType src1_exp0 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_0]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp1 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_1]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp2 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_2]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp3 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_3]*src_scale - new_max_val);

        BLKC_ASSIGN_CAST(src_exp1, idx_0, src1_exp0);
        BLKC_ASSIGN_CAST(src_exp1, idx_1, src1_exp1);
        BLKC_ASSIGN_CAST(src_exp1, idx_2, src1_exp2);
        BLKC_ASSIGN_CAST(src_exp1, idx_3, src1_exp3);
        typename tileSum::DType src1_exp_sum = src1_exp0 + src1_exp1 + src1_exp2 + src1_exp3;

        upd_sum += src0_exp_sum + src1_exp_sum;
    }
    blkv_get_tile_ptr(new_sum)[idx_sum] = upd_sum;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum>
void __vec__ src_exp_2src_with_local_sum_dynamic(
                        typename tileSum::TileDType __out__ local_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,

                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale,
                        const size_t src_valid_col
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_max = i * tileMax::RowStride;

    const typename tileMax::DType new_max_val = blkv_get_tile_ptr(new_max)[idx_max];
    typename tileSum::DType upd_sum = 0;
    // #pragma clang loop unroll(full)
    // for(size_t j=0;j<tileSrc::Cols;j+=4){
    for(size_t j=0;j<src_valid_col;j+=4){
        size_t idx_0 = j * tileSrc::ColStride + i;
        size_t idx_1 = (j+1) * tileSrc::ColStride + i;
        size_t idx_2 = (j+2) * tileSrc::ColStride + i;
        size_t idx_3 = (j+3) * tileSrc::ColStride + i;
        typename tileSrc::DType src0_exp0 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_0]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp1 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_1]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp2 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_2]*src_scale - new_max_val);
        typename tileSrc::DType src0_exp3 = blkv_fexp(blkv_get_tile_ptr(src0)[idx_3]*src_scale - new_max_val);

        BLKC_ASSIGN_CAST(src_exp0, idx_0, src0_exp0);
        BLKC_ASSIGN_CAST(src_exp0, idx_1, src0_exp1);
        BLKC_ASSIGN_CAST(src_exp0, idx_2, src0_exp2);
        BLKC_ASSIGN_CAST(src_exp0, idx_3, src0_exp3);
        typename tileSum::DType src0_exp_sum = src0_exp0 + src0_exp1 + src0_exp2 + src0_exp3;

        typename tileSrc::DType src1_exp0 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_0]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp1 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_1]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp2 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_2]*src_scale - new_max_val);
        typename tileSrc::DType src1_exp3 = blkv_fexp(blkv_get_tile_ptr(src1)[idx_3]*src_scale - new_max_val);

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

/*
template <typename dtype, int qD, int vD, int kTm, int kTk>
__attribute__((noinline)) void flash_attention_dynamic_unroll(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, int Sq, int Skv) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<-1, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, -1>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<-1, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<-1, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), -1, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, -1>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk, -1, -1>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_left = TileLeft<dtype, kTm, kTk, -1, -1>; 

    using tileO_out  = TileAcc<float, kTm, vD, -1, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor, -1, vD>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor, -1, vD>;

    using tileV      = TileRight<dtype, kTk, vD, -1, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]


    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    const int rQ = Sq % kTm;
    const int rK = Skv % kTk;

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ) {
        int dyn_Xdim = (i+Xdim) < (Qb-1) ? Xdim:1;
        int dyn_m = (i+1) * kTm > Sq ? rQ:kTm;
        tileQ tQ[Xdim]; for (auto& x : tQ) { x = tileQ(dyn_m);}

        for(int x=0;x<dyn_Xdim;x++){
            size_t offset_Q = (i+x) * tileQ::Rows * qD;
            gmQ gQ(q_ptr+offset_Q, Sq);
            TCOPYIN(tQ[x], gQ);
        }

        tileMax tMax[Xdim]; for (auto& x : tMax) { x = tileMax(dyn_m);}
        for(int x=0;x<dyn_Xdim;x++){
            TEXPANDSCALAR(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim]; for (auto& x : tSum) { x = tileSum(dyn_m);}
        for(int x=0;x<dyn_Xdim;x++){
            TEXPANDSCALAR(tSum[x], 0);
        }

        tileO_out tPV_out(dyn_m);
        tileO tO[Xdim]; for (auto& x : tO) { x = tileO(dyn_m);}
        tileO tPV[Xdim]; for (auto& x : tPV) { x = tileO(dyn_m);}
        tileScale tScale[Xdim]; for (auto& x : tScale) { x = tileScale(dyn_m);}

        for (int j = 0; j < Kb; ) {
            int dyn_Ydim = (j+Ydim)   < (Kb-1)? Ydim: 
                           (j+Ydim/2) < (Kb-1)? Ydim/2:  
                           (j+Ydim/4) < (Kb-1)? Ydim/4:1;
            int dyn_k = (j+1) * kTk > Skv ? rK:kTk;

            tileK tK[Ydim];for (auto& x : tK) { x = tileK(dyn_k);}

            for(int y=0;y<dyn_Ydim;y++){
                size_t offset_K = (j+y) * tileK::Cols * qD;
                gmK gK(k_ptr+offset_K, Skv);
                TCOPYIN(tK[y], gK);
            }

            tileW tW[Xdim][Ydim];for (auto& row : tW) for (auto& x : row) { x = tileW(dyn_m, dyn_k);}
            for(int x=0;x<dyn_Xdim;x++){
                for(int y=0;y<dyn_Ydim;y++){
                    tileW_out tW_out(dyn_m, dyn_k);
                    MATMUL(tW_out, tQ[x], tK[y]);
                    // Nz -> ColMajor
                    TCVT(tW[x][y], tW_out);
                }
            }

            tileMax tNewMax[Xdim];for (auto& x : tNewMax) { x = tileMax(dyn_m);}
            tileSum tNewSum[Xdim];for (auto& x : tNewSum) { x = tileSum(dyn_m);}


            tileW_cast tExpW[Xdim][Ydim];for (auto& row : tExpW) for (auto& x : row) { x = tileW_cast(dyn_m, dyn_k);}

            if (dyn_Ydim == 1){
                for(int x=0;x<dyn_Xdim;x++){
                    flashsoftmax_dn_mout_cast_kernel_dynamic_unroll<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tW[x][0].GetValidRow(), 1, 1>>>(
                                                    tScale[x].data(),
                                                    tNewMax[x].data(),
                                                    tNewSum[x].data(),
                                                    tExpW[x][0].data(),
                                                    tW[x][0].data(),
                                                    tMax[x].data(),
                                                    tSum[x].data(),
                                                    scale, tW[0][0].GetValidCol());
                }
            }else if(dyn_Ydim == 2){
                for(int x=0;x<dyn_Xdim;x++){
                    new_max_2src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale, tW[0][0].GetValidCol());
                    src_exp_2src_with_new_sum_dynamic<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tW[0][0].GetValidRow(), 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale, tW[0][0].GetValidCol());
                }
            }else if(dyn_Ydim == 4){
                tileSum tLocalSum[Xdim][2]; for (auto& row : tLocalSum) for (auto& x : row) { x = tileSum(dyn_m);}
                for(int x=0;x<dyn_Xdim;x++){
                    new_max_4src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data(),
                                                                scale, tW[0][0].GetValidCol());
                    
                    src_exp_2src_with_local_sum_dynamic<tileW, tileW_cast, tileMax, tileSum><<<tW[0][0].GetValidRow(), 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                                                   tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale, tW[0][0].GetValidCol());
                    src_exp_2src_with_local_sum_dynamic<tileW, tileW_cast, tileMax, tileSum><<<tW[0][0].GetValidRow(), 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                                                   tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale, tW[0][2].GetValidCol());
                    new_sum_of_2_loc_sum<tileScale, tileSum><<<tSum[0].GetValidRow(), 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                }
            }

            tileV tV[Ydim];for (auto& x : tV) { x = tileV(dyn_k);}

            for(int y=0;y<dyn_Ydim;y++){
                size_t offset_V = (j+y) * tileV::Rows * vD;
                gmV gV(v_ptr+offset_V, Skv);
                TCOPYIN(tV[y], gV);
            }

            // ColMajor -> Nz
            // 计算当前块的加权输出: O_j = W * V
            tileW_left tW_left[Xdim][Ydim];for (auto& row : tW_left) for (auto& x : row) { x = tileW_left(dyn_m, dyn_k);}
            for(int x=0;x<dyn_Xdim;x++){
                for(int y=0;y<dyn_Ydim;y++){
                    TCVT_DN2NZ_DYN(tW_left[x][y], tExpW[x][y]);
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
                for(int x=0;x<dyn_Xdim;x++){
                    tO[x] = tPV[x];
                }
            }else if(j<(Kb-dyn_Ydim)){
                for(int x=0;x<dyn_Xdim;x++){
                    global_update_dynamic<tileO, tileScale><<<tO[0].GetValidRow(), 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tO[0].GetValidCol());
                }
            }
            // 更新最大值状态
            for(int x=0;x<dyn_Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }

            j+=dyn_Ydim;
        }

        tileO_cast tO_cast[Xdim];for (auto& x : tO_cast) { x = tileO_cast(dyn_m);}
        for (int x = 0; x < dyn_Xdim; ++x) {
            normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tO[0].GetValidRow(), tO[0].GetValidCol(), 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
        }
        // 写回全局内存

        for (int x = 0; x < dyn_Xdim; ++x) {
            size_t offset_O = (i+x) * tileO_cast::Rows * vD;
            gmO dstO(out_ptr+offset_O, Sq);
            TCOPYOUT(dstO, tO_cast[x]);
        }

        i+=dyn_Xdim;
    }
}
*/

#define CODE1(Xdim)  \
        tileQ tQ[Xdim];                                             \
        _Pragma("clang loop unroll(full)")                          \
        for (auto& x : tQ) { x = tileQ(dyn_m);}                     \
                                                                    \
        _Pragma("clang loop unroll(full)")                          \
        for(int x=0;x<Xdim;x++){                                    \
            size_t offset_Q = (i+x) * tileQ::Rows * qD;             \
            gmQ gQ(q_ptr+offset_Q, Sq);                             \
            TCOPYIN(tQ[x], gQ);                                     \
        }                                                           \
                                                                    \
        tileMax tMax[Xdim];                                         \
        _Pragma("clang loop unroll(full)")                          \
        for (auto& x : tMax) { x = tileMax(dyn_m);}                 \
        _Pragma("clang loop unroll(full)")                          \
        for(int x=0;x<Xdim;x++){                                    \
            TEXPANDSCALAR(tMax[x], -1e30f);                         \
        }                                                           \
                                                                    \
        tileSum tSum[Xdim];                                         \
        _Pragma("clang loop unroll(full)")                          \
        for (auto& x : tSum) { x = tileSum(dyn_m);}                 \
        _Pragma("clang loop unroll(full)")                          \
        for(int x=0;x<Xdim;x++){                                    \
            TEXPANDSCALAR(tSum[x], 0);                              \
        }                                                           \
                                                                    \
        tileO_out tPV_out(dyn_m);                                   \
        tileO tO[Xdim];                                        \
        _Pragma("clang loop unroll(full)")                     \
        for (auto& x : tO) { x = tileO(dyn_m);}                \
        _Pragma("clang loop unroll(full)")                     \
        for(int x=0;x<Xdim;x++){                                    \
            TEXPANDSCALAR(tO[x], 0);                                \
        }                                                           \
        tileO tPV[Xdim];                                       \
        _Pragma("clang loop unroll(full)")                     \
        for (auto& x : tPV) { x = tileO(dyn_m);}               \
        _Pragma("clang loop unroll(full)")                     \
        for(int x=0;x<Xdim;x++){                                    \
            TEXPANDSCALAR(tPV[x], 0);                                \
        }                                                           \
        tileScale tScale[Xdim];                                \
        _Pragma("clang loop unroll(full)")                     \
        for (auto& x : tScale) { x = tileScale(dyn_m);}        \
        _Pragma("clang loop unroll(full)")                     \
        for(int x=0;x<Xdim;x++){                                    \
            TEXPANDSCALAR(tScale[x], 0);                                \
        }

#define CODE2(Xdim, Ydim) \
            tileK tK[Ydim];                                         \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& x : tK) { x = tileK(dyn_k);}                 \
            _Pragma("clang loop unroll(full)")                      \
            for(int y=0;y<Ydim;y++){                                \
                size_t offset_K = (j+y) * tileK::Cols * qD;         \
                gmK gK(k_ptr+offset_K, Skv);                        \
                TCOPYIN(tK[y], gK);                                 \
            }                                                       \
            tileW tW[Xdim][Ydim];                                   \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& row : tW)                                    \
                _Pragma("clang loop unroll(full)")                  \
                for (auto& x : row) { x = tileW(dyn_m, dyn_k);}     \
            _Pragma("clang loop unroll(full)")                      \
            for(int x=0;x<Xdim;x++){                \
                _Pragma("clang loop unroll(full)")  \
                for(int y=0;y<Ydim;y++){            \
                    tileW_out tW_out(dyn_m, dyn_k); \
                    MATMUL(tW_out, tQ[x], tK[y]);   \
                    /* Nz -> ColMajor */            \
                    TCVT(tW[x][y], tW_out);         \
                }                                   \
            }                                       \
            tileMax tNewMax[Xdim];                                  \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& x : tNewMax) { x = tileMax(dyn_m);}          \
            tileSum tNewSum[Xdim];                                  \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& x : tNewSum) { x = tileSum(dyn_m);}          \
            tileW_cast tExpW[Xdim][Ydim];for (auto& row : tExpW)    \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& x : row) { x = tileW_cast(dyn_m, dyn_k);}    \
            if (Ydim == 1){ \
                _Pragma("clang loop unroll(full)")                  \
                for(int x=0;x<Xdim;x++){ \
                    flashsoftmax_dn_mout_cast_kernel_dynamic_unroll<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tW[x][0].GetValidRow(), 1, 1>>>( \
                                                    tScale[x].data(), \
                                                    tNewMax[x].data(), \
                                                    tNewSum[x].data(), \
                                                    tExpW[x][0].data(), \
                                                    tW[x][0].data(), \
                                                    tMax[x].data(), \
                                                    tSum[x].data(), \
                                                    scale, tW[0][0].GetValidCol()); \
                } \
            }else if(Ydim == 2){ \
                _Pragma("clang loop unroll(full)")                  \
                for(int x=0;x<Xdim;x++){ \
                    new_max_2src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>( \
                                                                tScale[x].data(), \
                                                                tNewMax[x].data(), \
                                                                tW[x][0].data(), tW[x][1].data(), \
                                                                tMax[x].data(), \
                                                                scale, tW[0][0].GetValidCol()); \
                    src_exp_2src_with_new_sum_dynamic<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tW[0][0].GetValidRow(), 1, 1>>>( \
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(), \
                                                                tW[x][0].data(), tW[x][1].data(), \
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(), \
                                                                scale, tW[0][0].GetValidCol()); \
                } \
            }else if(Ydim == 4){ \
                tileSum tLocalSum[Xdim][2];                     \
                _Pragma("clang loop unroll(full)")              \
                for (auto& row : tLocalSum)                     \
                    _Pragma("clang loop unroll(full)")          \
                    for (auto& x : row) { x = tileSum(dyn_m);}  \
                _Pragma("clang loop unroll(full)")              \
                for(int x=0;x<Xdim;x++){ \
                    new_max_4src_dynamic<tileW, tileMax><<<tMax[x].GetValidRow(), 1, 1>>>( \
                                                                tScale[x].data(),  \
                                                                tNewMax[x].data(),  \
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(), \
                                                                tMax[x].data(), \
                                                                scale, tW[0][0].GetValidCol()); \
                    src_exp_2src_with_local_sum_dynamic<tileW, tileW_cast, tileMax, tileSum><<<tW[0][0].GetValidRow(), 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),      \
                                                                                                   tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale, tW[0][0].GetValidCol());             \
                    src_exp_2src_with_local_sum_dynamic<tileW, tileW_cast, tileMax, tileSum><<<tW[0][0].GetValidRow(), 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),      \
                                                                                                   tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale, tW[0][2].GetValidCol());             \
                    new_sum_of_2_loc_sum<tileScale, tileSum><<<tSum[0].GetValidRow(), 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data()); \
                } \
            } \
            tileV tV[Ydim];                                         \
            _Pragma("clang loop unroll(full)")                      \
            for (auto& x : tV) { x = tileV(dyn_k);}                 \
            _Pragma("clang loop unroll(full)")                      \
            for(int y=0;y<Ydim;y++){ \
                size_t offset_V = (j+y) * tileV::Rows * vD; \
                gmV gV(v_ptr+offset_V, Skv); \
                TCOPYIN(tV[y], gV); \
            } \
            /* ColMajor -> Nz */ \
            /* 计算当前块的加权输出: O_j = W * V */ \
            tileW_left tW_left[Xdim][Ydim];                             \
            _Pragma("clang loop unroll(full)")                          \
            for (auto& row : tW_left)                                   \
                _Pragma("clang loop unroll(full)")                      \
                for (auto& x : row) { x = tileW_left(dyn_m, dyn_k);}    \
            _Pragma("clang loop unroll(full)")                          \
            for(int x=0;x<Xdim;x++){ \
                _Pragma("clang loop unroll(full)")                      \
                for(int y=0;y<Ydim;y++){ \
                    TMOV_DN2NZ_DYN(tW_left[x][y], tExpW[x][y]); \
                    /*TCVT(tW_left[x][y], tExpW[x][y]);*/ \
                    if(y==0){ \
                        MATMUL(tPV_out, tW_left[x][y], tV[y]); \
                    }else{ \
                        MATMACC(tPV_out, tW_left[x][y], tV[y]); \
                    } \
                } \
                TCVT(tPV[x], tPV_out); \
            } \
            /* update */ \
            if(j==0){ \
                _Pragma("clang loop unroll(full)")                      \
                for(int x=0;x<Xdim;x++){ \
                    tO[x] = tPV[x]; \
                } \
            }else if(j<(Kb-Ydim)){ \
                _Pragma("clang loop unroll(full)")                      \
                for(int x=0;x<Xdim;x++){ \
                    global_update_dynamic<tileO, tileScale><<<tO[0].GetValidRow(), 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tO[0].GetValidCol()); \
                } \
            } \
            /* 更新最大值状态 */ \
            _Pragma("clang loop unroll(full)")                      \
            for(int x=0;x<Xdim;x++){ \
                tMax[x] = tNewMax[x]; \
                tSum[x] = tNewSum[x]; \
            }

#define CODE3(Xdim) \
        tileO_cast tO_cast[Xdim];                               \
        _Pragma("clang loop unroll(full)")                      \
        for (auto& x : tO_cast) { x = tileO_cast(dyn_m);} \
        _Pragma("clang loop unroll(full)")                      \
        for (int x = 0; x < Xdim; ++x) { \
            normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tO[0].GetValidRow(), tO[0].GetValidCol(), 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data()); \
        } \
        /*写回全局内存*/ \
        _Pragma("clang loop unroll(full)")                      \
        for (int x = 0; x < Xdim; ++x) { \
            size_t offset_O = (i+x) * tileO_cast::Rows * vD; \
            gmO dstO(out_ptr+offset_O, Sq); \
            TCOPYOUT(dstO, tO_cast[x]); \
        }

template <typename dtype, int qD, int vD, int kTm, int kTk>
__attribute__((noinline)) void flash_attention_dynamic_unroll(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, int Sq, int Skv) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<-1, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, -1>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<-1, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<-1, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), -1, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, -1>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk, -1, -1>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, -1, -1>;
    using tileW_left = TileLeft<dtype, kTm, kTk, -1, -1>; 

    using tileO_out  = TileAcc<float, kTm, vD, -1, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor, -1, vD>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor, -1, vD>;

    using tileV      = TileRight<dtype, kTk, vD, -1, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, -1, 1>; // [kTm×1]


    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    const int rQ = Sq % kTm;
    const int rK = Skv % kTk;

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ) {
        int dyn_m = (i+1) * kTm > Sq ? rQ:kTm;

        if( (i+Xdim) < (Qb-1) ){ // dyn_Xdim = Xdim
            CODE1(Xdim);
            for (int j = 0; j < Kb; ) {
                int dyn_k = (j+1) * kTk > Skv ? rK:kTk;
                if((j+Ydim) < (Kb-1)){ // dyn_Ydim = Ydim
                    CODE2(Xdim, Ydim);
                    j+=Ydim;
                }else if((j+Ydim/2) < (Kb-1)){ // dyn_Ydim = Ydim/2
                    CODE2(Xdim, Ydim/2);
                    j+=Ydim/2;
                }else if((j+Ydim/4) < (Kb-1)){ // dyn_Ydim = Ydim/4
                    CODE2(Xdim, Ydim/4);
                    j+=Ydim/4;
                }else{ // dyn_Ydim = 1
                    CODE2(Xdim, 1);
                    j+=1;
                }
            }
            CODE3(Xdim);
            i+=Xdim;
        }else{  // dyn_Xdim = 1
            CODE1(1);
            for (int j = 0; j < Kb; ) {
                int dyn_k = (j+1) * kTk > Skv ? rK:kTk;
                if((j+Ydim) < (Kb-1)){ // dyn_Ydim = Ydim
                    CODE2(1, Ydim);
                    j+=Ydim;
                }else if((j+Ydim/2) < (Kb-1)){ // dyn_Ydim = Ydim/2
                    CODE2(1, Ydim/2);
                    j+=Ydim/2;
                }else if((j+Ydim/4) < (Kb-1)){ // dyn_Ydim = Ydim/4
                    CODE2(1, Ydim/4);
                    j+=Ydim/4;
                }else{ // dyn_Ydim = 1
                    CODE2(1, 1);
                    j+=1;
                }
            }
            CODE3(1);
            i+=1;
        }
    }
}
