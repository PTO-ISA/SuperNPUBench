#ifndef FA_HIF4_HPP
#define FA_HIF4_HPP

#include "template_asm.h"
#include "linx_blkc.h"
#include "utils/layout_transform.hpp"

// This kernel process 64 elements
template<typename tileData, typename tileScale, typename tileSrc>
void __vec__ tohif4(
                typename tileData::TileDType __out__ p_new,
                typename tileScale::TileDType __out__ p_scale,
                const typename tileSrc::TileDType __in__ src_exp){
    uint16_t x = blkv_get_index_x();
    uint16_t y = blkv_get_index_y();
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileData::DType *p_new_ptr = blkv_get_tile_ptr(p_new);
    __vbuf__ typename tileScale::DType *p_scale_ptr = blkv_get_tile_ptr(p_scale);
    // typename tileSrc::DType upd_max = -1e10f;
    // typename tileSrc::DType scale_factor = 1/3.5;
    __bf16 upd_max = -1e10f;
    __bf16 scale_factor = 1/3.5;

    // row max for every 64 element
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<64;j+=4){
        size_t base_idx = x * tileSrc::RowStride + y * 64;
        uint16_t src_idx_0 =  base_idx + j + 0;
        uint16_t src_idx_1 =  base_idx + j + 1;
        uint16_t src_idx_2 =  base_idx + j + 2;
        uint16_t src_idx_3 =  base_idx + j + 3;
        // typename tileSrc::DType max_01 = blkv_bf16_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        // typename tileSrc::DType max_23 = blkv_bf16_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        // typename tileSrc::DType max_0123 = blkv_bf16_max(max_01, max_23);
        // __bf16 max_01 = blkv_bf16_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        // __bf16 a = src_ptr[src_idx_0];
        // __bf16 b = src_ptr[src_idx_1];
        __bf16 max_01;
        __bf16 max_23;
        __bf16 max_0123;
        blkv_bf16_fmax(max_01, src_ptr[src_idx_0], src_ptr[src_idx_1]);
        blkv_bf16_fmax(max_23, src_ptr[src_idx_2], src_ptr[src_idx_3]);
        blkv_bf16_fmax(max_0123, max_01, max_23);
        blkv_bf16_fmax(upd_max, upd_max, max_0123);
    }
    upd_max = blkv_bf16_mul(upd_max, scale_factor);
    size_t mx_idx = x * tileScale::RowStride + y; // Zn2Zz
    // p_scale[mx_idx] = upd_max; // TODO: cast fp16 to E6M2, use E4M2 temporarily
    p_scale[mx_idx] = linx_cvt_as<unsigned char>(upd_max);
    p_scale[mx_idx + 1] = 0; //E1_8 && E1_16 set to zeros
    p_scale[mx_idx + 2] = 0;
    p_scale[mx_idx + 3] = 0;

    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<64;j+=2){
        size_t base_idx = x * tileSrc::RowStride + y * 64;
        uint16_t src_idx =  base_idx + j;
        // typename tileSrc::DType src0 = blkv_bf16_div(src_ptr[src_idx] , upd_max);
        // typename tileSrc::DType src1 = blkv_bf16_div(src_ptr[src_idx + 1] , upd_max);
        __bf16 src0;
        __bf16 src1;
        blkv_bf16_fdiv(src0, src_ptr[src_idx] , upd_max);
        blkv_bf16_fdiv(src1, src_ptr[src_idx + 1] , upd_max);
        linx_cvt_package(p_new_ptr[base_idx + j/2], src0, src1);
    }
}

template<typename tileData, typename tileScale, typename tileSrc>
void __vec__ tohif4_bf16x2(
                typename tileData::TileDType __out__ p_new,
                typename tileScale::TileDType __out__ p_scale,
                const typename tileSrc::TileDType __in__ src_exp){
    uint16_t x = blkv_get_index_x();
    uint16_t y = blkv_get_index_y();
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ __bf16x2 *srcx2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_ptr);
    __vbuf__ typename tileData::DType *p_new_ptr = blkv_get_tile_ptr(p_new);
    __vbuf__ typename tileScale::DType *p_scale_ptr = blkv_get_tile_ptr(p_scale);
    __bf16x2 upd_max;
    __bf16x2 scale_factor;
    linx_cvt_package(scale_factor, 1/3.5f, 1/3.5f);

    // size_t base_idx = x * tileSrc::RowStride + y * 64;
    // group max for every 64 element
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<32;j+=4){
        size_t src_idx =  x * tileSrc::RowStride + y * 64 + j;
        // const uint16_t src_idx_0 =  x * tileSrc::RowStride + y * 64 + j + 0;
        // const uint16_t src_idx_1 =  x * tileSrc::RowStride + y * 64 + j + 1;
        // const uint16_t src_idx_2 =  x * tileSrc::RowStride + y * 64 + j + 2;
        // const uint16_t src_idx_3 =  x * tileSrc::RowStride + y * 64 + j + 3;
        __bf16x2 max_01;
        __bf16x2 max_23;
        __bf16x2 max_0123;
        // blkv_bf16x2_fmax(max_01, srcx2_ptr[src_idx_0], srcx2_ptr[src_idx_1]);
        // blkv_bf16x2_fmax(max_23, srcx2_ptr[src_idx_2], srcx2_ptr[src_idx_3]);
        blkv_bf16x2_fmax(max_01, srcx2_ptr[src_idx], srcx2_ptr[src_idx+1]);
        blkv_bf16x2_fmax(max_23, srcx2_ptr[src_idx+2], srcx2_ptr[src_idx+3]);
        blkv_bf16x2_fmax(max_0123, max_01, max_23);
        blkv_bf16x2_fmax(upd_max, upd_max, max_0123);
        // __bf16 max_0123_0 = (max_0123 >> 16) & 0xffff;
        // uint32_t tmp = linx_cvt_as<uint32_t>(max_0123);
        // __bf16 max_0123_0 = (tmp >> 16) & 0xffff;
        // __bf16 max_0123_1 = (tmp & 0xffff);
        // blkv_bf16_fmax(upd_max, max_0123_0, max_0123_1);
    }
    upd_max = blkv_bf16x2_mul(upd_max, scale_factor);
    size_t mx_idx = x * tileScale::RowStride + y; // Zn2Zz
    // p_scale[mx_idx] = upd_max;
    // p_scale[mx_idx] = linx_cvt_as<__fp8_e4m3x2>(upd_max); // TODO: cast fp16x2 to E6M2x2 ****
    p_scale[mx_idx + 1] = 0; //E1_8 && E1_16 set to zeros
    // p_scale[mx_idx + 2] = 0;
    // p_scale[mx_idx + 3] = 0;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<32;j+=2){ // 64 elements in total
        size_t src_idx =  x * tileSrc::RowStride + y * 64 + j;
        __bf16x2 src0;
        __bf16x2 src1;
        blkv_bf16x2_fdiv(src0, srcx2_ptr[src_idx] , upd_max);
        blkv_bf16x2_fdiv(src1, srcx2_ptr[src_idx + 1] , upd_max);
        linx_cvt(p_new_ptr[src_idx], src0); // bf16x2-> hif4x2
        linx_cvt(p_new_ptr[src_idx + 1], src1); // bf16x2-> hif4x2
    }
}

template<typename tileO, typename tileSum, typename tileScale>
void __vec__ normalize_with_last_update_nocast(
                        typename tileO::TileDType __out__ out_cast,
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

    blkv_get_tile_ptr(out_cast)[idx] = ((blkv_get_tile_ptr(last_pv)[idx] + blkv_get_tile_ptr(old_out)[idx] * blkv_get_tile_ptr(last_scale)[idx_scale]) /  blkv_get_tile_ptr(sum)[idx_sum]);
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale, int qD>
__attribute__((nospill))
void __vec__ flashsoftmax_dn_mout_cast_kernel(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum
                                ){
    const typename tileSrc::DType src_scale = 1.0f / sqrt((float)qD);
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);

    size_t max_idx = i*tileMax::RowStride;

    // typename tileMax::DType upd_max = old_max_ptr[max_idx];
    __bf16 upd_max;
    linx_cvt(upd_max, old_max_ptr[max_idx]); //float->bf16, TODO, check max type is bf16 or float
    // calc tile rowmax
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        // typename tileMax::DType max_01 = blkv_bf16_fmax(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        // typename tileMax::DType max_23 = blkv_bf16_fmax(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        // typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        __bf16 max_01, max_23, max_0123;
        blkv_bf16_fmax(max_01, src_ptr[src_idx_0], src_ptr[src_idx_1]);
        blkv_bf16_fmax(max_23, src_ptr[src_idx_2], src_ptr[src_idx_3]);
        blkv_bf16_fmax(max_0123, max_01, max_23);
        blkv_bf16_fmax(upd_max, upd_max, max_0123);
        // upd_max = blkv_max(upd_max, max_0123);
    }
    // scale with norm (sqrt(d_k))
    blkv_bf16_fmul(upd_max, upd_max, src_scale);
    // upd_max = upd_max * linx_cvt_as<float>(src_scale);
    new_max_ptr[max_idx] = linx_cvt_as<float>(upd_max);

    // recalculate scale of softmax
    typename tileScale::DType scale = blkv_fexp(old_max_ptr[max_idx] - linx_cvt_as<float>(upd_max));
    // __bf16 scale;
    // blkv_bf16_fsub();
    // blkv_bf16_fexp();
    size_t scale_idx = i*tileScale::RowStride;
    scale_ptr[scale_idx] = scale;

    size_t sum_idx = i*tileSum::RowStride;
    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;

    // calculate row sum of softmax, l_i
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        // typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        typename tileSrc::DType exp_src_0, exp_src_1, exp_src_2, exp_src_3;
        blkv_bf16_fmul(exp_src_0, src_ptr[src_idx_0], src_scale);
        blkv_bf16_fsub(exp_src_0, exp_src_0, upd_max);
        blkv_bf16_fexp(exp_src_0, exp_src_0);
        blkv_bf16_fmul(exp_src_1, src_ptr[src_idx_1], src_scale);
        blkv_bf16_fsub(exp_src_1, exp_src_1, upd_max);
        blkv_bf16_fexp(exp_src_1, exp_src_1);
        blkv_bf16_fmul(exp_src_2, src_ptr[src_idx_2], src_scale);
        blkv_bf16_fsub(exp_src_2, exp_src_2, upd_max);
        blkv_bf16_fexp(exp_src_2, exp_src_2);
        blkv_bf16_fmul(exp_src_3, src_ptr[src_idx_3], src_scale);
        blkv_bf16_fsub(exp_src_3, exp_src_3, upd_max);
        blkv_bf16_fexp(exp_src_3, exp_src_3);
        blkv_bf16_fadd(exp_src_0, exp_src_0, exp_src_1);
        blkv_bf16_fadd(exp_src_2, exp_src_2, exp_src_3);
        blkv_bf16_fadd(exp_src_0, exp_src_0, exp_src_2);
        upd_sum += linx_cvt_as<float>(exp_src_0);
        // blkv_bf16_fadd(upd_sum, upd_sum, exp_src_0);
        // typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        // typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        // typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        // upd_sum += exp_src_0123;
        if constexpr (std::is_same_v<typename tileSrc_cast::DType, __bf16>) {
            // TODO, convert before comparing with max
            BLKC_ASSIGN_CAST(src_exp, src_idx_0, exp_src_0);
            BLKC_ASSIGN_CAST(src_exp, src_idx_1, exp_src_1);
            BLKC_ASSIGN_CAST(src_exp, src_idx_2, exp_src_2);
            BLKC_ASSIGN_CAST(src_exp, src_idx_3, exp_src_3);
        } else {
            // static_assert(tileSrc_cast::TileDType == __bf16x2);
            uint16_t idx0 = i * tileSrc::RowStride + j/2 * tileSrc::ColStride;
            uint16_t idx1 = i * tileSrc::RowStride + (j+2)/2 * tileSrc::ColStride;
            linx_cvt_package(src_exp_ptr[idx0], exp_src_0, exp_src_1);
            linx_cvt_package(src_exp_ptr[idx1], exp_src_2, exp_src_3);
        }

    }
    new_sum_ptr[sum_idx] = upd_sum;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileScale, int qD>
__attribute__((nospill))
void __vec__ pkg_rowmax(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    // __vbuf__ __bf16x2 *srcx2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_ptr);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_max;
    __bf16 old_max_bf160, old_max_bf161;
    linx_cvt(old_max_bf160, old_max_ptr[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(old_max_bf161, old_max_ptr[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(upd_max, old_max_bf160, old_max_bf161);

    // calc tile rowmax
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        __bf16x2 src0, src1, src2, src3;
        linx_cvt_package(src0, src_ptr[src_idx_00], src_ptr[src_idx_01]);
        linx_cvt_package(src1, src_ptr[src_idx_10], src_ptr[src_idx_11]);
        linx_cvt_package(src2, src_ptr[src_idx_20], src_ptr[src_idx_21]);
        linx_cvt_package(src3, src_ptr[src_idx_30], src_ptr[src_idx_31]);
        blkv_bf16x2_fmax(src0, src0, src1);
        blkv_bf16x2_fmax(src2, src2, src3);
        blkv_bf16x2_fmax(src0, src0, src2);
        blkv_bf16x2_fmax(upd_max, upd_max, src0);

        // uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        // uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        // uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        // uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        // blkv_bf16x2_fmax(src0, srcx2_ptr[src_idx_00], srcx2_ptr[src_idx_10]);
        // blkv_bf16x2_fmax(src1, srcx2_ptr[src_idx_20], srcx2_ptr[src_idx_30]);
        // blkv_bf16x2_fmax(src0, src0, src1);
        // blkv_bf16x2_fmax(upd_max, upd_max, src0);
    }

    // scale with norm (sqrt(d_k))
    blkv_bf16x2_mul(upd_max, upd_max, src_scale);
    // uint32_t max_u32 = linx_cvt_as<uint32_t>(upd_max); //*** TODO: check how to assign */
    union { __bf16x2 vec; uint32_t u32; } max_u;
    max_u.vec = upd_max;
    __bf16 max0 = (max_u.u32 >> 16) & 0xffff;
    __bf16 max1 = (max_u.u32 & 0xffff);

    // uint32_t *max_u32 = (uint32_t*)&upd_max;
    // __bf16 max0 = ((*max_u32) >> 16) & 0xffff;
    // __bf16 max1 = ((*max_u32) & 0xffff);
    new_max_ptr[i*2*tileMax::RowStride] = linx_cvt_as<float>(max0);
    new_max_ptr[(i*2+1)*tileMax::RowStride] = linx_cvt_as<float>(max1);

    // recalculate scale of softmax
    __bf16x2 scale, old_max_bf16x2;
    linx_cvt_package(old_max_bf16x2, old_max_bf160, old_max_bf161);
    blkv_bf16x2_fsub(old_max_bf16x2, old_max_bf16x2, upd_max); 
    blkv_bf16x2_fexp(scale, old_max_bf16x2);
    // opt1
    union { __bf16x2 vec; uint32_t u32; } scale_u;
    scale_u.vec = scale;
    __bf16 scale0 = (scale_u.u32 >> 16) & 0xffff;
    __bf16 scale1 = (scale_u.u32 & 0xffff);
    uint32_t scale_idx00 = i*2*tileScale::RowStride;
    uint32_t scale_idx01 = (i*2+1)*tileScale::RowStride;
    // uint32_t *scale_u32 = (uint32_t*)&scale;
    // __bf16 scale0 = ((*scale_u32) >> 16) & 0xffff;
    // __bf16 scale1 = ((*scale_u32) & 0xffff);
    // unpack bf16x2 to 2xfloat
    scale_ptr[scale_idx00] = linx_cvt_as<float>(scale0);
    scale_ptr[scale_idx01] = linx_cvt_as<float>(scale1);
}

// ------- Ydim==4 --------
template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileScale, int qD>
// __attribute__((nospill))
void __vec__ pkg_rowmax_4src(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileMax::TileDType __in__ old_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType *src3_ptr = blkv_get_tile_ptr(src3);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_max;
    __bf16 old_max_bf160, old_max_bf161;
    linx_cvt(old_max_bf160, old_max_ptr[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(old_max_bf161, old_max_ptr[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(upd_max, old_max_bf160, old_max_bf161);

    // calc tile rowmax
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        
        __bf16x2 s0_0, s0_1, s0_2, s0_3;
        linx_cvt_package(s0_0, src0_ptr[src_idx_00], src0_ptr[src_idx_01]);
        linx_cvt_package(s0_1, src0_ptr[src_idx_10], src0_ptr[src_idx_11]);
        linx_cvt_package(s0_2, src0_ptr[src_idx_20], src0_ptr[src_idx_21]);
        linx_cvt_package(s0_3, src0_ptr[src_idx_30], src0_ptr[src_idx_31]);
        blkv_bf16x2_fmax(s0_0, s0_0, s0_1);
        blkv_bf16x2_fmax(s0_2, s0_2, s0_3);
        blkv_bf16x2_fmax(s0_0, s0_0, s0_2);

        __bf16x2 s1_0, s1_1, s1_2, s1_3;
        linx_cvt_package(s1_0, src1_ptr[src_idx_00], src1_ptr[src_idx_01]);
        linx_cvt_package(s1_1, src1_ptr[src_idx_10], src1_ptr[src_idx_11]);
        linx_cvt_package(s1_2, src1_ptr[src_idx_20], src1_ptr[src_idx_21]);
        linx_cvt_package(s1_3, src1_ptr[src_idx_30], src1_ptr[src_idx_31]);
        blkv_bf16x2_fmax(s1_0, s1_0, s1_1);
        blkv_bf16x2_fmax(s1_2, s1_2, s1_3);
        blkv_bf16x2_fmax(s1_0, s1_0, s1_2);

        __bf16x2 s2_0, s2_1, s2_2, s2_3;
        linx_cvt_package(s2_0, src2_ptr[src_idx_00], src2_ptr[src_idx_01]);
        linx_cvt_package(s2_1, src2_ptr[src_idx_10], src2_ptr[src_idx_11]);
        linx_cvt_package(s2_2, src2_ptr[src_idx_20], src2_ptr[src_idx_21]);
        linx_cvt_package(s2_3, src2_ptr[src_idx_30], src2_ptr[src_idx_31]);
        blkv_bf16x2_fmax(s2_0, s2_0, s2_1);
        blkv_bf16x2_fmax(s2_2, s2_2, s2_3);
        blkv_bf16x2_fmax(s2_0, s2_0, s2_2);

        __bf16x2 s3_0, s3_1, s3_2, s3_3;
        linx_cvt_package(s3_0, src3_ptr[src_idx_00], src3_ptr[src_idx_01]);
        linx_cvt_package(s3_1, src3_ptr[src_idx_10], src3_ptr[src_idx_11]);
        linx_cvt_package(s3_2, src3_ptr[src_idx_20], src3_ptr[src_idx_21]);
        linx_cvt_package(s3_3, src3_ptr[src_idx_30], src3_ptr[src_idx_31]);
        blkv_bf16x2_fmax(s3_0, s3_0, s3_1);
        blkv_bf16x2_fmax(s3_2, s3_2, s3_3);
        blkv_bf16x2_fmax(s3_0, s3_0, s3_2);

        blkv_bf16x2_fmax(upd_max, upd_max, s0_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s1_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s2_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s3_0);
    }

    // scale with norm (sqrt(d_k))
    blkv_bf16x2_mul(upd_max, upd_max, src_scale);
    union { __bf16x2 vec; uint32_t u32; } max_u;
    max_u.vec = upd_max;
    __bf16 max0 = (max_u.u32 >> 16) & 0xffff;
    __bf16 max1 = (max_u.u32 & 0xffff);

    new_max_ptr[i*2*tileMax::RowStride] = linx_cvt_as<float>(max0);
    new_max_ptr[(i*2+1)*tileMax::RowStride] = linx_cvt_as<float>(max1);

    // recalculate scale of softmax
    __bf16x2 scale, old_max_bf16x2;
    linx_cvt_package(old_max_bf16x2, old_max_bf160, old_max_bf161);
    blkv_bf16x2_fsub(old_max_bf16x2, old_max_bf16x2, upd_max); 
    blkv_bf16x2_fexp(scale, old_max_bf16x2);

    union { __bf16x2 vec; uint32_t u32; } scale_u;
    scale_u.vec = scale;
    __bf16 scale0 = (scale_u.u32 >> 16) & 0xffff;
    __bf16 scale1 = (scale_u.u32 & 0xffff);
    uint32_t scale_idx00 = i*2*tileScale::RowStride;
    uint32_t scale_idx01 = (i*2+1)*tileScale::RowStride;

    scale_ptr[scale_idx00] = linx_cvt_as<float>(scale0);
    scale_ptr[scale_idx01] = linx_cvt_as<float>(scale1);
}

// calculate rowmax and recalculate scale of tiled softmax
template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileScale, int qD>
// __attribute__((nospill))
void __vec__ pkg_rowmax_4srcx2(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileSrc::TileDType __in__ src2,
                        const typename tileSrc::TileDType __in__ src3,
                        const typename tileMax::TileDType __in__ old_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);
    __vbuf__ typename tileSrc::DType *src2_ptr = blkv_get_tile_ptr(src2);
    __vbuf__ typename tileSrc::DType *src3_ptr = blkv_get_tile_ptr(src3);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);

    // 将4个src指针转换为双宽度的 __bf16x2 指针
    __vbuf__ __bf16x2 *src0_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src0_ptr);
    __vbuf__ __bf16x2 *src1_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src1_ptr);
    __vbuf__ __bf16x2 *src2_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src2_ptr);
    __vbuf__ __bf16x2 *src3_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src3_ptr);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_max;
    __bf16 old_max_bf160, old_max_bf161;
    linx_cvt(old_max_bf160, old_max_ptr[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(old_max_bf161, old_max_ptr[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(upd_max, old_max_bf160, old_max_bf161);

    // calc tile rowmax
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        // 只需要计算偶数行的索引，因为 __bf16x2 访存会同时带出奇数行的数据
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        
        __bf16x2 s0_0, s0_1;
        blkv_bf16x2_fmax(s0_0, src0_x2_ptr[src_idx_00], src0_x2_ptr[src_idx_10]);
        blkv_bf16x2_fmax(s0_1, src0_x2_ptr[src_idx_20], src0_x2_ptr[src_idx_30]);
        blkv_bf16x2_fmax(s0_0, s0_0, s0_1);

        __bf16x2 s1_0, s1_1;
        blkv_bf16x2_fmax(s1_0, src1_x2_ptr[src_idx_00], src1_x2_ptr[src_idx_10]);
        blkv_bf16x2_fmax(s1_1, src1_x2_ptr[src_idx_20], src1_x2_ptr[src_idx_30]);
        blkv_bf16x2_fmax(s1_0, s1_0, s1_1);

        __bf16x2 s2_0, s2_1;
        blkv_bf16x2_fmax(s2_0, src2_x2_ptr[src_idx_00], src2_x2_ptr[src_idx_10]);
        blkv_bf16x2_fmax(s2_1, src2_x2_ptr[src_idx_20], src2_x2_ptr[src_idx_30]);
        blkv_bf16x2_fmax(s2_0, s2_0, s2_1);

        __bf16x2 s3_0, s3_1;
        blkv_bf16x2_fmax(s3_0, src3_x2_ptr[src_idx_00], src3_x2_ptr[src_idx_10]);
        blkv_bf16x2_fmax(s3_1, src3_x2_ptr[src_idx_20], src3_x2_ptr[src_idx_30]);
        blkv_bf16x2_fmax(s3_0, s3_0, s3_1);

        blkv_bf16x2_fmax(upd_max, upd_max, s0_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s1_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s2_0);
        blkv_bf16x2_fmax(upd_max, upd_max, s3_0);
    }

    // scale with norm (sqrt(d_k))
    blkv_bf16x2_mul(upd_max, upd_max, src_scale);
    union { __bf16x2 vec; uint32_t u32; } max_u;
    max_u.vec = upd_max;
    __bf16 max0 = (max_u.u32 >> 16) & 0xffff;
    __bf16 max1 = (max_u.u32 & 0xffff);

    new_max_ptr[i*2*tileMax::RowStride] = linx_cvt_as<float>(max0);
    new_max_ptr[(i*2+1)*tileMax::RowStride] = linx_cvt_as<float>(max1);

    // recalculate scale of softmax
    __bf16x2 scale, old_max_bf16x2;
    linx_cvt_package(old_max_bf16x2, old_max_bf160, old_max_bf161);
    blkv_bf16x2_fsub(old_max_bf16x2, old_max_bf16x2, upd_max); 
    blkv_bf16x2_fexp(scale, old_max_bf16x2);

    union { __bf16x2 vec; uint32_t u32; } scale_u;
    scale_u.vec = scale;
    __bf16 scale0 = (scale_u.u32 >> 16) & 0xffff;
    __bf16 scale1 = (scale_u.u32 & 0xffff);
    uint32_t scale_idx00 = i*2*tileScale::RowStride;
    uint32_t scale_idx01 = (i*2+1)*tileScale::RowStride;

    scale_ptr[scale_idx00] = linx_cvt_as<float>(scale0);
    scale_ptr[scale_idx01] = linx_cvt_as<float>(scale1);
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, int qD>
void __vec__ rowsum_2src_with_local_sum(
                        typename tileSum::TileDType __out__ local_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileSum::DType *local_sum_ptr = blkv_get_tile_ptr(local_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp0_ptr = blkv_get_tile_ptr(src_exp0);
    __vbuf__ typename tileSrc_cast::DType *src_exp1_ptr = blkv_get_tile_ptr(src_exp1);
    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_sum, new_max_val;
    __bf16 new_max_bf16_0, new_max_bf16_1;
    
    // Initialize local sum to 0
    linx_cvt_package(upd_sum, 0.0f, 0.0f);

    linx_cvt(new_max_bf16_0, blkv_get_tile_ptr(new_max)[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(new_max_bf16_1, blkv_get_tile_ptr(new_max)[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(new_max_val, new_max_bf16_0, new_max_bf16_1);

    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        
        // Process src0
        __bf16x2 s0_0, s0_1, s0_2, s0_3;
        __bf16x2 sum01_0, sum23_0, sum0123_0;
        linx_cvt_package(s0_0, src0_ptr[src_idx_00], src0_ptr[src_idx_01]);
        linx_cvt_package(s0_1, src0_ptr[src_idx_10], src0_ptr[src_idx_11]);
        linx_cvt_package(s0_2, src0_ptr[src_idx_20], src0_ptr[src_idx_21]);
        linx_cvt_package(s0_3, src0_ptr[src_idx_30], src0_ptr[src_idx_31]);
        
        blkv_bf16x2_fmsub(s0_0, s0_0, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_1, s0_1, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_2, s0_2, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_3, s0_3, src_scale, new_max_val);
        blkv_bf16x2_fexp(s0_0, s0_0);
        blkv_bf16x2_fexp(s0_1, s0_1);
        blkv_bf16x2_fexp(s0_2, s0_2);
        blkv_bf16x2_fexp(s0_3, s0_3);
        blkv_bf16x2_fadd(sum01_0, s0_0, s0_1);
        blkv_bf16x2_fadd(sum23_0, s0_2, s0_3);
        blkv_bf16x2_fadd(sum0123_0, sum01_0, sum23_0);
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_0);
        
        BLKC_ASSIGN_CAST(src_exp0, src_idx_00, s0_0);
        BLKC_ASSIGN_CAST(src_exp0, src_idx_10, s0_1);
        BLKC_ASSIGN_CAST(src_exp0, src_idx_20, s0_2);
        BLKC_ASSIGN_CAST(src_exp0, src_idx_30, s0_3);

        // Process src1
        __bf16x2 s1_0, s1_1, s1_2, s1_3;
        __bf16x2 sum01_1, sum23_1, sum0123_1;
        linx_cvt_package(s1_0, src1_ptr[src_idx_00], src1_ptr[src_idx_01]);
        linx_cvt_package(s1_1, src1_ptr[src_idx_10], src1_ptr[src_idx_11]);
        linx_cvt_package(s1_2, src1_ptr[src_idx_20], src1_ptr[src_idx_21]);
        linx_cvt_package(s1_3, src1_ptr[src_idx_30], src1_ptr[src_idx_31]);
        
        blkv_bf16x2_fmsub(s1_0, s1_0, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_1, s1_1, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_2, s1_2, src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_3, s1_3, src_scale, new_max_val);
        blkv_bf16x2_fexp(s1_0, s1_0);
        blkv_bf16x2_fexp(s1_1, s1_1);
        blkv_bf16x2_fexp(s1_2, s1_2);
        blkv_bf16x2_fexp(s1_3, s1_3);
        blkv_bf16x2_fadd(sum01_1, s1_0, s1_1);
        blkv_bf16x2_fadd(sum23_1, s1_2, s1_3);
        blkv_bf16x2_fadd(sum0123_1, sum01_1, sum23_1);
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_1);
        
        BLKC_ASSIGN_CAST(src_exp1, src_idx_00, s1_0);
        BLKC_ASSIGN_CAST(src_exp1, src_idx_10, s1_1);
        BLKC_ASSIGN_CAST(src_exp1, src_idx_20, s1_2);
        BLKC_ASSIGN_CAST(src_exp1, src_idx_30, s1_3);
    }

    union { __bf16x2 vec; uint32_t u32; } sum_u;
    sum_u.vec = upd_sum;
    __bf16 sum0 = (sum_u.u32 >> 16) & 0xffff;
    __bf16 sum1 = (sum_u.u32 & 0xffff);
    
    local_sum_ptr[(i*2)*tileSum::RowStride] = sum0;
    local_sum_ptr[(i*2+1)*tileSum::RowStride] = sum1;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, int qD>
void __vec__ rowsum_2src_with_local_sumx2(
                        typename tileSum::TileDType __out__ local_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileSum::DType *local_sum_ptr = blkv_get_tile_ptr(local_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp0_ptr = blkv_get_tile_ptr(src_exp0);
    __vbuf__ typename tileSrc_cast::DType *src_exp1_ptr = blkv_get_tile_ptr(src_exp1);
    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);

    // 将 src 指针转换为双宽度的 __bf16x2 指针
    __vbuf__ __bf16x2 *src0_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src0_ptr);
    __vbuf__ __bf16x2 *src1_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src1_ptr);
    __vbuf__ __bf16x2 *src_exp0_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_exp0_ptr);
    __vbuf__ __bf16x2 *src_exp1_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_exp1_ptr);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_sum, new_max_val;
    __bf16 new_max_bf16_0, new_max_bf16_1;
    
    // Initialize local sum to 0
    linx_cvt_package(upd_sum, 0.0f, 0.0f);

    linx_cvt(new_max_bf16_0, blkv_get_tile_ptr(new_max)[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(new_max_bf16_1, blkv_get_tile_ptr(new_max)[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(new_max_val, new_max_bf16_0, new_max_bf16_1);

    // row sum
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        // 只需要偶数行的索引，双宽访存会自动带出奇数行数据
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        
        // Process src0
        __bf16x2 s0_0, s0_1, s0_2, s0_3;
        __bf16x2 sum01_0, sum23_0, sum0123_0;
        
        // 直接将内存读取作为 fmsub 的输入操作数
        blkv_bf16x2_fmsub(s0_0, src0_x2_ptr[src_idx_00], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_1, src0_x2_ptr[src_idx_10], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_2, src0_x2_ptr[src_idx_20], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_3, src0_x2_ptr[src_idx_30], src_scale, new_max_val);
        
        blkv_bf16x2_fexp(s0_0, s0_0);
        blkv_bf16x2_fexp(s0_1, s0_1);
        blkv_bf16x2_fexp(s0_2, s0_2);
        blkv_bf16x2_fexp(s0_3, s0_3);
        blkv_bf16x2_fadd(sum01_0, s0_0, s0_1);
        blkv_bf16x2_fadd(sum23_0, s0_2, s0_3);
        blkv_bf16x2_fadd(sum0123_0, sum01_0, sum23_0);
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_0);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_00, s0_0);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_10, s0_1);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_20, s0_2);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_30, s0_3);

        // Process src1
        __bf16x2 s1_0, s1_1, s1_2, s1_3;
        __bf16x2 sum01_1, sum23_1, sum0123_1;
        
        blkv_bf16x2_fmsub(s1_0, src1_x2_ptr[src_idx_00], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_1, src1_x2_ptr[src_idx_10], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_2, src1_x2_ptr[src_idx_20], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_3, src1_x2_ptr[src_idx_30], src_scale, new_max_val);
        
        blkv_bf16x2_fexp(s1_0, s1_0);
        blkv_bf16x2_fexp(s1_1, s1_1);
        blkv_bf16x2_fexp(s1_2, s1_2);
        blkv_bf16x2_fexp(s1_3, s1_3);
        blkv_bf16x2_fadd(sum01_1, s1_0, s1_1);
        blkv_bf16x2_fadd(sum23_1, s1_2, s1_3);
        blkv_bf16x2_fadd(sum0123_1, sum01_1, sum23_1);
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_1);
        
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_00, s1_0);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_10, s1_1);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_20, s1_2);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_30, s1_3);
    }

    union { __bf16x2 vec; uint32_t u32; } sum_u;
    sum_u.vec = upd_sum;
    __bf16 sum0 = (sum_u.u32 >> 16) & 0xffff;
    __bf16 sum1 = (sum_u.u32 & 0xffff);
    
    local_sum_ptr[(i*2)*tileSum::RowStride] = sum0;
    local_sum_ptr[(i*2+1)*tileSum::RowStride] = sum1;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, int qD>
void __vec__ rowsum_2src_with_local_expx2(
                        // typename tileSum::TileDType __out__ local_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp0,
                        typename tileSrc_cast::TileDType __out__ src_exp1,
                        const typename tileSrc::TileDType __in__ src0,
                        const typename tileSrc::TileDType __in__ src1,
                        const typename tileMax::TileDType __in__ new_max){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    // __vbuf__ typename tileSum::DType *local_sum_ptr = blkv_get_tile_ptr(local_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp0_ptr = blkv_get_tile_ptr(src_exp0);
    __vbuf__ typename tileSrc_cast::DType *src_exp1_ptr = blkv_get_tile_ptr(src_exp1);
    __vbuf__ typename tileSrc::DType *src0_ptr = blkv_get_tile_ptr(src0);
    __vbuf__ typename tileSrc::DType *src1_ptr = blkv_get_tile_ptr(src1);

    // 将 src 指针转换为双宽度的 __bf16x2 指针
    __vbuf__ __bf16x2 *src0_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src0_ptr);
    __vbuf__ __bf16x2 *src1_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src1_ptr);
    __vbuf__ __bf16x2 *src_exp0_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_exp0_ptr);
    __vbuf__ __bf16x2 *src_exp1_x2_ptr = reinterpret_cast<__vbuf__ __bf16x2*>(src_exp1_ptr);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_sum, new_max_val;
    __bf16 new_max_bf16_0, new_max_bf16_1;
    
    // Initialize local sum to 0
    linx_cvt_package(upd_sum, 0.0f, 0.0f);

    linx_cvt(new_max_bf16_0, blkv_get_tile_ptr(new_max)[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(new_max_bf16_1, blkv_get_tile_ptr(new_max)[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(new_max_val, new_max_bf16_0, new_max_bf16_1);

    // row sum
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        // 只需要偶数行的索引，双宽访存会自动带出奇数行数据
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        
        // Process src0
        __bf16x2 s0_0, s0_1, s0_2, s0_3;
        __bf16x2 sum01_0, sum23_0, sum0123_0;
        
        // 直接将内存读取作为 fmsub 的输入操作数
        blkv_bf16x2_fmsub(s0_0, src0_x2_ptr[src_idx_00], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_1, src0_x2_ptr[src_idx_10], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_2, src0_x2_ptr[src_idx_20], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s0_3, src0_x2_ptr[src_idx_30], src_scale, new_max_val);
        
        blkv_bf16x2_fexp(s0_0, s0_0);
        blkv_bf16x2_fexp(s0_1, s0_1);
        blkv_bf16x2_fexp(s0_2, s0_2);
        blkv_bf16x2_fexp(s0_3, s0_3);
        // blkv_bf16x2_fadd(sum01_0, s0_0, s0_1);
        // blkv_bf16x2_fadd(sum23_0, s0_2, s0_3);
        // blkv_bf16x2_fadd(sum0123_0, sum01_0, sum23_0);
        // blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_0);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_00, s0_0);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_10, s0_1);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_20, s0_2);
        blkc_assign_elem(src_exp0_x2_ptr, src_idx_30, s0_3);

        // Process src1
        __bf16x2 s1_0, s1_1, s1_2, s1_3;
        __bf16x2 sum01_1, sum23_1, sum0123_1;
        
        blkv_bf16x2_fmsub(s1_0, src1_x2_ptr[src_idx_00], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_1, src1_x2_ptr[src_idx_10], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_2, src1_x2_ptr[src_idx_20], src_scale, new_max_val);
        blkv_bf16x2_fmsub(s1_3, src1_x2_ptr[src_idx_30], src_scale, new_max_val);
        
        blkv_bf16x2_fexp(s1_0, s1_0);
        blkv_bf16x2_fexp(s1_1, s1_1);
        blkv_bf16x2_fexp(s1_2, s1_2);
        blkv_bf16x2_fexp(s1_3, s1_3);
        // blkv_bf16x2_fadd(sum01_1, s1_0, s1_1);
        // blkv_bf16x2_fadd(sum23_1, s1_2, s1_3);
        // blkv_bf16x2_fadd(sum0123_1, sum01_1, sum23_1);
        // blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123_1);
        
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_00, s1_0);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_10, s1_1);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_20, s1_2);
        blkc_assign_elem(src_exp1_x2_ptr, src_idx_30, s1_3);
    }

    // union { __bf16x2 vec; uint32_t u32; } sum_u;
    // sum_u.vec = upd_sum;
    // __bf16 sum0 = (sum_u.u32 >> 16) & 0xffff;
    // __bf16 sum1 = (sum_u.u32 & 0xffff);
    
    // local_sum_ptr[(i*2)*tileSum::RowStride] = sum0;
    // local_sum_ptr[(i*2+1)*tileSum::RowStride] = sum1;
}

template<typename tileScale, typename tileSum>
void __vec__ new_sum_of_2_loc_sum_bf16x2(
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

    // X1Y1 logic uses bf16x2, processing 2 rows at a time
    uint32_t sum_idx0 = (i*2)*tileSum::RowStride;
    uint32_t sum_idx1 = (i*2+1)*tileSum::RowStride;

    new_sum_ptr[sum_idx0] = old_sum_ptr[sum_idx0] * scale_ptr[sum_idx0] + local_sum_0_ptr[sum_idx0] + local_sum_1_ptr[sum_idx0];
    new_sum_ptr[sum_idx1] = old_sum_ptr[sum_idx1] * scale_ptr[sum_idx1] + local_sum_0_ptr[sum_idx1] + local_sum_1_ptr[sum_idx1];
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale, int qD>
// __attribute__((nospill))
void __vec__ rowsum(
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ rescale){
    uint16_t i = blkv_get_index_x();
    __bf16x2 src_scale;
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);

    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));
    __bf16x2 upd_sum, scale, new_max_val;
    __bf16 old_sum_bf16_0, old_sum_bf16_1;
    __bf16 scale_bf16_0, scale_bf16_1;
    __bf16 new_max_bf16_0, new_max_bf16_1;
    // old_sum * rescale + new_sum
    linx_cvt(old_sum_bf16_0, old_sum_ptr[i*2*tileSum::RowStride]); //float->bf16
    linx_cvt(old_sum_bf16_1, old_sum_ptr[(i*2+1)*tileSum::RowStride]); 
    linx_cvt_package(upd_sum, old_sum_bf16_0, old_sum_bf16_1);
    linx_cvt(scale_bf16_0, scale_ptr[i*2*tileSum::RowStride]); //float->bf16
    linx_cvt(scale_bf16_1, scale_ptr[(i*2+1)*tileSum::RowStride]); 
    linx_cvt_package(scale, scale_bf16_0, scale_bf16_1);
    blkv_bf16x2_fmul(upd_sum, upd_sum, scale);  
    linx_cvt(new_max_bf16_0, blkv_get_tile_ptr(new_max)[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(new_max_bf16_1, blkv_get_tile_ptr(new_max)[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(new_max_val, new_max_bf16_0, new_max_bf16_1);

    // calculate row sum of softmax, l_i
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        __bf16x2 src0, src1, src2, src3;
        __bf16x2 sum01, sum23, sum0123;
        linx_cvt_package(src0, src_ptr[src_idx_00], src_ptr[src_idx_01]);
        linx_cvt_package(src1, src_ptr[src_idx_10], src_ptr[src_idx_11]);
        linx_cvt_package(src2, src_ptr[src_idx_20], src_ptr[src_idx_21]);
        linx_cvt_package(src3, src_ptr[src_idx_30], src_ptr[src_idx_31]);
        blkv_bf16x2_fmsub(src0, src0, src_scale, new_max_val);
        blkv_bf16x2_fmsub(src1, src1, src_scale, new_max_val);
        blkv_bf16x2_fmsub(src2, src2, src_scale, new_max_val);
        blkv_bf16x2_fmsub(src3, src3, src_scale, new_max_val);
        blkv_bf16x2_fexp(src0, src0);
        blkv_bf16x2_fexp(src1, src1);
        blkv_bf16x2_fexp(src2, src2);
        blkv_bf16x2_fexp(src3, src3);
        blkv_bf16x2_fadd(sum01, src0, src1);
        blkv_bf16x2_fadd(sum23, src2, src3);
        blkv_bf16x2_fadd(sum0123, sum01, sum23);
        // typename tileSrc::DType exp_src_0 = blkv_bf16x2_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_1 = blkv_bf16x2_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_2 = blkv_bf16x2_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_3 = blkv_bf16x2_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        // typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        // typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        // upd_sum += exp_src_0123;
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123);
        // src_exp_ptr[src_idx_00] = src0;
        // src_exp_ptr[src_idx_10] = src1;
        // src_exp_ptr[src_idx_20] = src2;
        // src_exp_ptr[src_idx_30] = src3;
        BLKC_ASSIGN_CAST(src_exp, src_idx_00, src0);
        BLKC_ASSIGN_CAST(src_exp, src_idx_10, src1);
        BLKC_ASSIGN_CAST(src_exp, src_idx_20, src2);
        BLKC_ASSIGN_CAST(src_exp, src_idx_30, src3);
    }

    // uint32_t sum_u32 = linx_cvt_as<uint32_t>(upd_sum);
    union { __bf16x2 vec; uint32_t u32; } sum_u;
    sum_u.vec = upd_sum;
    __bf16 sum0 = (sum_u.u32 >> 16) & 0xffff;
    __bf16 sum1 = (sum_u.u32 & 0xffff);
    
    new_sum_ptr[(i*2)*tileSum::RowStride] = sum0;
    new_sum_ptr[(i*2+1)*tileSum::RowStride] = sum1;
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale, int qD>
__attribute__((nospill))
void __vec__ flashsoftmax_dn_mout_cast_kernel_bf16x2(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum
                                ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);

    // uint32_t max_idx00 = i*2*tileMax::RowStride;
    // uint32_t max_idx01 = (i*2+1)*tileMax::RowStride;
    __bf16x2 src_scale;
    linx_cvt_package(src_scale, 1.0f / sqrt((float)qD), 1.0f / sqrt((float)qD));

    // typename tileMax::DType upd_max = old_max_ptr[max_idx];
    __bf16x2 upd_max;
    __bf16 old_max_bf160, old_max_bf161;
    linx_cvt(old_max_bf160, old_max_ptr[i*2*tileMax::RowStride]); //float->bf16
    linx_cvt(old_max_bf161, old_max_ptr[(i*2+1)*tileMax::RowStride]); 
    linx_cvt_package(upd_max, old_max_bf160, old_max_bf161);

    // calc tile rowmax
    // #pragma clang loop unroll(full)
    // for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
    //     uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
    //     uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
    //     uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
    //     uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
    //     uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
    //     uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
    //     uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
    //     uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
    //     __bf16x2 src0, src1, src2, src3;
    //     linx_cvt_package(src0, src_ptr[src_idx_00], src_ptr[src_idx_01]);
    //     linx_cvt_package(src1, src_ptr[src_idx_10], src_ptr[src_idx_11]);
    //     linx_cvt_package(src2, src_ptr[src_idx_20], src_ptr[src_idx_21]);
    //     linx_cvt_package(src3, src_ptr[src_idx_30], src_ptr[src_idx_31]);
    //     // typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
    //     // typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
    //     // typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
    //     // upd_max = blkv_max(upd_max, max0123);
    //     // __bf16x2 max_01, max_23, max_0123;
    //     // blkv_bf16x2_fmax(max_01, src0, src1);
    //     // blkv_bf16x2_fmax(max_23, src2, src3);
    //     // blkv_bf16x2_fmax(max_0123, max_01, max_23);
    //     // blkv_bf16x2_fmax(upd_max, upd_max, max_0123);
    //     // __bf16x2 max_01, max_23, max_0123;
    //     blkv_bf16x2_fmax(src0, src0, src1);
    //     blkv_bf16x2_fmax(src2, src2, src3);
    //     blkv_bf16x2_fmax(src0, src0, src2);
    //     blkv_bf16x2_fmax(upd_max, upd_max, src0);
    // }

    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=1){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        __bf16x2 src0;
        linx_cvt_package(src0, src_ptr[src_idx_00], src_ptr[src_idx_01]);
        blkv_bf16x2_fmax(upd_max, src0, upd_max);
    }
    // scale with norm (sqrt(d_k))
    // upd_max = upd_max * src_scale;
    blkv_bf16x2_mul(upd_max, upd_max, src_scale);
    // uint32_t max_u32 = linx_cvt_as<uint32_t>(upd_max); //*** TODO: check how to assign */
    uint32_t *max_u32 = (uint32_t*)&upd_max;
    __bf16 max0 = ((*max_u32) >> 16) & 0xffff;
    __bf16 max1 = ((*max_u32) & 0xffff);
    new_max_ptr[i*2*tileMax::RowStride] = linx_cvt_as<float>(max0);
    new_max_ptr[(i*2+1)*tileMax::RowStride] = linx_cvt_as<float>(max1);
    // BLKC_ASSIGN_CAST(new_max, max_idx, upd_max);

    // recalculate scale of softmax
    __bf16x2 scale, old_max_bf16x2;
    linx_cvt_package(old_max_bf16x2, old_max_bf160, old_max_bf161);
    blkv_bf16x2_fsub(old_max_bf16x2, old_max_bf16x2, upd_max); 
    blkv_bf16x2_fexp(scale, old_max_bf16x2);
    uint32_t scale_idx00 = i*2*tileScale::RowStride;
    uint32_t scale_idx01 = (i*2+1)*tileScale::RowStride;
    uint32_t *scale_u32 = (uint32_t*)&scale;
    __bf16 scale0 = ((*scale_u32) >> 16) & 0xffff;
    __bf16 scale1 = ((*scale_u32) & 0xffff);
    // unpack bf16x2 to 2xfloat
    scale_ptr[scale_idx00] = linx_cvt_as<float>(scale0);
    scale_ptr[scale_idx01] = linx_cvt_as<float>(scale1);
    // BLKC_ASSIGN_CAST(src_exp, scale_idx, scale);

    // uint32_t sum_idx = i*2*tileSum::RowStride;

    // typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;
    // uint16_t sum_idx00 = i*2*tileSum::RowStride;
    // uint16_t sum_idx01 = (i*2+1)*tileSum::RowStride;
    __bf16x2 upd_sum;
    __bf16 old_sum_bf16_0, old_sum_bf16_1;
    linx_cvt(old_sum_bf16_0, old_sum_ptr[i*2*tileSum::RowStride]); //float->bf16
    linx_cvt(old_sum_bf16_1, old_sum_ptr[(i*2+1)*tileSum::RowStride]); 
    linx_cvt_package(upd_sum, old_sum_bf16_0, old_sum_bf16_1);
    blkv_bf16x2_fmul(upd_sum, upd_sum, scale);  // *** TODO

    // calculate row sum of softmax, l_i
    #pragma clang loop unroll(full)
    for(uint16_t j=0;j<tileSrc::ValidCol;j+=4){
        uint32_t src_idx_00 =  (2*i) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_01 =  (2*i + 1) * tileSrc::RowStride + j * tileSrc::ColStride;
        uint32_t src_idx_10 =  (2*i) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_11 =  (2*i + 1) * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        uint32_t src_idx_20 =  (2*i) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_21 =  (2*i + 1) * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        uint32_t src_idx_30 =  (2*i) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        uint32_t src_idx_31 =  (2*i + 1) * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        __bf16x2 src0, src1, src2, src3;
        __bf16x2 sum01, sum23, sum0123;
        linx_cvt_package(src0, src_ptr[src_idx_00], src_ptr[src_idx_01]);
        linx_cvt_package(src1, src_ptr[src_idx_10], src_ptr[src_idx_11]);
        linx_cvt_package(src2, src_ptr[src_idx_20], src_ptr[src_idx_21]);
        linx_cvt_package(src3, src_ptr[src_idx_30], src_ptr[src_idx_31]);
        blkv_bf16x2_fmul(src0, src0, src_scale);
        blkv_bf16x2_fmul(src1, src1, src_scale);
        blkv_bf16x2_fmul(src2, src2, src_scale);
        blkv_bf16x2_fmul(src3, src3, src_scale);
        blkv_bf16x2_fsub(src0, src0, upd_max);
        blkv_bf16x2_fsub(src1, src1, upd_max);
        blkv_bf16x2_fsub(src2, src2, upd_max);
        blkv_bf16x2_fsub(src3, src3, upd_max);
        blkv_bf16x2_fexp(src0, src0);
        blkv_bf16x2_fexp(src1, src1);
        blkv_bf16x2_fexp(src2, src2);
        blkv_bf16x2_fexp(src3, src3);
        blkv_bf16x2_fadd(sum01, src0, src1);
        blkv_bf16x2_fadd(sum23, src2, src3);
        blkv_bf16x2_fadd(sum0123, sum01, sum23);
        // typename tileSrc::DType exp_src_0 = blkv_bf16x2_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_1 = blkv_bf16x2_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_2 = blkv_bf16x2_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_3 = blkv_bf16x2_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        // typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        // typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        // typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        // upd_sum += exp_src_0123;
        blkv_bf16x2_fadd(upd_sum, upd_sum, sum0123);
        // src_exp_ptr[src_idx_00] = src0;
        // src_exp_ptr[src_idx_10] = src1;
        // src_exp_ptr[src_idx_20] = src2;
        // src_exp_ptr[src_idx_30] = src3;
        BLKC_ASSIGN_CAST(src_exp, src_idx_00, src0);
        BLKC_ASSIGN_CAST(src_exp, src_idx_10, src1);
        BLKC_ASSIGN_CAST(src_exp, src_idx_20, src2);
        BLKC_ASSIGN_CAST(src_exp, src_idx_30, src3);
    }

    // uint32_t sum_u32 = linx_cvt_as<uint32_t>(upd_sum);
    uint32_t *sum_u32 = (uint32_t*)&upd_sum;
    __bf16 sum0 = ((*sum_u32) >> 16) & 0xffff;
    __bf16 sum1 = ((*sum_u32) & 0xffff);
    new_sum_ptr[(i*2)*tileSum::RowStride] = sum0;
    new_sum_ptr[(i*2+1)*tileSum::RowStride] = sum1;
    // new_sum_ptr[sum_idx] = upd_sum;
    // BLKC_ASSIGN_CAST(new_sum, sum_idx, upd_sum);
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

    // typename tileScale::DType local_scale;
    // local_scale.data = blkv_get_tile_ptr(scale)[idx_scale].data;
    typename tileScale::DType local_scale = blkv_get_tile_ptr(scale)[idx_scale];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileO::ValidCol;j++){
        size_t idx = j * tileO::ColStride + i * tileO::RowStride;
        blkv_get_tile_ptr(update_out)[idx] = blkv_get_tile_ptr(old_out)[idx] * local_scale + blkv_get_tile_ptr(pv)[idx];
        // blkv_get_tile_ptr(update_out)[idx] =  blkv_get_tile_ptr(old_out)[idx] + blkv_get_tile_ptr(pv)[idx];
        // __bf16x2 upd
        // BLKC_ASSIGN_CAST(update_out, idx, exp_src_0);
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

    BLKC_ASSIGN_CAST(
        out_cast,
        idx,
        (blkv_get_tile_ptr(last_pv)[idx] +
         blkv_get_tile_ptr(old_out)[idx] * blkv_get_tile_ptr(last_scale)[idx_scale]) /
        blkv_get_tile_ptr(sum)[idx_sum]
    );
}

template<typename tileO_cast, typename tileO, typename tileSum>
void __vec__ normalize_softmax(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ old_out,
                        // const typename tileO::TileDType __in__ last_pv,
                        // const typename tileScale::TileDType __in__ last_scale,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;

    BLKC_ASSIGN_CAST(
        out_cast,
        idx,
        (blkv_get_tile_ptr(old_out)[idx] ) / blkv_get_tile_ptr(sum)[idx_sum]
    );
}


template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint32_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint32_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    // using itQMX = global_iterator<gm_QMX, tile_QMX>;
    // using itKMX = global_iterator<gm_KMX, tile_KMX>;
    // using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    // itQMX gIterQMX(scale_q);
    // itKMX gIterKMX(scale_k);
    // itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                // auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                MGATHER(tQMX[x], gQMX, nd2zz_offset);
                // TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    // auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    // TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            static_assert(Xdim==1);
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){

                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    // src_exp_2src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tW[x][0].data(), tW[x][1].data(),
                    //                                             tNewMax[x].data(),
                    //                                             scale);
                    // new_sum_2src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                    //                                             tNewSum[x].data(),
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tSum[x].data(),
                    //                                             tScale[x].data()
                    //                                             );
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);
                //     // src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                //     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //     //                                             tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //     //                                             tNewMax[x].data(),
                //     //                                             scale);
                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     // new_sum_4src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                //     //                                             tNewSum[x].data(),
                //     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //     //                                             tSum[x].data(),
                //     //                                             tScale[x].data()
                //     //                                             );
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    // auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tV[y], gV);
                    gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    MGATHER(tVMX[y], gVMX, nd2nn_offset); 
                    // TCOPYIN(tVMX[y], gVMX);
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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                }
                TCVT(tPV[x], tPV_out); // float
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
        // normalize使用float
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif

    }
}

// template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
// void flash_attention_2d_unroll_hif4_optsoftmax(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
//     // 全局张量形状
//     static_assert(qD == vD);
//     using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
//     using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
//     using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
//     using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
//     using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
//     using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
//     using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
//     // tile 寄存器形状
//     using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
//     using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
//     using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
//     using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
//     using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
//     using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

//     using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
//     using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
//     using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
//     // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
//     using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
//     using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
//     using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
//     using tileW_left = TileLeft<dtype, kTm, kTk>;

//     using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
//     using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
//     using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

//     using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
//     using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
//     using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
//     using tile_ND2ZZOffset = Tile<Location::Vec, uint32_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
//     using tile_ND2NNOffset = Tile<Location::Vec, uint32_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

//     // 全局迭代器
//     using itQ = global_iterator<gmQ, tileQ>;
//     using itK = global_iterator<gmK, tileK>;
//     using itV = global_iterator<gmV, tileV>;
//     // using itQMX = global_iterator<gm_QMX, tile_QMX>;
//     // using itKMX = global_iterator<gm_KMX, tile_KMX>;
//     // using itVMX = global_iterator<gm_VMX, tile_VMX>;
//     using itO = global_iterator<gmO, tileO>;

//     itQ gIterQ(q_ptr);
//     itK gIterK(k_ptr);
//     itV gIterV(v_ptr);
//     gm_QMX gQMX(scale_q);
//     gm_KMX gKMX(scale_k);
//     gm_VMX gVMX(scale_v);
//     // itQMX gIterQMX(scale_q);
//     // itKMX gIterKMX(scale_k);
//     // itVMX gIterVMX(scale_v);
//     itO gIterO(out_ptr);
//     tile_ND2ZZOffset nd2zz_offset;
//     tile_ND2NNOffset nd2nn_offset;

//     // linx_cvt_package(maxv, -1e30f, -1e30f);
//     // linx_cvt_package(sumv, 0.f, 0.f);
//     // const float scale = 1.0f / sqrt((float)qD);
//     const int Qb = (Sq + kTm - 1) / kTm;
//     const int Kb = (Skv + kTk - 1) / kTk;

//     #ifdef _2D_UNROLL
//     static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
//     static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
//     static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
//     #endif

//     // 对每个 Q-block (i)
//     for (int i = 0; i < Qb; i+=Xdim) {

//         tileQ tQ[Xdim];
//         tile_QMX tQMX[Xdim];
//         // load tile Q,  TODO: add ND2ZZ transform for QMX
//         #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
//             #pragma clang loop unroll(full)
//             for(int x=0;x<Xdim;x+=2){
//                 auto gQ = gIterQ(i+x,0);
//                 TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
//             }
//         #else
//             // TODO: add ND2ZZ transform
//             #pragma clang loop unroll(full)
//             for(int x=0;x<Xdim;x++){
//                 auto gQ = gIterQ(i+x,0);
//                 // auto gQMX = gIterQMX(i+x,0);
//                 TCOPYIN(tQ[x], gQ);
//                 gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
//                 MGATHER(tQMX[x], gQMX, nd2zz_offset);
//                 // TCOPYIN(tQMX[x], gQMX);
//             }
//         #endif

//         tileMax tMax[Xdim];
//         #pragma clang loop unroll(full)
//         for(int x=0;x<Xdim;x++){
//             TEXPANDSCALAR(tMax[x], -1e30f);
//         }

//         tileSum tSum[Xdim];
//         #pragma clang loop unroll(full)
//         for(int x=0;x<Xdim;x++){
//             TEXPANDSCALAR(tSum[x], 0);
//         }

//         tileO_out tPV_out;
//         tileO tO[Xdim], tPV[Xdim];
//         tileScale tScale[Xdim];

//         #pragma clang loop unroll(full)
//         for (int j = 0; j < Kb; j+=Ydim) {

//             tileK tK[Ydim];
//             tile_KMX tKMX[Ydim];
//             // load tile K
//             #ifdef MULTI_LDST
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y+=4){
//                     auto gK = gIterK(0, j+y);
//                     TLOAD4_DN2ZN(tK[y+3], tK[y+2], tK[y+1], tK[y], gK);
//                 }
//             #else
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y++){
//                     auto gK = gIterK(0, j+y);
//                     // auto gKM = gIterKMX(0, j+y);
//                     TCOPYIN(tK[y], gK);
//                     gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
//                     MGATHER(tKMX[y], gKMX, nd2nn_offset); 
//                     // TCOPYIN(tKMX[y], gKM);
//                 }
//             #endif

//             // calculate QK
//             tileW tW[Xdim][Ydim];
//             #pragma clang loop unroll(full)
//             for(int x=0;x<Xdim;x++){
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y++){
//                     tileW_out tW_out;
//                     // MATMULMX(tW_out, tQ[x], tK[y]);
//                     MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
//                     // Nz -> Dn
//                     TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
//                 }
//             }

//             tileMax tNewMax[Xdim];
//             tileSum tNewSum[Xdim];

//             tileW_cast tExpW[Xdim][Ydim];
//             tileP tP[Xdim][Ydim];
//             tile_PMX tP_scale[Xdim][Ydim];
//             tileP_hif4 tP_hif4[Xdim][Ydim];
//             static_assert(Xdim==1);
//             #if Ydim == 1
//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){

//                     static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
//                     // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
//                     if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
//                         // << kTm, 1, 1>
//                         flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
//                                                         tScale[x].data(),
//                                                         tNewMax[x].data(),
//                                                         tNewSum[x].data(),
//                                                         tExpW[x][0].data(),
//                                                         tW[x][0].data(), // 
//                                                         tMax[x].data(),
//                                                         tSum[x].data());
//                         tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
//                     } else {
//                         // << kTm, 1, 1>
//                         // flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
//                         //                                 tScale[x].data(),
//                         //                                 tNewMax[x].data(),
//                         //                                 tNewSum[x].data(),
//                         //                                 tExpW[x][0].data(),
//                         //                                 tW[x][0].data(), // 
//                         //                                 tMax[x].data(),
//                         //                                 tSum[x].data());

//                         pkg_rowmax<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
//                             tScale[x].data(), tNewMax[x].data(), tW[x][0].data(), tMax[x].data());
//                         rowsum<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
//                             tNewSum[x].data(), tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), tSum[x].data(), tScale[x].data()
//                         );
//                         tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
//                     }
//                     // cvt P_new from BF16 to S1P2
//                     // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
//                 }
//             #elif Ydim == 2
//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){
//                     new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
//                                                                 tScale[x].data(),
//                                                                 tNewMax[x].data(),
//                                                                 tW[x][0].data(), tW[x][1].data(),
//                                                                 tMax[x].data(),
//                                                                 scale);
//                     // src_exp_2src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
//                     //                                             tExpW[x][0].data(), tExpW[x][1].data(),
//                     //                                             tW[x][0].data(), tW[x][1].data(),
//                     //                                             tNewMax[x].data(),
//                     //                                             scale);
//                     // new_sum_2src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
//                     //                                             tNewSum[x].data(),
//                     //                                             tExpW[x][0].data(), tExpW[x][1].data(),
//                     //                                             tSum[x].data(),
//                     //                                             tScale[x].data()
//                     //                                             );
//                     src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
//                                                                 tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
//                                                                 tW[x][0].data(), tW[x][1].data(),
//                                                                 tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
//                                                                 scale);
//                 }
//             #elif Ydim == 4
//                 tileSum tLocalSum[Xdim][2];
//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){
//                     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
//                                                                 tScale[x].data(), 
//                                                                 tNewMax[x].data(), 
//                                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
//                                                                 tMax[x].data(),
//                                                                 scale);
//                     // src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
//                     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
//                     //                                             tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
//                     //                                             tNewMax[x].data(),
//                     //                                             scale);
                    
//                     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
//                                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
//                     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
//                                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
//                     // new_sum_4src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
//                     //                                             tNewSum[x].data(),
//                     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
//                     //                                             tSum[x].data(),
//                     //                                             tScale[x].data()
//                     //                                             );
//                     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
//                 }
//             #elif Ydim == 8
//                 tileMax tLocalMax[Xdim][2];
//                 tileSum tLocalSum[Xdim][4];

//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){    
//                     #pragma clang loop unroll(full)
//                     for(int k=0;k<2;k++){
//                         local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
//                     }
//                     new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

//                     #pragma clang loop unroll(full)
//                     for(int k=0;k<4;k++){
//                         src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
//                                                                                                        tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

//                     }
//                     new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
//                 }
//             #elif Ydim == 16
//                 tileMax tLocalMax[Xdim][4];
//                 tileSum tLocalSum[Xdim][4];

//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){       
//                     for(int k=0;k<4;k++){
//                         local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
//                     }
//                     new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


//                     #pragma clang loop unroll(full)
//                     for(int k=0;k<4;k++){
//                         src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
//                                             tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
//                                             tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
//                                             tNewMax[x].data(),
//                                             scale);
//                     }

//                     #pragma clang loop unroll(full)
//                     for(int k=0;k<4;k++){
//                         local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
//                     }
//                     new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
//                 }
//             #else
//                 #ifdef _2D_UNROLL
//                 static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
//                 #endif
//             #endif

//             // load tV
//             tileV tV[Ydim];
//             tile_VMX tVMX[Ydim];
//             #ifdef MULTI_LDST
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y+=4){
//                     auto gV = gIterV(j+y, 0);
//                     TLOAD4_ND2ZN(tV[y+3], tV[y+2], tV[y+1], tV[y], gV);
//                 }
//             #else
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y++){
//                     auto gV = gIterV(j+y, 0);
//                     // auto gVMX = gIterVMX(j+y, 0);
//                     TCOPYIN(tV[y], gV);
//                     gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
//                     MGATHER(tVMX[y], gVMX, nd2nn_offset); 
//                     // TCOPYIN(tVMX[y], gVMX);
//                 }
//             #endif

//             // ColMajor -> Nz
//             // 计算当前块的加权输出: O_j = W * V
//             tileW_left tW_left[Xdim][Ydim];
//             #pragma clang loop unroll(full)
//             for(int x=0;x<Xdim;x++){
//                 #pragma clang loop unroll(full)
//                 for(int y=0;y<Ydim;y++){
//                     if constexpr (type_traits<typename tileO_cast::DType>::bits == 4) {
//                         tileW_cast tW_left_tmp;
//                         TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
//                         TCVT(tW_left[x][y], tW_left_tmp);
// 		            } else {
//                         TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
// 		            }
//                     //TCVT(tW_left[x][y], tExpW[x][y]);
//                     if(y==0){
//                         MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
//                     }else{
//                         MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
//                     }
//                 }
//                 TCVT(tPV[x], tPV_out); // float
//             }

//             // update
//             if(j==0){
//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){
//                     tO[x] = tPV[x];
//                 }
//             }else if(j<(Kb-Ydim)){
//                 #pragma clang loop unroll(full)
//                 for(int x=0;x<Xdim;x++){
//                     global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
//                 }
//             }
//             // 更新最大值状态
//             #pragma clang loop unroll(full)
//             for(int x=0;x<Xdim;x++){
//                 tMax[x] = tNewMax[x];
//                 tSum[x] = tNewSum[x];
//             }
//         }

//         tileO_cast tO_cast[Xdim];
//         #pragma clang loop unroll(full)
//         // normalize使用float
//         for (int x = 0; x < Xdim; ++x) {
//             if constexpr (type_traits<typename tileO_cast::DType>::bits == 4) {
//             normalize_with_last_update_nocast<tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
//             TMOV_NORM(tO_cast[x], tO[x]);
//             } else {
//             normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
//             }
//         }
//         // 写回全局内存

//         #ifdef MULTI_LDST
//             #pragma clang loop unroll(full)
//             for (int x = 0; x < Xdim; x+=2) {
//                 auto dstO = gIterO(i+x, 0);
//                 TSTORE2_DN2DN(dstO, tO_cast[x+1], tO_cast[x]);
//             }
//         #else
//             #pragma clang loop unroll(full)
//             for (int x = 0; x < Xdim; ++x) {
//                 auto dstO = gIterO(i+x, 0);
//                 TCOPYOUT(dstO, tO_cast[x]);//TMOV
//             }
//         #endif

//     }
// }


template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4_nogather(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itQMX = global_iterator<gm_QMX, tile_QMX>;
    using itKMX = global_iterator<gm_KMX, tile_KMX>;
    using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    itQMX gIterQMX(scale_q);
    itKMX gIterKMX(scale_k);
    itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                // gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                // MGATHER(tQMX[x], gQMX, nd2zz_offset);
                TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    // gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    // MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){

                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        static_assert(std::is_same_v<typename tileW_cast::DType, __bf16x2>);
                        flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    // src_exp_2src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tW[x][0].data(), tW[x][1].data(),
                    //                                             tNewMax[x].data(),
                    //                                             scale);
                    // new_sum_2src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                    //                                             tNewSum[x].data(),
                    //                                             tExpW[x][0].data(), tExpW[x][1].data(),
                    //                                             tSum[x].data(),
                    //                                             tScale[x].data()
                    //                                             );
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);
                //     // src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                //     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //     //                                             tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //     //                                             tNewMax[x].data(),
                //     //                                             scale);
                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     // new_sum_4src<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                //     //                                             tNewSum[x].data(),
                //     //                                             tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //     //                                             tSum[x].data(),
                //     //                                             tScale[x].data()
                //     //                                             );
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tVMX[y], gVMX);
                    TCOPYIN(tV[y], gV);
                    // gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    // MGATHER(tVMX[y], gVMX, nd2nn_offset); 

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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                }
                TCVT(tPV[x], tPV_out); // float
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
        // normalize使用float
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif

    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4_optsoftmax(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileWbf16x2      = Tile<Location::Vec, __bf16x2, kTm, kTk/2, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itQMX = global_iterator<gm_QMX, tile_QMX>;
    using itKMX = global_iterator<gm_KMX, tile_KMX>;
    using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    itQMX gIterQMX(scale_q);
    itKMX gIterKMX(scale_k);
    itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                // gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                // MGATHER(tQMX[x], gQMX, nd2zz_offset);
                TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    // gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    // MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            tileWbf16x2 tWbf16x2[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    // tWbf16x2[x][0] = tW[x][0];
                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        static_assert(std::is_same_v<typename tileW_cast::DType, __bf16x2>);
                        // flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                        //                                 tScale[x].data(),
                        //                                 tNewMax[x].data(),
                        //                                 tNewSum[x].data(),
                        //                                 tExpW[x][0].data(),
                        //                                 tW[x][0].data(), // 
                        //                                 tMax[x].data(),
                        //                                 tSum[x].data());
                        // bf16tobf16x2();
                        pkg_rowmax<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tScale[x].data(), tNewMax[x].data(), tW[x][0].data(), tMax[x].data());
                        rowsum<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tNewSum[x].data(), tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), tSum[x].data(), tScale[x].data()
                        );
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);

                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }

                tileSum tLocalSum[Xdim][2];
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    pkg_rowmax_4src<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data());

                    rowsum_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data());
                    rowsum_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data());
                    
                    new_sum_of_2_loc_sum_bf16x2<tileScale, tileSum><<<tileSum::ValidRow/2, 1, 1>>>(
                                                                    tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());

                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][1].data(), tP_scale[x][1].data(), tExpW[x][1].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][2].data(), tP_scale[x][2].data(), tExpW[x][2].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][3].data(), tP_scale[x][3].data(), tExpW[x][3].data());
                }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tVMX[y], gVMX);
                    TCOPYIN(tV[y], gV);
                    // gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    // MGATHER(tVMX[y], gVMX, nd2nn_offset); 

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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                }
                TCVT(tPV[x], tPV_out); // float
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
        // normalize使用float
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif

    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_loadx2(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileWbf16x2      = Tile<Location::Vec, __bf16x2, kTm, kTk/2, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itQMX = global_iterator<gm_QMX, tile_QMX>;
    using itKMX = global_iterator<gm_KMX, tile_KMX>;
    using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    itQMX gIterQMX(scale_q);
    itKMX gIterKMX(scale_k);
    itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                // gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                // MGATHER(tQMX[x], gQMX, nd2zz_offset);
                TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    // gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    // MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            tileWbf16x2 tWbf16x2[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    // tWbf16x2[x][0] = tW[x][0];
                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        static_assert(std::is_same_v<typename tileW_cast::DType, __bf16x2>);
                        // flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                        //                                 tScale[x].data(),
                        //                                 tNewMax[x].data(),
                        //                                 tNewSum[x].data(),
                        //                                 tExpW[x][0].data(),
                        //                                 tW[x][0].data(), // 
                        //                                 tMax[x].data(),
                        //                                 tSum[x].data());
                        // bf16tobf16x2();
                        pkg_rowmax<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tScale[x].data(), tNewMax[x].data(), tW[x][0].data(), tMax[x].data());
                        rowsum<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tNewSum[x].data(), tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), tSum[x].data(), tScale[x].data()
                        );
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);

                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }

                tileSum tLocalSum[Xdim][2];
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    pkg_rowmax_4srcx2<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data());

                    rowsum_2src_with_local_sumx2<tileW, tileW_cast, tileMax, tileSum, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data());
                    rowsum_2src_with_local_sumx2<tileW, tileW_cast, tileMax, tileSum, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data());
                    
                    new_sum_of_2_loc_sum_bf16x2<tileScale, tileSum><<<tileSum::ValidRow/2, 1, 1>>>(
                                                                    tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());

                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][1].data(), tP_scale[x][1].data(), tExpW[x][1].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][2].data(), tP_scale[x][2].data(), tExpW[x][2].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][3].data(), tP_scale[x][3].data(), tExpW[x][3].data());
                }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tVMX[y], gVMX);
                    TCOPYIN(tV[y], gV);
                    // gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    // MGATHER(tVMX[y], gVMX, nd2nn_offset); 

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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                }
                TCVT(tPV[x], tPV_out); // float
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
        // normalize使用float
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif

    }
}

template <typename tile_shape>
void __vec__ get_iden_matrix(typename tile_shape::TileDType __out__ dst
                                 ) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int row_fract_nums =
      tile_shape::Rows / tile_shape::InnerRows;
    __bf16 upd_max;
    linx_cvt(upd_max, 1.0);
    #pragma clang loop unroll(full)
    for (size_t k = 0; k < row_fract_nums; k++) {
        size_t index =
            k * tile_shape::Cols * tile_shape::InnerRows + j * LaneNum + i;
        // blkv_get_tile_ptr(dst)[index] = val;
        BLKC_ASSIGN_CAST(dst, index, upd_max);
  }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileWbf16x2      = Tile<Location::Vec, __bf16x2, kTm, kTk/2, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    // using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    using tileW_cast = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    using tile_Expbf16 = TileLeft<__bf16, kTm, kTk>;
    using tile_iden = TileRight<__bf16, kTk, kTk, kTk, 1>; // vector [kTk x 1]
    using tile_rowsum = TileAcc<float, kTm, kTk, kTm, 1>;
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itQMX = global_iterator<gm_QMX, tile_QMX>;
    using itKMX = global_iterator<gm_KMX, tile_KMX>;
    using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    itQMX gIterQMX(scale_q);
    itKMX gIterKMX(scale_k);
    itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif
    tile_iden tIden;
    static constexpr size_t Y = tile_iden::Cols / (LaneNum / tile_iden::InnerRows);
    // linx_cvt(ones, float(1.0));
    get_iden_matrix<tile_iden><<<LaneNum, Y, 1>>>(tIden.data());
    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                // gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                // MGATHER(tQMX[x], gQMX, nd2zz_offset);
                TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    // gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    // MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            tileWbf16x2 tWbf16x2[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            // get_iden_matrix<<<>>>(tIden, 1.0);
            // TEXPANDSCALAR(tIden, 1.0);
            tile_Expbf16 tExpbf16;
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    // tWbf16x2[x][0] = tW[x][0];
                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        static_assert(std::is_same_v<typename tileW_cast::DType, __bf16x2>);
                        // flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                        //                                 tScale[x].data(),
                        //                                 tNewMax[x].data(),
                        //                                 tNewSum[x].data(),
                        //                                 tExpW[x][0].data(),
                        //                                 tW[x][0].data(), // 
                        //                                 tMax[x].data(),
                        //                                 tSum[x].data());
                        // bf16tobf16x2();
                        pkg_rowmax<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tScale[x].data(), tNewMax[x].data(), tW[x][0].data(), tMax[x].data());
                        rowsum<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tNewSum[x].data(), tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), tSum[x].data(), tScale[x].data()
                        );
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);

                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }

                tileSum tLocalSum[Xdim][2];
                tile_rowsum tRowsum[Xdim][2];
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    pkg_rowmax_4srcx2<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data());

                    
                    rowsum_2src_with_local_expx2<tileW, tileW_cast, tileMax, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tExpW[x][0].data(), tExpW[x][1].data(),
                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data());
                    // tExpbf16 = tExpW[x][0];
                    // rowsum卸载到cube
                    TMOV_DN2NZ(tExpbf16, tExpW[x][0]);
                    // tExpbf16 = tExpW[x][0];
                    MATMUL(tRowsum[x][0], tExpbf16, tIden);
                    TMOV_DN2NZ(tExpbf16, tExpW[x][1]);
                    MATMACC(tRowsum[x][0], tExpbf16, tIden);
                    TCVT(tLocalSum[x][0], tRowsum[x][0]);
                    rowsum_2src_with_local_expx2<tileW, tileW_cast, tileMax, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tExpW[x][2].data(), tExpW[x][3].data(),
                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data());
                    TMOV_DN2NZ(tExpbf16, tExpW[x][2]);
                    MATMUL(tRowsum[x][1], tExpbf16, tIden);
                    TMOV_DN2NZ(tExpbf16, tExpW[x][3]);
                    MATMACC(tRowsum[x][1], tExpbf16, tIden);
                    TCVT(tLocalSum[x][1], tRowsum[x][1]);
                    new_sum_of_2_loc_sum_bf16x2<tileScale, tileSum><<<tileSum::ValidRow/2, 1, 1>>>(
                                                                    tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());

                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][1].data(), tP_scale[x][1].data(), tExpW[x][1].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][2].data(), tP_scale[x][2].data(), tExpW[x][2].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][3].data(), tP_scale[x][3].data(), tExpW[x][3].data());
                }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tVMX[y], gVMX);
                    TCOPYIN(tV[y], gV);
                    // gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    // MGATHER(tVMX[y], gVMX, nd2nn_offset); 

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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                }
                TCVT(tPV[x], tPV_out); // float
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
        // normalize使用float
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif
    }
}

// 卸载global update
template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, uint32_t w_factor=64/4, typename casttype=__bf16>
void flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload2(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, uint8_t *scale_q, uint8_t *scale_k, uint8_t* scale_v) {
    // 全局张量形状
    static_assert(qD == vD);
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD/2>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD/2, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD/2>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD/2>>;  // O: [SxvD]
    using gm_QMX = global_tensor<uint8_t, RowMajor<Sq, qD/w_factor>>; 
    using gm_KMX = global_tensor<uint8_t, ColMajor<qD/w_factor, Skv>>; 
    using gm_VMX = global_tensor<uint8_t, RowMajor<Skv, vD/w_factor>>; 
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD)/2, kTm, qD/2>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD)/2, kTk, qD/2, kTk>;      // [vD×kTk]
    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tile_QMX = Tile<Location::Scaling, uint8_t, kTm, qD, BLayout::RowMajor, kTm, qD/w_factor, SLayout::NoneBox>; // ZZ, 需初始化为0
    using tile_KMX = Tile<Location::Scaling, uint8_t, qD, kTk, BLayout::ColMajor, qD/w_factor, kTk, SLayout::NoneBox>; // NN, 需初始化为0
    using tile_VMX = Tile<Location::Scaling, uint8_t, kTk, vD, BLayout::RowMajor, kTk, vD/w_factor, SLayout::NoneBox>; // NN, 需初始化为0

    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    using tileWbf16x2      = Tile<Location::Vec, __bf16x2, kTm, kTk/2, BLayout::ColMajor>; // transposed, [kTkxkTm]· 计算结果float, 转换成bp16x2，fixp输出，预期bf16x2， fixp做fp32->bf16
    // using tileW_cast = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    using tileW_cast = Tile<Location::Vec, __bf16, kTm, kTk, BLayout::ColMajor>;      // float32->bf16, P_i矩阵， exp(S_i-m_i)， // TODO， 改成bf16x2, 可优化掉
    using tile_Expbf16 = TileLeft<__bf16, kTm, kTk>;
    using tile_iden = TileRight<__bf16, kTk, kTk, kTk, 1>; // vector [kTk x 1]
    using tile_rowsum = TileAcc<float, kTm, kTk, kTm, 1>;
    // using tile_scale_vec = Tile<__bf16, kTm, kTm>;
    using tile_scale_cube = TileLeft<__bf16, kTm, kTm>;
    // using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor, 1, kTm*kTk/w_factor, SLayout::NoneBox>;
    using tile_PMX   = Tile<Location::Scaling, uint8_t, 1, kTm*kTk/w_factor, BLayout::RowMajor>;
    using tileP      = Tile<Location::Vec, casttype, kTm, kTk, BLayout::ColMajor>;        // bf16
    using tileP_hif4 = Tile<Location::Vec, __fp4_e1m2x2, kTm, kTk/2, BLayout::ColMajor>;      // bf16->E1M2x2
    using tileW_left = TileLeft<dtype, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>; // acc 固定float类型
    using tileO_old = Tile<Location::Vec, __bf16, kTm, vD, BLayout::ColMajor>;
    using tileO_r = TileRight<__bf16, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1] // 对齐accelerator, float类型
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tile_ND2ZZOffset = Tile<Location::Vec, uint16_t, 1, kTm*qD/w_factor, BLayout::RowMajor>;
    using tile_ND2NNOffset = Tile<Location::Vec, uint16_t, 1, qD/w_factor*kTk, BLayout::RowMajor>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itQMX = global_iterator<gm_QMX, tile_QMX>;
    using itKMX = global_iterator<gm_KMX, tile_KMX>;
    using itVMX = global_iterator<gm_VMX, tile_VMX>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    gm_QMX gQMX(scale_q);
    gm_KMX gKMX(scale_k);
    gm_VMX gVMX(scale_v);
    itQMX gIterQMX(scale_q);
    itKMX gIterKMX(scale_k);
    itVMX gIterVMX(scale_v);
    itO gIterO(out_ptr);
    tile_ND2ZZOffset nd2zz_offset;
    tile_ND2NNOffset nd2nn_offset;

    // linx_cvt_package(maxv, -1e30f, -1e30f);
    // linx_cvt_package(sumv, 0.f, 0.f);
    // const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif
    tile_iden tIden;
    tile_scale_cube tscale_cube;
    static constexpr size_t Y = tile_iden::Cols / (LaneNum / tile_iden::InnerRows);
    static constexpr size_t scaleY = tile_scale_cube::Cols / (LaneNum / tile_scale_cube::InnerRows);
    // linx_cvt(ones, float(1.0));
    get_iden_matrix<tile_iden><<<LaneNum, Y, 1>>>(tIden.data());

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        tile_QMX tQMX[Xdim];
        // load tile Q,  TODO: add ND2ZZ transform for QMX
        #ifdef MULTI_LDST // don't use, no need for multi tload/tstore 
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x+=2){
                auto gQ = gIterQ(i+x,0);
                TLOAD2_ND2NZ(tQ[x+1], tQ[x], gQ);
            }
        #else
            // TODO: add ND2ZZ transform
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                auto gQ = gIterQ(i+x,0);
                auto gQMX = gIterQMX(i+x,0);
                TCOPYIN(tQ[x], gQ);
                // gen_ND2ZZ_offset_Impl<gm_QMX, tile_QMX, tile_ND2ZZOffset>(gQMX, tQMX[x], nd2zz_offset, i+x, 0); 
                // MGATHER(tQMX[x], gQMX, nd2zz_offset);
                TCOPYIN(tQMX[x], gQMX);
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
            tile_KMX tKMX[Ydim];
            // load tile K
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
                    auto gKM = gIterKMX(0, j+y);
                    TCOPYIN(tK[y], gK);
                    // gen_ND2NN_offset_Impl<gm_KMX, tile_KMX, tile_ND2NNOffset>(gKMX, tKMX[y], nd2nn_offset, 0, j+y);
                    // MGATHER(tKMX[y], gKMX, nd2nn_offset); 
                    TCOPYIN(tKMX[y], gKM);
                }
            #endif

            // calculate QK
            tileW tW[Xdim][Ydim];
            tileWbf16x2 tWbf16x2[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    // MATMULMX(tW_out, tQ[x], tK[y]);
                    MATMULMX(tW_out, tQ[x], tQMX[x], tK[y], tKMX[y]);
                    // Nz -> Dn
                    TCVT(tW[x][y], tW_out); // acc cvt， cube执行， layout转换
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            // get_iden_matrix<<<>>>(tIden, 1.0);
            // TEXPANDSCALAR(tIden, 1.0);
            tile_Expbf16 tExpbf16;
            tileP tP[Xdim][Ydim];
            tile_PMX tP_scale[Xdim][Ydim];
            tileP_hif4 tP_hif4[Xdim][Ydim];
            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    // tWbf16x2[x][0] = tW[x][0];
                    static_assert(tileW::ValidCol % 64 == 0);   // TODO, temorarily support regular shape only.
                    // static_assert(std::is_same_v<typename tileW_cast::DType, __bf16>);
                    if constexpr (std::is_same_v<typename tileW_cast::DType, __bf16>) {
                        // << kTm, 1, 1>
                        flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow, 1, 1>>>(
                                                        tScale[x].data(),
                                                        tNewMax[x].data(),
                                                        tNewSum[x].data(),
                                                        tExpW[x][0].data(),
                                                        tW[x][0].data(), // 
                                                        tMax[x].data(),
                                                        tSum[x].data());
                        tohif4<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data()); // 64 = 1 group, tExpW(Zn, bf16), tP_scale(ZZ, E6M2 with zero E1_8 && E1_16)
                    } else {
                        // << kTm, 1, 1>
                        static_assert(std::is_same_v<typename tileW_cast::DType, __bf16x2>);
                        // flashsoftmax_dn_mout_cast_kernel_bf16x2<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                        //                                 tScale[x].data(),
                        //                                 tNewMax[x].data(),
                        //                                 tNewSum[x].data(),
                        //                                 tExpW[x][0].data(),
                        //                                 tW[x][0].data(), // 
                        //                                 tMax[x].data(),
                        //                                 tSum[x].data());
                        // bf16tobf16x2();
                        pkg_rowmax<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tScale[x].data(), tNewMax[x].data(), tW[x][0].data(), tMax[x].data());
                        rowsum<tileW, tileW_cast, tileMax, tileSum, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                            tNewSum[x].data(), tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), tSum[x].data(), tScale[x].data()
                        );
                        tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    }
                    // cvt P_new from BF16 to S1P2
                    // FCVT(tP_hif4, tP); // 改成fcvt， 放在tohif4里面, TODO, wait for support
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_2src_with_new_sum<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(), tSum[x].data(), tScale[x].data(),
                                                                scale);
                }
            #elif Ydim == 4
                // tileSum tLocalSum[Xdim][2];
                // #pragma clang loop unroll(full)
                // for(int x=0;x<Xdim;x++){
                //     new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                //                                                 tScale[x].data(), 
                //                                                 tNewMax[x].data(), 
                //                                                 tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                //                                                 tMax[x].data(),
                //                                                 scale);

                    
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][0].data(), tExpW[x][0].data(), tExpW[x][1].data(),
                //                                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                //     src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                //                                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                //     new_sum_of_2_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());
                // }

                tileSum tLocalSum[Xdim][2];
                tile_rowsum tRowsum[Xdim][2];
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    pkg_rowmax_4srcx2<tileW, tileW_cast, tileMax, tileScale, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data());

                    
                    rowsum_2src_with_local_expx2<tileW, tileW_cast, tileMax, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tExpW[x][0].data(), tExpW[x][1].data(),
                                                                    tW[x][0].data(), tW[x][1].data(), tNewMax[x].data());
                    // tExpbf16 = tExpW[x][0];
                    // rowsum卸载到cube
                    TMOV_DN2NZ(tExpbf16, tExpW[x][0]);
                    // tExpbf16 = tExpW[x][0];
                    MATMUL(tRowsum[x][0], tExpbf16, tIden);
                    TMOV_DN2NZ(tExpbf16, tExpW[x][1]);
                    MATMACC(tRowsum[x][0], tExpbf16, tIden);
                    TCVT(tLocalSum[x][0], tRowsum[x][0]);
                    rowsum_2src_with_local_expx2<tileW, tileW_cast, tileMax, qD><<<tileW::ValidRow/2, 1, 1>>>(
                                                                    tExpW[x][2].data(), tExpW[x][3].data(),
                                                                    tW[x][2].data(), tW[x][3].data(), tNewMax[x].data());
                    TMOV_DN2NZ(tExpbf16, tExpW[x][2]);
                    MATMUL(tRowsum[x][1], tExpbf16, tIden);
                    TMOV_DN2NZ(tExpbf16, tExpW[x][3]);
                    MATMACC(tRowsum[x][1], tExpbf16, tIden);
                    TCVT(tLocalSum[x][1], tRowsum[x][1]);
                    new_sum_of_2_loc_sum_bf16x2<tileScale, tileSum><<<tileSum::ValidRow/2, 1, 1>>>(
                                                                    tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tSum[x].data(), tScale[x].data());

                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][0].data(), tP_scale[x][0].data(), tExpW[x][0].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][1].data(), tP_scale[x][1].data(), tExpW[x][1].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][2].data(), tP_scale[x][2].data(), tExpW[x][2].data());
                    tohif4_bf16x2<tileP_hif4, tile_PMX, tileW_cast><<<tileW_cast::ValidRow, tileW_cast::ValidCol/64, 1>>>(tP_hif4[x][3].data(), tP_scale[x][3].data(), tExpW[x][3].data());
                }
            #elif Ydim == 8
                tileMax tLocalMax[Xdim][2];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){    
                    #pragma clang loop unroll(full)
                    for(int k=0;k<2;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_2_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tMax[x].data());

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_2src_with_local_sum<tileW, tileW_cast, tileMax, tileSum><<<tileW::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][2*k].data(), tExpW[x][2*k+1].data(),
                                                                                                       tW[x][2*k].data(), tW[x][2*k+1].data(), tNewMax[x].data(), scale);

                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #elif Ydim == 16
                tileMax tLocalMax[Xdim][4];
                tileSum tLocalSum[Xdim][4];

                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){       
                    for(int k=0;k<4;k++){
                        local_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(tLocalMax[x][k].data(), tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(), scale);
                    }
                    new_max_of_4_loc_max<tileScale, tileMax><<<tileMax::ValidRow, 1, 1>>>(tScale[x].data(), tNewMax[x].data(), tLocalMax[x][0].data(), tLocalMax[x][1].data(), tLocalMax[x][2].data(), tLocalMax[x][3].data(), tMax[x].data());


                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        src_exp_4src<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                            tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data(),
                                            tW[x][4*k].data(), tW[x][4*k+1].data(), tW[x][4*k+2].data(), tW[x][4*k+3].data(),
                                            tNewMax[x].data(),
                                            scale);
                    }

                    #pragma clang loop unroll(full)
                    for(int k=0;k<4;k++){
                        local_sum_4src<tileW_cast, tileSum><<<tileSum::ValidRow, 1, 1>>>(tLocalSum[x][k].data(), tExpW[x][4*k].data(), tExpW[x][4*k+1].data(), tExpW[x][4*k+2].data(), tExpW[x][4*k+3].data());
                    }
                    new_sum_of_4_loc_sum<tileScale, tileSum><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tLocalSum[x][0].data(), tLocalSum[x][1].data(), tLocalSum[x][2].data(), tLocalSum[x][3].data(), tSum[x].data(), tScale[x].data());
                }
            #else
                #ifdef _2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4 || Ydim==8 || Ydim==16 , "Not Supprot Ydim != 1 or 2 or 4 or 8 or 16.");
                #endif
            #endif

            // load tV
            tileV tV[Ydim];
            tile_VMX tVMX[Ydim];
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
                    auto gVMX = gIterVMX(j+y, 0);
                    TCOPYIN(tVMX[y], gVMX);
                    TCOPYIN(tV[y], gV);
                    // gen_ND2NN_offset_Impl<gm_VMX, tile_VMX, tile_ND2NNOffset>(gVMX, tVMX[y], nd2nn_offset, j+y, 0);
                    // MGATHER(tVMX[y], gVMX, nd2nn_offset); 

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
                        TMOV_DN2NZ(tW_left_tmp, tP_hif4[x][y]);
                        TCVT(tW_left[x][y], tW_left_tmp);
		            } else {
                        TMOV_DN2NZ(tW_left[x][y], tP_hif4[x][y]);
		            }
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMULMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }else{
                        MATMACCMX(tPV_out, tW_left[x][y], tP_scale[x][y], tV[y], tVMX[y]);
                    }
                    // global update cube offload
                    tileO_old tPV_old;
                    tileO_r tPV_r;
                    tileO_out tPV_scale;
                    get_iden_matrix<tile_scale_cube><<<LaneNum, scaleY, 1>>>(tscale_cube.data()); // todo, change to diag matrix generation, currently use get_iden_matrix to approximate
                    TCVT(tPV_old, tPV_out);
                    TMOV_DN2NZ(tPV_r, tPV_old);
                    MATMUL(tPV_out, tscale_cube, tPV_r);
                }
                TCVT(tPV[x], tPV_out); // float
            }

            // update
            // if(j==0){
            //     #pragma clang loop unroll(full)
            //     for(int x=0;x<Xdim;x++){
            //         tO[x] = tPV[x];
            //     }
            // }else if(j<(Kb-Ydim)){
            //     #pragma clang loop unroll(full)
            //     for(int x=0;x<Xdim;x++){
            //         global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
            //     }
            // }
            // 更新最大值状态
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        tileO_cast tO_cast[Xdim];
        #pragma clang loop unroll(full)
        // normalize使用float
        for (int x = 0; x < Xdim; ++x) {
            if constexpr (type_traits<typename tileO_cast::DType>::bits == 4) {
                normalize_with_last_update_nocast<tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
                TMOV_NORM(tO_cast[x], tO[x]);
            } else {
                normalize_softmax<tileO_cast, tileO, tileSum><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast[x].data(), tPV[x].data(), tSum[x].data());
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
                TCOPYOUT(dstO, tO_cast[x]);//TMOV
            }
        #endif
    }
}
#endif