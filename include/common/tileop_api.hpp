#ifndef TILEOP_API_HPP
#define TILEOP_API_HPP

#include "common/tileop_api_impl.hpp"

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
              is_tile_data_v tile_shape_C>
void MATMACC(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  MATMACC_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_C>
void MATMUL(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  MATMUL_Impl(dst, src0, src1);
}

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_AX,
          is_tile_data_v tile_shape_B, is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMULMX(tile_shape_C &dst, tile_shape_A &src0, tile_shape_AX &src0_scale,
             tile_shape_B &src1, tile_shape_BX &src1_scale) {
  MATMULMX_Impl(dst, src0, src0_scale, src1, src1_scale);
}

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_BX, is_tile_data_v tile_shape_C>
void MATMULMXB(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1, tile_shape_BX &src1_scale) {
  MATMULMXB_Impl(dst, src0, src1, src1_scale);
}

template <is_tile_data_v tile_shape_A,  is_tile_data_v tile_shape_AX,
          is_tile_data_v tile_shape_B,  is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMACCMX(tile_shape_C &dst, tile_shape_A &src0,  tile_shape_AX &src0x,
               tile_shape_B &src1,  tile_shape_BX &src1x) {
  MATMACCMX_Impl(dst, src0, src0x, src1, src1x);
}

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B, 
          is_tile_data_v tile_shape_BX, is_tile_data_v tile_shape_C>
void MATMACCMXB(tile_shape_C &dst, tile_shape_A &src0,
                tile_shape_B &src1, tile_shape_BX &src1x) {
  MATMACCMXB_Impl(dst, src0, src1, src1x);
}

template <is_tile_data_v tile_shape> void TABS(tile_shape &dst, tile_shape &src) {
  TABS_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TADD_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TADDS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TADDS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape>
void TAND(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TAND_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1,
          is_tile_data_v tile_shape_in2, is_tile_data_v tile_shape_out>
void TASSEMBLE(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1,
               tile_shape_in2 &src2) {
  TASSEMBLE_Impl(dst, src0, src1, src2);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST(tile_shape_out &dst, tile_shape_in &src) {
  TCAST_Impl(dst, src);
}
template <is_tile_data_v tile_shape, typename T, int descending>
void TCI(tile_shape &dst, T s) {
  TCI_Impl<tile_shape, T, descending>(dst, s);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCMP(tile_shape_out &dst, tile_shape_in &src0, tile_shape_in &src1, CmpMode cmpMode) {
  TCMP_Impl(dst, src0, src1, cmpMode);
}
template <is_tile_data_v tile_shape>
void TCOPY(tile_shape &dst, tile_shape &src) {
  TCOPY_Impl(dst, src);
}
template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TCOPYIN(tile_shape &dst, gm_shape &src) {
  TCOPYIN_Impl(dst, src);
}
template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TCOPYOUT(gm_shape &dst, tile_shape &src) {
  TCOPYOUT_Impl(dst, src);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT(tile_shape_out &dst, tile_shape_in &src) {
  TCVT_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TDIV(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TDIV_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TDIVS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TDIVS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape> void TEXP(tile_shape &dst, tile_shape &src) {
  TEXP_Impl(dst, src);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDCOL(tile_shape_out &dst, tile_shape_in &src) {
  TEXPANDCOL_Impl(dst, src);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDROW(tile_shape_out &dst, tile_shape_in &src) {
  TEXPANDROW_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TEXPANDSCALAR(tile_shape &dst, typename tile_shape::DType s) {
  TEXPANDSCALAR_Impl(dst, s);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXTRACT(tile_shape_out &dst, tile_shape_in &src, uint16_t offset_i,
              uint16_t offset_j) {
  TEXTRACT_Impl(dst, src, offset_i, offset_j);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TFILLPAD(tile_shape_out &dst, tile_shape_in &src) {
  TFILLPAD_Impl(dst, src);
}
template <is_tile_data_v tile_shape_dst, is_tile_data_v tile_shape_src,
          is_tile_data_v tile_shape_indices>
void TGATHER(tile_shape_dst &dst, tile_shape_src &src,
             tile_shape_indices &indices) {
  TGATHER_Impl(dst, src, indices);
}
template <is_tile_data_v tile_shape>
void TMAX(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TMAX_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TMAXS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TMAXS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape>
void TMIN(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TMIN_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TMINS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TMINS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape>
void TMUL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TMUL_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TMULS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TMULS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape>
void TOR(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TOR_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TXOR(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TXOR_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TSLL(tile_shape &dst, tile_shape &src, unsigned sh) {
  TSLL_Impl(dst, src, sh);
}
template <is_tile_data_v tile_shape>
void TSRL(tile_shape &dst, tile_shape &src, unsigned sh) {
  TSRL_Impl(dst, src, sh);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in, typename T>
void TPAD(tile_shape_out &dst, const tile_shape_in &src, T pad_value,
          size_t up_pad, size_t left_pad, size_t down_pad, size_t right_pad) {
  TPAD_Impl(dst, src, pad_value, up_pad, left_pad, down_pad, right_pad);
}
template <is_tile_data_v tile_shape>
void TRECIP(tile_shape &dst, tile_shape &src) {
  TRECIP_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TREM(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TREM_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TRESHAPE(tile_shape_out &tile_out, tile_shape_in &tile_in) {
  TRESHAPE_Impl(tile_out, tile_in);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWMAX(tile_shape_out &dst, tile_shape_in &src) {
  TROWMAX_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TROWMAXEXPAND(tile_shape &dst, tile_shape &src) {
  TROWMAXEXPAND_Impl(dst, src);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM(tile_shape_out &dst, tile_shape_in &src) {
  TROWSUM_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TROWSUMEXPAND(tile_shape &dst, tile_shape &src) {
  TROWSUMEXPAND_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TRSQRT(tile_shape &dst, tile_shape &src) {
  TRSQRT_Impl(dst, src);
}
template <is_tile_data_v tile_shape_dst, is_tile_data_v tile_shape_src,
          is_tile_data_v tile_shape_indices>
void TSCATTER(tile_shape_dst &dst, tile_shape_src &src,
                   tile_shape_indices &indices) {
  TSCATTER_Impl(dst, src, indices);
}
template <is_tile_data_v tile_shape, is_tile_data_v tile_shape_index>
void TSELECT(tile_shape &dst, tile_shape_index &cond, tile_shape &src0,
             tile_shape &src1) {
  TSELECT_Impl(dst, cond, src0, src1);
}
template <is_tile_data_v tile_shape>
void TSQRT(tile_shape &dst, tile_shape &src) {
  TSQRT_Impl(dst, src);
}
template <is_tile_data_v tile_shape>
void TSUB(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TSUB_Impl(dst, src0, src1);
}
template <is_tile_data_v tile_shape>
void TSUBS(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  TSUBS_Impl(dst, src, s);
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TTRANS(tile_shape_out &dst, tile_shape_in &src) {
  TTRANS_Impl(dst, src);
}

#endif