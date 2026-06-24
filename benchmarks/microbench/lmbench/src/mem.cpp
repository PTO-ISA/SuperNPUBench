#include <common/pto_tileop.hpp>
#include <cstring>
#include "benchmark.h"

// #if DTYPE == "float64"
//     typedef double dtype;
// #elif DTYPE == "float32"
//     typedef float dtype;
// #elif DTYPE == "int8"
//     typedef int8_t dtype;
// #elif DTYPE == "int16"
//     typedef int16_t dtype;
// #elif DTYPE == "int32"
//     typedef int32_t dtype;
// #elif DTYPE == "int64"
//     typedef int64_t dtype;
// #else
//     typedef float dtype;
// #endif

typedef float dtype;

#ifndef STRD
#define STRD 16
#endif

#ifndef GROW
#define GROW 128
#endif

#ifndef GCOL
#define GCOL 128
#endif

#ifndef TROW
#define TROW 64
#endif

#ifndef TCOL
#define TCOL 64
#endif

#ifndef LOOP
#define LOOP 2
#endif

template<typename gm_shape>
void __mtc__ gm_rd(typename gm_shape::DType *src, const uint16_t stride, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j * stride;
  typename gm_shape::DType value;
  value = src[gm_idx];
   __asm__ __volatile__(
    ""
    :
    :"vr"(value)
    :
  );
}

template<typename gm_shape>
void __mtc__ gm_frd(typename gm_shape::DType *src, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j;
  typename gm_shape::DType value;
  value = src[gm_idx];
  __asm__ __volatile__(
    ""
    :
    :"vr"(value)
    :
  );
}

template<typename gm_shape>
void __mtc__ gm_wr(typename gm_shape::DType *dst, const uint16_t stride, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j * stride;
  typename gm_shape::DType value = 0;
  dst[gm_idx] = value;
}

template<typename gm_shape>
void __mtc__ gm_fwr(typename gm_shape::DType *dst, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j;
  typename gm_shape::DType value = 0;
  dst[gm_idx] = value;
}

template<typename gm_shape>
void __mtc__ gm_copy(typename gm_shape::DType *dst, typename gm_shape::DType  *src, const uint16_t stride, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j * stride;
  typename gm_shape::DType value;
  value = src[gm_idx];
  dst[gm_idx] = value;
}

template<typename gm_shape>
void __mtc__ gm_fcopy(typename gm_shape::DType *dst, typename gm_shape::DType *src, const Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>::TileDType __out__ placeholder){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t gm_idx = i * gm_shape::RowStride + j;
  typename gm_shape::DType value;
  value = src[gm_idx];
  dst[gm_idx] = value;
}

template<typename tile_shape>
void __vec__ tile_rd(typename tile_shape::TileDType __in__ src, const uint16_t stride){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j * stride;
  typename tile_shape::DType value;
  value = blkv_get_tile_ptr(src)[tl_idx];
  __asm__ __volatile__(
    ""
    :
    :"vr"(value)
    :
  );
}

template<typename tile_shape>
void __vec__ tile_frd(typename tile_shape::TileDType __in__ src){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j;
  typename tile_shape::DType value;
  value = blkv_get_tile_ptr(src)[tl_idx];
  __asm__ __volatile__(
    ""
    :
    :"vr"(value)
    :
  );
}

template<typename tile_shape>
void __vec__ tile_wr(typename tile_shape::TileDType __out__ dst, const uint16_t  stride){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j * stride;
  typename tile_shape::DType value = 0;
  blkv_get_tile_ptr(dst)[tl_idx] = value;
}

template<typename tile_shape>
void __vec__ tile_fwr(typename tile_shape::TileDType __out__ dst){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j;
  typename tile_shape::DType value = 0;
  blkv_get_tile_ptr(dst)[tl_idx] = value;
}

template<typename tile_shape>
void __vec__ tile_copy(typename tile_shape::TileDType __out__ dst, typename tile_shape::TileDType __in__ src, const uint16_t stride){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j * stride;
  typename tile_shape::DType value;
  value = blkv_get_tile_ptr(src)[tl_idx];
  blkv_get_tile_ptr(dst)[tl_idx] = value;
}

template<typename tile_shape>
void __vec__ tile_fcopy(typename tile_shape::TileDType __out__ dst, typename tile_shape::TileDType __in__ src){
  uint16_t j = blkv_get_index_x();
  uint16_t i = blkv_get_index_y();
  uint16_t tl_idx = i * tile_shape::RowStride + j;
  typename tile_shape::DType value;
  value = blkv_get_tile_ptr(src)[tl_idx];
  blkv_get_tile_ptr(dst)[tl_idx] = value;
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_nd(dtype *src) {
  using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, dtype, tile_row, tile_col, BLayout::RowMajor>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;
  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gsrc(src + offset);
      tile_shape tsrc;
      TLOAD(tsrc, gsrc);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_dn(dtype *src) {
  using gm_shape = global_tensor<dtype, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, dtype, tile_row, tile_col, BLayout::ColMajor>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;
  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gsrc(src + offset);
      tile_shape tsrc;
      TLOAD(tsrc, gsrc);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_nd2nz(dtype *src) {
  using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileLeft<dtype, tile_row, tile_col>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;
  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gsrc(src + offset);
      tile_shape tsrc;
      TLOAD(tsrc, gsrc);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_nd2zn(dtype *src) {
  using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileRight<dtype, tile_row, tile_col>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;
  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gsrc(src + offset);
      tile_shape tsrc;
      TLOAD(tsrc, gsrc);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_dn2nz(dtype *src) {
}


template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tload_dn2zn(dtype *src) {
  using gm_shape = global_tensor<dtype, ColMajor<gm_row, gm_col>>;
  using tile_shape = TileRight<dtype, tile_row, tile_col>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;
  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gsrc(src + offset);
      tile_shape tsrc;
      TLOAD(tsrc, gsrc);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_nd(dtype *dst) {
  using gm_shape   = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, dtype, tile_row, tile_col, BLayout::RowMajor>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;

  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gdst(dst + offset);
      tile_shape tdst(0);
      TSTORE(gdst, tdst);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_dn(dtype *dst) {
  using gm_shape   = global_tensor<dtype, ColMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, dtype, tile_row, tile_col, BLayout::ColMajor>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;

  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gdst(dst + offset);
      tile_shape tdst(0);
      TSTORE(gdst, tdst);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_nz2nd(dtype *dst) {
  using gm_shape   = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileLeft<dtype, tile_row, tile_col>;
  uint16_t brow = gm_shape::Rows / tile_shape::Rows;
  uint16_t bcol = gm_shape::Cols / tile_shape::Cols;

  #pragma clang loop unroll(full)
  for (int i = 0; i < brow; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < bcol; ++j) {
      uint16_t offset = i * (tile_shape::Rows * gm_shape::Cols) + j * tile_shape::Cols;
      gm_shape gdst(dst + offset);
      tile_shape tdst(0);
      TSTORE(gdst, tdst);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_nz2dn(dtype *dst) {

}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_zn2nd(dtype *dst) {

}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void tstore_zn2dn(dtype *dst) {

}

int main() {
  size_t gm_size = GROW * GCOL;
  size_t tile_size = TROW * TCOL;
  using gm_shape = global_tensor<dtype, RowMajor<GROW, GCOL>>;
  using tile_shape = Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor>;
  Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tsrc(0);

  BENCHSTART;
  #pragma clang loop unroll(full)
  for(int i=0;i<LOOP;i++){
    if(!strcmp(MODE, "gm_rd")){
      dtype src[gm_size];
      gm_shape gsrc(src);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      uint16_t real_col = gm_shape::Cols / (STRD/sizeof(dtype));
      gm_rd<gm_shape><<<real_col, gm_shape::Rows, 1>>>(gsrc.data(), static_cast<uint16_t>(STRD/sizeof(dtype)), tmp.data());
    }
    else if(!strcmp(MODE, "gm_frd")){
      dtype src[gm_size];
      gm_shape gsrc(src);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      gm_frd<gm_shape><<<gm_shape::Cols, gm_shape::Rows, 1>>>(gsrc.data(), tmp.data());
    }
    else if(!strcmp(MODE, "gm_wr")){
      dtype dst[gm_size];
      gm_shape gdst(dst);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      uint16_t real_col = gm_shape::Cols / (STRD/sizeof(dtype));
      gm_wr<gm_shape><<<real_col, gm_shape::Rows, 1>>>(gdst.data(), static_cast<uint16_t>(STRD/sizeof(dtype)), tmp.data());
    }
    else if(!strcmp(MODE, "gm_fwr")){
      dtype dst[gm_size];
      gm_shape gdst(dst);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      gm_fwr<gm_shape><<<gm_shape::Cols, gm_shape::Rows, 1>>>(gdst.data(), tmp.data());
    }
    else if(!strcmp(MODE, "gm_copy")){
      dtype src[gm_size];
      dtype dst[gm_size];
      gm_shape gsrc(src);
      gm_shape gdst(dst);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      uint16_t real_col = gm_shape::Cols / (STRD/sizeof(dtype));
      gm_copy<gm_shape><<<real_col, gm_shape::Rows, 1>>>(gdst.data(), gsrc.data(), static_cast<uint16_t>(STRD/sizeof(dtype)), tmp.data());
    }
    else if(!strcmp(MODE, "gm_fcopy")){
      dtype src[gm_size];
      dtype dst[gm_size];
      gm_shape gsrc(src);
      gm_shape gdst(dst);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tmp;
      gm_fcopy<gm_shape><<<gm_shape::Cols, gm_shape::Rows, 1>>>(gdst.data(), gsrc.data(), tmp.data());
    }
    else if(!strcmp(MODE, "tile_rd")){
      //Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tsrc(0);
      uint16_t real_col = tile_shape::Cols / (STRD/sizeof(dtype));
      tile_rd<tile_shape><<<real_col, tile_shape::Rows, 1>>>(tsrc.data(), static_cast<uint16_t>(STRD/sizeof(dtype)));
    }
    else if(!strcmp(MODE, "tile_frd")){
      //Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tsrc(0);
      tile_frd<tile_shape><<<tile_shape::Cols, tile_shape::Rows, 1>>>(tsrc.data());
    }
    else if(!strcmp(MODE, "tile_wr")){
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tdst;
      uint16_t real_col = tile_shape::Cols / (STRD/sizeof(dtype));
      tile_wr<tile_shape><<<real_col, tile_shape::Rows, 1>>>(tdst.data(), static_cast<uint16_t>(STRD/sizeof(dtype)));
    }
    else if(!strcmp(MODE, "tile_fwr")){
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tdst;
      tile_fwr<tile_shape><<<tile_shape::Cols, tile_shape::Rows, 1>>>(tdst.data());
    }
    else if(!strcmp(MODE, "tile_copy")){
      //Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tsrc(0);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tdst;
      uint16_t real_col =tile_shape::Cols / (STRD/sizeof(dtype));
      tile_copy<tile_shape><<<real_col, tile_shape::Rows, 1>>>(tdst.data(), tsrc.data(), static_cast<uint16_t>(STRD/sizeof(dtype)));
    }
    else if(!strcmp(MODE, "tile_fcopy")){
      //Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tsrc(0);
      Tile<Location::Vec, dtype, TROW, TCOL, BLayout::RowMajor> tdst;
      tile_fcopy<tile_shape><<<tile_shape::Cols, tile_shape::Rows, 1>>>(tdst.data(), tsrc.data());
    }
    else if(!strcmp(MODE, "tload_nd")){
      dtype src[gm_size];
      tload_nd<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tload_dn")){
      dtype src[gm_size];
      tload_dn<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tload_nd2nz")){
      dtype src[gm_size];
      tload_nd2nz<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tload_nd2zn")){
      dtype src[gm_size];
      tload_nd2zn<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tload_dn2nz")){
      // dtype src[gm_size];
      // tload_dn2nz<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tload_dn2zn")){
      dtype src[gm_size];
      tload_dn2zn<GROW, GCOL, TROW, TCOL>(src);
    }
    else if(!strcmp(MODE, "tstore_nd")){
      dtype dst[gm_size];
      tstore_nd<GROW, GCOL, TROW, TCOL>(dst);
    }
    else if(!strcmp(MODE, "tstore_dn")){
      dtype dst[gm_size];
      tstore_dn<GROW, GCOL, TROW, TCOL>(dst);
    }
    else if(!strcmp(MODE, "tstore_nz2nd")){
      dtype dst[gm_size];
      tstore_nz2nd<GROW, GCOL, TROW, TCOL>(dst);
    }
    // else if(!strcmp(MODE, "tstore_nz2dn")){
    //   dtype dst[gm_size];
    //   tstore_nz2dn<GROW, GCOL, TROW, TCOL>(dst);
    // }
    // else if(!strcmp(MODE, "tstore_zn2nd")){
    //   dtype dst[gm_size];
    //   tstore_zn2nd<GROW, GCOL, TROW, TCOL>(dst);
    // }
    // else if(!strcmp(MODE, "tstore_zn2dn")){
    //   dtype dst[gm_size];
    //   tstore_zn2dn<GROW, GCOL, TROW, TCOL>(dst);
    // }
  }
  BENCHEND;

  return 0;
}