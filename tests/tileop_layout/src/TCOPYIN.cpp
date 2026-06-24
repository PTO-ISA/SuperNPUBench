#include <common/pto_tileop.hpp>
#include <string> 

#ifdef LINX_PMC
#include <linxStartEnd.hpp>
#endif

#ifndef GM_ROW
#define GM_ROW 16
#endif

#ifndef GM_COL
#define GM_COL 16
#endif

#ifndef TR_ROW
#define TR_ROW 8
#endif

#ifndef TR_COL
#define TR_COL 8
#endif

#ifndef MODE
#define MODE "ND2ND"
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void copyin_nd2nd(float *src) {
  using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;
  //带mask tile_shape = Tile<Location::Vec, float, tile_row, tile_col, 11, 11>;
  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;
  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      int offset = i * (tile_row * gm_col) + j * tile_col;
      gm_shape s0(src + offset);
      tile_shape d0; 
      TCOPYIN(d0, s0);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void copyin_nd2nz(float *src) {
  using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileLeft<float, tile_row, tile_col>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;

  using iter = global_iterator<gm_shape, tile_shape>;

  iter gsrc(src);

  for (int i = 0; i < block_row; ++i) {
    for (int j = 0; j < block_col; ++j) {
      tile_shape d0; 
      auto g0 =  gsrc(i,j);
      TCOPYIN(d0, g0);
    }
  }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void copyin_dn2zn(float *src) {
  using gm_shape = global_tensor<float, ColMajor<gm_row, gm_col>>;
  using tile_shape = TileRight<float, tile_row, tile_col>;

  uint16_t block_row = gm_row / tile_row;
  uint16_t block_col = gm_col / tile_col;

  using iter = global_iterator<gm_shape, tile_shape>;

  iter gsrc(src);

  for (int i = 0; i < block_col; ++i) {
    for (int j = 0; j < block_row; ++j) {
      tile_shape d0; 
      auto g0 =  gsrc(i,j);
      TCOPYIN(d0, g0);
    }
  }
}

int main() {
  const uint16_t gm_row = GM_ROW;
  const uint16_t gm_col = GM_COL;
  const uint16_t tile_row = TR_ROW;
  const uint16_t tile_col = TR_COL;

  size_t gm_size = gm_row * gm_col;
  size_t tile_size = tile_row * tile_col;

  #ifdef LINX_PMC
  PMC_START();
  #endif

  float src[gm_size];

  if(!strcmp(MODE, "ND2ND")){
    copyin_nd2nd<gm_row, gm_col, tile_row, tile_col>(src);
  }else if(!strcmp(MODE, "ND2NZ")){
    copyin_nd2nz<gm_row, gm_col, tile_row, tile_col>(src);
  }else if(!strcmp(MODE, "DN2ZN")){
    copyin_dn2zn<gm_row, gm_col, tile_row, tile_col>(src);
  }

  #ifdef LINX_PMC
  PMC_END();
  #endif
  // asm volatile(
  //   "bstart.aux fall\n"
  //   "acrc 3\n"
  //   "c.bstop\n"
  //   :::);

  return 0;
}