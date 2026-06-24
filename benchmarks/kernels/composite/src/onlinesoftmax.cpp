#include <common/pto_tileop.hpp>
#include <cstring>
#include "softmax.hpp"
#include "benchmark.h"

#ifndef globM
#define globM 120
#endif

#ifndef globN
#define globN 120
#endif

#ifndef tilM
#define tilM  16
#endif

#ifndef tilN
#define tilN  16
#endif

using namespace pto;

template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void onlinesoftmax_test(dtype* dst, dtype* src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using gm_shape1 = global_tensor<dtype, RowMajor<kTM, 1>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
    using tile_shape1 = Tile<Location::Vec, dtype, kTM, kN, BLayout::RowMajor>;
    // using tile_shape2 = Tile<Location::Vec, dtype, kTM, 1, BLayout::RowMajor>;
    using tile_shape2 = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor, kTM, 1>;
    using tDst = Tile<Location::Vec, dtype, kTM, 2*kTN, BLayout::RowMajor>;

    int Mb = kM/kTM;
    int Nb = kN/kTN;

    for(int i=0;i<Mb;i++){
      tile_shape2 tsum(0);
      tile_shape2 tmax(-10000);
      tile_shape tsrc;
      for (int j = 0; j < Nb; j++) {
        uint32_t offset = i * kTM * kN + j * kTN;
        gm_shape gsrc(src + offset);
        // tile_shape tsrc;
        tDst tdst;
        TLOAD(tsrc, gsrc);
        OnlineSoftMax(tdst,tsrc,tmax,tsum);
        TEXTRACT(tsrc, tdst, 0, kTN);
        TEXTRACT(tmax, tdst, 0, 0);
        TEXTRACT(tsum, tdst, 0, 1);
      }
        int offset1 = i * kTM * kN;
        // gm_shape1 res(dst);
        // TSTORE(res,tmax);
        gm_shape gsrc1(src + offset1);
        gm_shape res(dst + offset1);
        tile_shape1 d4;
        tile_shape1 d5;
        tile_shape1 d6;

        TLOAD(d4, gsrc1);
        TEXPANDCOL(d5, tmax);
        TEXPANDCOL(d6, tsum);
        TSUB(d4, d4, d5);
        TEXP(d4, d4);
        TDIV(d4, d4, d6);
        TSTORE(res, d4);
    }
}

int main() {

  float src[globM*globN];
  float dst[globM*globN];
  // 循环赋值1.0
  for (size_t i = 0; i < globM*globN; ++i) {
    src[i]=sin((i + 1) / 100.0f);
    dst[i]=0.0f;
  }

  BENCHSTART;
  onlinesoftmax_test<float, globM, globN, tilM, tilN>(dst,src);
  BENCHEND;

  // for (uint16_t i = 0; i < globM*globN; i++) {
  //   std::cout << dst[i] << " ";
  // }
  // std::cout << std::endl;

  return 0;
}