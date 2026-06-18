#ifndef MATADD_KERNEL_HPP
#define MATADD_KERNEL_HPP

#include <common/pto_tileop.hpp>

using namespace pto;

// 将两个Matrix做element-wise的加法。
// 将`A`和`B`从global memory加载至tile-reg，最终拷贝回global memory `C`

template <const int kM, const int kN, const int kTM, const int kTN>
void matadd(float *c_ptr, float *a_ptr, float *b_ptr) {
  using gm_shape = global_tensor<float, RowMajor<kM, kN>>;
  using tile_shape = Tile<Location::Vec, float, kTM, kTN, BLayout::RowMajor>;
  using gm_iterator = global_iterator<gm_shape, tile_shape>;

  gm_iterator gAIter(a_ptr);
  gm_iterator gBIter(b_ptr);
  gm_iterator gCIter(c_ptr);
  const int Mb = (kM + kTM - 1) / kTM;
  const int Nb = (kN + kTN - 1) / kTN;

  for (int i = 0; i < Mb; ++i) {
    for (int j = 0; j < Nb; ++j) {
      auto gA = gAIter(i, j);
      auto gB = gBIter(i, j);
      auto gC = gCIter(i, j);

      tile_shape tA, tB, tC;
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
  }
}

#endif
