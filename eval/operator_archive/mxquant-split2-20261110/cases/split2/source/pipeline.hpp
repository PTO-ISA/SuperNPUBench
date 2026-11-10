#pragma once
#include "bulk.hpp"

namespace mxquant_pipeline {
using namespace mxquant_bulk;

// Each PE owns four 32-row bands. Increase the lookahead while retaining
// distinct input backing; no arithmetic or quantization changes.
template<int Variant>
inline void run(__fp8_e4m3 *out,__fp8_e8m0 *scales,const __bf16 *in) {
  const auto tid=get_thread_idx();
  if(tid>=4)return;
  in+=tid*128*256;out+=tid*128*256;scales+=tid*128*8;
  Input x0,x1,x2,x3;
  load(x0,in);
  load(x1,in+32*256);
  load(x2,in+2*32*256);
  if constexpr(Variant==6)load(x3,in+3*32*256);
  block<true>(x0,out,scales);
  if constexpr(Variant==5)load(x3,in+3*32*256);
  block<true>(x1,out+32*256,scales+32*8);
  block<true>(x2,out+2*32*256,scales+2*32*8);
  block<true>(x3,out+3*32*256,scales+3*32*8);
}
}
