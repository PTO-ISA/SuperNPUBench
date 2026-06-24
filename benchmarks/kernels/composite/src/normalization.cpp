#include <common/pto_tileop.hpp>
#include <cstring>
#include "normalization.hpp"

#ifndef globM
#define globM 320
#endif

#ifndef globN
#define globN 128
#endif

#ifndef tilM
#define tilM 64
#endif

#ifndef tilN
#define tilN 32
#endif

int main() {

  #ifdef LINX_PMC
  PMC_START();
  #endif

  float src[globM*globN];
  float dst[globM*globN];

  if(!strcmp(MODE, "layer")){
    layernorm<float, globM, globN, tilM, tilN>(dst, src);
  }else if(!strcmp(MODE, "rms")){
    rmsnorm<float, globM, globN, tilM, tilN>(dst, src);
  }

  #ifdef LINX_PMC
  PMC_END();
  #endif

  return 0;
}