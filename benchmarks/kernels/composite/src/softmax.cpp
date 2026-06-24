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

int main() {

  float src[globM*globN];
  float dst[globM*globN];

  BENCHSTART;
  softmax<float, globM, globN, tilM, tilN>(dst,src);
  BENCHEND;

  return 0;
}