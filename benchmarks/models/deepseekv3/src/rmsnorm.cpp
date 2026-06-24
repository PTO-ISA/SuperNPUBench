#include "ds_config.h"
#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "mla.hpp"

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
  Tensor<dtype, globM, globN> x;

  rmsnorm<dtype, x.shape[0], x.shape[1], tilM, tilN>(x.data(), x.data());
  return 0;
}