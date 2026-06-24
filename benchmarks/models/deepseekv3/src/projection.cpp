#include <common/pto_tileop.hpp>
#include "ds_config.h"
#include "tensor.h"
#include "mla.hpp"
#include "tensorwrite.hpp"

#ifndef BSZ
#define BSZ 2
#endif

#ifndef SLEN
#define SLEN 128
#endif

#ifndef DIM_IN
#define DIM_IN 128
#endif

#ifndef DIM_OUT
#define DIM_OUT 128
#endif

int main() {
  Tensor<dtype, BSZ, SLEN, DIM_IN> x;
  Tensor<dtype, DIM_IN, DIM_OUT> weight;
  Tensor<dtype, BSZ, SLEN, DIM_OUT> out;

  for (int i = 0; i < BSZ; ++i) {
    for (int j = 0; j < SLEN; ++j) {
      for (int k = 0; k < DIM_IN; ++k) {
          x(i, j, k) = static_cast<dtype>(i%2+1);
      }
    }
  }

  for (int i = 0; i < DIM_IN; ++i) {
      for (int j = 0; j < DIM_OUT; ++j) {
          weight(i, j) = static_cast<dtype>(1);
      }
  }

  #ifdef __cpu_sim__
  writeTensorToFile<dtype, BSZ*SLEN, DIM_IN>(x.data(), "x_before.txt");
  writeTensorToFile<dtype, DIM_IN, DIM_OUT>(weight.data(), "weight_before.txt");
  #endif
  
  projection<dtype, BSZ, SLEN, DIM_IN, DIM_OUT>(out, x, weight);

  #ifdef __cpu_sim__
  writeTensorToFile<dtype, BSZ*SLEN, DIM_OUT>(out.data(), "out_after.txt");
  #endif

  return 0;
}