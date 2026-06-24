#include <common/pto_tileop.hpp>
#include "ds_config.h"
#include "tensor.h"
#include "moe.hpp"
#include "tensorwrite.hpp"

#ifndef ROW
#define ROW 64
#endif

#ifndef DIM
#define DIM 128
#endif

#ifndef INTER_DIM
#define INTER_DIM 256
#endif

int main() {
  Tensor<dtype, ROW, DIM> x;
  Tensor<dtype, DIM, INTER_DIM> w1;
  Tensor<dtype, INTER_DIM, DIM> w2;
  Tensor<dtype, DIM, INTER_DIM> w3;
  Tensor<dtype, ROW, DIM> out;

  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < DIM; ++j) {
        x(i, j) = static_cast<dtype>(i%2+1);
    }
  }

  for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < INTER_DIM; ++j) {
          w1(i, j) = static_cast<dtype>(1);
          w2(j, i) = static_cast<dtype>(2);
          w3(i, j) = static_cast<dtype>(3);
      }
  }

  #ifdef __cpu_sim__
  writeTensorToFile<dtype, ROW, DIM>(x.data(), "x_before.txt");
  writeTensorToFile<dtype, DIM, INTER_DIM>(w1.data(), "weight1_before.txt");
  writeTensorToFile<dtype, INTER_DIM, DIM>(w2.data(), "weight2_before.txt");
  writeTensorToFile<dtype, DIM, INTER_DIM>(w3.data(), "weight3_before.txt");
  #endif
  
  MLP<dtype, ROW, DIM, INTER_DIM>(out.data(), x.data(), w1.data(), w2.data(), w3.data());

  #ifdef __cpu_sim__
  writeTensorToFile<dtype, ROW, DIM>(out.data(), "out_after.txt");
  #endif

  return 0;
}