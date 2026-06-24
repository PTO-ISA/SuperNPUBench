#include <common/pto_tileop.hpp>
#include "ds_config.h"
#include "tensor.h"
#include "moe.hpp"
#include "tensorwrite.hpp"

#ifndef ROW
#define ROW 64
#endif

#ifndef DIM
#define DIM 8
#endif

#ifndef IDX_DIM
#define IDX_DIM 4
#endif

#ifndef EXT_DIM
#define EXT_DIM 16
#endif

int main() {
    Tensor<dtype, ROW, DIM> x;
    Tensor<dtype, ROW, IDX_DIM> idx;
    Tensor<dtype, ROW, DIM, EXT_DIM> scatter_out;

    Tensor<dtype, ROW, DIM, EXT_DIM> mask_out;

    int tmp = 0;
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < DIM; ++j) {
            x(i, j) = static_cast<dtype>(1);
        }
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < IDX_DIM; ++j) {
            idx(i, j) = static_cast<dtype>(std::rand()%DIM);
        }
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < DIM; ++j) {
            for (int k = 0; k < EXT_DIM; ++k) {
                mask_out(i,j,k) = static_cast<dtype>(1);
            }
        }
    }

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, ROW, DIM>(x.data(), "x_before.txt");
    writeTensorToFile<dtype, ROW, IDX_DIM>(idx.data(), "idx_before.txt");
    #endif
  
    scatter_expand<dtype, ROW, DIM, IDX_DIM, EXT_DIM>(scatter_out.data(), idx.data(), x.data(), 0);

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, ROW*DIM, EXT_DIM>(scatter_out.data(), "scatter_after.txt");
    #endif

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, ROW*DIM, EXT_DIM>(mask_out.data(), "mask_data_before.txt");
    #endif

    mask_fill<dtype, ROW*DIM, EXT_DIM>(mask_out.data(), scatter_out.data(), static_cast<dtype>(-1.0f));

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, ROW*DIM, EXT_DIM>(mask_out.data(), "mask_data_after.txt");
    #endif

  return 0;
}