#include <common/pto_tileop.hpp>
#include "ds_config.h"
#include "tensor.h"
#include "moe.hpp"
#include "tensorwrite.hpp"

#ifndef ROW
#define ROW 64
#endif

#ifndef DIM
#define DIM 32
#endif

#ifndef TOPK
#define TOPK 8
#endif

int main(){
    Tensor<dtype, ROW, DIM> x;
    Tensor<dtype, ROW, TOPK> weight;
    Tensor<dtype, ROW, TOPK> indices;

    int tmp = 0;
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < DIM; ++j) {
            //x(i, j) = static_cast<dtype>(rand()%32);
            // x(i, j) = static_cast<dtype>((tmp+5)%16);
            // tmp++;
            x(i, j) = static_cast<dtype>(tmp++);
        }
    }

    // printf("input x: \n");
    // for (int i = 0; i < ROW; ++i) {
    //     for (int j = 0; j < DIM; ++j) {
    //         printf("%3.0f ", x(i, j));
    //     }
    //     printf("\n");
    // }

    topk<dtype, ROW, DIM, TOPK>(weight.data(), indices.data(), x.data());

    // printf("weight out: \n");
    // for (int i = 0; i < ROW; ++i) {
    //     for (int j = 0; j < TOPK; ++j) {
    //         printf("%3.0f ", weight(i, j));
    //     }
    //     printf("\n");
    // }

    // printf("indices out: \n");
    // for (int i = 0; i < ROW; ++i) {
    //     for (int j = 0; j < TOPK; ++j) {
    //         printf("%3.0f ", indices(i, j));
    //     }
    //     printf("\n");
    // }

    //for illustration
    // int src[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    // int dst[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    // for (uint16_t stage = 2; stage <= 16; stage <<= 1) {
    //     for (uint16_t step = stage >> 1; step > 0; step >>= 1) {
    //         for(int j=0;j<16;j++){
    //             uint16_t partner = j ^ step;
    //             if (j < partner) {
    //                 if ((partner & stage) == 0) {
    //                     if (src[j] <
    //                         src[partner]) {
    //                         dst[j] =
    //                             src[partner];
    //                         dst[partner] =
    //                             src[j];
    //                     }
    //                 } else {
    //                     if (src[j] >
    //                         src[partner]) {
    //                         dst[j] =
    //                             src[partner];
    //                         dst[partner] =
    //                             src[j];
    //                     }
    //                 }
    //             }
    //         }

    //         for (int j=0;j<16;j++){
    //             src[j] = dst[j];
    //         }
    //         printf("stage:%d step:%d\n", stage, step);
    //         for (int j=0;j<16;j++) {
    //             printf("%d ", src[j]);
    //         }
    //         printf("\n");
    //     }
    // }

    return 0;
}