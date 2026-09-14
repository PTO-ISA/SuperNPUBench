#include <common/pto_tileop.hpp>
#include <cstring>
#include <cstdint>
#include "common.h"
#include "benchmark.h"

#ifndef IN_H
#define IN_H 4
#endif

#ifndef IN_W
#define IN_W 4
#endif

#ifndef IN_C
#define IN_C 16
#endif

#ifndef OUT_C
#define OUT_C 16
#endif

#ifndef tilM
#define tilM 16
#endif

#ifndef tilN
#define tilN 16
#endif

#ifndef tilK
#define tilK 16
#endif

#include "conv2d/conv2d.hpp"

int main() {
    float input_nchw[IN_C * IN_H * IN_W];
    float weight[OUT_C * IN_C];
    float output[OUT_C * IN_H * IN_W];

    for (int i = 0; i < IN_C * IN_H * IN_W; ++i)
        input_nchw[i] = (float)(i + 1);
    for (int i = 0; i < OUT_C * IN_C; ++i)
        weight[i] = (float)(i + 1);

    conv2d_1x1_tileop<float, IN_C, IN_H, IN_W, OUT_C,
                      tilM, tilN, tilK>(output, input_nchw, weight);

    return 0;
}
