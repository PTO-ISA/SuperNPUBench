#include "benchmark.h"
#include "single_thread/conv2d/v300_conv2d.hpp"

#ifndef CONV_IN_C
#define CONV_IN_C 16
#endif

#ifndef CONV_IN_H
#define CONV_IN_H 4
#endif

#ifndef CONV_IN_W
#define CONV_IN_W 4
#endif

#ifndef CONV_OUT_C
#define CONV_OUT_C 16
#endif

#ifndef CONV_TILE_M
#define CONV_TILE_M 16
#endif

#ifndef CONV_TILE_N
#define CONV_TILE_N 16
#endif

#ifndef CONV_TILE_K
#define CONV_TILE_K 16
#endif

namespace {

constexpr int kInputElements = CONV_IN_C * CONV_IN_H * CONV_IN_W;
constexpr int kWeightElements = CONV_OUT_C * CONV_IN_C;
constexpr int kOutputElements = CONV_OUT_C * CONV_IN_H * CONV_IN_W;

alignas(4096) float input[kInputElements];
alignas(4096) float weight[kWeightElements];
alignas(4096) float output[kOutputElements];

}  // namespace

int main() {
#ifdef CONV_PATTERN_VERIFY
    constexpr int kSpatialElements = CONV_IN_H * CONV_IN_W;
    static_assert(CONV_IN_C == CONV_OUT_C,
                  "pattern verification requires an identity weight matrix");
    for (int channel = 0; channel < CONV_IN_C; ++channel) {
        for (int spatial = 0; spatial < kSpatialElements; ++spatial) {
            input[channel * kSpatialElements + spatial] =
                static_cast<float>(channel * 100 + spatial);
        }
    }
    for (int out_channel = 0; out_channel < CONV_OUT_C; ++out_channel) {
        for (int in_channel = 0; in_channel < CONV_IN_C; ++in_channel) {
            weight[out_channel * CONV_IN_C + in_channel] =
                out_channel == in_channel ? 1.0f : 0.0f;
        }
    }
#else
    for (int i = 0; i < kInputElements; ++i) {
        input[i] = 1.0f;
    }
    for (int i = 0; i < kWeightElements; ++i) {
        weight[i] = 1.0f;
    }
#endif

    BENCHSTART;
    supernpu::conv2d::conv2d_1x1<
        float,
        CONV_IN_C, CONV_IN_H, CONV_IN_W, CONV_OUT_C,
        CONV_TILE_M, CONV_TILE_N, CONV_TILE_K>(output, input, weight);
    BENCHEND;

    return 0;
}
