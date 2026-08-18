#include <common/pto_tileop.hpp>

#include "benchmark.h"
#include "fileop.h"
#include "norm/rms_norm_pto.hpp"

#include <cstdint>

#ifndef RMS_M
#define RMS_M 16
#endif

#ifndef RMS_N
#define RMS_N 256
#endif

#ifndef RMS_TM
#define RMS_TM 8
#endif

#ifndef RMS_TN
#define RMS_TN 128
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    float input_buffer[RMS_M * RMS_N + 2 * ALIGN];
    float weight_buffer[RMS_N + 2 * ALIGN];
    float output_buffer[RMS_M * RMS_N + 2 * ALIGN];

    float *input = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(input_buffer) & ALIGN_MASK) + ALIGN);
    float *weight = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(weight_buffer) & ALIGN_MASK) + ALIGN);
    float *output = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(output_buffer) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define INPUT_PATH CHK_DIR "/input.bin"
#define WEIGHT_PATH CHK_DIR "/weight.bin"
#define OUTPUT_PATH CHK_DIR "/output.bin"
    readBinaryFile(
        INPUT_PATH, reinterpret_cast<uint8_t *>(input),
        RMS_M * RMS_N * sizeof(float));
    readBinaryFile(
        WEIGHT_PATH, reinterpret_cast<uint8_t *>(weight),
        RMS_N * sizeof(float));
#else
    for (int i = 0; i < RMS_M * RMS_N; ++i) {
        input[i] = static_cast<float>((i % 17) - 8) * 0.125f;
    }
    for (int i = 0; i < RMS_N; ++i) {
        weight[i] = 1.0f + static_cast<float>(i % 5) * 0.0625f;
    }
#endif

    BENCHSTART;
    supernpu::norm::rms_norm<RMS_M, RMS_N, RMS_TM, RMS_TN>(
        output, input, weight);
    BENCHEND;

#ifdef RES_CHECK
    writeBinaryFile(
        OUTPUT_PATH, reinterpret_cast<uint8_t *>(output),
        RMS_M * RMS_N * sizeof(float));
#endif

    return 0;
}
