#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "broadcast/broadcast_vec_039.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 16
#endif

#ifndef kInners
#define kInners 16
#endif

#ifndef MAX_DIMs
#define MAX_DIMs 3
#endif

#ifndef IN_SHAPEs
#define IN_SHAPEs 8192,1,16
#endif

#ifndef OUT_SHAPEs
#define OUT_SHAPEs 8192,8,16
#endif

#ifndef IN_DIMs
#define IN_DIMs 3
#endif

#ifndef OUT_DIMs
#define OUT_DIMs 3
#endif

#ifndef gIMs
#define gIMs 8192*1*16
#endif

#ifndef gOMs
#define gOMs 8192*8*16
#endif

#ifndef OFFSET_INs
#define OFFSET_INs  11
#endif
#ifndef OFFSET_OUTs
#define OFFSET_OUTs 17
#endif

int main() {
    using dtype = DType;
    size_t in_shape[MAX_DIMs]  = {IN_SHAPEs};
    size_t out_shape[MAX_DIMs] = {OUT_SHAPEs};

    dtype  input_buf[gIMs  + OFFSET_INs];
    dtype  output_buf[gOMs + OFFSET_OUTs];

    dtype* input  = input_buf  + OFFSET_INs;
    dtype* output = output_buf + OFFSET_OUTs;

    // for (size_t i = 0; i < (size_t)gIMs; i++) {
    //     if constexpr (sizeof(dtype) == 2) {
    //         ((uint16_t*)input)[i] = (uint16_t)(i % 1000);
    //     } else {
    //         input[i] = (dtype)(i % 1000);
    //     }
    // }

    broadcast<dtype, MAX_DIMs, IN_DIMs, OUT_DIMs, gIMs, gOMs, tMs, kInners>(
        input, output, in_shape, out_shape);

    // size_t pass_count = 0;
    // for (size_t b = 0; b < in_shape[0]; b++) {
    //     for (size_t n = 0; n < out_shape[1]; n++) {
    //         for (size_t k = 0; k < in_shape[2]; k++) {
    //             size_t in_idx  = b * in_shape[2] + k;
    //             size_t out_idx = b * out_shape[1] * out_shape[2] + n * out_shape[2] + k;

    //             dtype expected;
    //             if constexpr (sizeof(dtype) == 2) {
    //                 expected = (dtype)(uint16_t)(in_idx % 1000);
    //             } else {
    //                 expected = (dtype)(in_idx % 1000);
    //             }

    //             dtype actual = output[out_idx];
    //             if (actual != expected) {
    //                 if (pass_count < 10) {
    //                     printf("FAIL: [%zu][%zu][%zu] expected=%d, got=%d\n",
    //                            b, n, k, (int)expected, (int)actual);
    //                 }
    //                 pass_count++;
    //             }
    //         }
    //     }
    // }
    // if (pass_count == 0) {
    //     printf("PASS: all %zu elements correct\n", (size_t)gOMs);
    // } else {
    //     printf("FAIL: %zu mismatches\n", pass_count);
    // }

    #ifdef RES_CHECK
    #define INPUT_PATH CHK_DIR "/input.bin"
    #define OUTPUT_PATH CHK_DIR "/output.bin"
    readBinaryFile(INPUT_PATH, (uint8_t*)input, gIMs * sizeof(dtype));
    writeBinaryFile(OUTPUT_PATH, (uint8_t*)output, gOMs * sizeof(dtype));
    #endif

    // return (pass_count == 0) ? 0 : 1;
}