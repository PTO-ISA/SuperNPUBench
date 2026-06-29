#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "broadcast/broadcast_vec_019.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 16
#endif

#ifndef kInners
#define kInners 49
#endif

#ifndef MAX_DIMs
#define MAX_DIMs 3
#endif

#ifndef IN_SHAPEs
#define IN_SHAPEs 1280,1,49
#endif

#ifndef OUT_SHAPEs
#define OUT_SHAPEs 1280,8,49
#endif

#ifndef IN_DIMs
#define IN_DIMs 3
#endif

#ifndef OUT_DIMs
#define OUT_DIMs 3
#endif

#ifndef gIMs
#define gIMs 1280*1*49
#endif

#ifndef gOMs
#define gOMs 1280*8*49
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

    broadcast<dtype, MAX_DIMs, IN_DIMs, OUT_DIMs, gIMs, gOMs, tMs, kInners>(
        input, output, in_shape, out_shape);

    #ifdef RES_CHECK
    #define INPUT_PATH CHK_DIR "/input.bin"
    #define OUTPUT_PATH CHK_DIR "/output.bin"
    readBinaryFile(INPUT_PATH, (uint8_t*)input, gIMs * sizeof(dtype));
    writeBinaryFile(OUTPUT_PATH, (uint8_t*)output, gOMs * sizeof(dtype));
    #endif
}