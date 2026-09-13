#include <common/pto_tileop.hpp>
#include <cstring>
#include <cstdint>
#include "common.h"
#include "benchmark.h"

#ifndef IN_H
#define IN_H 16
#endif

#ifndef IN_W
#define IN_W 16
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

#ifdef CONV_FP16
using datatype = __half;
#else
using datatype = float;
#endif
using output_type = float;

#ifdef RES_CHECK
#include "fileop.h"
#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024
#endif

#ifdef EMBED_DATA
#include "conv2d_embed_data.h"
#endif

int main() {
#if defined(EMBED_DATA)
    datatype* input_nchw = (datatype*)embed_input;
    datatype* weight = (datatype*)embed_weight;
    output_type output[OUT_C * IN_H * IN_W];
#elif defined(RES_CHECK)
    datatype input_buf[IN_C * IN_H * IN_W + 2 * ALIGN];
    datatype weight_buf[OUT_C * IN_C + 2 * ALIGN];
    output_type output_buf[OUT_C * IN_H * IN_W + 2 * ALIGN];

    datatype* input_nchw = (datatype*)(((uint64_t)input_buf & ALIGN_MASK) + ALIGN);
    datatype* weight = (datatype*)(((uint64_t)weight_buf & ALIGN_MASK) + ALIGN);
    output_type* output = (output_type*)(((uint64_t)output_buf & ALIGN_MASK) + ALIGN);

    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)input_nchw, IN_C * IN_H * IN_W * sizeof(datatype));
    readBinaryFile(SRC1_PATH, (uint8_t*)weight, OUT_C * IN_C * sizeof(datatype));
#else
    datatype input_nchw[IN_C * IN_H * IN_W];
    datatype weight[OUT_C * IN_C];
    output_type output[OUT_C * IN_H * IN_W];

#ifdef CONV_FP16
    volatile uint32_t* vi = (volatile uint32_t*)input_nchw;
    for (int i = 0; i < IN_C * IN_H * IN_W / 2; ++i)
        vi[i] = 0x3C003C00u;
    volatile uint32_t* vw = (volatile uint32_t*)weight;
    for (int i = 0; i < OUT_C * IN_C / 2; ++i)
        vw[i] = 0x3C003C00u;
#else
    volatile uint32_t* vi = (volatile uint32_t*)input_nchw;
    for (int i = 0; i < IN_C * IN_H * IN_W; ++i)
        vi[i] = 0x40000000u;
    volatile uint32_t* vw = (volatile uint32_t*)weight;
    for (int i = 0; i < OUT_C * IN_C; ++i)
        vw[i] = 0x3F800000u;
#endif
#endif

    conv2d_1x1_tileop<datatype, IN_C, IN_H, IN_W, OUT_C,
                      tilM, tilN, tilK>(output, input_nchw, weight);

#ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)output, OUT_C * IN_H * IN_W * sizeof(output_type));
#endif

    return 0;
}
