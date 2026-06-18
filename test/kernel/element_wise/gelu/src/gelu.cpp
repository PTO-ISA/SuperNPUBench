#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "element_wise/gelu.hpp"


#ifndef DTYPE
#define DTYPE int32_t
#endif

#ifndef tMs
#define tMs 512
#endif

#ifndef gMs
#define gMs (24 * 512 * 1024)
#endif

#ifndef Approximate
#define Approximate false
#endif

#ifndef DTYPE
#define DTYPE true
#endif
// 【静态随机偏移】手动写死，模拟内存地址不是数组起始位置
#define OFFSET_IN  11   // 输入静态偏移
#define OFFSET_OUT 17   // 输出静态偏移
// ============================================================================
// main
// ============================================================================
int main() {
    
    // printf("start !!!!!!!!!!!!!!!!!");
    using dtype = DTYPE;
    // ==========================
    // 申明空间，留出静态偏移空间
    // ==========================
    dtype  input_buf[gMs  + OFFSET_IN];   // 前面留空，模拟随机地址
    dtype  output_buf[gMs + OFFSET_OUT];

    // ==========================
    // 【静态随机地址】
    // ==========================
    dtype* input  = input_buf  + OFFSET_IN;
    dtype* output = output_buf + OFFSET_OUT;

    // ==========================
    // 从 bin 文件读取输入数据
    // ==========================

    // #ifdef RES_CHECK
    //     // #ifndef CHK_DIR
    //     // #define CHK_DIR "/remote/lms01/l00948608/project/jcore_benchmark/JanusCoreBench/test/gelu/src/data"
    //     // #endif
    // #define INPUT_PATH CHK_DIR "/input.bin"
    // #define OUTPUT_PATH CHK_DIR "/output.bin"
    // printf("input path \n");
    // printf(INPUT_PATH);
    // printf("\n");
    // // printf(OUTPUT_PATH);
    // readBinaryFile(INPUT_PATH, (uint8_t*)input, gMs * sizeof(dtype));
    // // printf("input %.4f \n", input[0]);
    // // printf("input %.4f \n", input[1]);
    // // printf("input %.4f \n", input[2]);
    // #endif
    
    gelu<dtype, gMs, tMs>(input, output, Approximate);
    // printf("output %.4f \n", output[0]);
    // printf("output %.4f \n", output[1]);
    // printf("output %.4f \n", output[2]);

    // #ifdef RES_CHECK
    // writeBinaryFile(OUTPUT_PATH, (uint8_t*)output, gMs * sizeof(dtype));
    // #endif
}