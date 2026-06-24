#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "regular_memory_aaccelss/broadcast_tst.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 512
#endif

#ifndef MAX_DIMs
#define MAX_DIMs 3
#endif

#ifndef IN_SHAPEs
#define IN_SHAPEs 64,1,49
#endif

#ifndef OUT_SHAPEs
#define OUT_SHAPEs 64,8,49
#endif

#ifndef IN_DIMs
#define IN_DIMs 3
#endif

#ifndef OUT_DIMs
#define OUT_DIMs 3
#endif

#ifndef gIMs
#define gIMs 64*1*49    //4 014 080
#endif

#ifndef gOMs
#define gOMs 64*8*49    //8,028,160
#endif


// 【静态随机偏移】手动写死，模拟内存地址不是数组起始位置
#ifndef OFFSET_INs
#define OFFSET_INs  11   // 输入静态偏移
#endif
#ifndef OFFSET_OUTs
#define OFFSET_OUTs 17   // 输出静态偏移
#endif
// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;
    size_t in_shape[MAX_DIMs] = {IN_SHAPEs};
    size_t out_shape[MAX_DIMs] = {OUT_SHAPEs};

    // ==========================
    // 申明空间，留出静态偏移空间
    // ==========================
    dtype  input_buf[gIMs  + OFFSET_INs];   // 前面留空，模拟随机地址
    dtype  output_buf[gOMs + OFFSET_OUTs];

    // ==========================
    // 【静态随机地址】
    // ==========================
    dtype* input  = input_buf  + OFFSET_INs;
    dtype* output = output_buf + OFFSET_OUTs;
    dtype  golden[gOMs];
    dtype  offset[gOMs];

    // ==========================
    // 从 bin 文件读取输入数据
    // ==========================
    // printf("---cpp: start to read input data---");
    // printf("---cpp: start to read input data---");

    #ifdef RES_CHECK
    #define INPUT_PATH CHK_DIR "/input.bin"
    #define GOLDEN_PATH CHK_DIR "/output.bin"
    #define OUTPUT_PATH CHK_DIR "/output.bin"
    #define OFFSET_PATH CHK_DIR "/offset.bin"
    // #define INPUT_PATH "./input.bin"
    // #define OUTPUT_PATH "./output.bin"
    readBinaryFile(INPUT_PATH, (uint8_t*)input, gIMs * sizeof(dtype));
    readBinaryFile(GOLDEN_PATH, (uint8_t*)golden, gOMs * sizeof(dtype));

    #endif
    
    broadcast_tst<dtype, MAX_DIMs, IN_DIMs, OUT_DIMs, gIMs, gOMs, tMs>(input, output, golden, offset, in_shape, out_shape);

    #ifdef RES_CHECK
    writeBinaryFile(OFFSET_PATH, (uint8_t*)offset, gOMs * sizeof(dtype));
    writeBinaryFile(OUTPUT_PATH, (uint8_t*)output, gOMs * sizeof(dtype));
    #endif

}