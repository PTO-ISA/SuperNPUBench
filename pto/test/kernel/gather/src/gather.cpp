#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "gather/gather.hpp"


#ifndef DType
#define DType float
#endif

#ifndef OType
#define OType int32_t
#endif



#ifndef gKs
#define gKs 754
#endif

#ifndef gMs
#define gMs 17
#endif

#ifndef gNs
#define gNs 4096
#endif

#ifndef tMs
#define tMs 32
#endif

#ifndef tNs
#define tNs 32
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
    using otype = OType;
    // ==========================
    // 申明空间，留出静态偏移空间
    // ==========================
    
    // printf("gKs = %d\n",gKs);
    // printf("gMs = %d\n",gMs);
    // printf("gNs = %d\n",gNs);
    // printf("gMs * sizeof(otype) = %d\n",gMs * sizeof(otype));
    // printf("gKs * gNs * sizeof(dtype) = %d\n",gKs * gNs * sizeof(dtype));
    
    dtype  input_buf[gKs * gNs + OFFSET_INs];   // 前面留空，模拟随机地址
    otype  input_offset_buf[gMs + OFFSET_INs];
    dtype  output_buf[gNs * gMs + OFFSET_OUTs];

    // ==========================
    // 【静态随机地址】
    // ==========================
    dtype* input  = input_buf  + OFFSET_INs;
    otype* input_offset  = input_offset_buf  + OFFSET_INs;
    dtype* output = output_buf + OFFSET_OUTs;

    // ==========================
    // 从 bin 文件读取输入数据
    // ==========================
    // printf("---cpp: start to read input data---");
    // printf("---cpp: start to read input data---");

    #ifdef RES_CHECK
    #define INPUT_PATH CHK_DIR "/input.bin"
    #define INPUT_OFFSET_PATH CHK_DIR "/input_offset.bin"
    #define OUTPUT_PATH CHK_DIR "/output.bin"
    readBinaryFile(INPUT_PATH, (uint8_t*)input, gKs * gNs * sizeof(dtype));
    readBinaryFile(INPUT_OFFSET_PATH, (uint8_t*)input_offset, gMs * sizeof(otype));

    // printf("input_offset[200] = %d\n",input_offset[200]);
    // printf("input_offset[1] = %d\n",input_offset[1]);
    // printf("input_offset[2] = %d\n",input_offset[2]);
    // printf("input_offset[3] = %d\n",input_offset[3]);

    // printf("input[200] = %f\n",input[200]);
    // printf("input[1] = %f\n",input[1]);
    // printf("input[2] = %f\n",input[2]);
    // printf("input[3] = %f\n",input[3]);
    printf("input[351*17] = %f\n",input[351*17]);
    #endif
    
    gather<dtype, otype, gKs, gMs, gNs, tMs, tNs>(input, input_offset, output);

    #ifdef RES_CHECK
    writeBinaryFile(OUTPUT_PATH, (uint8_t*)output, gMs * gNs * sizeof(dtype));
    #endif

}