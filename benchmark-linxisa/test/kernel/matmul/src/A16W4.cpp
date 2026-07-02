#include <common/pto_tileop.hpp>
#include <cstring>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef globM 
#define globM 120
#endif

#ifndef globN
#define globN 120
#endif

#ifndef globK
#define globK 120
#endif

#ifndef tilM
#define tilM  16
#endif

#ifndef tilN
#define tilN  16
#endif

#ifndef tilK
#define tilK  16
#endif

#ifndef Batch
#define Batch  1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

#include "matmul/matmul_mx.hpp"


int main() {
    // bf16*fp4
    using fp4_t = __fp4_e2m1x2;
    using bf16_t = __bf16;
    static_assert(tilM % 2 == 0); // 暂时假定tile是偶数的，方便取地址，奇数tile实现需要末尾padding 0对齐地址
    static_assert(tilN % 2 == 0);
    static_assert(tilK == (128));
    bf16_t src0[(globM+1)/2*(globK+1)/2*4]; // 保证是偶数M,K,N,奇数MKN末尾pad0
    fp4_t src1[(globK+1)/2*(globN+1)/2*4];
    float src2[(globK+1)/2*(globN+1)/2*4];
    float dst[(globM+1)/2*(globN+1)/2*4];

    #ifdef RES_CHECK
      #define SRC0_PATH CHK_DIR "/src0.bin"
      #define SRC1_PATH CHK_DIR "/src1.bin"
      readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(bf16_t));
      readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(fp4_t));
      readBinaryFile(SRC1_PATH, (uint8_t*)src2, globK*globN*sizeof(float));
    #endif

    BENCHSTART;
    matmul_mp<bf16_t, fp4_t, globM, globN, globK, tilM, tilN, 128, 2>(dst, src0, src1, src2); 
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
    #endif

  return 0;
}