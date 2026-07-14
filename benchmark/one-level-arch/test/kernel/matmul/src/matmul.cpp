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

#include "matmul/matmul.hpp"

int main() {

  #if defined(VEC)
    float src0[globM*globK];
    float src1[globK*globN];
    float dst[globM*globN];

    BENCHSTART;
    matmul_vec<globM, globN, globK, tilM, tilN, tilK>(dst,src0, src1);
    BENCHEND;

  #elif defined(FRAC)
    float src0[globM*globK];
    float src1[globK*globN];
    float dst[globM*globN];

    BENCHSTART;
    matmul_frac<globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    BENCHEND;

  #elif defined(MASK_FP32) || defined(MASK_FP32_REUSEA) || defined(MASK_FP32_REUSEB) || defined(MASK_FP32_REUSEAB) || defined(MASK_FP32_DYNAMIC) || \
        defined(MASK_FP32_DYNAMIC_REUSE)
    // char* p;
    // p = (char *)malloc(globM*globK*sizeof(__fp8_e4m3)+2*ALIGN);
    // src0 = (__fp8_e4m3 *)((uint64_t)p & ALIGN_MASK);
    // p = (char *)malloc(globK*globN*sizeof(__fp8_e4m3)+2*ALIGN);
    // src1 = (__fp8_e4m3 *)((uint64_t)p & ALIGN_MASK);
    // p = (char *)malloc(globM*globN*sizeof(float)+2*ALIGN);
    // dst  = (float*)((uint64_t)p & ALIGN_MASK);
    // printf("src0 is %p src1 is %p dst is %p\n", src0, src1, dst);

    float src0[globM*globK];
    float src1[globK*globN];
    float dst [globM*globN];

    #ifdef RES_CHECK
    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(float));
    readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(float));
    #endif

    BENCHSTART;
    #if defined(MASK_FP32) 
      matmul_mask_tileop<float, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP32_REUSEA)
      matmul_mask_reuseA_tileop<float, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP32_REUSEB)
      matmul_mask_reuseB_tileop<float, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP32_REUSEAB)
      matmul_mask_reuseAB_tileop<float, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP32_DYNAMIC)
      matmul_dynamic_new<float, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #elif defined(MASK_FP32_DYNAMIC_REUSE)
      matmul_dynamic_reuseA<float, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #endif
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
    #endif

  #elif defined(MASK_FP16_REUSEA_OPT) || (MASK_FP16_REUSEA_OPT2)|| (MASK_FP16_REUSEB_OPT2)|| defined(MASK_FP16) || defined(MASK_FP16_REUSEA) || defined(MASK_FP16_REUSEB) || defined(MASK_FP16_REUSEAB) || defined(MASK_FP16_DYNAMIC) || \
        defined(MASK_FP16_DYNAMIC_REUSE)
    using datatype = __half;
    // using datatype = __bf16;
    using output_type = float;
    datatype *src0;
    datatype *src1;
    output_type *dst;
    
    datatype src0p[globM*globK + 2*ALIGN];
    datatype src1p[globK*globN + 2*ALIGN];
    output_type      dstp[globM*globN + 2*ALIGN];

    src0 = (datatype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    src1 = (datatype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    dst  = (output_type*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
      #define SRC0_PATH CHK_DIR "/src0.bin"
      #define SRC1_PATH CHK_DIR "/src1.bin"
      readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(datatype));
      readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(datatype));
    #endif

    BENCHSTART;
    #if defined(MASK_FP16) 
      matmul_mask_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEA)
      matmul_mask_reuseA_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEA_OPT)
      matmul_mask_reuseA_OPT_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEA_OPT2)
      matmul_mask_reuseA_OPT2_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEB_OPT2)
      matmul_mask_reuseB_OPT2_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEB)
      matmul_mask_reuseB_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_REUSEAB)
      matmul_mask_reuseAB_tileop<datatype, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP16_DYNAMIC)
      matmul_dynamic_new<datatype, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #elif defined(MASK_FP16_DYNAMIC_REUSE)
      matmul_dynamic_reuseA<datatype, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #endif
    BENCHEND;

  #elif defined(MASK_FP8) || defined(MASK_FP8_2LVL) || defined(MASK_FP8_MULTI4_B) || defined(MASK_FP8_MULTI4_AB) || defined(MASK_FP8_REUSEA) || \
        defined(MASK_FP8_REUSEB) || defined(MASK_FP8_REUSEAB) || defined(MASK_FP8_DYNAMIC) || defined(MASK_FP8_DYNAMIC_REUSE)
    __fp8_e4m3 *src0;
    __fp8_e4m3 *src1;
    float *dst;
    
    __fp8_e4m3 src0p[globM*globK + 2*ALIGN];
    __fp8_e4m3 src1p[globK*globN + 2*ALIGN];
    float      dstp[globM*globN + 2*ALIGN];

    src0 = (__fp8_e4m3 *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    src1 = (__fp8_e4m3 *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    dst  = (float*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
      #define SRC0_PATH CHK_DIR "/src0.bin"
      #define SRC1_PATH CHK_DIR "/src1.bin"
      readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(__fp8_e4m3));
      readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(__fp8_e4m3));
    #endif

    BENCHSTART;
    #if defined(MASK_FP8) 
      matmul_mask_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_2LVL)
      matmul_mask_2lvl_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_MULTI4_B)
      matmul_mask_multi4_B_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_MULTI4_AB)
      matmul_mask_multi4_AB_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_REUSEA)
      matmul_mask_reuseA_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_REUSEB)
      matmul_mask_reuseB_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_REUSEAB)
      matmul_mask_reuseAB_tileop<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1);
    #elif defined(MASK_FP8_DYNAMIC)
      matmul_dynamic_new<__fp8_e4m3, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #elif defined(MASK_FP8_DYNAMIC_REUSE)
      matmul_dynamic_reuseA<__fp8_e4m3, tilM, tilN, tilK>(dst, src0, src1, globM, globN, globK);
    #endif
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
    #endif

  #elif defined(MX_FP8)
    __fp8_e4m3 *src0;
    __fp8_e4m3 *src1;
    uint8_t *src0_mx;
    uint8_t *src1_mx;
    float *dst;
    
    __fp8_e4m3 src0p[globM*globK + 2*ALIGN];
    __fp8_e4m3 src1p[globK*globN + 2*ALIGN];
    uint8_t src0_mxp[(globM*globK + 2*ALIGN)/32];
    uint8_t src1_mxp[(globK*globN + 2*ALIGN)/32];
    float      dstp[globM*globN + 2*ALIGN];

    src0 = (__fp8_e4m3 *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    src1 = (__fp8_e4m3 *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    src0_mx = (uint8_t *)(((uint64_t)src0_mxp & ALIGN_MASK) + ALIGN);
    src1_mx = (uint8_t *)(((uint64_t)src1_mxp & ALIGN_MASK) + ALIGN);
    dst  = (float*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
      #define SRC0_PATH CHK_DIR "/src0.bin"
      #define SRC1_PATH CHK_DIR "/src1.bin"
      readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(__fp8_e4m3));
      readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(__fp8_e4m3));
    #endif

    BENCHSTART;
    matmul_mx<__fp8_e4m3, globM, globN, globK, tilM, tilN, tilK>(dst, src0, src1, src0_mx, src1_mx);
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
    #endif

  #else
    #error "No Available Macro!";
  #endif

  return 0;
}
