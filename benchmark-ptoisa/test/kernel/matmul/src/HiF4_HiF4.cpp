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
  using fp4_t = __fp4_hif4x2;
  constexpr int globKv = globK / 2;
  constexpr int tilKv = tilK / 2;
  static_assert(tilM % 2 == 0); // 暂时假定tile是偶数的，方便取地址，奇数tile实现需要末尾padding 0对齐地址
  static_assert(tilN % 2 == 0);
  static_assert(tilKv % 2 == 0);
  fp4_t src0p[(globM+1)/2*(globK+1)/2*4 + 2*ALIGN]; // 保证是偶数M,K,N,奇数MKN末尾pad0, ALGIN保证地址256B对齐
  fp4_t src1p[(globK+1)/2*(globN+1)/2*4 + 2*ALIGN];
  uint8_t src0_mxp[(globM+1)/2*(globK+1)/2*4/16 + 2*ALIGN];  //scaling matrix  64元素共享 4 Bytes
  uint8_t src1_mxp[(globM+1)/2*(globK+1)/2*4/16 + 2*ALIGN];
  float      dstp[(globM+1)/2*(globN+1)/2*4];
  fp4_t *src0 = (fp4_t *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
  fp4_t *src1 = (fp4_t *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
  uint8_t *src0_mx = (uint8_t *)(((uint64_t)src0_mxp & ALIGN_MASK) + ALIGN);
  uint8_t *src1_mx = (uint8_t *)(((uint64_t)src1_mxp & ALIGN_MASK) + ALIGN);
  float *dst  = (float*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

  #ifdef RES_CHECK
    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(fp4_t));
    readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(fp4_t));
  #endif

  BENCHSTART;
  #ifdef NOMX_NOGATHER
    matmul_fp_notcvt<fp4_t, globM, globN, globKv, tilM, tilN, tilKv, fp4_t, 1, 16>(dst, src0, src1, src0_mx, src1_mx);
  #elif MX_NOGATHER
    matmul_mxfp_notcvt<fp4_t, globM, globN, globKv, tilM, tilN, tilKv, fp4_t, 1, 16>(dst, src0, src1, src0_mx, src1_mx);
  #elif MX_NOGATHER_REUSEA
    matmul_mxfp_notcvt_reuseA<fp4_t, globM, globN, globKv, tilM, tilN, tilKv, fp4_t, 1, 16>(dst, src0, src1, src0_mx, src1_mx);
  #else
    matmul_mxfp<fp4_t, globM, globN, globKv, tilM, tilN, tilKv, fp4_t, 1, 16>(dst, src0, src1, src0_mx, src1_mx);
  #endif
  // matmul_mxfp_notcvt<fp4_t, globM, globN, globKv, tilM, tilN, tilKv, fp4_t, 1, 16>(dst, src0, src1, src0_mx, src1_mx);
  BENCHEND;

  #ifdef RES_CHECK
  #define RES_PATH CHK_DIR "/res.bin"
  writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
  #endif

  return 0;
}