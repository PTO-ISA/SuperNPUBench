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

#include "single_thread/matmul/matmul_mx.hpp"


int main() {
  using fp4_t = __fp4_hif4x2;
  static_assert(globK % 64 == 0 && tilK % 64 == 0);
  static_assert((globM * globK) % 2 == 0);
  static_assert((globK * globN) % 2 == 0);

  alignas(4096) static fp4_t src0[globM * globK / 2];
  alignas(4096) static fp4_t src1[globK * globN / 2];
  alignas(4096) static uint32_t src0_mx[globM * globK / 64];
  alignas(4096) static uint32_t src1_mx[globK / 64 * globN];
  alignas(4096) static float dst[globM * globN];

  #ifdef RES_CHECK
    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK/2);
    readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN/2);
  #endif

  BENCHSTART;
  #if defined(MX_NOGATHER)
    matmul_hif4x2_mx<globM, globN, globK, tilM, tilN, tilK, false>(
        dst, src0, src1, src0_mx, src1_mx);
  #elif MX_NOGATHER_REUSEA
    matmul_hif4x2_mx<globM, globN, globK, tilM, tilN, tilK, true>(
        dst, src0, src1, src0_mx, src1_mx);
  #else
    #error "HiF4x2 requires Matrix-MX; select MX_NOGATHER or MX_NOGATHER_REUSEA"
  #endif
  BENCHEND;

  #ifdef RES_CHECK
  #define RES_PATH CHK_DIR "/res.bin"
  writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
  #endif

  return 0;
}
