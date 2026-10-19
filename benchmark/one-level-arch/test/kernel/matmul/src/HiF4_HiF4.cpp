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

#include "basic_op/matmul/matmul_mx.hpp"


#if defined(MATMUL_INPUT_DATA) && defined(RES_CHECK)
#error "select one input source"
#endif
#ifdef MATMUL_INPUT_DATA
#include "matmul_data.inc"
#endif

int main() {
  using fp4_t = __fp4_hif4x2;
  static_assert(globK % 64 == 0 && tilK % 64 == 0);
  static_assert((globM * globK) % 2 == 0);
  static_assert((globK * globN) % 2 == 0);

#ifdef MATMUL_INPUT_DATA
  auto *src0 = reinterpret_cast<fp4_t *>(matmul_src0);
  auto *src1 = reinterpret_cast<fp4_t *>(matmul_src1);
  auto *src0_mx = reinterpret_cast<uint32_t *>(matmul_src0_mx);
  auto *src1_mx = reinterpret_cast<uint32_t *>(matmul_src1_mx);
  auto *dst = reinterpret_cast<float *>(matmul_dst);
  static_assert(sizeof(matmul_src0) == globM * globK / 2);
  static_assert(sizeof(matmul_src1) == globK * globN / 2);
  static_assert(sizeof(matmul_src0_mx) == globM * globK / 64 * 4);
  static_assert(sizeof(matmul_src1_mx) == globK / 64 * globN * 4);
  static_assert(sizeof(matmul_dst) == globM * globN * 4);
#else
  alignas(4096) static fp4_t src0[globM * globK / 2];
  alignas(4096) static fp4_t src1[globK * globN / 2];
  alignas(4096) static uint32_t src0_mx[globM * globK / 64];
  alignas(4096) static uint32_t src1_mx[globK / 64 * globN];
  alignas(4096) static float dst[globM * globN];
#endif

  #ifdef RES_CHECK
    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    #define SRC0_MX_PATH CHK_DIR "/src0_mx.bin"
    #define SRC1_MX_PATH CHK_DIR "/src1_mx.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK/2);
    readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN/2);
    readBinaryFile(SRC0_MX_PATH, (uint8_t*)src0_mx, sizeof(src0_mx));
    readBinaryFile(SRC1_MX_PATH, (uint8_t*)src1_mx, sizeof(src1_mx));
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

#ifdef MATMUL_INPUT_DATA
  matmul_status = 0x600d600du;
#endif
  return 0;
}
