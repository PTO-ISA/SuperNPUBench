#include <common/pto_tileop.hpp>
#include <cstring>
#include "gemm.hpp"
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef globM 
#define globM 128
#endif

#ifndef globN
#define globN 128
#endif

#ifndef globK
#define globK 128
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

#ifndef hasBias
#define hasBias true
#endif

#ifndef withRelu
#define withRelu true
#endif

int main() {

  float src0[globM*globK];
  float src1[globK*globN];
  float src2[globN];
  float dst[globM*globN];

  #ifdef RES_CHECK
  #define SRC0_PATH CHK_DIR "/src0.bin"
  #define SRC1_PATH CHK_DIR "/src1.bin"
  #define SRC2_PATH CHK_DIR "/src2.bin"
  readBinaryFile(SRC0_PATH, (uint8_t*)src0, globM*globK*sizeof(float));
  readBinaryFile(SRC1_PATH, (uint8_t*)src1, globK*globN*sizeof(float));
  readBinaryFile(SRC2_PATH, (uint8_t*)src2, globN*sizeof(float));
  #endif

  BENCHSTART;
  gemm<float, globM, globN, globK, tilM, tilN, tilK, withRelu, hasBias>(dst, src0, src1, src2);
  BENCHEND;

  #ifdef RES_CHECK
  #define RES_PATH CHK_DIR "/res.bin"
  writeBinaryFile(RES_PATH, (uint8_t*)dst, globM*globN*sizeof(float));
  #endif

  return 0;
}