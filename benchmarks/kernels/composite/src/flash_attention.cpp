#include <common/pto_tileop.hpp>
#include <cstring>
#include "flash_attention.hpp"
#include "fileop.h"
#include "benchmark.h"

#ifndef BatSz
#define BatSz 1
#endif

#ifndef globS
#define globS 320
#endif

#ifndef globD
#define globD 128
#endif

#ifndef tilM
#define tilM 64
#endif

#ifndef tilK
#define tilK 32
#endif

int main() {

  float q[BatSz*globS*globD];
  float k[BatSz*globS*globD];
  float v[BatSz*globS*globD];
  float out[BatSz*globS*globD];

  #ifdef RES_CHECK
  #define SRCQ_PATH CHK_DIR "/srcq.bin"
  #define SRCK_PATH CHK_DIR "/srck.bin"
  #define SRCV_PATH CHK_DIR "/srcv.bin"
  readBinaryFile(SRCQ_PATH, (uint8_t*)q, BatSz*globS*globD*sizeof(float));
  readBinaryFile(SRCK_PATH, (uint8_t*)k, BatSz*globS*globD*sizeof(float));
  readBinaryFile(SRCV_PATH, (uint8_t*)v, BatSz*globS*globD*sizeof(float));
  #endif

  #ifdef __linx
  BENCHSTART;
  #endif

  if(!strcmp(MODE, "RM")){
    flash_attention_rm<BatSz*globS, globD, globD, tilM, tilK>(out, q, k, v);
  }
  else if(!strcmp(MODE, "ALL")){
    flash_attention<BatSz*globS, globD, globD, tilM, tilK>(out, q, k, v);
  }
  else if(!strcmp(MODE, "ALL_OPT")){
    flash_attention_opt<BatSz*globS, globD, globD, tilM, tilK>(out, q, k, v);
  }
  else if(!strcmp(MODE, "ALL_OPT2")){
    flash_attention_opt2<BatSz*globS, globD, globD, tilM, tilK>(out, q, k, v);
  }
  else if(!strcmp(MODE, "DYNAMIC")){
    flash_attention_dynamic<float, globD, globD, tilM, tilK>(out, q, k, v, BatSz*globS, BatSz*globS);
  }
  // else if(!strcmp(MODE, "FRAC")){
  //   flash_attention_frac<BatSz*globS, globD, globD, tilM, tilK>(out, q, k, v);
  // }

  #ifdef __linx
  BENCHEND;
  #endif
  
  #ifdef RES_CHECK
  #define RES_PATH CHK_DIR "/res.bin"
  writeBinaryFile(RES_PATH, (uint8_t*)out, BatSz*globS*globD*sizeof(float));
  #endif

  return 0;
}