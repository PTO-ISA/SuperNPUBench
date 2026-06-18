#include <common/pto_tileop.hpp>
#include <cstring>
#include "linear.hpp"
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef DIN1 
#define DIN1 120
#endif

#ifndef DIN2
#define DIN2 120
#endif

#ifndef DOUT
#define DOUT 120
#endif

#ifndef hasBias
#define hasBias true
#endif

int main() {
  if(!strcmp(MODE, "bis")){
    float weight[DOUT*DIN1*DIN2];
    float in1[DIN1];
    float in2[DIN2];
    float bias[DOUT];
    float out[DOUT];

    #ifdef RES_CHECK
    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    #define SRC2_PATH CHK_DIR "/src2.bin"
    #define SRC3_PATH CHK_DIR "/src3.bin"
    readBinaryFile(SRC0_PATH, (uint8_t*)weight, DOUT*DIN1*DIN2*sizeof(float));
    readBinaryFile(SRC1_PATH, (uint8_t*)in1,  DIN1*sizeof(float));
    readBinaryFile(SRC2_PATH, (uint8_t*)in2,  DIN2*sizeof(float));
    readBinaryFile(SRC3_PATH, (uint8_t*)bias, DOUT*sizeof(float));
    #endif

    BENCHSTART;
    BiLinear<float, DIN1, DIN2, DOUT, hasBias>(out, weight, in1, in2, bias);
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, DOUT*sizeof(float));
    #endif
  }

  return 0;
}