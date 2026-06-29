#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "fa/fa_2d_unroll.hpp"

#define B 1
#define H 1

#ifndef Tsq
#define Sq 512
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 512
#else
#define Skv Tskv
#endif

#define qD 128
#define vD 128

#ifndef Tm
#define kTm 128
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 128
#else
#define kTk Tk
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    using dtype = __half;
    
    dtype qp[B*H*Sq*qD + 2*ALIGN];
    dtype kp[B*H*Skv*qD + 2*ALIGN];
    dtype vp[B*H*Skv*vD + 2*ALIGN];
    dtype outp[B*H*Sq*vD + 2*ALIGN];

    dtype* q = (dtype *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    dtype* k = (dtype *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    dtype* v = (dtype *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    dtype* out = (dtype *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
    #define SRCQ_PATH CHK_DIR "/srcq.bin"
    #define SRCK_PATH CHK_DIR "/srck.bin"
    #define SRCV_PATH CHK_DIR "/srcv.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, B*H*Sq*qD*sizeof(dtype));
    readBinaryFile(SRCK_PATH, (uint8_t*)k, B*H*Skv*qD*sizeof(dtype));
    readBinaryFile(SRCV_PATH, (uint8_t*)v, B*H*Skv*vD*sizeof(dtype));
    #endif

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            flash_attention_2d_unroll<dtype, Sq, Skv, qD, vD, kTm, kTk>(
                out + i*H*Sq*vD + j*Sq*vD,
                q + i*H*Sq*qD + j*Sq*qD,
                k + i*H*Skv*qD + j*Skv*qD,
                v + i*H*Skv*vD + j*Skv*vD
            );
        }
    }
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B*H*Sq*vD*sizeof(dtype));
    #endif

    return 0;
}
