#include <common/pto_tileop.hpp>
// #include "../../include/accelerator_fusion.h"
#include "benchmark.h"
#include "fileop.h"
#include "fa_mx/fa_hif4.hpp"

#define B 1
#define H 1    //2

#ifndef Tsq
#define Sq 512 //1024
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 512 //1024
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

#ifndef MBUF
#define MBUF 2

// #define HIF4_BF16x2
// #define HIF4
// #define HIF4_NOGATHER
#define STR(x) #x
#endif

int main(){
    // using typep = __half;
    using typep = __fp4_e1m2x2;
    typep qp[B*H*Sq*qD + 2*ALIGN];
    typep kp[B*H*Skv*qD + 2*ALIGN];
    typep vp[B*H*Skv*vD + 2*ALIGN];
    typep outp[B*H*Sq*vD + 2*ALIGN];
    uint8_t qmx[B*H*Sq*qD + 2*ALIGN];
    uint8_t kmx[B*H*Sq*qD + 2*ALIGN];
    uint8_t vmx[B*H*Sq*vD + 2*ALIGN];

    typep* q = (typep *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    typep* k = (typep *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    typep* v = (typep *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    typep* out = (typep *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
    #define SRCQ_PATH CHK_DIR "/srcq.bin"
    #define SRCK_PATH CHK_DIR "/srck.bin"
    #define SRCV_PATH CHK_DIR "/srcv.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, B*H*S*qD*sizeof(__half));
    readBinaryFile(SRCK_PATH, (uint8_t*)k, B*H*S*qD*sizeof(__half));
    readBinaryFile(SRCV_PATH, (uint8_t*)v, B*H*S*vD*sizeof(__half));
    #endif

    // uint32_t a;
    // uint32_t *b = &a;
    // float *c = (float*)b;
    // c = c + 1;
    // printf("%f\n", *c);

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            #ifdef BF16
            // Sq256_Skv8192_
                static_assert(0);
                // flash_attention_2d_unroll_hif4<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(BF16x2)
                static_assert(0);
                // flash_attention_2d_unroll_hif4<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(BF16x2_NOGATHER)
                static_assert(0);
                // flash_attention_2d_unroll_hif4_nogather<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(BF16_NOGATHER)
                static_assert(0);
                // flash_attention_2d_unroll_hif4_nogather<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(OPT)
                // support X1 Y4
                flash_attention_2d_unroll_hif4_optsoftmax<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(OPT_LOAD)
                // opt lhi -> lwi
                flash_attention_2d_unroll_hif4_optsoftmax_loadx2<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(OPT_OFFLOAD)
                // support X1 Y4
                flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            #elif defined(OPT_OFFLOAD2)
                // support X1 Y4
                flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload2<__fp4_e1m2x2, Sq, Skv, qD, vD, kTm, kTk, 16, __bf16x2>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, qmx, kmx, vmx);
            
            #endif
        }
    }
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B*H*S*vD*sizeof(__half));
    #endif
}