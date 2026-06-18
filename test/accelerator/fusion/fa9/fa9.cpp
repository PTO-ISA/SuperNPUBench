#include <common/pto_tileop.hpp>
#include "../../include/accelerator_fusion.h"
#include "benchmark.h"

//need with philox
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
#endif

int main(){
    __half qp[B*H*Sq*qD + 2*ALIGN];
    __half kp[B*H*Skv*qD + 2*ALIGN];
    __half vp[B*H*Skv*vD + 2*ALIGN];
    __half outp[B*H*Sq*vD + 2*ALIGN];

    __half* q = (__half *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    __half* k = (__half *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    __half* v = (__half *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    __half* out = (__half *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            #ifdef OPT1
            flash_attention_opt1<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT2)
            flash_attention_opt2<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT3)
            flash_attention_opt3<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(MULTI_TILE_OPT3)
            flash_attention_multitile_opt3<__half, Sq, Skv, qD, vD, kTm, kTk, MBUF>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(DCORE)
            flash_attention_dcore<__half, Sq, Skv, qD, vD, kTm, kTk, MBUF>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(_2D_UNROLL)
            flash_attention_2d_unroll<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(_2D_UNROLL_PTO)
            flash_attention_2d_unroll_pto<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT4)
            flash_attention_opt4<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #else
            flash_attention<__half, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #endif
        }
    }
    BENCHEND;
}