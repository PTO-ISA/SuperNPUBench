#ifndef BLKC_ASSIGN_CAST_COMMON
#define BLKC_ASSIGN_CAST_COMMON BLKC_ASSIGN_PTR
#endif

#include <common/pto_tileop.hpp>
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(MULTI_TILE_OPT3) || defined(DCORE) || defined(FA_DYNAMIC) || defined(FA_DYNAMIC_UNROLL) || defined(_2D_UNROLL) || defined(_UNALIGN_2D_UNROLL) || defined(_TEMPLATE_2D_UNROLL) || defined(_2D_UNROLL_PTO) || defined(MANUAL) || defined(OPT4)
#include "../../../../kernels/fa/fa_fusion.h"
#else
#include "../../../../kernels/fa/flash_attention.hpp"
#endif
#include "src/benchmark.h"

#ifdef RES_CHECK
#include "fileop.h"
#endif

#ifndef FA_DTYPE
#define FA_DTYPE __half
#endif

#ifndef B
#define B 1
#endif

#ifndef H
#define H 1
#endif

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

#ifndef FA_QD
#define FA_QD 64
#endif

#ifndef FA_VD
#define FA_VD 64
#endif

#define qD FA_QD
#define vD FA_VD

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
    using dtype = FA_DTYPE;

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
            #ifdef OPT1
            flash_attention_opt1<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT2)
            flash_attention_opt2<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT3)
            flash_attention_opt3<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(MULTI_TILE_OPT3)
            flash_attention_multitile_opt3<dtype, Sq, Skv, qD, vD, kTm, kTk, MBUF>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(DCORE)
            flash_attention_dcore<dtype, Sq, Skv, qD, vD, kTm, kTk, MBUF>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(FA_DYNAMIC)
            flash_attention_dynamic<dtype, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, Sq, Skv);
            #elif defined(FA_DYNAMIC_UNROLL)
            flash_attention_dynamic_unroll<dtype, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD, Sq, Skv);
            #elif defined(_2D_UNROLL)
            flash_attention_2d_unroll<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(_UNALIGN_2D_UNROLL)
            flash_attention_unalign_2d_unroll<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(_TEMPLATE_2D_UNROLL)
            flash_attention_template_2d_unroll<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(_2D_UNROLL_PTO)
            flash_attention_2d_unroll_pto<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(MANUAL)
            flash_attention_manual<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #elif defined(OPT4)
            flash_attention_opt4<dtype, Sq, Skv, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #else
            flash_attention<Sq, qD, vD, kTm, kTk>(out, q+i*H*Sq*qD+j*Sq*qD, k+i*H*Skv*qD+j*Skv*qD, v+i*H*Skv*vD+j*Skv*vD);
            #endif
        }
    }
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B*H*Sq*vD*sizeof(dtype));
    #endif
}
