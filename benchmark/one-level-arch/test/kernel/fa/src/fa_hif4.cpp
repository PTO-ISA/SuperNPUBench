#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "single_thread/fa/fa_hif4.hpp"

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
#define kTm 16
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 32
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
    // Match the active multi-thread HIF4 path: packed-x2 matrix operands and
    // BF16 vector/output state. Packed storage halves the contiguous matrix
    // dimension; CUBE descriptors retain the corresponding logical M/N/K.
    using typep = __fp4_hif4x2;
    using out_type = __bf16;
    typep qp[B*H*Sq*(qD/2) + 2*ALIGN];
    typep kp[B*H*Skv*(qD/2) + 2*ALIGN];
    typep vp[B*H*(Skv/2)*vD + 2*ALIGN];
    out_type outp[B*H*Sq*vD + 2*ALIGN];
    // HiF4 Matrix-MX uses one raw U32 scale carrier per 64 logical K
    // elements, matching the current matmul HIF4X2 contract.
    uint32_t qmx[B*H*Sq*(qD/64) + 2*ALIGN];
    uint32_t kmx[B*H*Skv*(qD/64) + 2*ALIGN];
    uint32_t vmx[B*H*(Skv/64)*vD + 2*ALIGN];

    typep* q = (typep *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    typep* k = (typep *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    typep* v = (typep *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    out_type* out = (out_type *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    #ifdef RES_CHECK
    #define SRCQ_PATH CHK_DIR "/srcq.bin"
    #define SRCK_PATH CHK_DIR "/srck.bin"
    #define SRCV_PATH CHK_DIR "/srcv.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t*)q, B*H*Sq*(qD/2)*sizeof(typep));
    readBinaryFile(SRCK_PATH, (uint8_t*)k, B*H*Skv*(qD/2)*sizeof(typep));
    readBinaryFile(SRCV_PATH, (uint8_t*)v, B*H*(Skv/2)*vD*sizeof(typep));
    #endif

    // uint32_t a;
    // uint32_t *b = &a;
    // float *c = (float*)b;
    // c = c + 1;
    // printf("%f\n", *c);

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            typep *q_block = q + i*H*Sq*(qD/2) + j*Sq*(qD/2);
            typep *k_block = k + i*H*Skv*(qD/2) + j*Skv*(qD/2);
            typep *v_block = v + i*H*(Skv/2)*vD + j*(Skv/2)*vD;
            out_type *o_block = out + i*H*Sq*vD + j*Sq*vD;
            #ifdef BF16
                flash_attention_2d_unroll_hif4<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(BF16x2)
                flash_attention_2d_unroll_hif4<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(BF16x2_NOGATHER)
                flash_attention_2d_unroll_hif4_nogather<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(BF16_NOGATHER)
                flash_attention_2d_unroll_hif4_nogather<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(OPT)
                flash_attention_2d_unroll_hif4_optsoftmax<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(OPT_LOAD)
                flash_attention_2d_unroll_hif4_optsoftmax_loadx2<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(OPT_OFFLOAD)
                flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            #elif defined(OPT_OFFLOAD2)
                flash_attention_2d_unroll_hif4_optsoftmax_cubeoffload2<typep, Sq, Skv, qD, vD, kTm, kTk, 16, out_type>(o_block, q_block, k_block, v_block, qmx, kmx, vmx);
            
            #endif
        }
    }
    BENCHEND;

    #ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*)out, B*H*Sq*vD*sizeof(out_type));
    #endif
}
