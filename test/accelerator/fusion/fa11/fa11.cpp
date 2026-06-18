#include <common/pto_tileop.hpp>
#include "../../include/accelerator_fusion.h"
#include "benchmark.h"

//need with bytemask
#define B 1
#define H 1    //2
#define S 512 //1024
#define D 128

#define kTm 64
#define kTk 64

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    float inputp[B*H*S*S + 2*ALIGN];
    float maxp[B*H*S*1 + 2*ALIGN];
    float sump[B*H*S*1 + 2*ALIGN];
    __half outputp[B*H*S*D + 2*ALIGN];
    float input_scale = 1.0;

    float* input = (float *)(((uint64_t)inputp & ALIGN_MASK) + ALIGN);
    float* max = (float *)(((uint64_t)maxp & ALIGN_MASK) + ALIGN);
    float* sum = (float *)(((uint64_t)sump & ALIGN_MASK) + ALIGN);
    __half* output = (__half *)(((uint64_t)outputp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    #pragma clang loop unroll(full)
    for(int i=0;i<B;i++){
        #pragma clang loop unroll(full)
        for(int j=0;j<H;j++){
            flashsoftmax<S, D, kTm, kTk>(input, max+i*H*S*1+j*S*1, sum+i*H*S*1+j*S*1, &input_scale, nullptr, output+i*H*S*D+j*S*D);
        }
    }
    BENCHEND;
}