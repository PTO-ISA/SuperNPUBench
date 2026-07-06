#include "elem_bench.hpp"
#include "benchmark.h"

int main() {
    __half a[16*16], b[16*16], c[16*16];
    for (int i = 0; i < 16*16; i++) { a[i] = (__half)(i%10*0.1f); b[i] = (__half)((i+1)%10*0.1f+0.1f); c[i] = 0; }
    BENCHSTART;
    tmax_fp16_16x16(c, a, b);
    BENCHEND;
    return 0;
}
