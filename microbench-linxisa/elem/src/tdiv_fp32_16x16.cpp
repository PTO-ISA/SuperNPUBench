#include "elem_bench.hpp"
#include "benchmark.h"

int main() {
    float a[16*16], b[16*16], c[16*16];
    for (int i = 0; i < 16*16; i++) { a[i] = (float)(i%10*0.1f); b[i] = (float)((i+1)%10*0.1f+0.1f); c[i] = 0; }
    BENCHSTART;
    tdiv_fp32_16x16(c, a, b);
    BENCHEND;
    return 0;
}
