#include "elem_bench.hpp"
#include "benchmark.h"

int main() {
    int16_t a[16*16], b[16*16], c[16*16];
    for (int i = 0; i < 16*16; i++) { a[i] = (int16_t)(i%10*0.1f); b[i] = (int16_t)((i+1)%10*0.1f+0.1f); c[i] = 0; }
    BENCHSTART;
    tsub_int16_16x16(c, a, b);
    BENCHEND;
    return 0;
}
