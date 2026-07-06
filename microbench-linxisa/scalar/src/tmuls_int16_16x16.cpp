#include "scalar_bench.hpp"
#include "benchmark.h"

int main() {
    int16_t a[16*16], c[16*16];
    for (int i = 0; i < 16*16; i++) a[i] = (int16_t)(i%10*0.1f);
    for (int i = 0; i < 16*16; i++) c[i] = 0;
    BENCHSTART;
    tmuls_int16_16x16(c, a, (int16_t)2.0f);
    BENCHEND;
    return 0;
}
