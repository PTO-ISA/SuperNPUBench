#include "scalar_bench.hpp"
#include "benchmark.h"

int main() {
    int32_t a[16*16], c[16*16];
    for (int i = 0; i < 16*16; i++) a[i] = (int32_t)(i%10*0.1f);
    for (int i = 0; i < 16*16; i++) c[i] = 0;
    BENCHSTART;
    tdivs_int32_16x16(c, a, (int32_t)2.0f);
    BENCHEND;
    return 0;
}
