#include "layout_bench.hpp"
#include "benchmark.h"

int main() {
    float a[32*32], c[32*32];
    for (int i = 0; i < 32*32; i++) { a[i] = (float)(i%10*0.1f); c[i] = 0; }
    BENCHSTART;
    trans_fp32_32x32(c, a);
    BENCHEND;
    return 0;
}
