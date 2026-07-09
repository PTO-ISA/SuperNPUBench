#include "scalar_bench.hpp"
// auto-generated: i2f (cv) i32->f32 throughput
int main() {
    int32_t b[16];
    for (int i = 0; i < 16; ++i) b[i] = (int32_t)(i * 0.7 + 1);
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_cv<int32_t, float>(b);
    BENCHEND;
    sink = r;
    return 0;
}
