#include "scalar_bench.hpp"
// auto-generated: f2i (cv) f32->i32 throughput
int main() {
    float b[16];
    for (int i = 0; i < 16; ++i) b[i] = (float)(i * 0.7 + 1);
    volatile int32_t sink = (int32_t)0;
    BENCHSTART;
    int32_t r = bench_cv<float, int32_t>(b);
    BENCHEND;
    sink = r;
    return 0;
}
