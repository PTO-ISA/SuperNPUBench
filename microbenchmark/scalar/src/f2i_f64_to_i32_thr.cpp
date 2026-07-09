#include "scalar_bench.hpp"
// auto-generated: f2i (cv) f64->i32 throughput
int main() {
    double b[16];
    for (int i = 0; i < 16; ++i) b[i] = (double)(i * 0.7 + 1);
    volatile int32_t sink = (int32_t)0;
    BENCHSTART;
    int32_t r = bench_cv<double, int32_t>(b);
    BENCHEND;
    sink = r;
    return 0;
}
