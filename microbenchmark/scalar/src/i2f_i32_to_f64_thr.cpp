#include "scalar_bench.hpp"
// auto-generated: i2f (cv) i32->f64 throughput
int main() {
    int32_t b[16];
    for (int i = 0; i < 16; ++i) b[i] = (int32_t)(i * 0.7 + 1);
    volatile double sink = (double)0;
    BENCHSTART;
    double r = bench_cv<int32_t, double>(b);
    BENCHEND;
    sink = r;
    return 0;
}
