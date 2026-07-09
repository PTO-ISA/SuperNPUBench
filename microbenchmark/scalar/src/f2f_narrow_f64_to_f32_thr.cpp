#include "scalar_bench.hpp"
// auto-generated: f2f_narrow (cv) f64->f32 throughput
int main() {
    double b[16];
    for (int i = 0; i < 16; ++i) b[i] = (double)(i * 0.7 + 1);
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_cv<double, float>(b);
    BENCHEND;
    sink = r;
    return 0;
}
