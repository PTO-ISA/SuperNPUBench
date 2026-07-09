#include "scalar_bench.hpp"
// auto-generated: f2f_widen (cv) f32->f64 throughput
int main() {
    float b[16];
    for (int i = 0; i < 16; ++i) b[i] = (float)(i * 0.7 + 1);
    volatile double sink = (double)0;
    BENCHSTART;
    double r = bench_cv<float, double>(b);
    BENCHEND;
    sink = r;
    return 0;
}
