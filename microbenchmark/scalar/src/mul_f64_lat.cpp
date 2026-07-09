#include "scalar_bench.hpp"
// auto-generated: mul (bin) f64 lat
int main() {
    double a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (double)(i * 0.7 + 1); b[i] = (double)(i * 0.3 + 2); }
    volatile double sink = (double)0;
    BENCHSTART;
    double r = bench_latency<double>(a, b, [](auto x,auto y){return x*y;});
    BENCHEND;
    sink = r;
    return 0;
}
