#include "scalar_bench.hpp"
// auto-generated: abs (un) f64 thr
int main() {
    double a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (double)(i * 0.7 + 1); b[i] = (double)(i * 0.3 + 2); }
    volatile double sink = (double)0;
    BENCHSTART;
    double r = bench_throughput<double>(a, b, [](auto x,auto y){auto t=x+y; return t<0?-t:t;});
    BENCHEND;
    sink = r;
    return 0;
}
