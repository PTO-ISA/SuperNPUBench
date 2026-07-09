#include "scalar_bench.hpp"
// auto-generated: st (st) f64 throughput
int main() {
    double out[16], val = (double)5;
    for (int i = 0; i < 16; ++i) out[i] = (double)0;
    BENCHSTART;
    bench_store<double>(out, val);
    BENCHEND;
    volatile double sink = out[0];
    return 0;
}
