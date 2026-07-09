#include "scalar_bench.hpp"
// auto-generated: sub (bin) f32 thr
int main() {
    float a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (float)(i * 0.7 + 1); b[i] = (float)(i * 0.3 + 2); }
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_throughput<float>(a, b, [](auto x,auto y){return x-y;});
    BENCHEND;
    sink = r;
    return 0;
}
