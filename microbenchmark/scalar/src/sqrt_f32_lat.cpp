#include "scalar_bench.hpp"
// auto-generated: sqrt (un) f32 lat
int main() {
    float a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (float)(i * 0.7 + 1); b[i] = (float)(i * 0.3 + 2); }
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_latency<float>(a, b, [](auto x,auto y){return (float)std::sqrt((double)(x+y));});
    BENCHEND;
    sink = r;
    return 0;
}
