#include "scalar_bench.hpp"
// auto-generated: popc (un) i32 lat
int main() {
    int32_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int32_t)(i * 0.7 + 1); b[i] = (int32_t)(i * 0.3 + 2); }
    volatile int32_t sink = (int32_t)0;
    BENCHSTART;
    int32_t r = bench_latency<int32_t>(a, b, [](auto x,auto y){return (int32_t)__builtin_popcountll((unsigned long long)(x+y));});
    BENCHEND;
    sink = r;
    return 0;
}
