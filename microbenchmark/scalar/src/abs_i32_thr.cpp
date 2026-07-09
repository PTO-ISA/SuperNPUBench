#include "scalar_bench.hpp"
// auto-generated: abs (un) i32 thr
int main() {
    int32_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int32_t)(i * 0.7 + 1); b[i] = (int32_t)(i * 0.3 + 2); }
    volatile int32_t sink = (int32_t)0;
    BENCHSTART;
    int32_t r = bench_throughput<int32_t>(a, b, [](auto x,auto y){auto t=x+y; return t<0?-t:t;});
    BENCHEND;
    sink = r;
    return 0;
}
