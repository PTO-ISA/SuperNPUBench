#include "scalar_bench.hpp"
// auto-generated: max (bin) i64 thr
int main() {
    int64_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int64_t)(i * 0.7 + 1); b[i] = (int64_t)(i * 0.3 + 2); }
    volatile int64_t sink = (int64_t)0;
    BENCHSTART;
    int64_t r = bench_throughput<int64_t>(a, b, [](auto x,auto y){return x<y?y:x;});
    BENCHEND;
    sink = r;
    return 0;
}
