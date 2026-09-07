#include "scalar_bench.hpp"
// auto-generated: st (st) i64 throughput
int main() {
    int64_t out[16], val = (int64_t)5;
    for (int i = 0; i < 16; ++i) out[i] = (int64_t)0;
    BENCHSTART;
    bench_store<int64_t>(out, val);
    BENCHEND;
    volatile int64_t sink = out[0];
#ifdef RES_CHECK
    for (int i = 0; i < 16; ++i) if (out[i] != val) return 1;
#endif
    return 0;
}
